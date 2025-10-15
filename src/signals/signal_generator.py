"""
Real-time Trading Signal Generator
實時交易信號生成引擎
Cloud DE - Task RT-001
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import json
from pathlib import Path
import logging
from enum import Enum
import warnings

warnings.filterwarnings("ignore")

# Import internal modules
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rl_trading.ppo_agent import ActorCritic
from src.rl_trading.trading_env import TradingEnvironment
from src.ml_models.lstm_price_predictor import LSTMPricePredictor
from src.indicators.advanced_indicators import AdvancedIndicators

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """信號強度等級"""

    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class TradingSignal:
    """交易信號數據結構"""

    timestamp: datetime
    symbol: str
    action: str  # BUY, SELL, HOLD
    strength: float  # 0-100
    confidence: float  # 0-1
    price: float
    predicted_return: float
    stop_loss: float
    take_profit: float
    risk_score: float
    sources: Dict[str, Any]  # 各模型貢獻
    metadata: Dict[str, Any]


class SignalGenerator:
    """
    實時信號生成器

    整合多個模型和指標生成交易信號
    """

    def __init__(
        self,
        ppo_model_path: str = "reports/ml_models/ppo_trader_final.pt",
        lstm_model_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        """
        初始化信號生成器

        Args:
            ppo_model_path: PPO 模型路徑
            lstm_model_path: LSTM 模型路徑
            config_path: 配置文件路徑
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 載入配置
        self.config = self._load_config(config_path)

        # 載入模型
        self.ppo_model = self._load_ppo_model(ppo_model_path)
        self.lstm_predictor = LSTMPricePredictor() if lstm_model_path else None

        # 技術指標計算器
        self.indicator_calculator = AdvancedIndicators()

        # 信號歷史
        self.signal_history = []

        # 權重配置
        self.weights = {"ppo": 0.35, "lstm": 0.25, "technical": 0.25, "momentum": 0.15}

        logger.info(f"Signal Generator initialized on {self.device}")

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """載入配置"""
        default_config = {
            "min_confidence": 0.6,
            "risk_limit": 0.02,  # 2% 風險限制
            "stop_loss_pct": 0.02,  # 2% 止損
            "take_profit_pct": 0.05,  # 5% 止盈
            "signal_threshold": 60,  # 信號強度閾值
            "max_position_size": 0.1,  # 最大倉位 10%
            "indicators": ["CCI", "RSI", "MACD", "BB", "Volume"],
        }

        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                custom_config = json.load(f)
                default_config.update(custom_config)

        return default_config

    def _load_ppo_model(self, model_path: str) -> Optional[ActorCritic]:
        """載入 PPO 模型"""
        try:
            if not Path(model_path).exists():
                logger.warning(f"PPO model not found at {model_path}")
                return None

            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            # 重建模型
            obs_dim = 11 * 20  # window_size * features
            action_dim = 4  # HOLD, BUY, SELL, CLOSE

            model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
            model.eval()

            logger.info("PPO model loaded successfully")
            return model

        except Exception as e:
            logger.error(f"Failed to load PPO model: {e}")
            return None

    def generate_signal(
        self, data: pd.DataFrame, symbol: str, current_position: float = 0
    ) -> TradingSignal:
        """
        生成交易信號

        Args:
            data: OHLCV 數據
            symbol: 股票代碼
            current_position: 當前持倉

        Returns:
            交易信號
        """
        # 確保數據足夠
        if len(data) < 100:
            return self._create_neutral_signal(symbol, data.iloc[-1]["close"])

        # 1. PPO 模型預測
        ppo_signal = self._get_ppo_signal(data, current_position)

        # 2. LSTM 價格預測
        lstm_signal = (
            self._get_lstm_signal(data) if self.lstm_predictor else {"action": "HOLD", "score": 50}
        )

        # 3. 技術指標信號
        technical_signal = self._get_technical_signal(data)

        # 4. 動量信號
        momentum_signal = self._get_momentum_signal(data)

        # 5. 整合信號
        final_signal = self._integrate_signals(
            {
                "ppo": ppo_signal,
                "lstm": lstm_signal,
                "technical": technical_signal,
                "momentum": momentum_signal,
            }
        )

        # 6. 計算風險指標
        risk_metrics = self._calculate_risk_metrics(data, final_signal)

        # 7. 生成最終信號
        current_price = float(data.iloc[-1]["close"])

        signal = TradingSignal(
            timestamp=datetime.now(),
            symbol=symbol,
            action=final_signal["action"],
            strength=final_signal["strength"],
            confidence=final_signal["confidence"],
            price=current_price,
            predicted_return=final_signal.get("predicted_return", 0),
            stop_loss=risk_metrics["stop_loss"],
            take_profit=risk_metrics["take_profit"],
            risk_score=risk_metrics["risk_score"],
            sources={
                "ppo": ppo_signal,
                "lstm": lstm_signal,
                "technical": technical_signal,
                "momentum": momentum_signal,
            },
            metadata={
                "volatility": risk_metrics["volatility"],
                "sharpe_ratio": risk_metrics.get("sharpe_ratio", 0),
                "max_drawdown": risk_metrics.get("max_drawdown", 0),
            },
        )

        # 記錄信號
        self.signal_history.append(signal)

        return signal

    def _get_ppo_signal(self, data: pd.DataFrame, current_position: float) -> Dict:
        """獲取 PPO 模型信號"""
        if self.ppo_model is None:
            return {"action": "HOLD", "score": 50, "confidence": 0.5}

        try:
            # 準備觀察數據
            obs = self._prepare_ppo_observation(data, current_position)
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            obs_tensor = obs_tensor.unsqueeze(0)

            # 獲取動作概率
            with torch.no_grad():
                action_logits, value = self.ppo_model(obs_tensor)
                probs = torch.softmax(action_logits, dim=-1)
                action = torch.argmax(probs, dim=-1).item()
                confidence = float(probs[0, action])

            # 轉換動作到信號
            action_map = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE"}
            signal_action = action_map[action]

            # 計算信號強度
            if signal_action == "BUY":
                score = 50 + confidence * 50
            elif signal_action == "SELL":
                score = 50 - confidence * 50
            else:
                score = 50

            return {
                "action": signal_action,
                "score": score,
                "confidence": confidence,
                "value": float(value),
            }

        except Exception as e:
            logger.error(f"PPO signal generation failed: {e}")
            return {"action": "HOLD", "score": 50, "confidence": 0.5}

    def _get_lstm_signal(self, data: pd.DataFrame) -> Dict:
        """獲取 LSTM 預測信號"""
        try:
            # 預測未來價格
            future_prices = self.lstm_predictor.predict(data)
            current_price = data.iloc[-1]["close"]

            # 計算預期收益
            expected_return = (future_prices[-1] - current_price) / current_price

            # 生成信號
            if expected_return > 0.02:  # 2% 上漲
                action = "BUY"
                score = min(100, 50 + expected_return * 1000)
            elif expected_return < -0.02:  # 2% 下跌
                action = "SELL"
                score = max(0, 50 + expected_return * 1000)
            else:
                action = "HOLD"
                score = 50

            return {
                "action": action,
                "score": score,
                "expected_return": expected_return,
                "confidence": min(1.0, abs(expected_return) * 20),
            }

        except Exception as e:
            logger.error(f"LSTM signal generation failed: {e}")
            return {"action": "HOLD", "score": 50, "confidence": 0.5}

    def _get_technical_signal(self, data: pd.DataFrame) -> Dict:
        """獲取技術指標信號"""
        try:
            # 計算各種指標
            indicators = self.indicator_calculator.calculate_all(data)

            signals = []

            # CCI 信號
            cci = indicators.get("CCI_20", 0)
            if cci > 100:
                signals.append(("SELL", abs(cci - 100) / 100))
            elif cci < -100:
                signals.append(("BUY", abs(cci + 100) / 100))

            # RSI 信號
            rsi = indicators.get("RSI_14", 50)
            if rsi > 70:
                signals.append(("SELL", (rsi - 70) / 30))
            elif rsi < 30:
                signals.append(("BUY", (30 - rsi) / 30))

            # MACD 信號
            macd = indicators.get("MACD", 0)
            macd_signal = indicators.get("MACD_Signal", 0)
            if macd > macd_signal:
                signals.append(("BUY", min(1, abs(macd - macd_signal) * 100)))
            elif macd < macd_signal:
                signals.append(("SELL", min(1, abs(macd - macd_signal) * 100)))

            # 整合信號
            if not signals:
                return {"action": "HOLD", "score": 50, "confidence": 0.5}

            buy_signals = [s for s in signals if s[0] == "BUY"]
            sell_signals = [s for s in signals if s[0] == "SELL"]

            if len(buy_signals) > len(sell_signals):
                action = "BUY"
                confidence = np.mean([s[1] for s in buy_signals])
                score = 50 + confidence * 50
            elif len(sell_signals) > len(buy_signals):
                action = "SELL"
                confidence = np.mean([s[1] for s in sell_signals])
                score = 50 - confidence * 50
            else:
                action = "HOLD"
                score = 50
                confidence = 0.5

            return {
                "action": action,
                "score": score,
                "confidence": confidence,
                "indicators": indicators,
            }

        except Exception as e:
            logger.error(f"Technical signal generation failed: {e}")
            return {"action": "HOLD", "score": 50, "confidence": 0.5}

    def _get_momentum_signal(self, data: pd.DataFrame) -> Dict:
        """獲取動量信號"""
        try:
            # 計算短期和長期動量
            returns_5d = (
                (data["close"].iloc[-1] / data["close"].iloc[-5] - 1) if len(data) >= 5 else 0
            )
            returns_20d = (
                (data["close"].iloc[-1] / data["close"].iloc[-20] - 1) if len(data) >= 20 else 0
            )

            # 成交量動量
            volume_ratio = (
                data["volume"].iloc[-5:].mean() / data["volume"].iloc[-20:].mean()
                if len(data) >= 20
                else 1
            )

            # 生成信號
            momentum_score = returns_5d * 100 + returns_20d * 50

            if momentum_score > 5 and volume_ratio > 1.2:
                action = "BUY"
                score = min(100, 50 + momentum_score)
            elif momentum_score < -5 and volume_ratio > 1.2:
                action = "SELL"
                score = max(0, 50 + momentum_score)
            else:
                action = "HOLD"
                score = 50

            return {
                "action": action,
                "score": score,
                "confidence": min(1.0, abs(momentum_score) / 10),
                "momentum": momentum_score,
                "volume_ratio": volume_ratio,
            }

        except Exception as e:
            logger.error(f"Momentum signal generation failed: {e}")
            return {"action": "HOLD", "score": 50, "confidence": 0.5}

    def _integrate_signals(self, signals: Dict[str, Dict]) -> Dict:
        """整合多個信號源"""
        # 加權平均分數
        weighted_score = 0
        total_weight = 0

        for source, weight in self.weights.items():
            if source in signals and signals[source]:
                weighted_score += signals[source].get("score", 50) * weight
                total_weight += weight

        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = 50

        # 決定最終動作
        if final_score > self.config["signal_threshold"]:
            action = "BUY"
            strength = final_score
        elif final_score < (100 - self.config["signal_threshold"]):
            action = "SELL"
            strength = 100 - final_score
        else:
            action = "HOLD"
            strength = 50

        # 計算信心度
        confidences = [s.get("confidence", 0.5) for s in signals.values() if s]
        confidence = np.mean(confidences) if confidences else 0.5

        # 如果信心度太低，改為 HOLD
        if confidence < self.config["min_confidence"]:
            action = "HOLD"
            strength = 50

        return {
            "action": action,
            "strength": strength,
            "confidence": confidence,
            "predicted_return": signals.get("lstm", {}).get("expected_return", 0),
        }

    def _calculate_risk_metrics(self, data: pd.DataFrame, signal: Dict) -> Dict:
        """計算風險指標"""
        current_price = float(data.iloc[-1]["close"])

        # 計算歷史波動率
        returns = data["close"].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # 年化波動率

        # ATR 止損
        high_low = data["high"] - data["low"]
        high_close = np.abs(data["high"] - data["close"].shift())
        low_close = np.abs(data["low"] - data["close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]

        # 設置止損止盈
        if signal["action"] == "BUY":
            stop_loss = current_price - 2 * atr
            take_profit = current_price + 3 * atr
        elif signal["action"] == "SELL":
            stop_loss = current_price + 2 * atr
            take_profit = current_price - 3 * atr
        else:
            stop_loss = current_price * (1 - self.config["stop_loss_pct"])
            take_profit = current_price * (1 + self.config["take_profit_pct"])

        # 風險評分
        risk_score = min(100, volatility * 100)

        # 夏普比率
        if returns.std() > 0:
            sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0

        # 最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_score": risk_score,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "atr": atr,
        }

    def _prepare_ppo_observation(self, data: pd.DataFrame, current_position: float) -> np.ndarray:
        """準備 PPO 模型的觀察數據"""
        # 獲取最近 20 個時間步的數據
        window_size = 20
        recent_data = data.tail(window_size)

        # 計算特徵
        features = []
        for _, row in recent_data.iterrows():
            # 價格特徵
            features.extend(
                [
                    row["open"] / row["close"],
                    row["high"] / row["close"],
                    row["low"] / row["close"],
                    row["volume"] / data["volume"].mean(),
                ]
            )

        # 添加技術指標
        indicators = self.indicator_calculator.calculate_all(data)
        for key in ["RSI_14", "CCI_20", "MACD"]:
            if key in indicators:
                features.append(indicators[key] / 100)  # 標準化

        # 添加持倉信息
        features.extend(
            [
                current_position / 100,  # 標準化持倉
                1.0,  # 現金比例（簡化）
                0.0,  # 未實現盈虧（簡化）
                len(data) / 1000,  # 時間步標準化
            ]
        )

        # 填充到固定長度
        target_length = 11 * 20  # PPO 模型期望的輸入維度
        if len(features) < target_length:
            features.extend([0] * (target_length - len(features)))
        elif len(features) > target_length:
            features = features[:target_length]

        return np.array(features, dtype=np.float32)

    def _create_neutral_signal(self, symbol: str, price: float) -> TradingSignal:
        """創建中性信號"""
        return TradingSignal(
            timestamp=datetime.now(),
            symbol=symbol,
            action="HOLD",
            strength=50,
            confidence=0.5,
            price=price,
            predicted_return=0,
            stop_loss=price * 0.98,
            take_profit=price * 1.02,
            risk_score=50,
            sources={},
            metadata={},
        )

    def get_signal_history(
        self, symbol: Optional[str] = None, days: int = 7
    ) -> List[TradingSignal]:
        """獲取歷史信號"""
        cutoff_date = datetime.now() - timedelta(days=days)

        filtered_signals = [s for s in self.signal_history if s.timestamp >= cutoff_date]

        if symbol:
            filtered_signals = [s for s in filtered_signals if s.symbol == symbol]

        return filtered_signals

    def evaluate_performance(self) -> Dict[str, float]:
        """評估信號性能"""
        if len(self.signal_history) < 10:
            return {"status": "insufficient_data"}

        # 計算準確率
        correct_signals = 0
        total_actionable = 0

        for signal in self.signal_history:
            if signal.action != "HOLD":
                total_actionable += 1
                # 這裡簡化評估邏輯
                if signal.predicted_return > 0 and signal.action == "BUY":
                    correct_signals += 1
                elif signal.predicted_return < 0 and signal.action == "SELL":
                    correct_signals += 1

        accuracy = correct_signals / total_actionable if total_actionable > 0 else 0

        # 計算平均信心度
        avg_confidence = np.mean([s.confidence for s in self.signal_history])

        # 計算信號分佈
        action_counts = {}
        for signal in self.signal_history:
            action_counts[signal.action] = action_counts.get(signal.action, 0) + 1

        return {
            "total_signals": len(self.signal_history),
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "action_distribution": action_counts,
            "avg_strength": np.mean([s.strength for s in self.signal_history]),
        }


class SignalWebSocketServer:
    """
    WebSocket 信號推送服務器
    """

    def __init__(self, signal_generator: SignalGenerator, port: int = 8765):
        """
        初始化 WebSocket 服務器

        Args:
            signal_generator: 信號生成器
            port: 服務器端口
        """
        self.signal_generator = signal_generator
        self.port = port
        self.clients = set()
        self.is_running = False

    async def handler(self, websocket, path):
        """處理 WebSocket 連接"""
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)

    async def broadcast_signals(self, symbols: List[str], interval: int = 60):
        """
        廣播交易信號

        Args:
            symbols: 股票列表
            interval: 更新間隔（秒）
        """
        while self.is_running:
            for symbol in symbols:
                # 獲取最新數據（這裡簡化處理）
                # 實際應從數據源獲取
                data = self._get_latest_data(symbol)

                if data is not None:
                    # 生成信號
                    signal = self.signal_generator.generate_signal(data, symbol)

                    # 準備消息
                    message = {
                        "type": "signal",
                        "data": {
                            "symbol": signal.symbol,
                            "action": signal.action,
                            "strength": signal.strength,
                            "confidence": signal.confidence,
                            "price": signal.price,
                            "stop_loss": signal.stop_loss,
                            "take_profit": signal.take_profit,
                            "timestamp": signal.timestamp.isoformat(),
                        },
                    }

                    # 廣播給所有客戶端
                    if self.clients:
                        await asyncio.gather(
                            *[client.send(json.dumps(message)) for client in self.clients],
                            return_exceptions=True,
                        )

            await asyncio.sleep(interval)

    def _get_latest_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """獲取最新數據（簡化版）"""
        # 實際實現應從數據源獲取
        # 這裡返回 None 作為示例
        return None

    async def start(self, symbols: List[str]):
        """啟動服務器"""
        import websockets

        self.is_running = True

        async with websockets.serve(self.handler, "localhost", self.port):
            logger.info(f"WebSocket server started on port {self.port}")
            await self.broadcast_signals(symbols)

    def stop(self):
        """停止服務器"""
        self.is_running = False


if __name__ == "__main__":
    print("Real-time Signal Generator - Cloud DE Task RT-001")
    print("=" * 50)

    # 初始化信號生成器
    generator = SignalGenerator()

    # 生成測試數據
    dates = pd.date_range(start="2024-01-01", periods=100, freq="5min")
    test_data = pd.DataFrame(
        {
            "open": np.random.randn(100).cumsum() + 100,
            "high": np.random.randn(100).cumsum() + 101,
            "low": np.random.randn(100).cumsum() + 99,
            "close": np.random.randn(100).cumsum() + 100,
            "volume": np.random.randint(10000, 100000, 100),
        },
        index=dates,
    )

    # 生成信號
    signal = generator.generate_signal(test_data, "TEST")

    print(f"\nGenerated Signal:")
    print(f"  Symbol: {signal.symbol}")
    print(f"  Action: {signal.action}")
    print(f"  Strength: {signal.strength:.2f}")
    print(f"  Confidence: {signal.confidence:.2%}")
    print(f"  Stop Loss: ${signal.stop_loss:.2f}")
    print(f"  Take Profit: ${signal.take_profit:.2f}")

    print("\n✓ Signal Generator ready for real-time trading!")
