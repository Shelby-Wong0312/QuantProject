"""
Moving Average Crossover Strategy
移動平均交叉策略 - 經典技術分析策略
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from ..base_strategy import BaseStrategy
from ..strategy_interface import (
    TradingSignal, SignalType, StrategyConfig, StrategyPerformance, 
    Position, RiskMetrics, StrategyStatus
)
from ...indicators.trend_indicators import SMA, EMA

logger = logging.getLogger(__name__)


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    移動平均交叉策略
    
    策略邏輯：
    1. 使用快線(短期MA)和慢線(長期MA)
    2. 快線上穿慢線 = 買入信號
    3. 快線下穿慢線 = 賣出信號
    4. 支援多種MA類型：SMA、EMA
    5. 整合風險管理和濾波器
    
    適用範圍：
    - 趨勢跟蹤
    - 中長期交易
    - 4000+股票批量處理
    """
    
    def __init__(self, config: StrategyConfig, initial_capital: float = 100000):
        """
        初始化移動平均交叉策略
        
        Args:
            config: 策略配置
            initial_capital: 初始資金
        """
        super().__init__(config, initial_capital)
        
        # 策略狀態
        self.fast_ma_indicator = None
        self.slow_ma_indicator = None
        self.previous_fast_ma = {}
        self.previous_slow_ma = {}
        self.previous_signals = {}
        
        logger.info(f"MovingAverageCrossoverStrategy initialized: {self.name}")
    
    def _initialize_parameters(self) -> None:
        """
        初始化策略參數
        """
        # 默認參數
        default_params = {
            'fast_period': 20,        # 快線週期
            'slow_period': 50,        # 慢線週期
            'ma_type': 'SMA',         # MA類型: SMA, EMA
            'volume_filter': True,    # 成交量過濾
            'volume_multiplier': 1.2, # 成交量倍數
            'trend_filter': True,     # 趨勢過濾
            'min_trend_strength': 0.6, # 最小趨勢強度
            'signal_strength_threshold': 0.7,  # 信號強度閾值
            'exit_on_reverse_cross': True,     # 反向交叉出場
            'atr_position_sizing': True,       # 基於ATR的倉位管理
            'max_position_pct': 0.02          # 最大單倉位比例
        }
        
        # 合併用戶參數
        self.params = {**default_params, **self.config.parameters}
        
        # 初始化MA指標
        if self.params['ma_type'].upper() == 'EMA':
            self.fast_ma_indicator = EMA(period=self.params['fast_period'])
            self.slow_ma_indicator = EMA(period=self.params['slow_period'])
        else:
            self.fast_ma_indicator = SMA(period=self.params['fast_period'])
            self.slow_ma_indicator = SMA(period=self.params['slow_period'])
        
        logger.info(f"Strategy parameters: {self.params}")
    
    def calculate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        計算交易信號
        
        Args:
            data: OHLCV數據
            
        Returns:
            交易信號列表
        """
        signals = []
        
        try:
            if len(data) < max(self.params['fast_period'], self.params['slow_period']) + 10:
                logger.warning(f"Insufficient data for MA calculation: {len(data)} bars")
                return signals
            
            # 計算移動平均線
            fast_ma_data = self.fast_ma_indicator.calculate(data)
            slow_ma_data = self.slow_ma_indicator.calculate(data)
            
            if fast_ma_data.empty or slow_ma_data.empty:
                return signals
            
            # 獲取最新的MA值
            fast_ma_col = fast_ma_data.columns[0] if len(fast_ma_data.columns) > 0 else f"{self.params['ma_type']}_{self.params['fast_period']}"
            slow_ma_col = slow_ma_data.columns[0] if len(slow_ma_data.columns) > 0 else f"{self.params['ma_type']}_{self.params['slow_period']}"
            
            fast_ma = fast_ma_data[fast_ma_col].iloc[-1]
            slow_ma = slow_ma_data[slow_ma_col].iloc[-1]
            
            # 檢查是否有足夠的歷史數據進行交叉判斷
            if len(fast_ma_data) < 2 or len(slow_ma_data) < 2:
                return signals
            
            prev_fast_ma = fast_ma_data[fast_ma_col].iloc[-2]
            prev_slow_ma = slow_ma_data[slow_ma_col].iloc[-2]
            
            # 檢測交叉
            current_price = data['close'].iloc[-1]
            timestamp = pd.Timestamp.now()
            
            # 多頭交叉：快線上穿慢線
            if prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma:
                if self._validate_buy_signal(data, fast_ma, slow_ma):
                    signal_strength = self._calculate_signal_strength(data, 'BUY', fast_ma, slow_ma)
                    
                    if signal_strength >= self.params['signal_strength_threshold']:
                        signal = TradingSignal(
                            symbol=data.index.name or 'UNKNOWN',
                            signal_type=SignalType.BUY,
                            strength=signal_strength,
                            strategy_name=self.name,
                            timestamp=timestamp,
                            price=current_price,
                            metadata={
                                'fast_ma': fast_ma,
                                'slow_ma': slow_ma,
                                'prev_fast_ma': prev_fast_ma,
                                'prev_slow_ma': prev_slow_ma,
                                'crossover_type': 'golden_cross',
                                'trend_strength': self._calculate_trend_strength(data),
                                'volume_ratio': self._calculate_volume_ratio(data)
                            }
                        )
                        signals.append(signal)
            
            # 空頭交叉：快線下穿慢線
            elif prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma:
                if self._validate_sell_signal(data, fast_ma, slow_ma):
                    signal_strength = self._calculate_signal_strength(data, 'SELL', fast_ma, slow_ma)
                    
                    if signal_strength >= self.params['signal_strength_threshold']:
                        signal = TradingSignal(
                            symbol=data.index.name or 'UNKNOWN',
                            signal_type=SignalType.SELL,
                            strength=signal_strength,
                            strategy_name=self.name,
                            timestamp=timestamp,
                            price=current_price,
                            metadata={
                                'fast_ma': fast_ma,
                                'slow_ma': slow_ma,
                                'prev_fast_ma': prev_fast_ma,
                                'prev_slow_ma': prev_slow_ma,
                                'crossover_type': 'death_cross',
                                'trend_strength': self._calculate_trend_strength(data),
                                'volume_ratio': self._calculate_volume_ratio(data)
                            }
                        )
                        signals.append(signal)
        
        except Exception as e:
            logger.error(f"Error calculating MA crossover signals: {e}")
        
        return signals
    
    def _validate_buy_signal(self, data: pd.DataFrame, fast_ma: float, slow_ma: float) -> bool:
        """
        驗證買入信號
        
        Args:
            data: 市場數據
            fast_ma: 快線值
            slow_ma: 慢線值
            
        Returns:
            是否有效的買入信號
        """
        try:
            # 基本交叉驗證
            if fast_ma <= slow_ma:
                return False
            
            # 成交量過濾
            if self.params['volume_filter']:
                volume_ratio = self._calculate_volume_ratio(data)
                if volume_ratio < self.params['volume_multiplier']:
                    logger.debug("Buy signal filtered: insufficient volume")
                    return False
            
            # 趨勢過濾
            if self.params['trend_filter']:
                trend_strength = self._calculate_trend_strength(data)
                if trend_strength < self.params['min_trend_strength']:
                    logger.debug("Buy signal filtered: weak trend")
                    return False
            
            # 價格位置過濾（價格應該接近或高於慢線）
            current_price = data['close'].iloc[-1]
            if current_price < slow_ma * 0.98:  # 2%容忍度
                logger.debug("Buy signal filtered: price too far below slow MA")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating buy signal: {e}")
            return False
    
    def _validate_sell_signal(self, data: pd.DataFrame, fast_ma: float, slow_ma: float) -> bool:
        """
        驗證賣出信號
        
        Args:
            data: 市場數據
            fast_ma: 快線值
            slow_ma: 慢線值
            
        Returns:
            是否有效的賣出信號
        """
        try:
            # 基本交叉驗證
            if fast_ma >= slow_ma:
                return False
            
            # 成交量過濾
            if self.params['volume_filter']:
                volume_ratio = self._calculate_volume_ratio(data)
                if volume_ratio < self.params['volume_multiplier']:
                    logger.debug("Sell signal filtered: insufficient volume")
                    return False
            
            # 下跌趨勢確認
            if self.params['trend_filter']:
                trend_strength = self._calculate_trend_strength(data)
                # 對於賣出信號，我們要求負的趨勢強度
                if trend_strength > -self.params['min_trend_strength']:
                    logger.debug("Sell signal filtered: insufficient downtrend")
                    return False
            
            # 價格位置過濾（價格應該接近或低於慢線）
            current_price = data['close'].iloc[-1]
            if current_price > slow_ma * 1.02:  # 2%容忍度
                logger.debug("Sell signal filtered: price too far above slow MA")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating sell signal: {e}")
            return False
    
    def _calculate_signal_strength(self, data: pd.DataFrame, signal_type: str, 
                                 fast_ma: float, slow_ma: float) -> float:
        """
        計算信號強度
        
        Args:
            data: 市場數據
            signal_type: 信號類型
            fast_ma: 快線值
            slow_ma: 慢線值
            
        Returns:
            信號強度 (0-1)
        """
        try:
            strength_factors = []
            
            # 1. MA分離度 (0-0.3)
            ma_separation = abs(fast_ma - slow_ma) / slow_ma
            separation_strength = min(0.3, ma_separation * 100)
            strength_factors.append(separation_strength)
            
            # 2. 趨勢強度 (0-0.3)
            trend_strength = abs(self._calculate_trend_strength(data))
            trend_factor = min(0.3, trend_strength)
            strength_factors.append(trend_factor)
            
            # 3. 成交量確認 (0-0.2)
            volume_ratio = self._calculate_volume_ratio(data)
            volume_strength = min(0.2, (volume_ratio - 1) * 0.2)
            strength_factors.append(max(0, volume_strength))
            
            # 4. 價格動量 (0-0.2)
            momentum_strength = self._calculate_momentum_strength(data, signal_type)
            strength_factors.append(momentum_strength)
            
            # 基礎強度
            total_strength = 0.6 + sum(strength_factors)
            
            return min(1.0, max(0.0, total_strength))
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            return 0.5
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """
        計算趨勢強度
        
        Args:
            data: 市場數據
            
        Returns:
            趨勢強度 (-1 到 1)
        """
        try:
            if len(data) < 20:
                return 0
            
            # 使用價格變化率計算趨勢
            prices = data['close'].tail(20)
            x = np.arange(len(prices))
            
            # 線性回歸斜率
            slope = np.polyfit(x, prices, 1)[0]
            
            # 標準化斜率
            avg_price = prices.mean()
            normalized_slope = slope / avg_price
            
            # 限制在 -1 到 1 之間
            return np.clip(normalized_slope * 100, -1, 1)
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0
    
    def _calculate_volume_ratio(self, data: pd.DataFrame) -> float:
        """
        計算成交量比率
        
        Args:
            data: 市場數據
            
        Returns:
            成交量比率
        """
        try:
            if 'volume' not in data.columns or len(data) < 20:
                return 1.0
            
            current_volume = data['volume'].iloc[-1]
            avg_volume = data['volume'].tail(20).mean()
            
            if avg_volume == 0:
                return 1.0
            
            return current_volume / avg_volume
            
        except Exception as e:
            logger.error(f"Error calculating volume ratio: {e}")
            return 1.0
    
    def _calculate_momentum_strength(self, data: pd.DataFrame, signal_type: str) -> float:
        """
        計算動量強度
        
        Args:
            data: 市場數據
            signal_type: 信號類型
            
        Returns:
            動量強度 (0-0.2)
        """
        try:
            if len(data) < 5:
                return 0
            
            # 計算短期動量
            prices = data['close'].tail(5)
            momentum = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
            
            # 根據信號類型調整
            if signal_type == 'BUY':
                return max(0, min(0.2, momentum * 2))
            else:
                return max(0, min(0.2, -momentum * 2))
                
        except Exception as e:
            logger.error(f"Error calculating momentum strength: {e}")
            return 0
    
    def get_position_size(self, signal: TradingSignal, portfolio_value: float, 
                         current_price: float) -> float:
        """
        計算倉位大小
        
        Args:
            signal: 交易信號
            portfolio_value: 組合價值
            current_price: 當前價格
            
        Returns:
            倉位大小
        """
        try:
            # 基礎倉位計算
            base_position_value = portfolio_value * self.params['max_position_pct']
            base_shares = base_position_value / current_price
            
            # 根據信號強度調整
            strength_multiplier = signal.strength
            adjusted_shares = base_shares * strength_multiplier
            
            # 根據信號類型決定方向
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                return adjusted_shares
            elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                return -adjusted_shares
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def apply_risk_management(self, position: Position, 
                            market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        應用風險管理
        
        Args:
            position: 當前倉位
            market_data: 市場數據
            
        Returns:
            風險管理動作
        """
        action = {
            'action': 'hold',
            'new_size': position.size,
            'reason': 'No action required',
            'stop_loss': position.stop_loss,
            'take_profit': position.take_profit
        }
        
        try:
            current_price = market_data['close'].iloc[-1]
            
            # 反向交叉出場
            if self.params['exit_on_reverse_cross']:
                signals = self.calculate_signals(market_data)
                for signal in signals:
                    if signal.symbol == position.symbol:
                        # 如果是多頭倉位收到賣出信號
                        if position.size > 0 and signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                            action.update({
                                'action': 'close',
                                'new_size': 0,
                                'reason': 'Reverse crossover exit'
                            })
                            break
                        # 如果是空頭倉位收到買入信號
                        elif position.size < 0 and signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                            action.update({
                                'action': 'close',
                                'new_size': 0,
                                'reason': 'Reverse crossover exit'
                            })
                            break
            
            # 止損檢查
            if position.stop_loss is not None:
                if (position.size > 0 and current_price <= position.stop_loss) or \
                   (position.size < 0 and current_price >= position.stop_loss):
                    action.update({
                        'action': 'close',
                        'new_size': 0,
                        'reason': 'Stop loss triggered'
                    })
            
            # 止盈檢查
            if position.take_profit is not None:
                if (position.size > 0 and current_price >= position.take_profit) or \
                   (position.size < 0 and current_price <= position.take_profit):
                    action.update({
                        'action': 'close',
                        'new_size': 0,
                        'reason': 'Take profit triggered'
                    })
            
        except Exception as e:
            logger.error(f"Error in risk management: {e}")
        
        return action
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """
        獲取策略摘要
        
        Returns:
            策略摘要字典
        """
        summary = self.get_strategy_info()
        
        # 添加MA特定信息
        summary.update({
            'strategy_type': 'Moving Average Crossover',
            'fast_period': self.params['fast_period'],
            'slow_period': self.params['slow_period'],
            'ma_type': self.params['ma_type'],
            'filters_enabled': {
                'volume_filter': self.params['volume_filter'],
                'trend_filter': self.params['trend_filter']
            },
            'risk_management': {
                'exit_on_reverse_cross': self.params['exit_on_reverse_cross'],
                'max_position_pct': self.params['max_position_pct']
            }
        })
        
        return summary


if __name__ == "__main__":
    print("Moving Average Crossover Strategy")
    print("=" * 50)
    
    # 測試策略
    config = StrategyConfig(
        name="MA_Crossover_Test",
        parameters={
            'fast_period': 20,
            'slow_period': 50,
            'ma_type': 'SMA'
        }
    )
    
    strategy = MovingAverageCrossoverStrategy(config)
    
    # 生成測試數據
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    np.random.seed(42)
    
    # 模擬有趨勢的價格數據
    price_base = 100
    price_trend = np.cumsum(np.random.randn(200) * 0.02 + 0.001)
    prices = price_base + price_trend * 10
    
    test_data = pd.DataFrame({
        'open': prices + np.random.randn(200) * 0.5,
        'high': prices + np.abs(np.random.randn(200)) * 1.0,
        'low': prices - np.abs(np.random.randn(200)) * 1.0,
        'close': prices,
        'volume': np.random.randint(1000, 5000, 200)
    }, index=dates)
    
    test_data.index.name = 'AAPL'
    
    # 計算信號
    signals = strategy.calculate_signals(test_data)
    
    print(f"Generated {len(signals)} signals:")
    for signal in signals:
        print(f"  {signal.timestamp.strftime('%Y-%m-%d')}: {signal.signal_type.value} "
              f"(strength: {signal.strength:.3f}, price: {signal.price:.2f})")
    
    # 策略摘要
    summary = strategy.get_strategy_summary()
    print(f"\nStrategy Summary:")
    print(f"  Type: {summary['strategy_type']}")
    print(f"  Fast Period: {summary['fast_period']}")
    print(f"  Slow Period: {summary['slow_period']}")
    print(f"  MA Type: {summary['ma_type']}")
    
    print("\n✓ Moving Average Crossover Strategy ready for production use!")