"""
Trading Environment for Reinforcement Learning
強化學習交易環境 - OpenAI Gym 標準實現
Cloud DE - Task DT-001
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import IntEnum

logger = logging.getLogger(__name__)


class Actions(IntEnum):
    """交易動作定義"""

    HOLD = 0  # 持有/不動作
    BUY = 1  # 買入
    SELL = 2  # 賣出
    CLOSE = 3  # 平倉


@dataclass
class TradingState:
    """交易狀態"""

    position: int = 0  # 持倉量
    entry_price: float = 0.0  # 進場價格
    cash: float = 10000.0  # 現金
    total_trades: int = 0  # 總交易次數
    winning_trades: int = 0  # 獲利交易
    current_step: int = 0  # 當前步數


class TradingEnvironment(gym.Env):
    """
    強化學習交易環境

    支援日內交易策略訓練，包含：
    - 真實市場模擬（滑點、手續費）
    - 多維度觀察空間
    - 風險調整獎勵函數
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        max_position: int = 100,
        window_size: int = 20,
        reward_scaling: float = 1e-4,
        render_mode: Optional[str] = None,
    ):
        """
        初始化交易環境

        Args:
            df: OHLCV 數據 DataFrame
            initial_balance: 初始資金
            commission: 手續費率
            slippage: 滑點
            max_position: 最大持倉量
            window_size: 觀察窗口大小
            reward_scaling: 獎勵縮放係數
            render_mode: 渲染模式
        """
        super().__init__()

        # 數據準備
        self.df = df.copy()
        self.prices = df["close"].values
        self.volumes = df["volume"].values
        self.high = df["high"].values
        self.low = df["low"].values
        self.open = df["open"].values

        # 環境參數
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.max_position = max_position
        self.window_size = window_size
        self.reward_scaling = reward_scaling
        self.render_mode = render_mode

        # 計算技術特徵
        self._prepare_features()

        # 定義動作空間
        self.action_space = spaces.Discrete(len(Actions))

        # 定義觀察空間
        # 包含：價格特徵 + 技術指標 + 持倉信息
        n_features = self.features.shape[1] + 4  # +4 for position info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, n_features), dtype=np.float32
        )

        # 初始化狀態
        self.state = None
        self.episode_start_idx = window_size
        self.max_steps = len(self.prices) - window_size - 1

        # 記錄
        self.history = []
        self.trades = []

    def _prepare_features(self):
        """準備技術特徵"""
        df = self.df.copy()

        # 價格變化率
        df["returns"] = df["close"].pct_change()

        # 價格位置（在當日範圍內）
        df["price_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-10)

        # 成交量比率
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

        # RSI
        df["rsi"] = self._calculate_rsi(df["close"], 14)

        # 波動率
        df["volatility"] = df["returns"].rolling(20).std()

        # MACD
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = exp1 - exp2
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_dif"] = df["macd"] - df["macd_signal"]

        # 布林帶
        sma = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        df["bb_upper"] = sma + (std * 2)
        df["bb_lower"] = sma - (std * 2)
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"] + 1e-10
        )

        # 選擇特徵
        feature_columns = [
            "returns",
            "price_position",
            "volume_ratio",
            "rsi",
            "volatility",
            "macd_diff",
            "bb_position",
        ]

        self.features = df[feature_columns].fillna(0).values

        # 標準化
        self.features = (self.features - np.mean(self.features, axis=0)) / (
            np.std(self.features, axis=0) + 1e-10
        )

    def _calculate_rsi(self, prices, period=14):
        """計算 RSI"""
        delta = pd.Series(prices).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50).values

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        重置環境

        Returns:
            observation: 初始觀察
            info: 額外信息
        """
        super().reset(seed=seed)

        # 重置狀態
        self.state = TradingState(cash=self.initial_balance)

        # 隨機選擇起始點
        self.current_step = self.episode_start_idx
        if options and "start_idx" in options:
            self.current_step = options["start_idx"]
        else:
            max_start = len(self.prices) - self.max_steps - self.window_size
            if max_start > self.episode_start_idx:
                self.current_step = self.np_random.integers(self.episode_start_idx, max_start)

        # 清空記錄
        self.history = []
        self.trades = []

        # 獲取初始觀察
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        執行一步環境

        Args:
            action: 交易動作

        Returns:
            observation: 新觀察
            reward: 獎勵
            terminated: 是否結束
            truncated: 是否截斷
            info: 額外信息
        """
        # 記錄前一步的資產價值
        prev_portfolio_value = self._get_portfolio_value()

        # 執行動作
        self._execute_action(action)

        # 更新步數
        self.current_step += 1
        self.state.current_step = self.current_step

        # 計算獎勵
        current_portfolio_value = self._get_portfolio_value()
        step_reward = (current_portfolio_value - prev_portfolio_value) * self.reward_scaling

        # 風險調整
        if self.state.position != 0:
            # 持倉風險懲罰
            position_risk = abs(self.state.position) / self.max_position * 0.01
            step_reward -= position_risk

        # 記錄歷史
        self.history.append(
            {
                "step": self.current_step,
                "action": action,
                "price": self.prices[self.current_step],
                "position": self.state.position,
                "cash": self.state.cash,
                "portfolio_value": current_portfolio_value,
                "reward": step_reward,
            }
        )

        # 檢查是否結束
        terminated = False
        truncated = False

        # 資金耗盡
        if current_portfolio_value <= self.initial_balance * 0.5:
            terminated = True
            step_reward -= 1.0  # 大懲罰

        # 達到最大步數
        if self.current_step >= len(self.prices) - 1:
            truncated = True

        # 獲取新觀察
        observation = self._get_observation()
        info = self._get_info()

        return observation, float(step_reward), terminated, truncated, info

    def _execute_action(self, action: int):
        """執行交易動作"""
        current_price = self.prices[self.current_step]

        if action == Actions.BUY:
            # 計算可買數量
            available_cash = self.state.cash * 0.95  # 保留5%現金
            max_shares = min(
                int(available_cash / (current_price * (1 + self.commission + self.slippage))),
                self.max_position - self.state.position,
            )

            if max_shares > 0:
                # 執行買入
                cost = max_shares * current_price * (1 + self.commission + self.slippage)
                self.state.cash -= cost
                self.state.position += max_shares
                self.state.entry_price = current_price
                self.state.total_trades += 1

                self.trades.append(
                    {
                        "step": self.current_step,
                        "type": "BUY",
                        "price": current_price,
                        "shares": max_shares,
                        "cost": cost,
                    }
                )

        elif action == Actions.SELL:
            if self.state.position > 0:
                # 賣出一半持倉
                shares_to_sell = self.state.position // 2
                if shares_to_sell > 0:
                    revenue = shares_to_sell * current_price * (1 - self.commission - self.slippage)
                    self.state.cash += revenue
                    self.state.position -= shares_to_sell

                    # 計算盈虧
                    pnl = (current_price - self.state.entry_price) * shares_to_sell
                    if pnl > 0:
                        self.state.winning_trades += 1

                    self.trades.append(
                        {
                            "step": self.current_step,
                            "type": "SELL",
                            "price": current_price,
                            "shares": shares_to_sell,
                            "revenue": revenue,
                            "pnl": pnl,
                        }
                    )

        elif action == Actions.CLOSE:
            if self.state.position != 0:
                # 平倉
                shares = abs(self.state.position)
                if self.state.position > 0:
                    # 平多倉
                    revenue = shares * current_price * (1 - self.commission - self.slippage)
                    self.state.cash += revenue
                    pnl = (current_price - self.state.entry_price) * shares
                else:
                    # 平空倉（如果支援做空）
                    cost = shares * current_price * (1 + self.commission + self.slippage)
                    self.state.cash -= cost
                    pnl = (self.state.entry_price - current_price) * shares

                if pnl > 0:
                    self.state.winning_trades += 1

                self.state.position = 0
                self.state.total_trades += 1

                self.trades.append(
                    {
                        "step": self.current_step,
                        "type": "CLOSE",
                        "price": current_price,
                        "shares": shares,
                        "pnl": pnl,
                    }
                )

    def _get_observation(self) -> np.ndarray:
        """獲取當前觀察"""
        # 獲取歷史窗口
        start_idx = self.current_step - self.window_size + 1
        end_idx = self.current_step + 1

        # 價格和特徵
        window_features = self.features[start_idx:end_idx]

        # 添加持倉信息
        position_info = np.zeros((self.window_size, 4))
        position_info[:, 0] = self.state.position / self.max_position  # 標準化持倉
        position_info[:, 1] = self.state.cash / self.initial_balance  # 標準化現金

        if self.state.position != 0:
            # 未實現盈虧
            current_price = self.prices[self.current_step]
            unrealized_pnl = (current_price - self.state.entry_price) / self.state.entry_price
            position_info[:, 2] = unrealized_pnl

        # 持倉時間
        position_info[:, 3] = self.state.current_step / self.max_steps

        # 合併特徵
        observation = np.hstack([window_features, position_info]).astype(np.float32)

        return observation

    def _get_portfolio_value(self) -> float:
        """計算投資組合價值"""
        current_price = self.prices[self.current_step]
        return self.state.cash + self.state.position * current_price

    def _get_info(self) -> Dict:
        """獲取額外信息"""
        portfolio_value = self._get_portfolio_value()

        info = {
            "portfolio_value": portfolio_value,
            "position": self.state.position,
            "cash": self.state.cash,
            "current_price": self.prices[self.current_step],
            "total_trades": self.state.total_trades,
            "win_rate": self.state.winning_trades / max(self.state.total_trades, 1),
            "current_step": self.current_step,
        }

        # 計算績效指標
        if len(self.history) > 0:
            returns = pd.Series([h["reward"] for h in self.history])
            info["total_reward"] = returns.sum()
            info["sharpe_ratio"] = np.sqrt(252) * returns.mean() / (returns.std() + 1e-10)

            values = [h["portfolio_value"] for h in self.history]
            peak = np.maximum.accumulate(values)
            dd = (values - peak) / peak
            info["max_drawdown"] = dd.min()

        return info

    def render(self):
        """渲染環境（可選）"""
        if self.render_mode == "human":
            info = self._get_info()
            print(f"Step: {self.current_step}")
            print(f"Price: ${info['current_price']:.2f}")
            print(f"Position: {info['position']}")
            print(f"Portfolio Value: ${info['portfolio_value']:.2f}")
            print(f"Win Rate: {info['win_rate']:.2%}")
            print("-" * 40)

    def close(self):
        """關閉環境"""
        pass


def check_env_valid():
    """檢查環境是否有效"""
    # 創建測試數據
    dates = pd.date_range(start="2024-01-01", periods=1000, freq="5min")
    test_data = pd.DataFrame(
        {
            "open": np.random.randn(1000).cumsum() + 100,
            "high": np.random.randn(1000).cumsum() + 101,
            "low": np.random.randn(1000).cumsum() + 99,
            "close": np.random.randn(1000).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, 1000),
        },
        index=dates,
    )

    # 確保 high >= low
    test_data["high"] = test_data[["open", "close", "high"]].max(axis=1)
    test_data["low"] = test_data[["open", "close", "low"]].min(axis=1)

    # 創建環境
    env = TradingEnvironment(test_data)

    # 檢查環境
    from stable_baselines3.common.env_checker import check_env

    try:
        check_env(env)
        print("Environment check passed!")
        return True
    except Exception as e:
        print(f"Environment check failed: {e}")
        return False


if __name__ == "__main__":
    print("Trading Environment for RL - Cloud DE Task DT-001")
    print("=" * 50)
    print("Testing environment validity...")

    if check_env_valid():
        print("\n✓ Environment is ready for training!")
    else:
        print("\n✗ Environment needs fixes.")
