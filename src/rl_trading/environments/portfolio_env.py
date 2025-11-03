"""
Multi-Asset Portfolio Trading Environment for Reinforcement Learning
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any
from datetime import datetime
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    """Portfolio state container"""

    cash: float
    positions: Dict[str, float]  # symbol -> shares
    weights: Dict[str, float]  # symbol -> weight
    total_value: float
    returns: float
    timestamp: datetime


class PortfolioTradingEnvironment(gym.Env):
    """
    OpenAI Gym environment for multi-asset portfolio trading

    State Space: Concatenated features for all assets + portfolio state
    Action Space: Continuous weights for each asset (sum to 1)
    """

    def __init__(
        self,
        symbols: List[str],
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        max_steps_per_episode: int = 252,
        lookback_window: int = 60,
        rebalance_frequency: int = 5,  # Rebalance every 5 days
        risk_free_rate: float = 0.02,
        features_per_asset: int = 20,
    ):
        """
        Initialize portfolio trading environment

        Args:
            symbols: List of tradable symbols
            initial_capital: Starting capital
            transaction_cost: Transaction cost rate
            slippage: Slippage rate
            max_steps_per_episode: Maximum steps per episode
            lookback_window: Historical data window for features
            rebalance_frequency: How often to allow rebalancing
            risk_free_rate: Risk-free rate for Sharpe calculation
            features_per_asset: Number of features per asset
        """
        super().__init__()

        self.symbols = symbols
        self.n_assets = len(symbols)
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_steps_per_episode = max_steps_per_episode
        self.lookback_window = lookback_window
        self.rebalance_frequency = rebalance_frequency
        self.risk_free_rate = risk_free_rate
        self.features_per_asset = features_per_asset

        # State space: (n_assets * features_per_asset) + portfolio features
        portfolio_features = 10  # cash ratio, total return, volatility, etc.
        self.state_dim = self.n_assets * self.features_per_asset + portfolio_features

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        # Action space: Weight allocation for each asset (including cash)
        # Actions sum to 1 (simplex constraint)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_assets + 1,), dtype=np.float32  # +1 for cash
        )

        # Data containers
        self.market_data = {}
        self.current_step = 0
        self.portfolio_state = None
        self.portfolio_history = []
        self.trade_history = []

        # Performance tracking
        self.episode_returns = []
        self.episode_volatility = []
        self.episode_sharpe = []

        logger.info(f"Initialized portfolio environment with {self.n_assets} assets")

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0

        # Initialize portfolio state
        self.portfolio_state = PortfolioState(
            cash=self.initial_capital,
            positions={symbol: 0 for symbol in self.symbols},
            weights={symbol: 0 for symbol in self.symbols},
            total_value=self.initial_capital,
            returns=0.0,
            timestamp=None,
        )
        self.portfolio_state.weights["cash"] = 1.0

        # Reset history
        self.portfolio_history = [self._get_portfolio_snapshot()]
        self.trade_history = []
        self.episode_returns = []
        self.episode_volatility = []
        self.episode_sharpe = []

        # Load market data if needed
        if not self.market_data:
            self._load_market_data()

        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment

        Args:
            action: Portfolio weights (including cash)

        Returns:
            observation: Next state
            reward: Step reward
            done: Episode finished
            info: Additional information
        """
        # Validate action
        action = self._validate_action(action)

        # Get current market prices
        current_prices = self._get_current_prices()

        # Check if rebalancing is allowed
        can_rebalance = self.current_step % self.rebalance_frequency == 0

        if can_rebalance:
            # Execute portfolio rebalancing
            trades = self._rebalance_portfolio(action, current_prices)
            self.trade_history.extend(trades)

        # Update portfolio value
        self._update_portfolio_value(current_prices)

        # Calculate step return
        step_return = self._calculate_step_return()
        self.episode_returns.append(step_return)

        # Calculate reward
        reward = self._calculate_reward(step_return)

        # Update state
        self.current_step += 1
        done = self.current_step >= self.max_steps_per_episode

        # Get next observation
        observation = self._get_observation()

        # Prepare info
        info = {
            "portfolio_value": self.portfolio_state.total_value,
            "weights": self.portfolio_state.weights.copy(),
            "step_return": step_return,
            "can_rebalance": can_rebalance,
            "trades_executed": len(trades) if can_rebalance else 0,
        }

        if done:
            info.update(self._get_episode_summary())

        # Record portfolio snapshot
        self.portfolio_history.append(self._get_portfolio_snapshot())

        return observation, reward, done, info

    def _validate_action(self, action: np.ndarray) -> np.ndarray:
        """Validate and normalize action to ensure it sums to 1"""
        # Clip to valid range
        action = np.clip(action, 0, 1)

        # Normalize to sum to 1
        action_sum = np.sum(action)
        if action_sum > 0:
            action = action / action_sum
        else:
            # Default to all cash
            action = np.zeros_like(action)
            action[-1] = 1.0  # Last element is cash

        return action

    def _get_observation(self) -> np.ndarray:
        """Get current observation state"""
        features = []

        # 1. Asset features (technical indicators, predictions, etc.)
        for symbol in self.symbols:
            asset_features = self._get_asset_features(symbol)
            features.extend(asset_features)

        # 2. Portfolio features
        portfolio_features = self._get_portfolio_features()
        features.extend(portfolio_features)

        return np.array(features, dtype=np.float32)

    def _get_asset_features(self, symbol: str) -> List[float]:
        """Get features for a single asset"""
        if symbol not in self.market_data:
            return [0.0] * self.features_per_asset

        self.market_data[symbol]

        # Get current and historical prices
        end_idx = self.current_step + self.lookback_window
        if end_idx >= len(data):
            return [0.0] * self.features_per_asset

        window_data = data.iloc[self.current_step : end_idx]

        features = []

        # Price features
        returns = window_data["close"].pct_change().dropna()
        features.append(returns.mean())  # Mean return
        features.append(returns.std())  # Volatility
        features.append(returns.skew())  # Skewness
        features.append(returns.kurt())  # Kurtosis

        # Technical indicators
        features.append(self._calculate_rsi(window_data["close"]))
        features.append(self._calculate_macd_signal(window_data["close"]))

        # Price position
        current_price = window_data["close"].iloc[-1]
        features.append(
            (current_price - window_data["close"].min())
            / (window_data["close"].max() - window_data["close"].min() + 1e-8)
        )

        # Volume features
        features.append(window_data["volume"].iloc[-1] / window_data["volume"].mean())

        # Moving averages
        ma_20 = window_data["close"].rolling(20).mean().iloc[-1]
        ma_50 = (
            window_data["close"].rolling(50).mean().iloc[-1]
            if len(window_data) >= 50
            else ma_20
        )
        features.append(current_price / ma_20 - 1)
        features.append(current_price / ma_50 - 1)

        # Momentum
        features.append(
            window_data["close"].iloc[-1] / window_data["close"].iloc[0] - 1
        )

        # Correlation with portfolio (placeholder)
        features.append(0.0)

        # Pad or truncate to match expected size
        while len(features) < self.features_per_asset:
            features.append(0.0)

        return features[: self.features_per_asset]

    def _get_portfolio_features(self) -> List[float]:
        """Get portfolio-level features"""
        features = []

        # Cash ratio
        features.append(self.portfolio_state.cash / self.portfolio_state.total_value)

        # Portfolio concentration (Herfindahl index)
        weights_squared = sum(w**2 for w in self.portfolio_state.weights.values())
        features.append(weights_squared)

        # Number of active positions
        active_positions = sum(
            1 for pos in self.portfolio_state.positions.values() if pos > 0
        )
        features.append(active_positions / self.n_assets)

        # Portfolio returns statistics
        if len(self.episode_returns) > 1:
            returns_array = np.array(self.episode_returns)
            features.append(returns_array.mean())
            features.append(returns_array.std())
            features.append(self._calculate_sharpe_ratio(returns_array))
            features.append(self._calculate_max_drawdown())
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        # Days since last rebalance
        features.append(
            (self.current_step % self.rebalance_frequency) / self.rebalance_frequency
        )

        # Portfolio value relative to initial
        features.append(self.portfolio_state.total_value / self.initial_capital - 1)

        # Market regime indicator (simple)
        market_returns = self._get_market_returns()
        features.append(1.0 if market_returns > 0 else -1.0)

        return features

    def _get_current_prices(self) -> Dict[str, float]:
        """Get current market prices for all assets"""
        prices = {}

        for symbol in self.symbols:
            if symbol in self.market_data:
                idx = self.current_step + self.lookback_window
                if idx < len(self.market_data[symbol]):
                    prices[symbol] = self.market_data[symbol].iloc[idx]["close"]
                else:
                    prices[symbol] = self.market_data[symbol].iloc[-1]["close"]
            else:
                prices[symbol] = 100.0  # Default price

        return prices

    def _rebalance_portfolio(
        self, target_weights: np.ndarray, current_prices: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Execute portfolio rebalancing to target weights"""
        trades = []

        # Calculate current portfolio value
        current_value = self.portfolio_state.total_value

        # Map action to symbol weights
        weight_dict = {}
        for i, symbol in enumerate(self.symbols):
            weight_dict[symbol] = target_weights[i]
        weight_dict["cash"] = target_weights[-1]

        # Calculate target positions
        target_positions = {}
        for symbol in self.symbols:
            target_value = current_value * weight_dict[symbol]
            target_shares = int(target_value / current_prices[symbol])
            target_positions[symbol] = target_shares

        # Execute trades
        for symbol in self.symbols:
            current_shares = self.portfolio_state.positions[symbol]
            target_shares = target_positions[symbol]

            if target_shares != current_shares:
                # Calculate trade
                shares_to_trade = target_shares - current_shares
                trade_value = abs(shares_to_trade * current_prices[symbol])

                # Apply transaction costs and slippage
                if shares_to_trade > 0:  # Buy
                    execution_price = current_prices[symbol] * (1 + self.slippage)
                    cost = trade_value * (1 + self.transaction_cost)
                else:  # Sell
                    execution_price = current_prices[symbol] * (1 - self.slippage)
                    cost = trade_value * (1 - self.transaction_cost)

                # Check if we have enough cash for buy
                if shares_to_trade > 0 and cost > self.portfolio_state.cash:
                    # Adjust shares to what we can afford
                    affordable_shares = int(
                        self.portfolio_state.cash
                        / (execution_price * (1 + self.transaction_cost))
                    )
                    shares_to_trade = min(shares_to_trade, affordable_shares)

                # Execute trade
                if shares_to_trade != 0:
                    if shares_to_trade > 0:
                        self.portfolio_state.cash -= (
                            shares_to_trade
                            * execution_price
                            * (1 + self.transaction_cost)
                        )
                    else:
                        self.portfolio_state.cash += (
                            abs(shares_to_trade)
                            * execution_price
                            * (1 - self.transaction_cost)
                        )

                    self.portfolio_state.positions[symbol] += shares_to_trade

                    trades.append(
                        {
                            "symbol": symbol,
                            "shares": shares_to_trade,
                            "price": execution_price,
                            "cost": abs(
                                shares_to_trade
                                * execution_price
                                * self.transaction_cost
                            ),
                            "timestamp": self.current_step,
                        }
                    )

        # Update portfolio weights
        self._update_portfolio_weights(current_prices)

        return trades

    def _update_portfolio_value(self, current_prices: Dict[str, float]):
        """Update portfolio value based on current prices"""
        total_value = self.portfolio_state.cash

        for symbol, shares in self.portfolio_state.positions.items():
            if shares > 0 and symbol in current_prices:
                total_value += shares * current_prices[symbol]

        self.portfolio_state.total_value = total_value

    def _update_portfolio_weights(self, current_prices: Dict[str, float]):
        """Update portfolio weight allocations"""
        total_value = self.portfolio_state.total_value

        # Asset weights
        for symbol in self.symbols:
            if symbol in self.portfolio_state.positions and symbol in current_prices:
                position_value = (
                    self.portfolio_state.positions[symbol] * current_prices[symbol]
                )
                self.portfolio_state.weights[symbol] = position_value / total_value
            else:
                self.portfolio_state.weights[symbol] = 0.0

        # Cash weight
        self.portfolio_state.weights["cash"] = self.portfolio_state.cash / total_value

    def _calculate_step_return(self) -> float:
        """Calculate return for current step"""
        if len(self.portfolio_history) < 2:
            return 0.0

        current_value = self.portfolio_state.total_value
        previous_value = self.portfolio_history[-1]["total_value"]

        return (current_value - previous_value) / previous_value

    def _calculate_reward(self, step_return: float) -> float:
        """
        Calculate reward based on risk-adjusted returns

        Considers:
        - Portfolio returns
        - Risk (volatility)
        - Sharpe ratio
        - Drawdown penalty
        - Diversification bonus
        """
        reward = 0.0

        # 1. Basic return component
        reward += step_return * 100  # Scale for better learning

        # 2. Risk-adjusted component (mini Sharpe)
        if len(self.episode_returns) >= 20:
            recent_returns = np.array(self.episode_returns[-20:])
            if recent_returns.std() > 0:
                sharpe = (
                    recent_returns.mean() - self.risk_free_rate / 252
                ) / recent_returns.std()
                reward += sharpe * 0.1

        # 3. Drawdown penalty
        if len(self.portfolio_history) > 1:
            drawdown = self._calculate_max_drawdown()
            if drawdown > 0.1:  # Penalize drawdowns > 10%
                reward -= drawdown * 10

        # 4. Diversification bonus
        # Use Herfindahl index (lower is more diversified)
        weights_squared = sum(
            w**2
            for s, w in self.portfolio_state.weights.items()
            if s != "cash" and w > 0
        )
        diversification_score = (
            1 - weights_squared
        )  # Higher score for more diversification
        reward += diversification_score * 0.05

        # 5. Transaction cost penalty (implicit in returns but emphasize)
        if len(self.trade_history) > 0:
            recent_trades = [
                t for t in self.trade_history if t["timestamp"] == self.current_step - 1
            ]
            if recent_trades:
                total_cost = sum(t["cost"] for t in recent_trades)
                reward -= total_cost / self.portfolio_state.total_value * 100

        return reward

    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - self.risk_free_rate / 252

        if returns.std() > 0:
            return np.sqrt(252) * excess_returns.mean() / returns.std()
        else:
            return 0.0

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if len(self.portfolio_history) < 2:
            return 0.0

        values = [h["total_value"] for h in self.portfolio_history]
        peak = values[0]
        max_dd = 0.0

        for value in values[1:]:
            peak = max(peak, value)
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)

        return max_dd

    def _get_market_returns(self) -> float:
        """Get average market returns"""
        returns = []

        for symbol in self.symbols:
            if symbol in self.market_data:
                idx = self.current_step + self.lookback_window
                if idx > 0 and idx < len(self.market_data[symbol]):
                    current = self.market_data[symbol].iloc[idx]["close"]
                    previous = self.market_data[symbol].iloc[idx - 1]["close"]
                    returns.append((current - previous) / previous)

        return np.mean(returns) if returns else 0.0

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period:
            return 50.0

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        if loss.iloc[-1] == 0:
            return 100.0

        rs = gain.iloc[-1] / loss.iloc[-1]
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd_signal(self, prices: pd.Series) -> float:
        """Calculate MACD signal"""
        if len(prices) < 26:
            return 0.0

        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()

        return (macd.iloc[-1] - signal.iloc[-1]) / prices.iloc[-1]

    def _get_portfolio_snapshot(self) -> Dict[str, Any]:
        """Get current portfolio snapshot"""
        return {
            "step": self.current_step,
            "total_value": self.portfolio_state.total_value,
            "cash": self.portfolio_state.cash,
            "positions": self.portfolio_state.positions.copy(),
            "weights": self.portfolio_state.weights.copy(),
            "returns": self.portfolio_state.returns,
        }

    def _get_episode_summary(self) -> Dict[str, Any]:
        """Get episode summary statistics"""
        returns_array = np.array(self.episode_returns)

        total_return = (
            self.portfolio_state.total_value - self.initial_capital
        ) / self.initial_capital

        summary = {
            "total_return": total_return,
            "annualized_return": (
                total_return * (252 / len(self.episode_returns))
                if self.episode_returns
                else 0
            ),
            "volatility": (
                returns_array.std() * np.sqrt(252) if len(returns_array) > 1 else 0
            ),
            "sharpe_ratio": self._calculate_sharpe_ratio(returns_array),
            "max_drawdown": self._calculate_max_drawdown(),
            "total_trades": len(self.trade_history),
            "final_weights": self.portfolio_state.weights.copy(),
        }

        # Win rate
        positive_returns = sum(1 for r in self.episode_returns if r > 0)
        summary["win_rate"] = (
            positive_returns / len(self.episode_returns) if self.episode_returns else 0
        )

        # Average trade cost
        if self.trade_history:
            total_cost = sum(t["cost"] for t in self.trade_history)
            summary["avg_trade_cost"] = total_cost / len(self.trade_history)
        else:
            summary["avg_trade_cost"] = 0

        return summary

    def _load_market_data(self):
        """Load market data for all symbols"""
        # This is a placeholder - in production, load real market data
        # For now, generate synthetic data

        dates = pd.date_range(start="2022-01-01", periods=500, freq="D")

        for symbol in self.symbols:
            # Generate correlated synthetic price data
            np.random.seed(hash(symbol) % 1000)

            returns = np.random.normal(0.0002, 0.02, len(dates))
            prices = 100 * np.exp(np.cumsum(returns))

            volume = np.random.lognormal(15, 0.5, len(dates))

            self.market_data[symbol] = pd.DataFrame(
                {
                    "open": prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
                    "high": prices * (1 + np.random.uniform(0, 0.02, len(dates))),
                    "low": prices * (1 + np.random.uniform(-0.02, 0, len(dates))),
                    "close": prices,
                    "volume": volume,
                },
                index=dates,
            )

        logger.info(f"Loaded market data for {len(self.symbols)} symbols")

    def set_market_data(self, market_data: Dict[str, pd.DataFrame]):
        """Set market data from external source"""
        self.market_data = market_data

    def render(self, mode="human"):
        """Render environment state"""
        if mode == "human":
            print(f"\n=== Step {self.current_step} ===")
            print(f"Portfolio Value: ${self.portfolio_state.total_value:,.2f}")
            print(f"Cash: ${self.portfolio_state.cash:,.2f}")
            print("\nPositions:")
            for symbol, shares in self.portfolio_state.positions.items():
                if shares > 0:
                    print(f"  {symbol}: {shares} shares")
            print("\nWeights:")
            for asset, weight in self.portfolio_state.weights.items():
                if weight > 0.01:
                    print(f"  {asset}: {weight:.1%}")

            if len(self.episode_returns) > 0:
                print(f"\nCurrent Return: {self.episode_returns[-1]:.2%}")
                print(
                    f"Total Return: {(self.portfolio_state.total_value/self.initial_capital - 1):.2%}"
                )
