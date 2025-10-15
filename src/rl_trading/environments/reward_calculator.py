"""
Reward function calculator for RL trading environment
"""

import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Configuration for reward calculation"""

    # Profit components
    use_simple_pnl: bool = False
    use_risk_adjusted_returns: bool = True

    # Risk penalties
    volatility_penalty: float = 0.1
    max_drawdown_penalty: float = 0.2
    exposure_penalty: float = 0.05

    # Transaction costs
    commission_rate: float = 0.001  # 0.1% per trade
    slippage_rate: float = 0.0005  # 0.05% slippage

    # Behavioral penalties
    overtrading_penalty: float = 0.01
    holding_reward: float = 0.0001  # Small reward for holding profitable positions

    # Risk-free rate for Sharpe ratio
    risk_free_rate: float = 0.02  # Annual 2%

    # Scaling factors
    reward_scaling: float = 1.0
    clip_reward: Optional[float] = 10.0


class RewardCalculator:
    """
    Calculate rewards for trading actions with risk adjustments
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        """
        Initialize reward calculator

        Args:
            config: Reward configuration
        """
        self.config = config or RewardConfig()

        # Track metrics for risk calculations
        self.returns_history = []
        self.equity_history = []
        self.trade_count = 0
        self.episode_trades = 0

    def calculate_reward(
        self,
        prev_portfolio_value: float,
        curr_portfolio_value: float,
        action: Dict[str, float],
        position_held: int,
        current_prices: Dict[str, float],
        info: Optional[Dict] = None,
    ) -> float:
        """
        Calculate reward for a single step

        Args:
            prev_portfolio_value: Portfolio value before action
            curr_portfolio_value: Portfolio value after action
            action: Action taken (with trade details)
            position_held: Number of steps position has been held
            current_prices: Current market prices
            info: Additional information

        Returns:
            Calculated reward
        """
        reward_components = {}

        # 1. Basic P&L component
        pnl = curr_portfolio_value - prev_portfolio_value
        returns = pnl / prev_portfolio_value if prev_portfolio_value > 0 else 0

        if self.config.use_simple_pnl:
            reward_components["pnl"] = pnl
        else:
            reward_components["returns"] = returns * 100  # Scale up returns

        # 2. Transaction costs
        if action["shares"] != 0:
            commission = action["value"] * self.config.commission_rate
            slippage = action["value"] * self.config.slippage_rate
            reward_components["transaction_costs"] = -(commission + slippage)
            self.episode_trades += 1

        # 3. Risk-adjusted returns (if enabled)
        if self.config.use_risk_adjusted_returns and len(self.returns_history) > 0:
            # Volatility penalty
            if len(self.returns_history) >= 20:
                volatility = np.std(self.returns_history[-20:])
                reward_components["volatility_penalty"] = (
                    -volatility * self.config.volatility_penalty
                )

            # Sharpe ratio component
            if len(self.returns_history) >= 252:  # One year of data
                sharpe = self._calculate_sharpe_ratio()
                reward_components["sharpe_bonus"] = sharpe * 0.1

        # 4. Drawdown penalty
        if len(self.equity_history) >= 2:
            drawdown = self._calculate_drawdown()
            if drawdown < -0.1:  # Penalty for >10% drawdown
                reward_components["drawdown_penalty"] = drawdown * self.config.max_drawdown_penalty

        # 5. Exposure penalty (encourage appropriate position sizing)
        if info and "portfolio_exposure" in info:
            exposure = info["portfolio_exposure"]
            if exposure > 0.8:  # Penalty for >80% exposure
                reward_components["exposure_penalty"] = (
                    -(exposure - 0.8) * self.config.exposure_penalty
                )

        # 6. Behavioral rewards/penalties
        # Overtrading penalty
        if self.episode_trades > 0:
            trade_frequency = self.episode_trades / max(1, len(self.returns_history))
            if trade_frequency > 0.5:  # More than 50% of steps involve trades
                reward_components["overtrading_penalty"] = -self.config.overtrading_penalty

        # Holding reward (for profitable positions)
        if position_held > 0 and returns > 0:
            reward_components["holding_reward"] = self.config.holding_reward * min(
                position_held, 100
            )

        # 7. Market timing bonus
        if action["shares"] > 0 and info and "price_trend" in info:
            # Reward buying in uptrend
            if info["price_trend"] > 0:
                reward_components["timing_bonus"] = 0.01
        elif action["shares"] < 0 and info and "price_trend" in info:
            # Reward selling in downtrend
            if info["price_trend"] < 0:
                reward_components["timing_bonus"] = 0.01

        # Calculate total reward
        total_reward = sum(reward_components.values())

        # Apply scaling
        total_reward *= self.config.reward_scaling

        # Clip reward if configured
        if self.config.clip_reward is not None:
            total_reward = np.clip(total_reward, -self.config.clip_reward, self.config.clip_reward)

        # Update history
        self.returns_history.append(returns)
        self.equity_history.append(curr_portfolio_value)

        # Log reward components for debugging
        logger.debug(f"Reward components: {reward_components}")
        logger.debug(f"Total reward: {total_reward}")

        return total_reward

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from returns history"""
        returns = np.array(self.returns_history[-252:])  # Last year

        # Annualized metrics
        mean_return = np.mean(returns) * 252
        std_return = np.std(returns) * np.sqrt(252)

        if std_return == 0:
            return 0

        sharpe = (mean_return - self.config.risk_free_rate) / std_return
        return sharpe

    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown"""
        if len(self.equity_history) < 2:
            return 0

        # Get running maximum
        running_max = np.maximum.accumulate(self.equity_history)
        drawdown = (self.equity_history[-1] - running_max[-1]) / running_max[-1]

        return drawdown

    def get_episode_metrics(self) -> Dict[str, float]:
        """Get metrics for the current episode"""
        if len(self.returns_history) == 0:
            return {}

        returns = np.array(self.returns_history)

        metrics = {
            "total_return": np.sum(returns),
            "mean_return": np.mean(returns),
            "volatility": np.std(returns),
            "sharpe_ratio": self._calculate_sharpe_ratio() if len(returns) > 20 else 0,
            "max_drawdown": np.min([self._calculate_drawdown()]),
            "trade_count": self.episode_trades,
            "win_rate": np.mean(returns > 0) if len(returns) > 0 else 0,
        }

        return metrics

    def reset(self):
        """Reset calculator for new episode"""
        self.returns_history = []
        self.equity_history = []
        self.episode_trades = 0

    def get_reward_info(self) -> Dict[str, any]:
        """Get information about reward configuration"""
        return {"config": self.config.__dict__, "metrics": self.get_episode_metrics()}
