"""
Risk management utilities for RL trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk management limits"""
    max_position_size: float = 0.3      # Max 30% in single position
    max_portfolio_exposure: float = 1.0  # Max 100% invested
    max_daily_loss: float = 0.02        # Max 2% daily loss
    max_drawdown: float = 0.10          # Max 10% drawdown
    max_trades_per_day: int = 10        # Max trades per day
    min_cash_buffer: float = 0.05       # Keep 5% cash buffer
    position_size_kelly: bool = True    # Use Kelly criterion for sizing
    kelly_fraction: float = 0.25        # Fraction of Kelly to use


class RiskManager:
    """
    Risk management for trading decisions
    """
    
    def __init__(self, limits: Optional[RiskLimits] = None):
        """
        Initialize risk manager
        
        Args:
            limits: Risk limits configuration
        """
        self.limits = limits or RiskLimits()
        
        # Track daily metrics
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.current_drawdown = 0.0
        
        # Historical data for calculations
        self.returns_history = []
        self.win_loss_history = []
        
    def check_trade_allowed(
        self,
        action: str,
        shares: int,
        price: float,
        portfolio_value: float,
        current_positions: Dict[str, float],
        cash: float
    ) -> Tuple[bool, str]:
        """
        Check if a trade is allowed under risk limits
        
        Args:
            action: 'BUY' or 'SELL'
            shares: Number of shares
            price: Current price
            portfolio_value: Total portfolio value
            current_positions: Current position values
            cash: Available cash
            
        Returns:
            Tuple of (allowed, reason)
        """
        # Check daily trade limit
        if self.daily_trades >= self.limits.max_trades_per_day:
            return False, "Daily trade limit reached"
        
        # Check daily loss limit
        if self.daily_pnl < -self.limits.max_daily_loss * portfolio_value:
            return False, "Daily loss limit reached"
        
        # Check drawdown limit
        if self.current_drawdown < -self.limits.max_drawdown:
            return False, "Maximum drawdown reached"
        
        if action == 'BUY':
            # Check position size limit
            position_value = shares * price
            if position_value > self.limits.max_position_size * portfolio_value:
                return False, "Position size exceeds limit"
            
            # Check portfolio exposure limit
            total_exposure = sum(current_positions.values()) + position_value
            if total_exposure > self.limits.max_portfolio_exposure * portfolio_value:
                return False, "Portfolio exposure exceeds limit"
            
            # Check cash buffer
            remaining_cash = cash - position_value
            if remaining_cash < self.limits.min_cash_buffer * portfolio_value:
                return False, "Insufficient cash buffer"
        
        return True, "Trade allowed"
    
    def calculate_position_size(
        self,
        signal_strength: float,
        confidence: float,
        portfolio_value: float,
        current_price: float,
        volatility: float
    ) -> int:
        """
        Calculate optimal position size
        
        Args:
            signal_strength: Strength of trading signal (-1 to 1)
            confidence: Confidence in signal (0 to 1)
            portfolio_value: Total portfolio value
            current_price: Current asset price
            volatility: Recent price volatility
            
        Returns:
            Recommended number of shares
        """
        if self.limits.position_size_kelly and len(self.win_loss_history) > 20:
            # Use Kelly criterion
            size_fraction = self._calculate_kelly_fraction()
        else:
            # Use fixed fraction based on signal
            size_fraction = abs(signal_strength) * confidence * 0.1  # Max 10% per trade
        
        # Apply risk limits
        size_fraction = min(size_fraction, self.limits.max_position_size)
        
        # Adjust for volatility (reduce size in high volatility)
        volatility_adj = 1 / (1 + volatility * 10)
        size_fraction *= volatility_adj
        
        # Calculate shares
        position_value = portfolio_value * size_fraction
        shares = int(position_value / current_price)
        
        return shares
    
    def _calculate_kelly_fraction(self) -> float:
        """Calculate position size using Kelly criterion"""
        if not self.win_loss_history:
            return 0.02  # Default small size
        
        # Calculate win probability and win/loss ratio
        wins = [r for r in self.win_loss_history if r > 0]
        losses = [r for r in self.win_loss_history if r < 0]
        
        if not wins or not losses:
            return 0.02
        
        win_prob = len(wins) / len(self.win_loss_history)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        # Kelly formula: f = p - q/b
        # where p = win probability, q = loss probability, b = win/loss ratio
        b = avg_win / avg_loss if avg_loss > 0 else 1
        q = 1 - win_prob
        
        kelly = win_prob - q / b
        
        # Apply Kelly fraction (use only a fraction of full Kelly)
        kelly = max(0, kelly * self.limits.kelly_fraction)
        
        # Cap at maximum position size
        return min(kelly, self.limits.max_position_size)
    
    def update_metrics(
        self,
        trade_pnl: float,
        portfolio_value: float,
        peak_value: float
    ):
        """
        Update risk metrics after a trade
        
        Args:
            trade_pnl: P&L from the trade
            portfolio_value: Current portfolio value
            peak_value: Peak portfolio value
        """
        # Update daily metrics
        self.daily_pnl += trade_pnl
        self.daily_trades += 1
        
        # Update drawdown
        self.current_drawdown = (portfolio_value - peak_value) / peak_value
        
        # Update history
        if trade_pnl != 0:
            self.returns_history.append(trade_pnl / portfolio_value)
            self.win_loss_history.append(trade_pnl)
        
        # Keep history size manageable
        if len(self.returns_history) > 1000:
            self.returns_history = self.returns_history[-1000:]
            self.win_loss_history = self.win_loss_history[-1000:]
    
    def calculate_var(
        self,
        portfolio_value: float,
        confidence_level: float = 0.95,
        time_horizon: int = 1
    ) -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            portfolio_value: Current portfolio value
            confidence_level: Confidence level (e.g., 0.95)
            time_horizon: Time horizon in days
            
        Returns:
            VaR amount
        """
        if len(self.returns_history) < 20:
            # Not enough data, use simple estimate
            return portfolio_value * 0.02 * np.sqrt(time_horizon)
        
        # Calculate historical VaR
        returns = np.array(self.returns_history)
        var_percentile = (1 - confidence_level) * 100
        var_return = np.percentile(returns, var_percentile)
        
        # Scale to time horizon
        var_return *= np.sqrt(time_horizon)
        
        return abs(var_return * portfolio_value)
    
    def get_risk_metrics(self, portfolio_value: float) -> Dict[str, float]:
        """
        Get current risk metrics
        
        Args:
            portfolio_value: Current portfolio value
            
        Returns:
            Dictionary of risk metrics
        """
        metrics = {
            'current_drawdown': self.current_drawdown,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'var_95': self.calculate_var(portfolio_value, 0.95),
            'var_99': self.calculate_var(portfolio_value, 0.99),
        }
        
        if self.returns_history:
            returns = np.array(self.returns_history)
            metrics['volatility'] = np.std(returns) * np.sqrt(252)
            metrics['downside_deviation'] = np.std(returns[returns < 0]) * np.sqrt(252) if any(returns < 0) else 0
            
            # Sortino ratio (uses downside deviation)
            if metrics['downside_deviation'] > 0:
                metrics['sortino_ratio'] = (np.mean(returns) * 252) / metrics['downside_deviation']
            else:
                metrics['sortino_ratio'] = 0
        
        return metrics
    
    def reset_daily_metrics(self):
        """Reset daily tracking metrics"""
        self.daily_trades = 0
        self.daily_pnl = 0.0
        logger.info("Daily risk metrics reset")
    
    def should_stop_trading(self, portfolio_value: float) -> Tuple[bool, str]:
        """
        Check if trading should be stopped due to risk limits
        
        Args:
            portfolio_value: Current portfolio value
            
        Returns:
            Tuple of (should_stop, reason)
        """
        # Check daily loss
        if self.daily_pnl < -self.limits.max_daily_loss * portfolio_value:
            return True, "Daily loss limit exceeded"
        
        # Check drawdown
        if self.current_drawdown < -self.limits.max_drawdown:
            return True, "Maximum drawdown exceeded"
        
        # Check trade count
        if self.daily_trades >= self.limits.max_trades_per_day:
            return True, "Daily trade limit reached"
        
        return False, ""