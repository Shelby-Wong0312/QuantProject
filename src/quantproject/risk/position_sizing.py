"""
Position Sizing Module
Kelly Criterion, Fixed Proportion, Risk Parity
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List


class PositionSizing:
    """Position sizing calculator for portfolio optimization"""
    
    @staticmethod
    def kelly_criterion(win_prob: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly criterion for optimal position sizing
        
        Args:
            win_prob: Probability of winning trade (0-1)
            avg_win: Average winning amount
            avg_loss: Average losing amount (positive value)
            
        Returns:
            Kelly fraction (0-1)
        """
        if avg_loss <= 0 or win_prob <= 0 or win_prob >= 1:
            return 0.0
            
        loss_prob = 1 - win_prob
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_prob, q = loss_prob
        b = avg_win / avg_loss
        
        kelly_fraction = (b * win_prob - loss_prob) / b
        
        # Cap at 25% for safety
        return max(0, min(kelly_fraction, 0.25))
    
    @staticmethod
    def kelly_from_returns(returns: Union[list, np.ndarray, pd.Series]) -> float:
        """
        Calculate Kelly criterion from return series
        
        Args:
            returns: Historical returns
            
        Returns:
            Kelly fraction
        """
        if isinstance(returns, (list, np.ndarray)):
            returns = pd.Series(returns)
            
        returns = returns.dropna()
        
        if len(returns) == 0:
            return 0.0
            
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        
        if len(winning_trades) == 0 or len(losing_trades) == 0:
            return 0.0
            
        win_prob = len(winning_trades) / len(returns)
        avg_win = winning_trades.mean()
        avg_loss = abs(losing_trades.mean())
        
        return PositionSizing.kelly_criterion(win_prob, avg_win, avg_loss)
    
    @staticmethod
    def fixed_proportion(capital: float, risk_per_trade: float = 0.02) -> float:
        """
        Calculate fixed proportion position size
        
        Args:
            capital: Total capital
            risk_per_trade: Risk percentage per trade (default 2%)
            
        Returns:
            Position size
        """
        return capital * risk_per_trade
    
    @staticmethod
    def volatility_adjusted(capital: float, 
                          volatility: float, 
                          target_volatility: float = 0.15) -> float:
        """
        Calculate volatility-adjusted position size
        
        Args:
            capital: Total capital
            volatility: Asset volatility
            target_volatility: Target portfolio volatility
            
        Returns:
            Position size
        """
        if volatility <= 0:
            return 0.0
            
        vol_ratio = target_volatility / volatility
        return capital * min(vol_ratio, 1.0)  # Cap at 100%
    
    @staticmethod
    def risk_parity_weights(volatilities: List[float]) -> List[float]:
        """
        Calculate risk parity weights
        
        Args:
            volatilities: List of asset volatilities
            
        Returns:
            List of weights
        """
        if not volatilities or any(vol <= 0 for vol in volatilities):
            n = len(volatilities)
            return [1/n] * n
            
        # Inverse volatility weighting
        inv_vols = [1/vol for vol in volatilities]
        total_inv_vol = sum(inv_vols)
        
        return [inv_vol/total_inv_vol for inv_vol in inv_vols]
    
    @staticmethod
    def max_position_size(capital: float, 
                         price: float, 
                         max_risk: float = 0.05) -> int:
        """
        Calculate maximum position size based on risk limit
        
        Args:
            capital: Total capital
            price: Asset price
            max_risk: Maximum risk percentage
            
        Returns:
            Maximum number of shares/contracts
        """
        if price <= 0:
            return 0
            
        max_capital_at_risk = capital * max_risk
        return int(max_capital_at_risk / price)
    
    @staticmethod
    def atr_position_sizing(capital: float, 
                           price: float, 
                           atr: float, 
                           risk_amount: float) -> int:
        """
        Calculate position size using ATR (Average True Range)
        
        Args:
            capital: Total capital
            price: Current price
            atr: Average True Range
            risk_amount: Amount willing to risk
            
        Returns:
            Position size in shares/contracts
        """
        if price <= 0 or atr <= 0:
            return 0
            
        # Position size = Risk Amount / (ATR * multiplier)
        # Using 2x ATR as stop distance
        stop_distance = 2 * atr
        position_size = risk_amount / stop_distance
        
        # Convert to number of shares
        shares = int(position_size / price)
        
        # Ensure we don't exceed capital
        max_shares = int(capital / price)
        
        return min(shares, max_shares)