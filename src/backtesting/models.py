"""
Transaction cost and slippage models for backtesting
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, Union
from abc import ABC, abstractmethod
import logging


logger = logging.getLogger(__name__)


class TransactionCostModel(ABC):
    """
    Abstract base class for transaction cost models
    """
    
    @abstractmethod
    def calculate_cost(
        self,
        order_size: float,
        price: float,
        instrument: str
    ) -> float:
        """Calculate transaction cost for an order"""
        pass


class FixedTransactionCost(TransactionCostModel):
    """
    Fixed transaction cost model
    """
    
    def __init__(self, cost_per_trade: float = 0.0, cost_per_share: float = 0.0):
        """
        Initialize fixed transaction cost model
        
        Args:
            cost_per_trade: Fixed cost per trade
            cost_per_share: Cost per share/unit traded
        """
        self.cost_per_trade = cost_per_trade
        self.cost_per_share = cost_per_share
    
    def calculate_cost(
        self,
        order_size: float,
        price: float,
        instrument: str
    ) -> float:
        """
        Calculate fixed transaction cost
        
        Args:
            order_size: Number of shares/units
            price: Price per share/unit
            instrument: Instrument identifier
            
        Returns:
            Total transaction cost
        """
        return self.cost_per_trade + (abs(order_size) * self.cost_per_share)


class PercentageTransactionCost(TransactionCostModel):
    """
    Percentage-based transaction cost model
    """
    
    def __init__(self, rate: float = 0.001):
        """
        Initialize percentage transaction cost model
        
        Args:
            rate: Cost as percentage of trade value (default: 0.1%)
        """
        self.rate = rate
    
    def calculate_cost(
        self,
        order_size: float,
        price: float,
        instrument: str
    ) -> float:
        """
        Calculate percentage-based transaction cost
        
        Args:
            order_size: Number of shares/units
            price: Price per share/unit
            instrument: Instrument identifier
            
        Returns:
            Total transaction cost
        """
        trade_value = abs(order_size) * price
        return trade_value * self.rate


class TieredTransactionCost(TransactionCostModel):
    """
    Tiered transaction cost model with volume-based rates
    """
    
    def __init__(self, tiers: Dict[float, float]):
        """
        Initialize tiered transaction cost model
        
        Args:
            tiers: Dictionary mapping volume thresholds to rates
                   Example: {0: 0.002, 10000: 0.0015, 50000: 0.001}
        """
        self.tiers = sorted(tiers.items())
    
    def calculate_cost(
        self,
        order_size: float,
        price: float,
        instrument: str
    ) -> float:
        """
        Calculate tiered transaction cost
        
        Args:
            order_size: Number of shares/units
            price: Price per share/unit
            instrument: Instrument identifier
            
        Returns:
            Total transaction cost
        """
        trade_value = abs(order_size) * price
        
        # Find applicable tier
        rate = self.tiers[0][1]  # Default to first tier
        for threshold, tier_rate in self.tiers:
            if trade_value >= threshold:
                rate = tier_rate
            else:
                break
        
        return trade_value * rate


class SlippageModel(ABC):
    """
    Abstract base class for slippage models
    """
    
    @abstractmethod
    def calculate_slippage(
        self,
        order_size: float,
        current_price: float,
        volume: float,
        volatility: float,
        spread: Optional[float] = None
    ) -> float:
        """Calculate slippage for an order"""
        pass


class FixedSlippage(SlippageModel):
    """
    Fixed slippage model
    """
    
    def __init__(self, slippage_bps: float = 5.0):
        """
        Initialize fixed slippage model
        
        Args:
            slippage_bps: Slippage in basis points (default: 5 bps)
        """
        self.slippage_rate = slippage_bps / 10000
    
    def calculate_slippage(
        self,
        order_size: float,
        current_price: float,
        volume: float,
        volatility: float,
        spread: Optional[float] = None
    ) -> float:
        """
        Calculate fixed slippage
        
        Args:
            order_size: Number of shares/units (positive for buy, negative for sell)
            current_price: Current market price
            volume: Market volume
            volatility: Price volatility
            spread: Bid-ask spread (optional)
            
        Returns:
            Slippage amount (positive makes execution worse)
        """
        # Buy orders execute at higher price, sell orders at lower price
        return abs(current_price * self.slippage_rate) * np.sign(order_size)


class LinearSlippage(SlippageModel):
    """
    Linear slippage model based on order size relative to volume
    """
    
    def __init__(self, impact_coefficient: float = 0.1):
        """
        Initialize linear slippage model
        
        Args:
            impact_coefficient: Price impact coefficient
        """
        self.impact_coefficient = impact_coefficient
    
    def calculate_slippage(
        self,
        order_size: float,
        current_price: float,
        volume: float,
        volatility: float,
        spread: Optional[float] = None
    ) -> float:
        """
        Calculate linear slippage based on market impact
        
        Args:
            order_size: Number of shares/units
            current_price: Current market price
            volume: Market volume
            volatility: Price volatility
            spread: Bid-ask spread (optional)
            
        Returns:
            Slippage amount
        """
        if volume == 0:
            # If no volume, use maximum slippage
            return current_price * 0.01 * np.sign(order_size)
        
        # Calculate participation rate
        participation_rate = abs(order_size) / volume
        
        # Linear impact
        price_impact = self.impact_coefficient * participation_rate * volatility
        
        # Add half spread if available
        if spread is not None:
            price_impact += spread / 2
        
        return current_price * price_impact * np.sign(order_size)


class SquareRootSlippage(SlippageModel):
    """
    Square-root market impact model (more realistic for large orders)
    """
    
    def __init__(
        self,
        temporary_impact: float = 0.1,
        permanent_impact: float = 0.05
    ):
        """
        Initialize square-root slippage model
        
        Args:
            temporary_impact: Temporary price impact coefficient
            permanent_impact: Permanent price impact coefficient
        """
        self.temporary_impact = temporary_impact
        self.permanent_impact = permanent_impact
    
    def calculate_slippage(
        self,
        order_size: float,
        current_price: float,
        volume: float,
        volatility: float,
        spread: Optional[float] = None
    ) -> float:
        """
        Calculate square-root slippage (Almgren-Chriss style)
        
        Args:
            order_size: Number of shares/units
            current_price: Current market price
            volume: Market volume (ADV - Average Daily Volume)
            volatility: Price volatility (daily)
            spread: Bid-ask spread (optional)
            
        Returns:
            Slippage amount
        """
        if volume == 0:
            return current_price * 0.01 * np.sign(order_size)
        
        # Normalize order size by ADV
        normalized_size = abs(order_size) / volume
        
        # Temporary impact (square-root)
        temp_impact = self.temporary_impact * volatility * np.sqrt(normalized_size)
        
        # Permanent impact (linear)
        perm_impact = self.permanent_impact * volatility * normalized_size
        
        # Total impact
        total_impact = temp_impact + perm_impact
        
        # Add half spread if available
        if spread is not None:
            total_impact += spread / (2 * current_price)
        
        return current_price * total_impact * np.sign(order_size)


class CombinedCostModel:
    """
    Combined model for both transaction costs and slippage
    """
    
    def __init__(
        self,
        transaction_cost_model: Optional[TransactionCostModel] = None,
        slippage_model: Optional[SlippageModel] = None
    ):
        """
        Initialize combined cost model
        
        Args:
            transaction_cost_model: Transaction cost model
            slippage_model: Slippage model
        """
        self.transaction_cost_model = transaction_cost_model or PercentageTransactionCost()
        self.slippage_model = slippage_model or FixedSlippage()
    
    def calculate_total_cost(
        self,
        order_size: float,
        current_price: float,
        instrument: str,
        volume: float = 1000000,
        volatility: float = 0.02,
        spread: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate total execution cost
        
        Args:
            order_size: Number of shares/units
            current_price: Current market price
            instrument: Instrument identifier
            volume: Market volume
            volatility: Price volatility
            spread: Bid-ask spread
            
        Returns:
            Dictionary with cost breakdown
        """
        # Calculate transaction cost
        transaction_cost = self.transaction_cost_model.calculate_cost(
            order_size, current_price, instrument
        )
        
        # Calculate slippage
        slippage = self.slippage_model.calculate_slippage(
            order_size, current_price, volume, volatility, spread
        )
        
        # Calculate execution price
        execution_price = current_price + slippage
        
        # Total cost
        total_cost = transaction_cost + abs(slippage * order_size)
        
        return {
            'transaction_cost': transaction_cost,
            'slippage_per_unit': slippage,
            'total_slippage': slippage * abs(order_size),
            'execution_price': execution_price,
            'total_cost': total_cost,
            'cost_percentage': total_cost / (abs(order_size) * current_price) * 100
        }