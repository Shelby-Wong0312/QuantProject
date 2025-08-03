"""
Action space definition for RL trading environment
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of trading actions"""
    HOLD = 0
    BUY_25 = 1      # Buy with 25% of available capital
    BUY_50 = 2      # Buy with 50% of available capital
    BUY_75 = 3      # Buy with 75% of available capital
    BUY_100 = 4     # Buy with 100% of available capital
    SELL_25 = 5     # Sell 25% of position
    SELL_50 = 6     # Sell 50% of position
    SELL_75 = 7     # Sell 75% of position
    SELL_100 = 8    # Sell 100% of position (close position)


class ActionSpace:
    """
    Action space for trading environment
    Supports both discrete and continuous actions
    """
    
    def __init__(
        self,
        action_type: str = 'discrete',
        allow_short: bool = False,
        max_position_size: float = 1.0,
        min_trade_size: float = 0.01,
        discrete_actions: Optional[List[ActionType]] = None
    ):
        """
        Initialize action space
        
        Args:
            action_type: 'discrete' or 'continuous'
            allow_short: Whether to allow short positions
            max_position_size: Maximum position size as fraction of capital
            min_trade_size: Minimum trade size as fraction of capital
            discrete_actions: List of allowed discrete actions
        """
        self.action_type = action_type
        self.allow_short = allow_short
        self.max_position_size = max_position_size
        self.min_trade_size = min_trade_size
        
        # Set up discrete actions
        if discrete_actions is None:
            self.discrete_actions = list(ActionType)
        else:
            self.discrete_actions = discrete_actions
        
        self.n_actions = len(self.discrete_actions) if action_type == 'discrete' else 1
        
        logger.info(f"Initialized {action_type} action space with {self.n_actions} actions")
        
    def sample(self) -> Union[int, float]:
        """Sample a random action"""
        if self.action_type == 'discrete':
            return np.random.randint(0, self.n_actions)
        else:
            # Continuous action: position target from -1 to 1
            return np.random.uniform(-1 if self.allow_short else 0, 1)
    
    def process_action(
        self,
        action: Union[int, float],
        current_position: float,
        current_cash: float,
        current_price: float,
        portfolio_value: float
    ) -> Dict[str, float]:
        """
        Process raw action into executable trade
        
        Args:
            action: Raw action from agent
            current_position: Current position in shares
            current_cash: Available cash
            current_price: Current asset price
            portfolio_value: Total portfolio value
            
        Returns:
            Dictionary with trade details
        """
        if self.action_type == 'discrete':
            return self._process_discrete_action(
                action, current_position, current_cash, current_price, portfolio_value
            )
        else:
            return self._process_continuous_action(
                action, current_position, current_cash, current_price, portfolio_value
            )
    
    def _process_discrete_action(
        self,
        action_idx: int,
        current_position: float,
        current_cash: float,
        current_price: float,
        portfolio_value: float
    ) -> Dict[str, float]:
        """Process discrete action"""
        
        # Get action type
        action = self.discrete_actions[action_idx]
        
        # Initialize trade details
        trade = {
            'action': action.name,
            'shares': 0,
            'value': 0,
            'new_position': current_position,
            'new_cash': current_cash,
            'feasible': True
        }
        
        # Calculate maximum shares we can buy
        max_buy_shares = int(current_cash / current_price)
        
        # Process action
        if action == ActionType.HOLD:
            # No trade
            pass
            
        elif action in [ActionType.BUY_25, ActionType.BUY_50, ActionType.BUY_75, ActionType.BUY_100]:
            # Buy actions
            fraction = {
                ActionType.BUY_25: 0.25,
                ActionType.BUY_50: 0.50,
                ActionType.BUY_75: 0.75,
                ActionType.BUY_100: 1.00
            }[action]
            
            # Calculate target shares
            target_value = current_cash * fraction
            target_shares = int(target_value / current_price)
            
            # Apply constraints
            shares_to_buy = min(target_shares, max_buy_shares)
            
            # Check minimum trade size
            if shares_to_buy * current_price < portfolio_value * self.min_trade_size:
                shares_to_buy = 0
                trade['feasible'] = False
            
            # Check maximum position size
            new_position_value = (current_position + shares_to_buy) * current_price
            if new_position_value > portfolio_value * self.max_position_size:
                # Adjust to stay within limit
                max_additional_value = portfolio_value * self.max_position_size - current_position * current_price
                shares_to_buy = max(0, int(max_additional_value / current_price))
            
            trade['shares'] = shares_to_buy
            trade['value'] = shares_to_buy * current_price
            trade['new_position'] = current_position + shares_to_buy
            trade['new_cash'] = current_cash - trade['value']
            
        elif action in [ActionType.SELL_25, ActionType.SELL_50, ActionType.SELL_75, ActionType.SELL_100]:
            # Sell actions
            if current_position <= 0:
                trade['feasible'] = False
            else:
                fraction = {
                    ActionType.SELL_25: 0.25,
                    ActionType.SELL_50: 0.50,
                    ActionType.SELL_75: 0.75,
                    ActionType.SELL_100: 1.00
                }[action]
                
                # Calculate shares to sell
                shares_to_sell = int(current_position * fraction)
                
                # Check minimum trade size
                if shares_to_sell * current_price < portfolio_value * self.min_trade_size:
                    # If selling all, ignore minimum
                    if action != ActionType.SELL_100:
                        shares_to_sell = 0
                        trade['feasible'] = False
                
                trade['shares'] = -shares_to_sell
                trade['value'] = shares_to_sell * current_price
                trade['new_position'] = current_position - shares_to_sell
                trade['new_cash'] = current_cash + trade['value']
        
        return trade
    
    def _process_continuous_action(
        self,
        action: float,
        current_position: float,
        current_cash: float,
        current_price: float,
        portfolio_value: float
    ) -> Dict[str, float]:
        """Process continuous action"""
        
        # Action is target position as fraction of portfolio value
        # Clip action to valid range
        if not self.allow_short:
            action = np.clip(action, 0, 1)
        else:
            action = np.clip(action, -1, 1)
        
        # Calculate target position value
        target_position_value = portfolio_value * action * self.max_position_size
        target_shares = target_position_value / current_price
        
        # Calculate change needed
        current_position_value = current_position * current_price
        position_change_value = target_position_value - current_position_value
        shares_change = int(position_change_value / current_price)
        
        # Initialize trade details
        trade = {
            'action': 'BUY' if shares_change > 0 else 'SELL' if shares_change < 0 else 'HOLD',
            'shares': shares_change,
            'value': abs(shares_change * current_price),
            'new_position': current_position + shares_change,
            'new_cash': current_cash - (shares_change * current_price),
            'feasible': True
        }
        
        # Check constraints
        if shares_change > 0:  # Buying
            # Check cash constraint
            max_buy_shares = int(current_cash / current_price)
            if shares_change > max_buy_shares:
                shares_change = max_buy_shares
                trade['shares'] = shares_change
                trade['value'] = shares_change * current_price
                trade['new_position'] = current_position + shares_change
                trade['new_cash'] = current_cash - trade['value']
        
        elif shares_change < 0:  # Selling
            # Check position constraint
            if not self.allow_short:
                shares_change = max(shares_change, -current_position)
                trade['shares'] = shares_change
                trade['value'] = abs(shares_change * current_price)
                trade['new_position'] = current_position + shares_change
                trade['new_cash'] = current_cash - (shares_change * current_price)
        
        # Check minimum trade size
        if abs(trade['value']) < portfolio_value * self.min_trade_size:
            trade['shares'] = 0
            trade['value'] = 0
            trade['new_position'] = current_position
            trade['new_cash'] = current_cash
            trade['action'] = 'HOLD'
            trade['feasible'] = False
        
        return trade
    
    def get_action_meanings(self) -> List[str]:
        """Get human-readable action descriptions"""
        if self.action_type == 'discrete':
            return [action.name for action in self.discrete_actions]
        else:
            return ["Continuous position target [-1, 1]"]
    
    def get_action_space_info(self) -> Dict[str, any]:
        """Get action space information"""
        return {
            'type': self.action_type,
            'n_actions': self.n_actions,
            'allow_short': self.allow_short,
            'max_position_size': self.max_position_size,
            'min_trade_size': self.min_trade_size,
            'action_meanings': self.get_action_meanings()
        }