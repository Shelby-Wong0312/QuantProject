"""
Portfolio tracking utilities for RL trading
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Trade information"""
    timestamp: datetime
    symbol: str
    action: str  # BUY or SELL
    shares: int
    price: float
    commission: float = 0.0
    slippage: float = 0.0
    
    @property
    def value(self) -> float:
        """Total trade value including costs"""
        base_value = self.shares * self.price
        if self.action == 'BUY':
            return base_value + self.commission + self.slippage
        else:
            return base_value - self.commission - self.slippage
    
    @property
    def cost_basis(self) -> float:
        """Cost per share including transaction costs"""
        if self.shares == 0:
            return 0
        return self.value / self.shares


@dataclass
class Position:
    """Position information"""
    symbol: str
    shares: int = 0
    avg_entry_price: float = 0.0
    realized_pnl: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    
    def update(self, trade: Trade):
        """Update position with new trade"""
        if trade.action == 'BUY':
            # Update average entry price
            total_value = self.shares * self.avg_entry_price + trade.value
            self.shares += trade.shares
            self.avg_entry_price = total_value / self.shares if self.shares > 0 else 0
        
        elif trade.action == 'SELL':
            if self.shares <= 0:
                logger.warning(f"Attempting to sell {trade.shares} shares with position {self.shares}")
                return
            
            # Calculate realized P&L
            shares_to_sell = min(trade.shares, self.shares)
            realized = shares_to_sell * (trade.price - self.avg_entry_price) - trade.commission - trade.slippage
            self.realized_pnl += realized
            
            # Update position
            self.shares -= shares_to_sell
            
            # Reset avg entry price if position closed
            if self.shares == 0:
                self.avg_entry_price = 0
        
        self.trades.append(trade)
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L"""
        if self.shares == 0:
            return 0
        return self.shares * (current_price - self.avg_entry_price)
    
    def get_total_pnl(self, current_price: float) -> float:
        """Calculate total P&L (realized + unrealized)"""
        return self.realized_pnl + self.get_unrealized_pnl(current_price)


class PortfolioTracker:
    """
    Track portfolio performance and positions
    """
    
    def __init__(self, initial_capital: float = 10000):
        """
        Initialize portfolio tracker
        
        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades_history: List[Trade] = []
        self.equity_curve: List[Dict] = []
        
        # Performance metrics
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'peak_equity': initial_capital
        }
        
    def execute_trade(
        self,
        symbol: str,
        action: str,
        shares: int,
        price: float,
        commission: float = 0.0,
        slippage: float = 0.0,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Execute a trade
        
        Args:
            symbol: Stock symbol
            action: 'BUY' or 'SELL'
            shares: Number of shares
            price: Execution price
            commission: Commission cost
            slippage: Slippage cost
            timestamp: Trade timestamp
            
        Returns:
            Success status
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Create trade object
        trade = Trade(
            timestamp=timestamp,
            symbol=symbol,
            action=action,
            shares=shares,
            price=price,
            commission=commission,
            slippage=slippage
        )
        
        # Check if trade is feasible
        if action == 'BUY':
            required_cash = trade.value
            if required_cash > self.cash:
                logger.warning(f"Insufficient cash: required {required_cash}, available {self.cash}")
                return False
        
        elif action == 'SELL':
            if symbol not in self.positions or self.positions[symbol].shares < shares:
                available = self.positions[symbol].shares if symbol in self.positions else 0
                logger.warning(f"Insufficient shares: trying to sell {shares}, available {available}")
                return False
        
        # Execute trade
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        
        position = self.positions[symbol]
        position.update(trade)
        
        # Update cash
        if action == 'BUY':
            self.cash -= trade.value
        else:
            self.cash += trade.price * trade.shares - trade.commission - trade.slippage
        
        # Record trade
        self.trades_history.append(trade)
        self.metrics['total_trades'] += 1
        
        # Update win/loss statistics
        if action == 'SELL' and position.realized_pnl > 0:
            self.metrics['winning_trades'] += 1
        elif action == 'SELL' and position.realized_pnl < 0:
            self.metrics['losing_trades'] += 1
        
        logger.info(f"Executed trade: {action} {shares} {symbol} @ ${price:.2f}")
        
        return True
    
    def update_equity(self, current_prices: Dict[str, float], timestamp: Optional[datetime] = None):
        """
        Update equity curve with current prices
        
        Args:
            current_prices: Current prices for each symbol
            timestamp: Current timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate total equity
        equity = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in current_prices and position.shares > 0:
                equity += position.shares * current_prices[symbol]
        
        # Update metrics
        self.metrics['peak_equity'] = max(self.metrics['peak_equity'], equity)
        drawdown = (equity - self.metrics['peak_equity']) / self.metrics['peak_equity']
        self.metrics['max_drawdown'] = min(self.metrics['max_drawdown'], drawdown)
        
        # Record equity
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity,
            'cash': self.cash,
            'positions_value': equity - self.cash,
            'drawdown': drawdown
        })
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Get current portfolio value"""
        value = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in current_prices and position.shares > 0:
                value += position.shares * current_prices[symbol]
        
        return value
    
    def get_position_summary(self, current_prices: Dict[str, float]) -> pd.DataFrame:
        """Get summary of all positions"""
        summaries = []
        
        for symbol, position in self.positions.items():
            if position.shares > 0:
                current_price = current_prices.get(symbol, position.avg_entry_price)
                
                summary = {
                    'symbol': symbol,
                    'shares': position.shares,
                    'avg_entry_price': position.avg_entry_price,
                    'current_price': current_price,
                    'market_value': position.shares * current_price,
                    'unrealized_pnl': position.get_unrealized_pnl(current_price),
                    'realized_pnl': position.realized_pnl,
                    'total_pnl': position.get_total_pnl(current_price),
                    'pnl_pct': (current_price / position.avg_entry_price - 1) * 100 if position.avg_entry_price > 0 else 0
                }
                
                summaries.append(summary)
        
        return pd.DataFrame(summaries)
    
    def get_performance_metrics(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate performance metrics"""
        current_equity = self.get_portfolio_value(current_prices)
        
        # Calculate total P&L
        total_pnl = sum(pos.get_total_pnl(current_prices.get(sym, pos.avg_entry_price)) 
                       for sym, pos in self.positions.items())
        
        metrics = {
            'initial_capital': self.initial_capital,
            'current_equity': current_equity,
            'total_return': (current_equity - self.initial_capital) / self.initial_capital,
            'total_pnl': total_pnl,
            'cash': self.cash,
            'positions_value': current_equity - self.cash,
            'total_trades': self.metrics['total_trades'],
            'winning_trades': self.metrics['winning_trades'],
            'losing_trades': self.metrics['losing_trades'],
            'win_rate': self.metrics['winning_trades'] / max(1, self.metrics['winning_trades'] + self.metrics['losing_trades']),
            'max_drawdown': self.metrics['max_drawdown'],
            'current_drawdown': (current_equity - self.metrics['peak_equity']) / self.metrics['peak_equity']
        }
        
        # Calculate Sharpe ratio if we have equity history
        if len(self.equity_curve) > 20:
            returns = pd.DataFrame(self.equity_curve)['equity'].pct_change().dropna()
            metrics['sharpe_ratio'] = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        else:
            metrics['sharpe_ratio'] = 0
        
        return metrics
    
    def reset(self):
        """Reset portfolio to initial state"""
        self.cash = self.initial_capital
        self.positions.clear()
        self.trades_history.clear()
        self.equity_curve.clear()
        
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'peak_equity': self.initial_capital
        }
        
        logger.info("Portfolio reset to initial state")