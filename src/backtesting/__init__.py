"""
Backtesting Engine Module

This module provides a comprehensive backtesting framework for:
- Strategy simulation and evaluation
- Portfolio management and tracking
- Performance metrics calculation
- Transaction cost modeling
"""

from .engine import BacktestEngine
from .portfolio import Portfolio, Position
from .performance import PerformanceAnalyzer
from .models import TransactionCostModel, SlippageModel
from .strategy_base import Strategy

__all__ = [
    'BacktestEngine',
    'Portfolio',
    'Position',
    'PerformanceAnalyzer',
    'TransactionCostModel',
    'SlippageModel',
    'Strategy'
]