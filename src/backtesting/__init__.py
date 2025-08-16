"""
Backtesting Framework
Complete backtesting system for quantitative trading strategies

Main Components:
- BacktestEngine: Core backtesting engine
- Portfolio: Portfolio management and tracking
- PerformanceAnalyzer: Performance metrics calculation
- run_backtest: Convenience function for quick backtesting

Example Usage:
    from src.backtesting import run_backtest, BacktestEngine, BacktestConfig
    
    # Quick backtest
    results = run_backtest(strategy, data, initial_capital=100000)
    
    # Advanced backtest
    config = BacktestConfig(initial_capital=100000, commission=0.001)
    engine = BacktestEngine(config)
    engine.add_data(data, 'AAPL')
    results = engine.run_backtest(strategy)
"""

from .backtest_engine import BacktestEngine, BacktestConfig, run_backtest
from .portfolio import Portfolio, Position, Trade
from .performance import PerformanceAnalyzer, calculate_performance_metrics

__all__ = [
    'BacktestEngine',
    'BacktestConfig', 
    'Portfolio',
    'Position',
    'Trade',
    'PerformanceAnalyzer',
    'run_backtest',
    'calculate_performance_metrics'
]

__version__ = '1.0.0'