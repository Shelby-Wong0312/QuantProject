"""
Traditional Trading Strategies Module
傳統交易策略模組
"""

from .ma_crossover import MovingAverageCrossoverStrategy
from .momentum_strategy import MomentumStrategy, create_momentum_strategy
from .mean_reversion import MeanReversionStrategy, create_mean_reversion_strategy
from .breakout_strategy import BreakoutStrategy, create_breakout_strategy
from .trend_following import TrendFollowingStrategy, create_trend_following_strategy

# 保持向後兼容性
try:
    from .breakout import BreakoutStrategy as OldBreakoutStrategy
    from .rsi_strategy import RSIStrategy
    from .bollinger_bands import BollingerBandsStrategy
except ImportError:
    pass

__all__ = [
    'MovingAverageCrossoverStrategy',
    'MomentumStrategy',
    'create_momentum_strategy',
    'MeanReversionStrategy', 
    'create_mean_reversion_strategy',
    'BreakoutStrategy',
    'create_breakout_strategy',
    'TrendFollowingStrategy',
    'create_trend_following_strategy'
]