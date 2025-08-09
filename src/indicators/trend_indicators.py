"""
Trend Indicators - Moving averages and trend following indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from .base_indicator import BaseIndicator
import logging

logger = logging.getLogger(__name__)


class SMA(BaseIndicator):
    """Simple Moving Average"""
    
    def __init__(self, period: int = 20, use_cache: bool = True):
        super().__init__(period, use_cache)
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Simple Moving Average"""
        if not self.validate(data):
            return pd.Series()
            
        cache_key = self._get_cache_key(data)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
            
        sma = data['close'].rolling(window=self.period).mean()
        
        self._save_to_cache(cache_key, sma)
        return sma
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on price crossing SMA"""
        sma = self.calculate(data)
        signals = pd.DataFrame(index=data.index)
        signals['sma'] = sma
        signals['price'] = data['close']
        
        # Buy when price crosses above SMA
        signals['buy'] = ((data['close'] > sma) & 
                          (data['close'].shift(1) <= sma.shift(1)))
        
        # Sell when price crosses below SMA
        signals['sell'] = ((data['close'] < sma) & 
                           (data['close'].shift(1) >= sma.shift(1)))
        
        signals['hold'] = ~(signals['buy'] | signals['sell'])
        
        return signals


class EMA(BaseIndicator):
    """Exponential Moving Average"""
    
    def __init__(self, period: int = 20, use_cache: bool = True):
        super().__init__(period, use_cache)
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Exponential Moving Average"""
        if not self.validate(data):
            return pd.Series()
            
        cache_key = self._get_cache_key(data)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
            
        ema = data['close'].ewm(span=self.period, adjust=False).mean()
        
        self._save_to_cache(cache_key, ema)
        return ema
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on price crossing EMA"""
        ema = self.calculate(data)
        signals = pd.DataFrame(index=data.index)
        signals['ema'] = ema
        signals['price'] = data['close']
        
        signals['buy'] = ((data['close'] > ema) & 
                          (data['close'].shift(1) <= ema.shift(1)))
        
        signals['sell'] = ((data['close'] < ema) & 
                           (data['close'].shift(1) >= ema.shift(1)))
        
        signals['hold'] = ~(signals['buy'] | signals['sell'])
        
        return signals


class WMA(BaseIndicator):
    """Weighted Moving Average"""
    
    def __init__(self, period: int = 20, use_cache: bool = True):
        super().__init__(period, use_cache)
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Weighted Moving Average"""
        if not self.validate(data):
            return pd.Series()
            
        cache_key = self._get_cache_key(data)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
            
        weights = np.arange(1, self.period + 1)
        wma = data['close'].rolling(self.period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
        
        self._save_to_cache(cache_key, wma)
        return wma
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on price crossing WMA"""
        wma = self.calculate(data)
        signals = pd.DataFrame(index=data.index)
        signals['wma'] = wma
        signals['price'] = data['close']
        
        signals['buy'] = ((data['close'] > wma) & 
                          (data['close'].shift(1) <= wma.shift(1)))
        
        signals['sell'] = ((data['close'] < wma) & 
                           (data['close'].shift(1) >= wma.shift(1)))
        
        signals['hold'] = ~(signals['buy'] | signals['sell'])
        
        return signals


class VWAP(BaseIndicator):
    """Volume Weighted Average Price"""
    
    def __init__(self, use_cache: bool = True):
        super().__init__(period=1, use_cache=use_cache)  # VWAP doesn't use period
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        if not self.validate(data):
            return pd.Series()
            
        cache_key = self._get_cache_key(data)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
            
        # Calculate typical price
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate VWAP
        cumulative_tpv = (typical_price * data['volume']).cumsum()
        cumulative_volume = data['volume'].cumsum()
        vwap = cumulative_tpv / cumulative_volume
        
        self._save_to_cache(cache_key, vwap)
        return vwap
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on price vs VWAP"""
        vwap = self.calculate(data)
        signals = pd.DataFrame(index=data.index)
        signals['vwap'] = vwap
        signals['price'] = data['close']
        
        # Buy when price is above VWAP (bullish)
        signals['buy'] = ((data['close'] > vwap) & 
                          (data['close'].shift(1) <= vwap.shift(1)))
        
        # Sell when price is below VWAP (bearish)
        signals['sell'] = ((data['close'] < vwap) & 
                           (data['close'].shift(1) >= vwap.shift(1)))
        
        signals['hold'] = ~(signals['buy'] | signals['sell'])
        
        return signals


class MovingAverageCrossover(BaseIndicator):
    """Moving Average Crossover Strategy"""
    
    def __init__(self, fast_period: int = 50, slow_period: int = 200, 
                 ma_type: str = 'EMA', use_cache: bool = True):
        super().__init__(period=slow_period, use_cache=use_cache)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.ma_type = ma_type.upper()
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate both moving averages"""
        if not self.validate(data):
            return pd.DataFrame()
            
        result = pd.DataFrame(index=data.index)
        
        if self.ma_type == 'SMA':
            result['fast_ma'] = data['close'].rolling(window=self.fast_period).mean()
            result['slow_ma'] = data['close'].rolling(window=self.slow_period).mean()
        elif self.ma_type == 'EMA':
            result['fast_ma'] = data['close'].ewm(span=self.fast_period, adjust=False).mean()
            result['slow_ma'] = data['close'].ewm(span=self.slow_period, adjust=False).mean()
        elif self.ma_type == 'WMA':
            # Fast WMA
            weights_fast = np.arange(1, self.fast_period + 1)
            result['fast_ma'] = data['close'].rolling(self.fast_period).apply(
                lambda x: np.dot(x, weights_fast) / weights_fast.sum(), raw=True
            )
            # Slow WMA
            weights_slow = np.arange(1, self.slow_period + 1)
            result['slow_ma'] = data['close'].rolling(self.slow_period).apply(
                lambda x: np.dot(x, weights_slow) / weights_slow.sum(), raw=True
            )
        else:
            raise ValueError(f"Unknown MA type: {self.ma_type}")
            
        return result
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Golden Cross and Death Cross signals"""
        ma_data = self.calculate(data)
        signals = pd.DataFrame(index=data.index)
        
        signals['fast_ma'] = ma_data['fast_ma']
        signals['slow_ma'] = ma_data['slow_ma']
        signals['price'] = data['close']
        
        # Golden Cross - buy signal
        signals['golden_cross'] = ((ma_data['fast_ma'] > ma_data['slow_ma']) & 
                                   (ma_data['fast_ma'].shift(1) <= ma_data['slow_ma'].shift(1)))
        
        # Death Cross - sell signal
        signals['death_cross'] = ((ma_data['fast_ma'] < ma_data['slow_ma']) & 
                                  (ma_data['fast_ma'].shift(1) >= ma_data['slow_ma'].shift(1)))
        
        signals['buy'] = signals['golden_cross']
        signals['sell'] = signals['death_cross']
        signals['hold'] = ~(signals['buy'] | signals['sell'])
        
        # Trend direction
        signals['trend'] = np.where(ma_data['fast_ma'] > ma_data['slow_ma'], 'bullish', 'bearish')
        
        return signals


class GoldenCross(MovingAverageCrossover):
    """Golden Cross Detection (50-day crosses above 200-day)"""
    
    def __init__(self, ma_type: str = 'EMA'):
        super().__init__(fast_period=50, slow_period=200, ma_type=ma_type)
        self.name = "GoldenCross"
        
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Get only Golden Cross signals"""
        all_signals = super().get_signals(data)
        signals = pd.DataFrame(index=data.index)
        signals['golden_cross'] = all_signals['golden_cross']
        signals['buy'] = signals['golden_cross']
        signals['sell'] = False
        signals['hold'] = ~signals['buy']
        return signals


class DeathCross(MovingAverageCrossover):
    """Death Cross Detection (50-day crosses below 200-day)"""
    
    def __init__(self, ma_type: str = 'EMA'):
        super().__init__(fast_period=50, slow_period=200, ma_type=ma_type)
        self.name = "DeathCross"
        
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Get only Death Cross signals"""
        all_signals = super().get_signals(data)
        signals = pd.DataFrame(index=data.index)
        signals['death_cross'] = all_signals['death_cross']
        signals['buy'] = False
        signals['sell'] = signals['death_cross']
        signals['hold'] = ~signals['sell']
        return signals