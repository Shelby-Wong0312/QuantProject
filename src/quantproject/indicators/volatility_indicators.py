"""
Volatility Indicators - Bollinger Bands, ATR, Keltner Channel, Donchian Channel
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from .base_indicator import BaseIndicator
import logging

logger = logging.getLogger(__name__)


class BollingerBands(BaseIndicator):
    """Bollinger Bands - Price volatility bands"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, use_cache: bool = True):
        super().__init__(period, use_cache)
        self.std_dev = std_dev
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands (Upper, Middle, Lower)"""
        if not self.validate(data):
            return pd.DataFrame()
            
        cache_key = self._get_cache_key(data, std_dev=self.std_dev)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
            
        # Calculate Middle Band (SMA)
        middle_band = data['close'].rolling(window=self.period).mean()
        
        # Calculate Standard Deviation
        std = data['close'].rolling(window=self.period).std()
        
        # Calculate Upper and Lower Bands
        upper_band = middle_band + (std * self.std_dev)
        lower_band = middle_band - (std * self.std_dev)
        
        # Calculate Band Width and %B
        band_width = upper_band - lower_band
        percent_b = (data['close'] - lower_band) / (upper_band - lower_band)
        
        result = pd.DataFrame({
            'upper_band': upper_band,
            'middle_band': middle_band,
            'lower_band': lower_band,
            'band_width': band_width,
            'percent_b': percent_b
        }, index=data.index)
        
        self._save_to_cache(cache_key, result)
        return result
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Bollinger Band breakout signals"""
        bb_data = self.calculate(data)
        signals = pd.DataFrame(index=data.index)
        
        signals['upper_band'] = bb_data['upper_band']
        signals['middle_band'] = bb_data['middle_band']
        signals['lower_band'] = bb_data['lower_band']
        signals['percent_b'] = bb_data['percent_b']
        
        # Breakout signals
        signals['upper_breakout'] = (data['close'] > bb_data['upper_band'])
        signals['lower_breakout'] = (data['close'] < bb_data['lower_band'])
        
        # Buy when price touches lower band and bounces (mean reversion)
        signals['buy'] = ((data['close'] > bb_data['lower_band']) & 
                         (data['close'].shift(1) <= bb_data['lower_band'].shift(1)))
        
        # Sell when price touches upper band and reverses
        signals['sell'] = ((data['close'] < bb_data['upper_band']) & 
                          (data['close'].shift(1) >= bb_data['upper_band'].shift(1)))
        
        signals['hold'] = ~(signals['buy'] | signals['sell'])
        
        # Squeeze detection (low volatility)
        signals['squeeze'] = bb_data['band_width'] < bb_data['band_width'].rolling(100).mean()
        
        return signals


class ATR(BaseIndicator):
    """Average True Range - Volatility indicator"""
    
    def __init__(self, period: int = 14, use_cache: bool = True):
        super().__init__(period, use_cache)
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        if not self.validate(data):
            return pd.Series()
            
        cache_key = self._get_cache_key(data)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
            
        # Calculate True Range
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate ATR (EMA of True Range)
        atr = true_range.ewm(alpha=1/self.period, adjust=False).mean()
        
        self._save_to_cache(cache_key, atr)
        return atr
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility-based signals"""
        atr = self.calculate(data)
        signals = pd.DataFrame(index=data.index)
        
        signals['atr'] = atr
        signals['atr_percent'] = (atr / data['close']) * 100
        
        # High volatility when ATR is above its average
        atr_sma = atr.rolling(window=50).mean()
        signals['high_volatility'] = atr > atr_sma * 1.5
        signals['low_volatility'] = atr < atr_sma * 0.5
        
        # Use ATR for stop-loss levels
        signals['stop_loss_long'] = data['close'] - (atr * 2)
        signals['stop_loss_short'] = data['close'] + (atr * 2)
        
        # Volatility expansion signal (potential breakout)
        signals['volatility_expansion'] = ((atr > atr.shift(1)) & 
                                          (atr.shift(1) > atr.shift(2)) & 
                                          (atr.shift(2) > atr.shift(3)))
        
        signals['buy'] = signals['volatility_expansion'] & (data['close'] > data['close'].shift(1))
        signals['sell'] = signals['volatility_expansion'] & (data['close'] < data['close'].shift(1))
        signals['hold'] = ~(signals['buy'] | signals['sell'])
        
        return signals


class KeltnerChannel(BaseIndicator):
    """Keltner Channel - Volatility-based channel"""
    
    def __init__(self, ema_period: int = 20, atr_period: int = 10, 
                 multiplier: float = 2.0, use_cache: bool = True):
        super().__init__(period=ema_period, use_cache=use_cache)
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.multiplier = multiplier
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Keltner Channel"""
        if not self.validate(data):
            return pd.DataFrame()
            
        cache_key = self._get_cache_key(data, atr_period=self.atr_period, 
                                       multiplier=self.multiplier)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
            
        # Calculate Middle Line (EMA)
        middle_line = data['close'].ewm(span=self.ema_period, adjust=False).mean()
        
        # Calculate ATR
        atr_indicator = ATR(period=self.atr_period)
        atr = atr_indicator.calculate(data)
        
        # Calculate Upper and Lower Channels
        upper_channel = middle_line + (atr * self.multiplier)
        lower_channel = middle_line - (atr * self.multiplier)
        
        result = pd.DataFrame({
            'upper_channel': upper_channel,
            'middle_line': middle_line,
            'lower_channel': lower_channel,
            'channel_width': upper_channel - lower_channel
        }, index=data.index)
        
        self._save_to_cache(cache_key, result)
        return result
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Keltner Channel signals"""
        kc_data = self.calculate(data)
        signals = pd.DataFrame(index=data.index)
        
        signals['upper_channel'] = kc_data['upper_channel']
        signals['middle_line'] = kc_data['middle_line']
        signals['lower_channel'] = kc_data['lower_channel']
        
        # Channel breakouts
        signals['upper_breakout'] = data['close'] > kc_data['upper_channel']
        signals['lower_breakout'] = data['close'] < kc_data['lower_channel']
        
        # Trend following signals
        signals['buy'] = ((data['close'] > kc_data['upper_channel']) & 
                         (data['close'].shift(1) <= kc_data['upper_channel'].shift(1)))
        
        signals['sell'] = ((data['close'] < kc_data['lower_channel']) & 
                          (data['close'].shift(1) >= kc_data['lower_channel'].shift(1)))
        
        signals['hold'] = ~(signals['buy'] | signals['sell'])
        
        # Trend strength
        position_in_channel = ((data['close'] - kc_data['lower_channel']) / 
                              (kc_data['upper_channel'] - kc_data['lower_channel']))
        signals['trend_strength'] = position_in_channel
        
        return signals


class DonchianChannel(BaseIndicator):
    """Donchian Channel - High/Low based channel"""
    
    def __init__(self, period: int = 20, use_cache: bool = True):
        super().__init__(period, use_cache)
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Donchian Channel"""
        if not self.validate(data):
            return pd.DataFrame()
            
        cache_key = self._get_cache_key(data)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
            
        # Calculate Upper Channel (Highest High)
        upper_channel = data['high'].rolling(window=self.period).max()
        
        # Calculate Lower Channel (Lowest Low)
        lower_channel = data['low'].rolling(window=self.period).min()
        
        # Calculate Middle Channel
        middle_channel = (upper_channel + lower_channel) / 2
        
        # Channel width
        channel_width = upper_channel - lower_channel
        
        result = pd.DataFrame({
            'upper_channel': upper_channel,
            'middle_channel': middle_channel,
            'lower_channel': lower_channel,
            'channel_width': channel_width
        }, index=data.index)
        
        self._save_to_cache(cache_key, result)
        return result
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Donchian Channel breakout signals"""
        dc_data = self.calculate(data)
        signals = pd.DataFrame(index=data.index)
        
        signals['upper_channel'] = dc_data['upper_channel']
        signals['middle_channel'] = dc_data['middle_channel']
        signals['lower_channel'] = dc_data['lower_channel']
        
        # Breakout signals (Turtle Trading style)
        signals['upper_breakout'] = ((data['close'] >= dc_data['upper_channel']) & 
                                     (data['close'].shift(1) < dc_data['upper_channel'].shift(1)))
        
        signals['lower_breakout'] = ((data['close'] <= dc_data['lower_channel']) & 
                                     (data['close'].shift(1) > dc_data['lower_channel'].shift(1)))
        
        # Buy on upper breakout (trend following)
        signals['buy'] = signals['upper_breakout']
        
        # Sell on lower breakout
        signals['sell'] = signals['lower_breakout']
        
        signals['hold'] = ~(signals['buy'] | signals['sell'])
        
        # Position in channel (0 = at lower, 1 = at upper)
        signals['position_in_channel'] = ((data['close'] - dc_data['lower_channel']) / 
                                          dc_data['channel_width']).fillna(0.5)
        
        # Narrow channel detection (consolidation)
        avg_width = dc_data['channel_width'].rolling(50).mean()
        signals['consolidation'] = dc_data['channel_width'] < avg_width * 0.7
        
        return signals