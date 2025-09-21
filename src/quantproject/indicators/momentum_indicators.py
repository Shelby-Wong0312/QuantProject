"""
Momentum Indicators - RSI, MACD, Stochastic and other momentum indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from .base_indicator import BaseIndicator
import logging

logger = logging.getLogger(__name__)


class RSI(BaseIndicator):
    """Relative Strength Index"""
    
    def __init__(self, period: int = 14, use_cache: bool = True):
        super().__init__(period, use_cache)
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate RSI"""
        if not self.validate(data):
            return pd.Series()
            
        cache_key = self._get_cache_key(data)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
            
        # Calculate price changes
        delta = data['close'].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=self.period, min_periods=1).mean()
        avg_loss = loss.rolling(window=self.period, min_periods=1).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Handle division by zero
        rsi = rsi.fillna(50)
        
        self._save_to_cache(cache_key, rsi)
        return rsi
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate overbought/oversold signals"""
        rsi = self.calculate(data)
        signals = pd.DataFrame(index=data.index)
        signals['rsi'] = rsi
        
        # Overbought when RSI > 70, Oversold when RSI < 30
        signals['overbought'] = rsi > 70
        signals['oversold'] = rsi < 30
        
        # Buy signal when RSI crosses above 30 (exiting oversold)
        signals['buy'] = (rsi > 30) & (rsi.shift(1) <= 30)
        
        # Sell signal when RSI crosses below 70 (exiting overbought)
        signals['sell'] = (rsi < 70) & (rsi.shift(1) >= 70)
        
        signals['hold'] = ~(signals['buy'] | signals['sell'])
        
        return signals


class MACD(BaseIndicator):
    """Moving Average Convergence Divergence"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, 
                 signal_period: int = 9, use_cache: bool = True):
        super().__init__(period=slow_period, use_cache=use_cache)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD, Signal, and Histogram"""
        if not self.validate(data):
            return pd.DataFrame()
            
        # Calculate EMAs
        ema_fast = data['close'].ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = data['close'].ewm(span=self.slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate Signal line
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        
        # Calculate Histogram
        histogram = macd_line - signal_line
        
        result = pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }, index=data.index)
        
        return result
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate MACD crossover signals"""
        macd_data = self.calculate(data)
        signals = pd.DataFrame(index=data.index)
        
        signals['macd'] = macd_data['macd']
        signals['signal'] = macd_data['signal']
        signals['histogram'] = macd_data['histogram']
        
        # Bullish signal when MACD crosses above Signal
        signals['bullish_cross'] = ((macd_data['macd'] > macd_data['signal']) & 
                                    (macd_data['macd'].shift(1) <= macd_data['signal'].shift(1)))
        
        # Bearish signal when MACD crosses below Signal
        signals['bearish_cross'] = ((macd_data['macd'] < macd_data['signal']) & 
                                    (macd_data['macd'].shift(1) >= macd_data['signal'].shift(1)))
        
        signals['buy'] = signals['bullish_cross']
        signals['sell'] = signals['bearish_cross']
        signals['hold'] = ~(signals['buy'] | signals['sell'])
        
        # Trend strength (positive histogram = bullish momentum)
        signals['momentum'] = np.where(macd_data['histogram'] > 0, 'bullish', 'bearish')
        
        return signals


class Stochastic(BaseIndicator):
    """Stochastic Oscillator"""
    
    def __init__(self, k_period: int = 14, d_period: int = 3, use_cache: bool = True):
        super().__init__(period=k_period, use_cache=use_cache)
        self.k_period = k_period
        self.d_period = d_period
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate %K and %D"""
        if not self.validate(data):
            return pd.DataFrame()
            
        # Calculate %K
        low_min = data['low'].rolling(window=self.k_period).min()
        high_max = data['high'].rolling(window=self.k_period).max()
        
        k_percent = 100 * ((data['close'] - low_min) / (high_max - low_min))
        
        # Calculate %D (3-period SMA of %K)
        d_percent = k_percent.rolling(window=self.d_period).mean()
        
        result = pd.DataFrame({
            'k': k_percent,
            'd': d_percent
        }, index=data.index)
        
        return result
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Stochastic signals"""
        stoch_data = self.calculate(data)
        signals = pd.DataFrame(index=data.index)
        
        signals['k'] = stoch_data['k']
        signals['d'] = stoch_data['d']
        
        # Overbought/Oversold
        signals['overbought'] = stoch_data['k'] > 80
        signals['oversold'] = stoch_data['k'] < 20
        
        # Crossover signals
        signals['bullish_cross'] = ((stoch_data['k'] > stoch_data['d']) & 
                                    (stoch_data['k'].shift(1) <= stoch_data['d'].shift(1)) & 
                                    (stoch_data['k'] < 80))
        
        signals['bearish_cross'] = ((stoch_data['k'] < stoch_data['d']) & 
                                    (stoch_data['k'].shift(1) >= stoch_data['d'].shift(1)) & 
                                    (stoch_data['k'] > 20))
        
        signals['buy'] = signals['bullish_cross']
        signals['sell'] = signals['bearish_cross']
        signals['hold'] = ~(signals['buy'] | signals['sell'])
        
        return signals


class WilliamsR(BaseIndicator):
    """Williams %R Indicator"""
    
    def __init__(self, period: int = 14, use_cache: bool = True):
        super().__init__(period, use_cache)
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Williams %R"""
        if not self.validate(data):
            return pd.Series()
            
        # Calculate Williams %R
        highest_high = data['high'].rolling(window=self.period).max()
        lowest_low = data['low'].rolling(window=self.period).min()
        
        williams_r = -100 * ((highest_high - data['close']) / (highest_high - lowest_low))
        
        return williams_r
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Williams %R signals"""
        williams_r = self.calculate(data)
        signals = pd.DataFrame(index=data.index)
        signals['williams_r'] = williams_r
        
        # Overbought when > -20, Oversold when < -80
        signals['overbought'] = williams_r > -20
        signals['oversold'] = williams_r < -80
        
        # Buy signal when exiting oversold
        signals['buy'] = (williams_r > -80) & (williams_r.shift(1) <= -80)
        
        # Sell signal when exiting overbought
        signals['sell'] = (williams_r < -20) & (williams_r.shift(1) >= -20)
        
        signals['hold'] = ~(signals['buy'] | signals['sell'])
        
        return signals


class CCI(BaseIndicator):
    """Commodity Channel Index"""
    
    def __init__(self, period: int = 20, use_cache: bool = True):
        super().__init__(period, use_cache)
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate CCI"""
        if not self.validate(data):
            return pd.Series()
            
        # Calculate Typical Price
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate SMA of Typical Price
        sma = typical_price.rolling(window=self.period).mean()
        
        # Calculate Mean Deviation
        mean_deviation = typical_price.rolling(window=self.period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x)))
        )
        
        # Calculate CCI
        cci = (typical_price - sma) / (0.015 * mean_deviation)
        
        return cci
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate CCI signals"""
        cci = self.calculate(data)
        signals = pd.DataFrame(index=data.index)
        signals['cci'] = cci
        
        # Overbought when > 100, Oversold when < -100
        signals['overbought'] = cci > 100
        signals['oversold'] = cci < -100
        
        # Buy signal when CCI crosses above -100
        signals['buy'] = (cci > -100) & (cci.shift(1) <= -100)
        
        # Sell signal when CCI crosses below 100
        signals['sell'] = (cci < 100) & (cci.shift(1) >= 100)
        
        signals['hold'] = ~(signals['buy'] | signals['sell'])
        
        # Trend strength
        signals['trend'] = np.where(cci > 0, 'bullish', 'bearish')
        
        return signals