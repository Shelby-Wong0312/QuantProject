# strategy/concrete_strategies/portfolio_three_level_strategy.py

import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime

class PortfolioThreeLevelStrategy:
    """
    投資組合三級交易策略
    
    Level 1: 單一指標信號（基礎信號，信賴度較低）
    Level 2: 雙指標共振（中等信賴度）
    Level 3: 三指標及以上共振（高信賴度）
    """
    
    def __init__(self, strategy_level=3):
        """
        初始化策略
        
        Args:
            strategy_level: 策略級別 (1, 2, 或 3)
        """
        self.strategy_level = strategy_level
        
        # 技術指標參數
        self.short_ma_period = 5
        self.long_ma_period = 20
        self.bias_period = 20
        self.bias_upper = 7.0
        self.bias_lower = -8.0
        self.kd_k = 14
        self.kd_d = 3
        self.kd_smooth = 3
        self.kd_overbought = 80
        self.kd_oversold = 20
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.bb_period = 20
        self.bb_std = 2.0
        self.ichi_tenkan = 9
        self.ichi_kijun = 26
        self.ichi_senkou_b = 52
        self.vol_ma_period = 20
        self.vol_multiplier = 1.5
        self.atr_period = 14
        self.atr_multiplier_sl = 2.0
        
    def calculate_indicators(self, data):
        """計算所有技術指標"""
        indicators = {}
        
        try:
            # 基本數據
            close = data['Close']
            high = data['High']
            low = data['Low']
            volume = data['Volume']
            
            # MA
            indicators['short_ma'] = ta.ema(close, length=self.short_ma_period)
            indicators['long_ma'] = ta.ema(close, length=self.long_ma_period)
            
            # BIAS
            bias_ma = ta.sma(close, length=self.bias_period)
            indicators['bias'] = ((close - bias_ma) / bias_ma) * 100
            
            # KD
            stoch = ta.stoch(high, low, close, k=self.kd_k, d=self.kd_d, smooth_k=self.kd_smooth)
            if stoch is not None and not stoch.empty:
                indicators['k_line'] = stoch[f'STOCHk_{self.kd_k}_{self.kd_d}_{self.kd_smooth}']
                indicators['d_line'] = stoch[f'STOCHd_{self.kd_k}_{self.kd_d}_{self.kd_smooth}']
            
            # MACD
            macd = ta.macd(close, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
            if macd is not None and not macd.empty:
                indicators['macd_hist'] = macd[f'MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
            
            # RSI
            indicators['rsi'] = ta.rsi(close, length=self.rsi_period)
            
            # Bollinger Bands
            bbands = ta.bbands(close, length=self.bb_period, std=self.bb_std)
            if bbands is not None and not bbands.empty:
                indicators['bb_upper'] = bbands[f'BBU_{self.bb_period}_{self.bb_std}']
                indicators['bb_lower'] = bbands[f'BBL_{self.bb_period}_{self.bb_std}']
            
            # Ichimoku
            ichimoku = ta.ichimoku(high, low, close, tenkan=self.ichi_tenkan, kijun=self.ichi_kijun, senkou=self.ichi_senkou_b)
            if ichimoku is not None and len(ichimoku) > 0 and not ichimoku[0].empty:
                indicators['senkou_a'] = ichimoku[0][f'ISA_{self.ichi_tenkan}']
                indicators['senkou_b'] = ichimoku[0][f'ISB_{self.ichi_kijun}']
            
            # Volume
            indicators['vol_ma'] = ta.sma(volume, length=self.vol_ma_period)
            
            # ATR
            indicators['atr'] = ta.atr(high, low, close, length=self.atr_period)
            
            # Fibonacci levels
            indicators['fib_levels'] = self.calculate_fibonacci_levels(high, low)
            
            # Dow Theory trend
            indicators['dow_trend'] = self.check_dow_theory_trend(high, low)
            
            # Candlestick patterns
            indicators['patterns'] = self.detect_candlestick_patterns(data)
            
        except Exception as e:
            print(f"指標計算錯誤: {e}")
            
        return indicators
    
    def calculate_fibonacci_levels(self, high_series, low_series):
        """計算斐波那契回撤位"""
        lookback = min(50, len(high_series))
        recent_high = high_series.iloc[-lookback:].max()
        recent_low = low_series.iloc[-lookback:].min()
        
        diff = recent_high - recent_low
        
        return {
            'high': recent_high,
            'low': recent_low,
            '0.5': recent_high - diff * 0.5,
            '0.618': recent_high - diff * 0.618
        }
    
    def check_dow_theory_trend(self, high_series, low_series):
        """檢查道氏理論趨勢"""
        if len(high_series) < 10:
            return 'neutral'
            
        # 簡化版本的高低點檢測
        highs = []
        lows = []
        
        for i in range(2, len(high_series)-2):
            if (high_series.iloc[i] > high_series.iloc[i-1] and 
                high_series.iloc[i] > high_series.iloc[i+1]):
                highs.append((i, high_series.iloc[i]))
                
            if (low_series.iloc[i] < low_series.iloc[i-1] and 
                low_series.iloc[i] < low_series.iloc[i+1]):
                lows.append((i, low_series.iloc[i]))
        
        if len(highs) >= 2 and len(lows) >= 2:
            if highs[-1][1] > highs[-2][1] and lows[-1][1] > lows[-2][1]:
                return 'uptrend'
            elif highs[-1][1] < highs[-2][1] and lows[-1][1] < lows[-2][1]:
                return 'downtrend'
                
        return 'neutral'
    
    def detect_candlestick_patterns(self, data):
        """檢測K線形態"""
        patterns = {
            'bullish_engulfing': False,
            'bearish_engulfing': False,
            'hammer': False,
            'shooting_star': False
        }
        
        if len(data) < 3:
            return patterns
            
        curr = data.iloc[-1]
        prev = data.iloc[-2]
        
        # 看漲吞噬
        if (prev['Close'] < prev['Open'] and curr['Close'] > curr['Open'] and
            curr['Open'] <= prev['Close'] and curr['Close'] >= prev['Open']):
            patterns['bullish_engulfing'] = True
            
        # 看跌吞噬
        if (prev['Close'] > prev['Open'] and curr['Close'] < curr['Open'] and
            curr['Open'] >= prev['Close'] and curr['Close'] <= prev['Open']):
            patterns['bearish_engulfing'] = True
            
        # 鎚子線和射擊之星
        body = abs(curr['Close'] - curr['Open'])
        lower_shadow = min(curr['Open'], curr['Close']) - curr['Low']
        upper_shadow = curr['High'] - max(curr['Open'], curr['Close'])
        
        if lower_shadow > body * 2 and upper_shadow < body * 0.5:
            patterns['hammer'] = True
        if upper_shadow > body * 2 and lower_shadow < body * 0.5:
            patterns['shooting_star'] = True
            
        return patterns
    
    def check_level1_signals(self, data, indicators):
        """檢查第一級信號（單一指標）"""
        signals = {'buy': False, 'sell': False, 'reasons': []}
        
        if len(data) < 2:
            return signals
            
        close = data['Close'].iloc[-1]
        close_prev = data['Close'].iloc[-2]
        volume = data['Volume'].iloc[-1]
        
        # 買入信號
        # KD黃金交叉在超賣區
        if ('k_line' in indicators and 'd_line' in indicators and
            len(indicators['k_line']) >= 2 and len(indicators['d_line']) >= 2):
            if (indicators['k_line'].iloc[-1] < self.kd_oversold and
                indicators['k_line'].iloc[-1] > indicators['d_line'].iloc[-1] and
                indicators['k_line'].iloc[-2] <= indicators['d_line'].iloc[-2]):
                signals['buy'] = True
                signals['reasons'].append('KD黃金交叉(超賣區)')
        
        # RSI從超賣區向上突破
        if 'rsi' in indicators and len(indicators['rsi']) >= 2:
            if (indicators['rsi'].iloc[-1] > self.rsi_oversold and
                indicators['rsi'].iloc[-2] <= self.rsi_oversold):
                signals['buy'] = True
                signals['reasons'].append('RSI突破超賣區')
        
        # MACD柱狀體由綠翻紅
        if 'macd_hist' in indicators and len(indicators['macd_hist']) >= 2:
            if (indicators['macd_hist'].iloc[-1] > 0 and
                indicators['macd_hist'].iloc[-2] <= 0):
                signals['buy'] = True
                signals['reasons'].append('MACD翻紅')
        
        # BIAS達到-8%以下
        if 'bias' in indicators and indicators['bias'].iloc[-1] < self.bias_lower:
            signals['buy'] = True
            signals['reasons'].append('BIAS超賣')
        
        # 價格觸及布林下軌後收回
        if 'bb_lower' in indicators and len(data) >= 2:
            if (close_prev < indicators['bb_lower'].iloc[-2] and
                close > indicators['bb_lower'].iloc[-1]):
                signals['buy'] = True
                signals['reasons'].append('BB下軌反彈')
        
        # MA黃金交叉
        if ('short_ma' in indicators and 'long_ma' in indicators and
            len(indicators['short_ma']) >= 2 and len(indicators['long_ma']) >= 2):
            if (indicators['short_ma'].iloc[-1] > indicators['long_ma'].iloc[-1] and
                indicators['short_ma'].iloc[-2] <= indicators['long_ma'].iloc[-2]):
                signals['buy'] = True
                signals['reasons'].append('MA黃金交叉')
        
        # 看漲吞噬
        if indicators.get('patterns', {}).get('bullish_engulfing', False):
            signals['buy'] = True
            signals['reasons'].append('看漲吞噬')
        
        # 賣出信號
        # KD死亡交叉在超買區
        if ('k_line' in indicators and 'd_line' in indicators and
            len(indicators['k_line']) >= 2 and len(indicators['d_line']) >= 2):
            if (indicators['k_line'].iloc[-1] > self.kd_overbought and
                indicators['k_line'].iloc[-1] < indicators['d_line'].iloc[-1] and
                indicators['k_line'].iloc[-2] >= indicators['d_line'].iloc[-2]):
                signals['sell'] = True
                signals['reasons'].append('KD死亡交叉(超買區)')
        
        # RSI從超買區向下跌破
        if 'rsi' in indicators and len(indicators['rsi']) >= 2:
            if (indicators['rsi'].iloc[-1] < self.rsi_overbought and
                indicators['rsi'].iloc[-2] >= self.rsi_overbought):
                signals['sell'] = True
                signals['reasons'].append('RSI跌破超買區')
        
        # MACD柱狀體由紅翻綠
        if 'macd_hist' in indicators and len(indicators['macd_hist']) >= 2:
            if (indicators['macd_hist'].iloc[-1] < 0 and
                indicators['macd_hist'].iloc[-2] >= 0):
                signals['sell'] = True
                signals['reasons'].append('MACD翻綠')
        
        # BIAS達到+7%以上
        if 'bias' in indicators and indicators['bias'].iloc[-1] > self.bias_upper:
            signals['sell'] = True
            signals['reasons'].append('BIAS超買')
        
        # 價格觸及布林上軌後收回
        if 'bb_upper' in indicators and len(data) >= 2:
            if (close_prev > indicators['bb_upper'].iloc[-2] and
                close < indicators['bb_upper'].iloc[-1]):
                signals['sell'] = True
                signals['reasons'].append('BB上軌壓力')
        
        # MA死亡交叉
        if ('short_ma' in indicators and 'long_ma' in indicators and
            len(indicators['short_ma']) >= 2 and len(indicators['long_ma']) >= 2):
            if (indicators['short_ma'].iloc[-1] < indicators['long_ma'].iloc[-1] and
                indicators['short_ma'].iloc[-2] >= indicators['long_ma'].iloc[-2]):
                signals['sell'] = True
                signals['reasons'].append('MA死亡交叉')
        
        # 看跌吞噬
        if indicators.get('patterns', {}).get('bearish_engulfing', False):
            signals['sell'] = True
            signals['reasons'].append('看跌吞噬')
        
        return signals
    
    def check_level2_signals(self, data, indicators):
        """檢查第二級信號（雙指標共振）"""
        signals = {'buy': False, 'sell': False, 'reasons': []}
        
        if len(data) < 2:
            return signals
            
        close = data['Close'].iloc[-1]
        volume = data['Volume'].iloc[-1]
        
        # 買入信號
        # 趨勢確認 + RSI回檔
        if ('short_ma' in indicators and 'long_ma' in indicators and 'rsi' in indicators):
            if (indicators['short_ma'].iloc[-1] > indicators['long_ma'].iloc[-1] and
                indicators['rsi'].iloc[-1] > self.rsi_oversold and
                indicators['rsi'].iloc[-2] <= self.rsi_oversold):
                signals['buy'] = True
                signals['reasons'].append('趨勢+RSI回檔')
        
        # 雲帶之上 + MACD翻紅
        if ('senkou_a' in indicators and 'senkou_b' in indicators and 'macd_hist' in indicators):
            cloud_top = max(indicators['senkou_a'].iloc[-1], indicators['senkou_b'].iloc[-1])
            if (close > cloud_top and
                indicators['macd_hist'].iloc[-1] > 0 and
                indicators['macd_hist'].iloc[-2] <= 0):
                signals['buy'] = True
                signals['reasons'].append('雲帶上方+MACD')
        
        # BIAS超賣 + 看漲吞噬
        if 'bias' in indicators and indicators.get('patterns', {}).get('bullish_engulfing', False):
            if indicators['bias'].iloc[-1] < self.bias_lower:
                signals['buy'] = True
                signals['reasons'].append('BIAS+看漲吞噬')
        
        # BB擠壓突破
        if ('bb_upper' in indicators and 'bb_lower' in indicators and 'vol_ma' in indicators):
            bb_width = indicators['bb_upper'].iloc[-1] - indicators['bb_lower'].iloc[-1]
            bb_width_prev = indicators['bb_upper'].iloc[-2] - indicators['bb_lower'].iloc[-2]
            if (bb_width < bb_width_prev * 0.8 and
                close > indicators['bb_upper'].iloc[-1] and
                volume > indicators['vol_ma'].iloc[-1] * self.vol_multiplier):
                signals['buy'] = True
                signals['reasons'].append('BB擠壓突破')
        
        # 斐波那契61.8% + 鎚子線
        if 'fib_levels' in indicators and 'atr' in indicators:
            if (abs(close - indicators['fib_levels']['0.618']) < indicators['atr'].iloc[-1] * 0.5 and
                indicators.get('patterns', {}).get('hammer', False)):
                signals['buy'] = True
                signals['reasons'].append('Fib61.8%+鎚子線')
        
        # 賣出信號
        # 趨勢確認 + RSI回落
        if ('short_ma' in indicators and 'long_ma' in indicators and 'rsi' in indicators):
            if (indicators['short_ma'].iloc[-1] < indicators['long_ma'].iloc[-1] and
                indicators['rsi'].iloc[-1] < self.rsi_overbought and
                indicators['rsi'].iloc[-2] >= self.rsi_overbought):
                signals['sell'] = True
                signals['reasons'].append('趨勢+RSI回落')
        
        # 雲帶之下 + MACD翻綠
        if ('senkou_a' in indicators and 'senkou_b' in indicators and 'macd_hist' in indicators):
            cloud_bottom = min(indicators['senkou_a'].iloc[-1], indicators['senkou_b'].iloc[-1])
            if (close < cloud_bottom and
                indicators['macd_hist'].iloc[-1] < 0 and
                indicators['macd_hist'].iloc[-2] >= 0):
                signals['sell'] = True
                signals['reasons'].append('雲帶下方+MACD')
        
        # BIAS超買 + 看跌吞噬
        if 'bias' in indicators and indicators.get('patterns', {}).get('bearish_engulfing', False):
            if indicators['bias'].iloc[-1] > self.bias_upper:
                signals['sell'] = True
                signals['reasons'].append('BIAS+看跌吞噬')
        
        # BB擠壓跌破
        if ('bb_upper' in indicators and 'bb_lower' in indicators and 'vol_ma' in indicators):
            bb_width = indicators['bb_upper'].iloc[-1] - indicators['bb_lower'].iloc[-1]
            bb_width_prev = indicators['bb_upper'].iloc[-2] - indicators['bb_lower'].iloc[-2]
            if (bb_width < bb_width_prev * 0.8 and
                close < indicators['bb_lower'].iloc[-1] and
                volume > indicators['vol_ma'].iloc[-1] * self.vol_multiplier):
                signals['sell'] = True
                signals['reasons'].append('BB擠壓跌破')
        
        # 斐波那契61.8% + 射擊之星
        if 'fib_levels' in indicators and 'atr' in indicators:
            if (abs(close - indicators['fib_levels']['0.618']) < indicators['atr'].iloc[-1] * 0.5 and
                indicators.get('patterns', {}).get('shooting_star', False)):
                signals['sell'] = True
                signals['reasons'].append('Fib61.8%+射擊之星')
        
        return signals
    
    def check_level3_signals(self, data, indicators):
        """檢查第三級信號（三指標及以上共振）"""
        signals = {'buy': False, 'sell': False, 'reasons': []}
        
        if len(data) < 2:
            return signals
            
        close = data['Close'].iloc[-1]
        volume = data['Volume'].iloc[-1]
        
        # 買入信號
        # 雲帶之上 + RSI回升 + BB擠壓突破
        if ('senkou_a' in indicators and 'senkou_b' in indicators and 
            'rsi' in indicators and 'bb_upper' in indicators and 'bb_lower' in indicators):
            cloud_top = max(indicators['senkou_a'].iloc[-1], indicators['senkou_b'].iloc[-1])
            bb_width = indicators['bb_upper'].iloc[-1] - indicators['bb_lower'].iloc[-1]
            bb_width_prev = indicators['bb_upper'].iloc[-2] - indicators['bb_lower'].iloc[-2]
            
            if (close > cloud_top and
                indicators['rsi'].iloc[-1] > self.rsi_oversold and
                indicators['rsi'].iloc[-2] <= self.rsi_oversold and
                bb_width < bb_width_prev * 0.8 and
                close > indicators['bb_upper'].iloc[-1]):
                signals['buy'] = True
                signals['reasons'].append('雲帶+RSI+BB')
        
        # MA黃金交叉 + MACD翻紅 + 成交量放大
        if ('short_ma' in indicators and 'long_ma' in indicators and 
            'macd_hist' in indicators and 'vol_ma' in indicators):
            if (indicators['short_ma'].iloc[-1] > indicators['long_ma'].iloc[-1] and
                indicators['short_ma'].iloc[-2] <= indicators['long_ma'].iloc[-2] and
                indicators['macd_hist'].iloc[-1] > 0 and
                volume > indicators['vol_ma'].iloc[-1] * self.vol_multiplier):
                signals['buy'] = True
                signals['reasons'].append('MA+MACD+成交量')
        
        # 道氏上升趨勢 + MA支撐 + KD黃金交叉
        if ('dow_trend' in indicators and 'long_ma' in indicators and 
            'k_line' in indicators and 'd_line' in indicators and 'atr' in indicators):
            if (indicators['dow_trend'] == 'uptrend' and
                abs(close - indicators['long_ma'].iloc[-1]) < indicators['atr'].iloc[-1] * 0.5 and
                indicators['k_line'].iloc[-1] < self.kd_oversold and
                indicators['k_line'].iloc[-1] > indicators['d_line'].iloc[-1]):
                signals['buy'] = True
                signals['reasons'].append('道氏+MA+KD')
        
        # 雲帶之上 + 斐波那契50% + 看漲吞噬
        if ('senkou_a' in indicators and 'senkou_b' in indicators and 
            'fib_levels' in indicators and 'atr' in indicators):
            cloud_top = max(indicators['senkou_a'].iloc[-1], indicators['senkou_b'].iloc[-1])
            if (close > cloud_top and
                abs(close - indicators['fib_levels']['0.5']) < indicators['atr'].iloc[-1] * 0.5 and
                indicators.get('patterns', {}).get('bullish_engulfing', False)):
                signals['buy'] = True
                signals['reasons'].append('雲帶+Fib50%+吞噬')
        
        # 賣出信號
        # 雲帶之下 + RSI回落 + BB擠壓跌破
        if ('senkou_a' in indicators and 'senkou_b' in indicators and 
            'rsi' in indicators and 'bb_upper' in indicators and 'bb_lower' in indicators):
            cloud_bottom = min(indicators['senkou_a'].iloc[-1], indicators['senkou_b'].iloc[-1])
            bb_width = indicators['bb_upper'].iloc[-1] - indicators['bb_lower'].iloc[-1]
            bb_width_prev = indicators['bb_upper'].iloc[-2] - indicators['bb_lower'].iloc[-2]
            
            if (close < cloud_bottom and
                indicators['rsi'].iloc[-1] < self.rsi_overbought and
                indicators['rsi'].iloc[-2] >= self.rsi_overbought and
                bb_width < bb_width_prev * 0.8 and
                close < indicators['bb_lower'].iloc[-1]):
                signals['sell'] = True
                signals['reasons'].append('雲帶+RSI+BB')
        
        # MA死亡交叉 + MACD翻綠 + 成交量放大
        if ('short_ma' in indicators and 'long_ma' in indicators and 
            'macd_hist' in indicators and 'vol_ma' in indicators):
            if (indicators['short_ma'].iloc[-1] < indicators['long_ma'].iloc[-1] and
                indicators['short_ma'].iloc[-2] >= indicators['long_ma'].iloc[-2] and
                indicators['macd_hist'].iloc[-1] < 0 and
                volume > indicators['vol_ma'].iloc[-1] * self.vol_multiplier):
                signals['sell'] = True
                signals['reasons'].append('MA+MACD+成交量')
        
        # 道氏下降趨勢 + MA壓力 + KD死亡交叉
        if ('dow_trend' in indicators and 'long_ma' in indicators and 
            'k_line' in indicators and 'd_line' in indicators and 'atr' in indicators):
            if (indicators['dow_trend'] == 'downtrend' and
                abs(close - indicators['long_ma'].iloc[-1]) < indicators['atr'].iloc[-1] * 0.5 and
                indicators['k_line'].iloc[-1] > self.kd_overbought and
                indicators['k_line'].iloc[-1] < indicators['d_line'].iloc[-1]):
                signals['sell'] = True
                signals['reasons'].append('道氏+MA+KD')
        
        # 雲帶之下 + 斐波那契50% + 看跌吞噬
        if ('senkou_a' in indicators and 'senkou_b' in indicators and 
            'fib_levels' in indicators and 'atr' in indicators):
            cloud_bottom = min(indicators['senkou_a'].iloc[-1], indicators['senkou_b'].iloc[-1])
            if (close < cloud_bottom and
                abs(close - indicators['fib_levels']['0.5']) < indicators['atr'].iloc[-1] * 0.5 and
                indicators.get('patterns', {}).get('bearish_engulfing', False)):
                signals['sell'] = True
                signals['reasons'].append('雲帶+Fib50%+吞噬')
        
        return signals
    
    def check_signals(self, data):
        """
        檢查交易信號
        
        Returns:
            dict: {'action': 'buy'/'sell'/None, 'reasons': [原因列表]}
        """
        if len(data) < max(60, self.ichi_senkou_b):  # 確保有足夠數據
            return {'action': None, 'reasons': []}
        
        # 計算所有指標
        indicators = self.calculate_indicators(data)
        
        # 根據策略級別檢查信號
        if self.strategy_level == 1:
            signals = self.check_level1_signals(data, indicators)
        elif self.strategy_level == 2:
            signals = self.check_level2_signals(data, indicators)
        else:  # level 3
            signals = self.check_level3_signals(data, indicators)
        
        # 決定行動
        if signals['buy'] and not signals['sell']:
            return {'action': 'buy', 'reasons': signals['reasons']}
        elif signals['sell'] and not signals['buy']:
            return {'action': 'sell', 'reasons': signals['reasons']}
        else:
            return {'action': None, 'reasons': []} 