# strategy/concrete_strategies/three_level_strategy.py

import pandas as pd
import pandas_ta as ta
import numpy as np
from strategy.abstract_strategy import AbstractStrategyBase, Signal

class ThreeLevelStrategy(AbstractStrategyBase):
    """
    三級交易策略系統
    
    Level 1: 單一指標信號（基礎信號，信賴度較低）
    Level 2: 雙指標共振（中等信賴度）
    Level 3: 三指標及以上共振（高信賴度）
    
    每次交易僅動用資金的1%
    """
    
    def _initialize_parameters(self):
        """初始化所有策略參數"""
        # MA 參數
        self.short_ma_period = self.parameters.get('short_ma_period', 5)
        self.long_ma_period = self.parameters.get('long_ma_period', 20)
        
        # BIAS 參數
        self.bias_period = self.parameters.get('bias_period', 20)
        self.bias_upper = self.parameters.get('bias_upper', 7.0)
        self.bias_lower = self.parameters.get('bias_lower', -8.0)
        
        # KD 參數 (Slow KD)
        self.kd_k = self.parameters.get('kd_k', 14)
        self.kd_d = self.parameters.get('kd_d', 3)
        self.kd_smooth = self.parameters.get('kd_smooth', 3)
        self.kd_overbought = self.parameters.get('kd_overbought', 80)
        self.kd_oversold = self.parameters.get('kd_oversold', 20)
        
        # MACD 參數
        self.macd_fast = self.parameters.get('macd_fast', 12)
        self.macd_slow = self.parameters.get('macd_slow', 26)
        self.macd_signal = self.parameters.get('macd_signal', 9)
        
        # RSI 參數
        self.rsi_period = self.parameters.get('rsi_period', 14)
        self.rsi_overbought = self.parameters.get('rsi_overbought', 70)
        self.rsi_oversold = self.parameters.get('rsi_oversold', 30)
        
        # Bollinger Bands 參數
        self.bb_period = self.parameters.get('bb_period', 20)
        self.bb_std = self.parameters.get('bb_std', 2.0)
        
        # Ichimoku 參數
        self.ichi_tenkan = self.parameters.get('ichi_tenkan', 9)
        self.ichi_kijun = self.parameters.get('ichi_kijun', 26)
        self.ichi_senkou_b = self.parameters.get('ichi_senkou_b', 52)
        
        # Volume 參數
        self.vol_ma_period = self.parameters.get('vol_ma_period', 20)
        self.vol_multiplier = self.parameters.get('vol_multiplier', 1.5)
        
        # ATR 風險管理參數
        self.atr_period = self.parameters.get('atr_period', 14)
        self.atr_multiplier_sl = self.parameters.get('atr_multiplier_sl', 2.0)
        self.atr_multiplier_tp = self.parameters.get('atr_multiplier_tp', 4.0)
        
        # 資金管理
        self.risk_per_trade = self.parameters.get('risk_per_trade', 0.01)  # 1%
        
        # 交易狀態
        self.in_position = False
        self.position_side = None
        self.entry_level = 0  # 記錄進場時的信號級別

    def calculate_bias(self, close, ma):
        """計算乖離率"""
        return ((close - ma) / ma) * 100
    
    def detect_candlestick_patterns(self, data_slice):
        """檢測K線形態"""
        patterns = {
            'bullish_engulfing': False,
            'bearish_engulfing': False,
            'hammer': False,
            'shooting_star': False
        }
        
        if len(data_slice) < 3:
            return patterns
            
        # 當前和前一根K線
        curr = data_slice.iloc[-1]
        prev = data_slice.iloc[-2]
        
        # 看漲吞噬
        if (prev['Close'] < prev['Open'] and  # 前一根是陰線
            curr['Close'] > curr['Open'] and  # 當前是陽線
            curr['Open'] <= prev['Close'] and  # 開盤價低於前收盤
            curr['Close'] >= prev['Open']):    # 收盤價高於前開盤
            patterns['bullish_engulfing'] = True
            
        # 看跌吞噬
        if (prev['Close'] > prev['Open'] and  # 前一根是陽線
            curr['Close'] < curr['Open'] and  # 當前是陰線
            curr['Open'] >= prev['Close'] and  # 開盤價高於前收盤
            curr['Close'] <= prev['Open']):    # 收盤價低於前開盤
            patterns['bearish_engulfing'] = True
            
        # 鎚子線（簡化版）
        body = abs(curr['Close'] - curr['Open'])
        lower_shadow = min(curr['Open'], curr['Close']) - curr['Low']
        upper_shadow = curr['High'] - max(curr['Open'], curr['Close'])
        
        if lower_shadow > body * 2 and upper_shadow < body * 0.5:
            patterns['hammer'] = True
            
        # 射擊之星（簡化版）
        if upper_shadow > body * 2 and lower_shadow < body * 0.5:
            patterns['shooting_star'] = True
            
        return patterns
    
    def calculate_fibonacci_levels(self, high_series, low_series):
        """計算斐波那契回撤位"""
        # 找出近期高低點（簡化版，使用最近50根K線）
        lookback = min(50, len(high_series))
        recent_high = high_series.iloc[-lookback:].max()
        recent_low = low_series.iloc[-lookback:].min()
        
        diff = recent_high - recent_low
        
        levels = {
            'high': recent_high,
            'low': recent_low,
            '0.236': recent_high - diff * 0.236,
            '0.382': recent_high - diff * 0.382,
            '0.5': recent_high - diff * 0.5,
            '0.618': recent_high - diff * 0.618,
            '0.786': recent_high - diff * 0.786
        }
        
        return levels
    
    def check_dow_theory_trend(self, high_series, low_series):
        """檢查道氏理論趨勢（HH-HL上升趨勢，LH-LL下降趨勢）"""
        if len(high_series) < 10:
            return 'neutral'
            
        # 找出最近的高低點
        highs = []
        lows = []
        
        for i in range(2, len(high_series)-2):
            # 高點：前後兩根K線的高點都低於當前
            if (high_series.iloc[i] > high_series.iloc[i-1] and 
                high_series.iloc[i] > high_series.iloc[i-2] and
                high_series.iloc[i] > high_series.iloc[i+1] and 
                high_series.iloc[i] > high_series.iloc[i+2]):
                highs.append((i, high_series.iloc[i]))
                
            # 低點：前後兩根K線的低點都高於當前
            if (low_series.iloc[i] < low_series.iloc[i-1] and 
                low_series.iloc[i] < low_series.iloc[i-2] and
                low_series.iloc[i] < low_series.iloc[i+1] and 
                low_series.iloc[i] < low_series.iloc[i+2]):
                lows.append((i, low_series.iloc[i]))
        
        # 需要至少兩個高點和兩個低點
        if len(highs) >= 2 and len(lows) >= 2:
            # 檢查最近的兩個高點和低點
            if highs[-1][1] > highs[-2][1] and lows[-1][1] > lows[-2][1]:
                return 'uptrend'  # HH & HL
            elif highs[-1][1] < highs[-2][1] and lows[-1][1] < lows[-2][1]:
                return 'downtrend'  # LH & LL
                
        return 'neutral'

    def check_level1_signals(self, indicators, patterns):
        """檢查第一級信號（單一指標）"""
        buy_signals = []
        sell_signals = []
        
        # KD黃金交叉在超賣區
        if (indicators['k_line'] < self.kd_oversold and 
            indicators['k_line'] > indicators['d_line'] and
            indicators['k_line_prev'] <= indicators['d_line_prev']):
            buy_signals.append('KD_golden_cross_oversold')
            
        # KD死亡交叉在超買區
        if (indicators['k_line'] > self.kd_overbought and 
            indicators['k_line'] < indicators['d_line'] and
            indicators['k_line_prev'] >= indicators['d_line_prev']):
            sell_signals.append('KD_death_cross_overbought')
            
        # RSI從超賣區向上突破
        if (indicators['rsi'] > self.rsi_oversold and 
            indicators['rsi_prev'] <= self.rsi_oversold):
            buy_signals.append('RSI_oversold_breakup')
            
        # RSI從超買區向下跌破
        if (indicators['rsi'] < self.rsi_overbought and 
            indicators['rsi_prev'] >= self.rsi_overbought):
            sell_signals.append('RSI_overbought_breakdown')
            
        # MACD柱狀體由綠翻紅
        if indicators['macd_hist'] > 0 and indicators['macd_hist_prev'] <= 0:
            buy_signals.append('MACD_hist_bullish')
            
        # MACD柱狀體由紅翻綠
        if indicators['macd_hist'] < 0 and indicators['macd_hist_prev'] >= 0:
            sell_signals.append('MACD_hist_bearish')
            
        # BIAS極端值
        if indicators['bias'] < self.bias_lower:
            buy_signals.append('BIAS_extreme_low')
        if indicators['bias'] > self.bias_upper:
            sell_signals.append('BIAS_extreme_high')
            
        # 布林通道信號
        if (indicators['price_prev'] < indicators['bb_lower_prev'] and 
            indicators['price'] > indicators['bb_lower']):
            buy_signals.append('BB_lower_bounce')
            
        if (indicators['price_prev'] > indicators['bb_upper_prev'] and 
            indicators['price'] < indicators['bb_upper']):
            sell_signals.append('BB_upper_rejection')
            
        # MA交叉
        if (indicators['short_ma'] > indicators['long_ma'] and 
            indicators['short_ma_prev'] <= indicators['long_ma_prev']):
            buy_signals.append('MA_golden_cross')
            
        if (indicators['short_ma'] < indicators['long_ma'] and 
            indicators['short_ma_prev'] >= indicators['long_ma_prev']):
            sell_signals.append('MA_death_cross')
            
        # K線形態
        if patterns['bullish_engulfing']:
            buy_signals.append('bullish_engulfing')
        if patterns['bearish_engulfing']:
            sell_signals.append('bearish_engulfing')
            
        return buy_signals, sell_signals
    
    def check_level2_signals(self, indicators, patterns):
        """檢查第二級信號（雙指標共振）"""
        buy_signals = []
        sell_signals = []
        
        # 趨勢確認 + RSI回檔
        if (indicators['short_ma'] > indicators['long_ma'] and 
            indicators['rsi'] > self.rsi_oversold and 
            indicators['rsi_prev'] <= self.rsi_oversold):
            buy_signals.append('Trend_RSI_pullback_buy')
            
        if (indicators['short_ma'] < indicators['long_ma'] and 
            indicators['rsi'] < self.rsi_overbought and 
            indicators['rsi_prev'] >= self.rsi_overbought):
            sell_signals.append('Trend_RSI_pullback_sell')
            
        # 雲帶 + MACD
        if (indicators['price'] > indicators['cloud_top'] and 
            indicators['macd_hist'] > 0 and 
            indicators['macd_hist_prev'] <= 0):
            buy_signals.append('Cloud_MACD_bullish')
            
        if (indicators['price'] < indicators['cloud_bottom'] and 
            indicators['macd_hist'] < 0 and 
            indicators['macd_hist_prev'] >= 0):
            sell_signals.append('Cloud_MACD_bearish')
            
        # BIAS + K線形態
        if indicators['bias'] < self.bias_lower and patterns['bullish_engulfing']:
            buy_signals.append('BIAS_candlestick_buy')
            
        if indicators['bias'] > self.bias_upper and patterns['bearish_engulfing']:
            sell_signals.append('BIAS_candlestick_sell')
            
        # 布林通道擠壓突破
        bb_width = indicators['bb_upper'] - indicators['bb_lower']
        bb_width_prev = indicators['bb_upper_prev'] - indicators['bb_lower_prev']
        bb_squeeze = bb_width < bb_width_prev * 0.8  # 帶寬收窄
        
        if (bb_squeeze and 
            indicators['price'] > indicators['bb_upper'] and 
            indicators['volume'] > indicators['vol_ma'] * self.vol_multiplier):
            buy_signals.append('BB_squeeze_breakout_up')
            
        if (bb_squeeze and 
            indicators['price'] < indicators['bb_lower'] and 
            indicators['volume'] > indicators['vol_ma'] * self.vol_multiplier):
            sell_signals.append('BB_squeeze_breakout_down')
            
        # 斐波那契 + K線形態
        fib_levels = indicators['fib_levels']
        price_at_618 = abs(indicators['price'] - fib_levels['0.618']) < indicators['atr'] * 0.5
        price_at_50 = abs(indicators['price'] - fib_levels['0.5']) < indicators['atr'] * 0.5
        
        if price_at_618 and patterns['hammer']:
            buy_signals.append('Fib_618_hammer')
            
        if price_at_618 and patterns['shooting_star']:
            sell_signals.append('Fib_618_shooting_star')
            
        return buy_signals, sell_signals
    
    def check_level3_signals(self, indicators, patterns):
        """檢查第三級信號（三指標及以上共振）"""
        buy_signals = []
        sell_signals = []
        
        # 雲帶 + RSI + 布林通道
        bb_width = indicators['bb_upper'] - indicators['bb_lower']
        bb_width_prev = indicators['bb_upper_prev'] - indicators['bb_lower_prev']
        bb_squeeze = bb_width < bb_width_prev * 0.8
        
        if (indicators['price'] > indicators['cloud_top'] and 
            indicators['rsi'] > self.rsi_oversold and 
            indicators['rsi_prev'] <= self.rsi_oversold and
            bb_squeeze and indicators['price'] > indicators['bb_upper']):
            buy_signals.append('Cloud_RSI_BB_bullish')
            
        if (indicators['price'] < indicators['cloud_bottom'] and 
            indicators['rsi'] < self.rsi_overbought and 
            indicators['rsi_prev'] >= self.rsi_overbought and
            bb_squeeze and indicators['price'] < indicators['bb_lower']):
            sell_signals.append('Cloud_RSI_BB_bearish')
            
        # MA交叉 + MACD + 成交量
        if (indicators['short_ma'] > indicators['long_ma'] and 
            indicators['short_ma_prev'] <= indicators['long_ma_prev'] and
            indicators['macd_hist'] > 0 and
            indicators['volume'] > indicators['vol_ma'] * self.vol_multiplier):
            buy_signals.append('MA_MACD_Volume_bullish')
            
        if (indicators['short_ma'] < indicators['long_ma'] and 
            indicators['short_ma_prev'] >= indicators['long_ma_prev'] and
            indicators['macd_hist'] < 0 and
            indicators['volume'] > indicators['vol_ma'] * self.vol_multiplier):
            sell_signals.append('MA_MACD_Volume_bearish')
            
        # 道氏理論 + MA支撐/壓力 + KD
        dow_trend = indicators['dow_trend']
        price_near_ma = abs(indicators['price'] - indicators['long_ma']) < indicators['atr'] * 0.5
        
        if (dow_trend == 'uptrend' and 
            price_near_ma and 
            indicators['k_line'] < self.kd_oversold and 
            indicators['k_line'] > indicators['d_line']):
            buy_signals.append('Dow_MA_KD_bullish')
            
        if (dow_trend == 'downtrend' and 
            price_near_ma and 
            indicators['k_line'] > self.kd_overbought and 
            indicators['k_line'] < indicators['d_line']):
            sell_signals.append('Dow_MA_KD_bearish')
            
        # 雲帶 + 斐波那契 + K線形態
        fib_levels = indicators['fib_levels']
        price_at_50 = abs(indicators['price'] - fib_levels['0.5']) < indicators['atr'] * 0.5
        
        if (indicators['price'] > indicators['cloud_top'] and 
            price_at_50 and 
            patterns['bullish_engulfing']):
            buy_signals.append('Cloud_Fib_Candlestick_bullish')
            
        if (indicators['price'] < indicators['cloud_bottom'] and 
            price_at_50 and 
            patterns['bearish_engulfing']):
            sell_signals.append('Cloud_Fib_Candlestick_bearish')
            
        return buy_signals, sell_signals

    def on_data(self, data_slice: pd.DataFrame) -> list:
        """處理市場數據並生成交易信號"""
        signals = []
        
        # 確保有足夠的數據
        min_length = max(self.long_ma_period, self.bias_period, self.macd_slow, 
                        self.ichi_senkou_b, self.vol_ma_period, self.atr_period, 50)
        if len(data_slice) < min_length + 5:
            return signals
            
        # 準備數據
        close = data_slice['Close']
        high = data_slice['High']
        low = data_slice['Low']
        volume = data_slice['Volume']
        
        # 計算所有技術指標
        try:
            # MA
            short_ma = ta.ema(close, length=self.short_ma_period)
            long_ma = ta.ema(close, length=self.long_ma_period)
            
            # BIAS
            bias_ma = ta.sma(close, length=self.bias_period)
            bias = self.calculate_bias(close, bias_ma)
            
            # KD
            stoch = ta.stoch(high, low, close, k=self.kd_k, d=self.kd_d, smooth_k=self.kd_smooth)
            if stoch is None or stoch.empty:
                return signals
            k_line = stoch[f'STOCHk_{self.kd_k}_{self.kd_d}_{self.kd_smooth}']
            d_line = stoch[f'STOCHd_{self.kd_k}_{self.kd_d}_{self.kd_smooth}']
            
            # MACD
            macd = ta.macd(close, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
            if macd is None or macd.empty:
                return signals
            macd_hist = macd[f'MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
            
            # RSI
            rsi = ta.rsi(close, length=self.rsi_period)
            
            # Bollinger Bands
            bbands = ta.bbands(close, length=self.bb_period, std=self.bb_std)
            if bbands is None or bbands.empty:
                return signals
            bb_upper = bbands[f'BBU_{self.bb_period}_{self.bb_std}']
            bb_middle = bbands[f'BBM_{self.bb_period}_{self.bb_std}']
            bb_lower = bbands[f'BBL_{self.bb_period}_{self.bb_std}']
            
            # Ichimoku
            ichimoku = ta.ichimoku(high, low, close, 
                                   tenkan=self.ichi_tenkan, 
                                   kijun=self.ichi_kijun, 
                                   senkou=self.ichi_senkou_b)
            if ichimoku is None or len(ichimoku) == 0 or ichimoku[0].empty:
                return signals
            tenkan_sen = ichimoku[0][f'ITS_{self.ichi_tenkan}']
            kijun_sen = ichimoku[0][f'IKS_{self.ichi_kijun}']
            senkou_a = ichimoku[0][f'ISA_{self.ichi_tenkan}']
            senkou_b = ichimoku[0][f'ISB_{self.ichi_kijun}']
            
            # Volume
            vol_ma = ta.sma(volume, length=self.vol_ma_period)
            
            # ATR
            atr = ta.atr(high, low, close, length=self.atr_period)
            
        except Exception as e:
            print(f"指標計算錯誤: {e}")
            return signals
            
        # 檢查所有指標是否有效
        if any(ind is None or (hasattr(ind, 'empty') and ind.empty) 
               for ind in [short_ma, long_ma, bias, rsi, vol_ma, atr]):
            return signals
            
        # 獲取當前和前一個值
        current_price = close.iloc[-1]
        current_timestamp = data_slice.index[-1]
        current_volume = volume.iloc[-1]
        
        # 準備指標字典
        indicators = {
            'price': current_price,
            'price_prev': close.iloc[-2],
            'short_ma': short_ma.iloc[-1],
            'short_ma_prev': short_ma.iloc[-2],
            'long_ma': long_ma.iloc[-1],
            'long_ma_prev': long_ma.iloc[-2],
            'bias': bias.iloc[-1],
            'k_line': k_line.iloc[-1],
            'k_line_prev': k_line.iloc[-2],
            'd_line': d_line.iloc[-1],
            'd_line_prev': d_line.iloc[-2],
            'macd_hist': macd_hist.iloc[-1],
            'macd_hist_prev': macd_hist.iloc[-2],
            'rsi': rsi.iloc[-1],
            'rsi_prev': rsi.iloc[-2],
            'bb_upper': bb_upper.iloc[-1],
            'bb_upper_prev': bb_upper.iloc[-2],
            'bb_lower': bb_lower.iloc[-1],
            'bb_lower_prev': bb_lower.iloc[-2],
            'cloud_top': max(senkou_a.iloc[-1], senkou_b.iloc[-1]),
            'cloud_bottom': min(senkou_a.iloc[-1], senkou_b.iloc[-1]),
            'volume': current_volume,
            'vol_ma': vol_ma.iloc[-1],
            'atr': atr.iloc[-1],
            'fib_levels': self.calculate_fibonacci_levels(high, low),
            'dow_trend': self.check_dow_theory_trend(high, low)
        }
        
        # 檢測K線形態
        patterns = self.detect_candlestick_patterns(data_slice)
        
        # 檢查各級別信號
        level1_buy, level1_sell = self.check_level1_signals(indicators, patterns)
        level2_buy, level2_sell = self.check_level2_signals(indicators, patterns)
        level3_buy, level3_sell = self.check_level3_signals(indicators, patterns)
        
        # 決定交易信號
        signal_level = 0
        signal_type = None
        signal_reasons = []
        
        # 優先檢查Level 3信號
        if level3_buy:
            signal_level = 3
            signal_type = 'BUY'
            signal_reasons = level3_buy
        elif level3_sell:
            signal_level = 3
            signal_type = 'SELL'
            signal_reasons = level3_sell
        # 其次檢查Level 2信號
        elif level2_buy:
            signal_level = 2
            signal_type = 'BUY'
            signal_reasons = level2_buy
        elif level2_sell:
            signal_level = 2
            signal_type = 'SELL'
            signal_reasons = level2_sell
        # 最後檢查Level 1信號（可選：只在沒有持倉時考慮）
        elif not self.in_position:
            if level1_buy:
                signal_level = 1
                signal_type = 'BUY'
                signal_reasons = level1_buy
            elif level1_sell:
                signal_level = 1
                signal_type = 'SELL'
                signal_reasons = level1_sell
        
        # 生成交易信號
        if not self.in_position and signal_level > 0:
            # 根據信號級別調整倉位大小
            position_size = self.risk_per_trade * signal_level  # Level 1: 1%, Level 2: 2%, Level 3: 3%
            
            if signal_type == 'BUY':
                sl_price = current_price - (atr.iloc[-1] * self.atr_multiplier_sl)
                tp_price = current_price + (atr.iloc[-1] * self.atr_multiplier_tp)
                
                signals.append(Signal(
                    timestamp=current_timestamp,
                    symbol=self.symbol,
                    action='BUY_ENTRY',
                    quantity=position_size,
                    price=current_price,
                    sl=sl_price,
                    tp=tp_price,
                    comment=f'L{signal_level}_Buy_{",".join(signal_reasons[:2])}'
                ))
                self.in_position = True
                self.position_side = 'long'
                self.entry_level = signal_level
                
            elif signal_type == 'SELL':
                sl_price = current_price + (atr.iloc[-1] * self.atr_multiplier_sl)
                tp_price = current_price - (atr.iloc[-1] * self.atr_multiplier_tp)
                
                signals.append(Signal(
                    timestamp=current_timestamp,
                    symbol=self.symbol,
                    action='SELL_ENTRY',
                    quantity=position_size,
                    price=current_price,
                    sl=sl_price,
                    tp=tp_price,
                    comment=f'L{signal_level}_Sell_{",".join(signal_reasons[:2])}'
                ))
                self.in_position = True
                self.position_side = 'short'
                self.entry_level = signal_level
                
        # 出場邏輯
        elif self.in_position:
            exit_signal = False
            exit_reason = ""
            
            # 根據進場級別決定出場條件
            if self.entry_level == 3:
                # Level 3進場：需要Level 2以上反向信號才出場
                if self.position_side == 'long' and (level2_sell or level3_sell):
                    exit_signal = True
                    exit_reason = f"L{3 if level3_sell else 2}_Exit_Sell"
                elif self.position_side == 'short' and (level2_buy or level3_buy):
                    exit_signal = True
                    exit_reason = f"L{3 if level3_buy else 2}_Exit_Buy"
                    
            elif self.entry_level == 2:
                # Level 2進場：任何反向信號都出場
                if self.position_side == 'long' and (level1_sell or level2_sell or level3_sell):
                    exit_signal = True
                    exit_reason = "Exit_Sell_Signal"
                elif self.position_side == 'short' and (level1_buy or level2_buy or level3_buy):
                    exit_signal = True
                    exit_reason = "Exit_Buy_Signal"
                    
            else:  # Level 1進場
                # Level 1進場：更嚴格的出場條件
                if self.position_side == 'long' and level1_sell:
                    exit_signal = True
                    exit_reason = "L1_Exit_Sell"
                elif self.position_side == 'short' and level1_buy:
                    exit_signal = True
                    exit_reason = "L1_Exit_Buy"
                    
            if exit_signal:
                signals.append(Signal(
                    timestamp=current_timestamp,
                    symbol=self.symbol,
                    action='CLOSE_POSITION',
                    price=current_price,
                    comment=exit_reason
                ))
                self.in_position = False
                self.position_side = None
                self.entry_level = 0
                
        return signals
        
    def get_indicator_definitions(self) -> dict:
        """返回用於繪圖的指標定義"""
        return {
            'short_ma': (ta.ema, {'length': self.short_ma_period}),
            'long_ma': (ta.ema, {'length': self.long_ma_period}),
            'rsi': (ta.rsi, {'length': self.rsi_period}),
            'bb_upper': (lambda x: ta.bbands(x, length=self.bb_period, std=self.bb_std)[f'BBU_{self.bb_period}_{self.bb_std}'], {}),
            'bb_lower': (lambda x: ta.bbands(x, length=self.bb_period, std=self.bb_std)[f'BBL_{self.bb_period}_{self.bb_std}'], {}),
        } 