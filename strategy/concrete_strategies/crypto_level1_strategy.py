# strategy/concrete_strategies/crypto_level1_strategy.py

import pandas as pd
import pandas_ta as ta
import numpy as np
from strategy.abstract_strategy import AbstractStrategyBase, Signal

class CryptoLevel1Strategy(AbstractStrategyBase):
    """
    虛擬貨幣一級交易策略
    
    Level 1: 單一指標信號觸發交易
    - 任何一個指標滿足條件就進行交易
    - 適合虛擬貨幣的高波動性市場
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
        
        # KD 參數
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
        
        # Volume 參數
        self.vol_ma_period = self.parameters.get('vol_ma_period', 20)
        self.vol_multiplier = self.parameters.get('vol_multiplier', 1.5)
        
        # ATR 風險管理參數
        self.atr_period = self.parameters.get('atr_period', 14)
        self.atr_multiplier_sl = self.parameters.get('atr_multiplier_sl', 2.0)
        self.atr_multiplier_tp = self.parameters.get('atr_multiplier_tp', 3.0)
        
        # 交易狀態
        self.in_position = False

    def calculate_bias(self, close, ma):
        """計算乖離率"""
        return ((close - ma) / ma) * 100

    def on_data(self, data_slice: pd.DataFrame) -> list:
        """處理市場數據並生成交易信號"""
        signals = []
        
        # 確保有足夠的數據
        min_length = max(self.long_ma_period, self.bias_period, self.macd_slow, 
                        self.bb_period, self.vol_ma_period, self.atr_period)
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
            bb_lower = bbands[f'BBL_{self.bb_period}_{self.bb_std}']
            
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
        
        # 檢查買入信號（任何一個條件滿足就觸發）
        buy_reasons = []
        
        if not self.in_position:
            # 1. KD黃金交叉在超賣區
            if (k_line.iloc[-1] < self.kd_oversold and 
                k_line.iloc[-1] > d_line.iloc[-1] and
                k_line.iloc[-2] <= d_line.iloc[-2]):
                buy_reasons.append('KD_golden_cross_oversold')
                
            # 2. RSI從超賣區向上突破
            if (rsi.iloc[-1] > self.rsi_oversold and 
                rsi.iloc[-2] <= self.rsi_oversold):
                buy_reasons.append('RSI_oversold_breakup')
                
            # 3. MACD柱狀體由綠翻紅
            if macd_hist.iloc[-1] > 0 and macd_hist.iloc[-2] <= 0:
                buy_reasons.append('MACD_hist_bullish')
                
            # 4. BIAS極端低值
            if bias.iloc[-1] < self.bias_lower:
                buy_reasons.append('BIAS_extreme_low')
                
            # 5. 布林通道下軌反彈
            if (close.iloc[-2] < bb_lower.iloc[-2] and 
                close.iloc[-1] > bb_lower.iloc[-1]):
                buy_reasons.append('BB_lower_bounce')
                
            # 6. MA黃金交叉
            if (short_ma.iloc[-1] > long_ma.iloc[-1] and 
                short_ma.iloc[-2] <= long_ma.iloc[-2]):
                buy_reasons.append('MA_golden_cross')
                
            # 7. 成交量突破
            if current_volume > vol_ma.iloc[-1] * self.vol_multiplier:
                buy_reasons.append('Volume_breakout')
            
            # 如果有任何買入信號，生成買入訂單
            if buy_reasons:
                # 計算止損和止盈
                stop_loss = current_price - atr.iloc[-1] * self.atr_multiplier_sl
                take_profit = current_price + atr.iloc[-1] * self.atr_multiplier_tp
                
                signal = Signal(
                    timestamp=current_timestamp,
                    symbol=self.symbol,
                    action='BUY_ENTRY',
                    price=current_price,
                    sl=stop_loss,
                    tp=take_profit,
                    comment=f'L1_Buy: {", ".join(buy_reasons)}'
                )
                signals.append(signal)
                self.in_position = True
                
        # 檢查賣出信號
        elif self.in_position:
            sell_reasons = []
            
            # 1. KD死亡交叉在超買區
            if (k_line.iloc[-1] > self.kd_overbought and 
                k_line.iloc[-1] < d_line.iloc[-1] and
                k_line.iloc[-2] >= d_line.iloc[-2]):
                sell_reasons.append('KD_death_cross_overbought')
                
            # 2. RSI從超買區向下跌破
            if (rsi.iloc[-1] < self.rsi_overbought and 
                rsi.iloc[-2] >= self.rsi_overbought):
                sell_reasons.append('RSI_overbought_breakdown')
                
            # 3. MACD柱狀體由紅翻綠
            if macd_hist.iloc[-1] < 0 and macd_hist.iloc[-2] >= 0:
                sell_reasons.append('MACD_hist_bearish')
                
            # 4. BIAS極端高值
            if bias.iloc[-1] > self.bias_upper:
                sell_reasons.append('BIAS_extreme_high')
                
            # 5. 布林通道上軌拒絕
            if (close.iloc[-2] > bb_upper.iloc[-2] and 
                close.iloc[-1] < bb_upper.iloc[-1]):
                sell_reasons.append('BB_upper_rejection')
                
            # 6. MA死亡交叉
            if (short_ma.iloc[-1] < long_ma.iloc[-1] and 
                short_ma.iloc[-2] >= long_ma.iloc[-2]):
                sell_reasons.append('MA_death_cross')
            
            # 如果有任何賣出信號，生成賣出訂單
            if sell_reasons:
                signal = Signal(
                    timestamp=current_timestamp,
                    symbol=self.symbol,
                    action='CLOSE_LONG_CONDITION',
                    price=current_price,
                    comment=f'L1_Sell: {", ".join(sell_reasons)}'
                )
                signals.append(signal)
                self.in_position = False
                
        return signals

    def get_indicator_definitions(self) -> dict:
        """返回指標定義"""
        return {
            'short_ma': (ta.ema, {'length': self.short_ma_period}),
            'long_ma': (ta.ema, {'length': self.long_ma_period}),
            'rsi': (ta.rsi, {'length': self.rsi_period}),
            'macd': (ta.macd, {'fast': self.macd_fast, 'slow': self.macd_slow, 'signal': self.macd_signal}),
            'bb': (ta.bbands, {'length': self.bb_period, 'std': self.bb_std}),
            'atr': (ta.atr, {'length': self.atr_period})
        } 