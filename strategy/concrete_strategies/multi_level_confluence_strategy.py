# strategy/concrete_strategies/multi_level_confluence_strategy.py

import pandas as pd
import pandas_ta as ta
import numpy as np
from strategy.abstract_strategy import AbstractStrategyBase, Signal

class MultiLevelConfluenceStrategy(AbstractStrategyBase):
    """
    多層級共振交易策略，整合了所有主要技術指標：
    MA, BIAS, KD, MACD, RSI, BB, Ichimoku, Volume, ATR
    
    交易邏輯：
    - Level 1 (弱共振): 3-4個指標一致
    - Level 2 (中共振): 5-6個指標一致  
    - Level 3 (強共振): 7個以上指標一致
    
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
        
        # Ichimoku 參數 (行業標準)
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
        self.position_side = None  # 'long' or 'short'

    def calculate_bias(self, close, ma):
        """計算乖離率"""
        return ((close - ma) / ma) * 100

    def on_data(self, data_slice: pd.DataFrame) -> list:
        """處理市場數據並生成交易信號"""
        signals = []
        
        # 確保有足夠的數據
        min_length = max(self.long_ma_period, self.bias_period, self.macd_slow, 
                        self.ichi_senkou_b, self.vol_ma_period, self.atr_period)
        if len(data_slice) < min_length + 5:  # 額外的緩衝
            return signals
            
        # 準備數據
        close = data_slice['Close']
        high = data_slice['High']
        low = data_slice['Low']
        volume = data_slice['Volume']
        
        # --- 計算所有技術指標 ---
        
        # 1. 移動平均線
        short_ma = ta.ema(close, length=self.short_ma_period)
        long_ma = ta.ema(close, length=self.long_ma_period)
        
        # 2. 乖離率
        bias_ma = ta.sma(close, length=self.bias_period)
        bias = self.calculate_bias(close, bias_ma)
        
        # 3. KD指標 (Slow Stochastic)
        stoch = ta.stoch(high, low, close, k=self.kd_k, d=self.kd_d, smooth_k=self.kd_smooth)
        if stoch is not None and not stoch.empty:
            k_line = stoch[f'STOCHk_{self.kd_k}_{self.kd_d}_{self.kd_smooth}']
            d_line = stoch[f'STOCHd_{self.kd_k}_{self.kd_d}_{self.kd_smooth}']
        else:
            return signals
            
        # 4. MACD
        macd = ta.macd(close, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        if macd is not None and not macd.empty:
            macd_line = macd[f'MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
            signal_line = macd[f'MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
            macd_hist = macd[f'MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
        else:
            return signals
            
        # 5. RSI
        rsi = ta.rsi(close, length=self.rsi_period)
        
        # 6. Bollinger Bands
        bbands = ta.bbands(close, length=self.bb_period, std=self.bb_std)
        if bbands is not None and not bbands.empty:
            bb_upper = bbands[f'BBU_{self.bb_period}_{self.bb_std}']
            bb_middle = bbands[f'BBM_{self.bb_period}_{self.bb_std}']
            bb_lower = bbands[f'BBL_{self.bb_period}_{self.bb_std}']
        else:
            return signals
            
        # 7. Ichimoku Cloud
        ichimoku = ta.ichimoku(high, low, close, 
                               tenkan=self.ichi_tenkan, 
                               kijun=self.ichi_kijun, 
                               senkou=self.ichi_senkou_b)
        if ichimoku is not None and len(ichimoku) > 0 and not ichimoku[0].empty:
            tenkan_sen = ichimoku[0][f'ITS_{self.ichi_tenkan}']
            kijun_sen = ichimoku[0][f'IKS_{self.ichi_kijun}']
            senkou_a = ichimoku[0][f'ISA_{self.ichi_tenkan}']
            senkou_b = ichimoku[0][f'ISB_{self.ichi_kijun}']
        else:
            return signals
            
        # 8. Volume
        vol_ma = ta.sma(volume, length=self.vol_ma_period)
        
        # 9. ATR
        atr = ta.atr(high, low, close, length=self.atr_period)
        
        # 檢查所有指標是否計算成功
        if any(ind is None or (hasattr(ind, 'empty') and ind.empty) 
               for ind in [short_ma, long_ma, bias, rsi, vol_ma, atr]):
            return signals
            
        # --- 獲取當前值 ---
        current_price = close.iloc[-1]
        current_timestamp = data_slice.index[-1]
        current_volume = volume.iloc[-1]
        
        # --- 計算各指標的多空信號 ---
        bullish_signals = 0
        bearish_signals = 0
        
        # 1. MA信號
        if short_ma.iloc[-1] > long_ma.iloc[-1]:
            bullish_signals += 1
        else:
            bearish_signals += 1
            
        # 2. BIAS信號
        if bias.iloc[-1] < self.bias_lower:  # 超賣
            bullish_signals += 1
        elif bias.iloc[-1] > self.bias_upper:  # 超買
            bearish_signals += 1
            
        # 3. KD信號
        if k_line.iloc[-1] < self.kd_oversold and k_line.iloc[-1] > d_line.iloc[-1]:
            bullish_signals += 1
        elif k_line.iloc[-1] > self.kd_overbought and k_line.iloc[-1] < d_line.iloc[-1]:
            bearish_signals += 1
            
        # 4. MACD信號
        if macd_hist.iloc[-1] > 0 and macd_hist.iloc[-1] > macd_hist.iloc[-2]:
            bullish_signals += 1
        elif macd_hist.iloc[-1] < 0 and macd_hist.iloc[-1] < macd_hist.iloc[-2]:
            bearish_signals += 1
            
        # 5. RSI信號
        if rsi.iloc[-1] < self.rsi_oversold:
            bullish_signals += 1
        elif rsi.iloc[-1] > self.rsi_overbought:
            bearish_signals += 1
            
        # 6. BB信號
        if current_price < bb_lower.iloc[-1]:
            bullish_signals += 1
        elif current_price > bb_upper.iloc[-1]:
            bearish_signals += 1
            
        # 7. Ichimoku信號
        if current_price > tenkan_sen.iloc[-1] and current_price > kijun_sen.iloc[-1]:
            bullish_signals += 1
        elif current_price < tenkan_sen.iloc[-1] and current_price < kijun_sen.iloc[-1]:
            bearish_signals += 1
            
        # 8. Volume信號
        if current_volume > vol_ma.iloc[-1] * self.vol_multiplier:
            # 成交量放大，加強當前趨勢
            if bullish_signals > bearish_signals:
                bullish_signals += 1
            elif bearish_signals > bullish_signals:
                bearish_signals += 1
                
        # --- 共振級別判定 ---
        total_bullish = bullish_signals
        total_bearish = bearish_signals
        
        confluence_level = 0
        if total_bullish >= 7:
            confluence_level = 3  # 強共振
        elif total_bullish >= 5:
            confluence_level = 2  # 中共振
        elif total_bullish >= 3:
            confluence_level = 1  # 弱共振
            
        # --- 交易邏輯 ---
        if not self.in_position:
            # 只在Level 2以上進場
            if confluence_level >= 2 and total_bullish > total_bearish:
                # 多頭進場
                sl_price = current_price - (atr.iloc[-1] * self.atr_multiplier_sl)
                tp_price = current_price + (atr.iloc[-1] * self.atr_multiplier_tp)
                
                signals.append(Signal(
                    timestamp=current_timestamp,
                    symbol=self.symbol,
                    action='BUY_ENTRY',
                    quantity=self.risk_per_trade,  # 1%資金
                    price=current_price,
                    sl=sl_price,
                    tp=tp_price,
                    comment=f'Long_L{confluence_level}_Signals:{total_bullish}'
                ))
                self.in_position = True
                self.position_side = 'long'
                
            elif total_bearish >= 5 and total_bearish > total_bullish:
                # 空頭進場
                sl_price = current_price + (atr.iloc[-1] * self.atr_multiplier_sl)
                tp_price = current_price - (atr.iloc[-1] * self.atr_multiplier_tp)
                
                signals.append(Signal(
                    timestamp=current_timestamp,
                    symbol=self.symbol,
                    action='SELL_ENTRY',
                    quantity=self.risk_per_trade,
                    price=current_price,
                    sl=sl_price,
                    tp=tp_price,
                    comment=f'Short_L{2 if total_bearish >= 5 else 1}_Signals:{total_bearish}'
                ))
                self.in_position = True
                self.position_side = 'short'
                
        else:
            # 出場邏輯：反向信號達到Level 1即出場
            if self.position_side == 'long' and total_bearish >= 3:
                signals.append(Signal(
                    timestamp=current_timestamp,
                    symbol=self.symbol,
                    action='CLOSE_POSITION',
                    price=current_price,
                    comment=f'Close_Long_BearSignals:{total_bearish}'
                ))
                self.in_position = False
                self.position_side = None
                
            elif self.position_side == 'short' and total_bullish >= 3:
                signals.append(Signal(
                    timestamp=current_timestamp,
                    symbol=self.symbol,
                    action='CLOSE_POSITION',
                    price=current_price,
                    comment=f'Close_Short_BullSignals:{total_bullish}'
                ))
                self.in_position = False
                self.position_side = None
                
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