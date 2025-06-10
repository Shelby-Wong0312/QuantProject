# strategy/concrete_strategies/confluence_strategy.py

import pandas as pd
import pandas_ta as ta
from strategy.abstract_strategy import AbstractStrategyBase, Signal

class ConfluenceStrategy(AbstractStrategyBase):
    """
    一個綜合了多種技術指標（MA, BIAS, KD, MACD, RSI, BB, Ichimoku, Volume, ATR）的
    高共振交易策略。
    """
    def _initialize_parameters(self):
        """從參數字典中提取並設定所有策略參數，並提供預設值。"""
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
        
        # Ichimoku Cloud 參數
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

        self.in_position = False # 簡化為單一倉位狀態

    def on_data(self, data_slice: pd.DataFrame) -> list:
        """處理市場數據，計算指標，並根據策略規則生成交易訊號。"""
        signals = []
        
        min_required_length = max(self.long_ma_period, self.bias_period, self.macd_slow, self.ichi_senkou_b)
        if len(data_slice) < min_required_length:
            return signals

        # 準備數據
        close = data_slice['Close']
        high = data_slice['High']
        low = data_slice['Low']
        volume = data_slice['Volume']
        
        # --- 計算所有指標 ---
        short_ma = ta.ema(close, length=self.short_ma_period)
        long_ma = ta.ema(close, length=self.long_ma_period)
        rsi = ta.rsi(close, length=self.rsi_period)
        macd = ta.macd(close, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        bbands = ta.bbands(close, length=self.bb_period, std=self.bb_std)
        ichimoku = ta.ichimoku(high=high, low=low, close=close, tenkan=self.ichi_tenkan, kijun=self.ichi_kijun, senkou=self.ichi_senkou_b)
        avg_volume = ta.sma(volume, length=self.vol_ma_period)
        atr = ta.atr(high=high, low=low, close=close, length=self.atr_period)

        if any(s is None for s in [short_ma, long_ma, rsi, macd, bbands, ichimoku, avg_volume, atr]) or bbands.empty or ichimoku[0].empty:
            return signals

        # --- 提取當前最新值 ---
        current_price = close.iloc[-1]
        current_timestamp = data_slice.index[-1]
        
        # --- 定義多頭共振條件 (Level 3+) ---
        is_trend_bullish = (current_price > ichimoku[0][f'ITS_{self.ichi_tenkan}'].iloc[-1]) and \
                           (current_price > ichimoku[0][f'IKS_{self.ichi_kijun}'].iloc[-1]) and \
                           (short_ma.iloc[-1] > long_ma.iloc[-1])
                           
        is_momentum_bullish = (rsi.iloc[-1] < self.rsi_overbought) and \
                              (macd[f'MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'].iloc[-1] > 0)

        is_buy_trigger = (close.iloc[-2] < bbands[f'BBL_{self.bb_period}_{self.bb_std}'].iloc[-2]) and \
                         (current_price > bbands[f'BBL_{self.bb_period}_{self.bb_std}'].iloc[-1])

        # --- 定義空頭共振條件 (Level 3+) ---
        is_trend_bearish = (current_price < ichimoku[0][f'ITS_{self.ichi_tenkan}'].iloc[-1]) and \
                           (current_price < ichimoku[0][f'IKS_{self.ichi_kijun}'].iloc[-1]) and \
                           (short_ma.iloc[-1] < long_ma.iloc[-1])

        is_momentum_bearish = (rsi.iloc[-1] > self.rsi_oversold) and \
                              (macd[f'MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'].iloc[-1] < 0)

        is_sell_trigger = (close.iloc[-2] > bbands[f'BBU_{self.bb_period}_{self.bb_std}'].iloc[-2]) and \
                          (current_price < bbands[f'BBU_{self.bb_period}_{self.bb_std}'].iloc[-1])

        # --- 產生信號 ---
        if not self.in_position:
            # 做多信號：趨勢向上 + 動能向上 + 均值回歸觸發
            if is_trend_bullish and is_momentum_bullish and is_buy_trigger:
                sl_price = current_price - (atr.iloc[-1] * self.atr_multiplier_sl)
                tp_price = current_price + (atr.iloc[-1] * self.atr_multiplier_tp)
                
                signals.append(Signal(
                    timestamp=current_timestamp, symbol=self.symbol, action='BUY_ENTRY', 
                    price=current_price, sl=sl_price, tp=tp_price, comment='Long_Confluence'
                ))
                self.in_position = True

            # 做空信號：趨勢向下 + 動能向下 + 均值回歸觸發
            elif is_trend_bearish and is_momentum_bearish and is_sell_trigger:
                sl_price = current_price + (atr.iloc[-1] * self.atr_multiplier_sl)
                tp_price = current_price - (atr.iloc[-1] * self.atr_multiplier_tp)

                signals.append(Signal(
                    timestamp=current_timestamp, symbol=self.symbol, action='SELL_ENTRY',
                    price=current_price, sl=sl_price, tp=tp_price, comment='Short_Confluence'
                ))
                self.in_position = True

        elif self.in_position:
            # 簡化出場：任一趨勢反轉即出場
            if is_trend_bearish or is_trend_bullish: # 如果趨勢與倉位相反
                signals.append(Signal(
                    timestamp=current_timestamp, symbol=self.symbol, action='CLOSE_POSITION',
                    price=current_price, comment='Close_On_Trend_Reversal'
                ))
                self.in_position = False

        return signals

    def get_indicator_definitions(self) -> dict:
        return {
            'short_ma': (ta.ema, {'length': self.short_ma_period}),
            'long_ma': (ta.ema, {'length': self.long_ma_period}),
            'rsi': (ta.rsi, {'length': self.rsi_period}),
        }