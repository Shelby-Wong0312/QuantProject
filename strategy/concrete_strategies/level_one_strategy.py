# strategy/concrete_strategies/level_one_strategy.py

import pandas as pd
import pandas_ta as ta
from strategy.abstract_strategy import AbstractStrategyBase, Signal

class LevelOneStrategy(AbstractStrategyBase):
    """
    實現了所有等級一指標的「或」邏輯策略。
    任何一個指標觸發信號，都會執行交易。
    """
    def _initialize_parameters(self):
        # 載入所有需要的指標參數
        self.short_ma_period = self.parameters.get('short_ma_period', 5)
        self.long_ma_period = self.parameters.get('long_ma_period', 20)
        self.bias_period = self.parameters.get('bias_period', 20)
        self.bias_upper = self.parameters.get('bias_upper', 7.0)
        self.bias_lower = self.parameters.get('bias_lower', -8.0)
        self.kd_k = self.parameters.get('kd_k', 14)
        self.kd_d = self.parameters.get('kd_d', 3)
        self.kd_smooth = self.parameters.get('kd_smooth', 3)
        self.kd_overbought = self.parameters.get('kd_overbought', 80)
        self.kd_oversold = self.parameters.get('kd_oversold', 20)
        self.macd_fast = self.parameters.get('macd_fast', 12)
        self.macd_slow = self.parameters.get('macd_slow', 26)
        self.macd_signal = self.parameters.get('macd_signal', 9)
        self.rsi_period = self.parameters.get('rsi_period', 14)
        self.rsi_overbought = self.parameters.get('rsi_overbought', 70)
        self.rsi_oversold = self.parameters.get('rsi_oversold', 30)
        self.bb_period = self.parameters.get('bb_period', 20)
        self.bb_std = self.parameters.get('bb_std', 2.0)
        self.atr_period = self.parameters.get('atr_period', 14)
        self.atr_multiplier_sl = self.parameters.get('atr_multiplier_sl', 2.0)
        self.live_trade_quantity = self.parameters.get('live_trade_quantity', 0.1)

        self.position = 0  # -1 for short, 0 for flat, 1 for long

    def on_data(self, data_slice: pd.DataFrame) -> list:
        signals = []
        
        # 確保數據長度足以計算最長的指標
        min_required_length = max(self.long_ma_period, self.bias_period, self.macd_slow)
        if len(data_slice) < min_required_length:
            return signals

        # 準備數據
        close = data_slice['Close']
        high = data_slice['High']
        low = data_slice['Low']
        opn = data_slice['Open']
        
        # --- 1. 計算所有指標 ---
        short_ma = ta.sma(close, length=self.short_ma_period)
        long_ma = ta.sma(close, length=self.long_ma_period)
        bias = ta.bias(close, length=self.bias_period)
        stoch = ta.stoch(high=high, low=low, close=close, k=self.kd_k, d=self.kd_d, smooth_k=self.kd_smooth)
        macd = ta.macd(close, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        rsi = ta.rsi(close, length=self.rsi_period)
        bbands = ta.bbands(close, length=self.bb_period, std=self.bb_std)
        atr = ta.atr(high=high, low=low, close=close, length=self.atr_period)

        # 檢查指標是否已生成
        if any(s is None for s in [short_ma, long_ma, bias, stoch, macd, rsi, bbands, atr]) or bbands.empty:
            return signals

        # --- 2. 定義所有做多信號條件 ---
        ma_golden_cross = (short_ma.iloc[-2] < long_ma.iloc[-2]) and (short_ma.iloc[-1] > long_ma.iloc[-1])
        bias_oversold = bias.iloc[-1] < self.bias_lower
        kd_golden_cross = (stoch.iloc[-2, 0] < stoch.iloc[-2, 1]) and (stoch.iloc[-1, 0] > stoch.iloc[-1, 1]) and (stoch.iloc[-1, 0] < self.kd_oversold)
        macd_bullish_flip = (macd.iloc[-2, 1] < 0) and (macd.iloc[-1, 1] > 0)
        rsi_oversold_cross = (rsi.iloc[-2] < self.rsi_oversold) and (rsi.iloc[-1] > self.rsi_oversold)
        bb_mean_reversion_buy = (close.iloc[-2] < bbands[f'BBL_{self.bb_period}_{self.bb_std}'].iloc[-2]) and (close.iloc[-1] > bbands[f'BBL_{self.bb_period}_{self.bb_std}'].iloc[-1])
        bullish_engulfing = (opn.iloc[-1] < close.iloc[-2]) and (close.iloc[-1] > opn.iloc[-2]) and (close.iloc[-2] < opn.iloc[-2]) and (close.iloc[-1] > opn.iloc[-1])

        # --- 3. 定義所有做空信號條件 ---
        ma_death_cross = (short_ma.iloc[-2] > long_ma.iloc[-2]) and (short_ma.iloc[-1] < long_ma.iloc[-1])
        bias_overbought = bias.iloc[-1] > self.bias_upper
        kd_death_cross = (stoch.iloc[-2, 0] > stoch.iloc[-2, 1]) and (stoch.iloc[-1, 0] < stoch.iloc[-1, 1]) and (stoch.iloc[-1, 0] > self.kd_overbought)
        macd_bearish_flip = (macd.iloc[-2, 1] > 0) and (macd.iloc[-1, 1] < 0)
        rsi_overbought_cross = (rsi.iloc[-2] > self.rsi_overbought) and (rsi.iloc[-1] < self.rsi_overbought)
        bb_mean_reversion_sell = (close.iloc[-2] > bbands[f'BBU_{self.bb_period}_{self.bb_std}'].iloc[-2]) and (close.iloc[-1] < bbands[f'BBU_{self.bb_period}_{self.bb_std}'].iloc[-1])
        bearish_engulfing = (opn.iloc[-1] > close.iloc[-2]) and (close.iloc[-1] < opn.iloc[-2]) and (close.iloc[-2] > opn.iloc[-2]) and (close.iloc[-1] < opn.iloc[-1])
        
        # --- 4. 整合邏輯 ---
        long_signal_triggered = any([ma_golden_cross, bias_oversold, kd_golden_cross, macd_bullish_flip, rsi_oversold_cross, bb_mean_reversion_buy, bullish_engulfing])
        short_signal_triggered = any([ma_death_cross, bias_overbought, kd_death_cross, macd_bearish_flip, rsi_overbought_cross, bb_mean_reversion_sell, bearish_engulfing])

        current_price = close.iloc[-1]
        current_timestamp = data_slice.index[-1]
        sl_offset = atr.iloc[-1] * self.atr_multiplier_sl

        if self.position == 0: # 如果目前空倉
            if long_signal_triggered:
                signals.append(Signal(
                    timestamp=current_timestamp, symbol=self.symbol, action='BUY_ENTRY', 
                    price=current_price, sl=current_price - sl_offset, quantity=self.live_trade_quantity,
                    comment='L1_Long_Entry'
                ))
                self.position = 1
            elif short_signal_triggered:
                signals.append(Signal(
                    timestamp=current_timestamp, symbol=self.symbol, action='SELL_ENTRY',
                    price=current_price, sl=current_price + sl_offset, quantity=self.live_trade_quantity,
                    comment='L1_Short_Entry'
                ))
                self.position = -1
        elif self.position == 1: # 如果持有多倉
            if short_signal_triggered: # 任何一個反向信號出現就平倉
                signals.append(Signal(
                    timestamp=current_timestamp, symbol=self.symbol, action='CLOSE_LONG_CONDITION',
                    price=current_price, comment='Close_Long_On_Short_Signal'
                ))
                self.position = 0
        elif self.position == -1: # 如果持有空倉
            if long_signal_triggered: # 任何一個反向信號出現就回補
                signals.append(Signal(
                    timestamp=current_timestamp, symbol=self.symbol, action='CLOSE_SHORT_CONDITION',
                    price=current_price, comment='Close_Short_On_Long_Signal'
                ))
                self.position = 0
        
        return signals

    def get_indicator_definitions(self) -> dict:
        return {} # 實盤交易不需要繪圖