# 檔案位置: strategies/concrete_strategies/enhanced_rsi_ma_kd_strategy.py

import pandas as pd
import pandas_ta as ta
from strategy.abstract_strategy import AbstractStrategyBase, Signal

class AbstractEnhancedRsiMaKdStrategy(AbstractStrategyBase):
    """
    一個具體的策略類別，實現了增強型的 RSI + MA + KD + Volume 策略。
    """
    def _initialize_parameters(self):
        """從參數字典中提取並設定所有策略參數，並提供預設值。"""
        self.rsi_period = self.parameters.get('rsi_period', 14)
        self.rsi_oversold = self.parameters.get('rsi_oversold', 30)
        self.rsi_overbought = self.parameters.get('rsi_overbought', 70)
        self.short_ma_period = self.parameters.get('short_ma_period', 10)
        self.long_ma_period = self.parameters.get('long_ma_period', 30)
        self.kd_k_length = self.parameters.get('kd_k_length', 14)
        self.kd_smooth_k = self.parameters.get('kd_smooth_k', 3)
        self.kd_oversold = self.parameters.get('kd_oversold', 20)
        self.kd_overbought = self.parameters.get('kd_overbought', 80)
        self.vol_lookback = self.parameters.get('vol_lookback', 20)
        self.vol_multiplier = self.parameters.get('vol_multiplier', 1.5)
        self.atr_period = self.parameters.get('atr_period', 14)
        self.atr_multiplier_sl = self.parameters.get('atr_multiplier_sl', 2.0)
        self.atr_multiplier_tp = self.parameters.get('atr_multiplier_tp', 3.0)

        self.in_long_position = False

    def on_data(self, data_slice: pd.DataFrame) -> list:
        """處理市場數據，計算指標，並根據策略規則生成交易訊號。"""
        signals = []
        
        min_required_length = max(self.long_ma_period, self.rsi_period, self.kd_k_length, self.vol_lookback, self.atr_period)
        if len(data_slice) < min_required_length:
            return signals

        close = data_slice['Close']
        high = data_slice['High']
        low = data_slice['Low']
        volume = data_slice['Volume']
        
        rsi_series = ta.rsi(close, length=self.rsi_period)
        short_ma = ta.ema(close, length=self.short_ma_period)
        long_ma = ta.ema(close, length=self.long_ma_period)
        stoch_df = ta.stoch(high=high, low=low, close=close, k=self.kd_k_length, d=self.kd_smooth_k, smooth_k=self.kd_smooth_k)
        avg_volume = ta.sma(volume, length=self.vol_lookback)
        atr_series = ta.atr(high=high, low=low, close=close, length=self.atr_period)

        if any(s is None for s in [rsi_series, short_ma, long_ma, stoch_df, avg_volume, atr_series]) or stoch_df.empty:
            return signals

        k_col_name = f'STOCHk_{self.kd_k_length}_{self.kd_smooth_k}_{self.kd_smooth_k}'
        d_col_name = f'STOCHd_{self.kd_k_length}_{self.kd_smooth_k}_{self.kd_smooth_k}'
        
        if k_col_name not in stoch_df.columns or d_col_name not in stoch_df.columns:
             return signals

        rsi_value = rsi_series.iloc[-1]
        k_val = stoch_df[k_col_name].iloc[-1]
        d_val = stoch_df[d_col_name].iloc[-1]
        k_prev = stoch_df[k_col_name].iloc[-2]
        d_prev = stoch_df[d_col_name].iloc[-2]
        
        is_rsi_oversold = rsi_value < self.rsi_oversold
        is_rsi_overbought = rsi_value > self.rsi_overbought
        
        golden_cross_ma = (short_ma.iloc[-2] < long_ma.iloc[-2]) and (short_ma.iloc[-1] > long_ma.iloc[-1])
        death_cross_ma = (short_ma.iloc[-2] > long_ma.iloc[-2]) and (short_ma.iloc[-1] < long_ma.iloc[-1])
        
        is_kd_oversold_cross = (k_val > d_val) and (k_prev < d_prev) and k_val < self.kd_oversold
        
        is_volume_spike = volume.iloc[-1] > (avg_volume.iloc[-1] * self.vol_multiplier)

        current_price = close.iloc[-1]
        current_timestamp = data_slice.index[-1]
        
        if not self.in_long_position:
            # --- 以下是修改的部分 ---
            # 將 'and' 修改為 'or'，任何一個條件滿足就觸發
            if golden_cross_ma or is_rsi_oversold or is_kd_oversold_cross or is_volume_spike:
                stop_loss_price_offset = atr_series.iloc[-1] * self.atr_multiplier_sl
                take_profit_price_offset = atr_series.iloc[-1] * self.atr_multiplier_tp
                
                sl_price = current_price - stop_loss_price_offset
                tp_price = current_price + take_profit_price_offset
                
                entry_signal = Signal(
                    timestamp=current_timestamp, 
                    symbol=self.symbol, 
                    action='BUY_ENTRY', 
                    price=current_price, 
                    sl=sl_price, 
                    tp=tp_price, 
                    comment='L_Entry_OR_Condition'
                )
                signals.append(entry_signal)
                self.in_long_position = True

        elif self.in_long_position:
            # 出場邏輯維持不變
            if death_cross_ma or is_rsi_overbought:
                exit_signal = Signal(
                    timestamp=current_timestamp,
                    symbol=self.symbol,
                    action='CLOSE_LONG_CONDITION',
                    price=current_price,
                    comment='L_Close_MA_RSI'
                )
                signals.append(exit_signal)
                self.in_long_position = False

        return signals

    def get_indicator_definitions(self) -> dict:
        return {
            'rsi': (ta.rsi, {'length': self.rsi_period}),
            'short_ma': (ta.ema, {'length': self.short_ma_period}),
            'long_ma': (ta.ema, {'length': self.long_ma_period}),
        }