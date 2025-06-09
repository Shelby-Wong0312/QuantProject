# 檔案位置: tests/unit/test_enhanced_rsi_ma_kd_strategy.py

import unittest
import pandas as pd
from strategy.concrete_strategies.enhanced_rsi_ma_kd_strategy import AbstractEnhancedRsiMaKdStrategy

class TestEnhancedRsiMaKdStrategyOrLogic(unittest.TestCase):

    def setUp(self):
        """設定基本的策略參數。"""
        self.strategy_params = {
            'symbol': 'BTC-USD', 'short_ma_period': 3, 'long_ma_period': 8,
            'rsi_period': 5, 'rsi_oversold': 30, 'kd_k_length': 5, 'kd_smooth_k': 3,
            'kd_oversold': 20, 'vol_lookback': 5, 'vol_multiplier': 1.5, 'atr_period': 5
        }
        self.strategy = AbstractEnhancedRsiMaKdStrategy(self.strategy_params)

    def test_buy_on_rsi_oversold_only(self):
        """測試僅 RSI 超賣時，是否能觸發買入訊號。"""
        data = {
            'Open':  [100]*10, 'High': [100]*10, 'Low': [100]*10,
            'Close': [100, 90, 80, 70, 60, 50, 40, 30, 25, 20],
            'Volume':[100]*10
        }
        data_slice = pd.DataFrame(data, index=pd.to_datetime(pd.date_range('2023-01-01', periods=10)))
        
        self.strategy.in_long_position = False
        signals = self.strategy.on_data(data_slice)
        
        self.assertEqual(len(signals), 1, "僅 RSI 超賣時應產生一個訊號")
        self.assertEqual(signals[0].action, 'BUY_ENTRY')

    def test_buy_on_golden_cross_only(self):
        """測試僅發生黃金交叉時，是否能觸發買入訊號。"""
        # --- 以下是修正的部分 ---
        # 重新設計的數據，確保黃金交叉正好發生在最後一刻
        data = {
            'Open':  [100]*10, 'High': [100]*10, 'Low': [100]*10,
            'Close': [100, 99, 98, 97, 96, 95, 95, 96, 97, 105],
            'Volume':[100]*10
        }
        data_slice = pd.DataFrame(data, index=pd.to_datetime(pd.date_range('2023-01-01', periods=10)))

        self.strategy.in_long_position = False
        signals = self.strategy.on_data(data_slice)
        
        self.assertEqual(len(signals), 1, "僅黃金交叉時應產生一個訊號")
        self.assertEqual(signals[0].action, 'BUY_ENTRY')

    def test_no_signal_when_no_condition_met(self):
        """測試在沒有任何條件滿足的盤整行情中，不應產生訊號。"""
        data = {
            'Open':  [100]*10, 'High': [101]*10, 'Low': [99]*10,
            'Close': [100.1, 100.2, 100.1, 100.3, 100.2, 100.1, 100.2, 100.3, 100.1, 100.2],
            'Volume':[100]*10
        }
        data_slice = pd.DataFrame(data, index=pd.to_datetime(pd.date_range('2023-01-01', periods=10)))

        self.strategy.in_long_position = False
        signals = self.strategy.on_data(data_slice)
        
        self.assertEqual(len(signals), 0, "無條件滿足時不應產生訊號")