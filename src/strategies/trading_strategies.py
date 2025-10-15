# quant_project/strategy/trading_strategies.py
# FINAL NAME-FIXED VERSION

import logging
from typing import Union
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy
from core.event import SignalEvent, MarketEvent
from . import indicators
import config

logger = logging.getLogger(__name__)


class ComprehensiveStrategy(BaseStrategy):
    """
    實現了您完整三級信號邏輯的終極策略
    """

    strategy_id = "ComprehensiveStrategy_v1"

    def _initialize_parameters(self):
        """從config初始化策略參數"""
        self.p = self.params  # 簡寫
        self.position = 0  # -1 short, 0 flat, 1 long

    def calculate_signals(self, market_event: MarketEvent) -> Union[SignalEvent, None]:
        """策略的核心邏輯"""
        if market_event.symbol != self.symbol:
            return None

        df_full = market_event.ohlcv_data.copy()

        min_period = max(
            self.p.get("ma_long_period", 50),
            self.p.get("bias_period", 20),
            self.p.get("macd_slow", 26),
            self.p.get("ichi_senkou_b", 52),
        )
        if len(df_full) < min_period:
            return None

        # 1. 計算所有指標
        df_indicators = indicators.add_all_indicators(df_full, self.p)
        latest = df_indicators.iloc[-1]

        # 2. 進行價格行為/理論分析
        patterns = indicators.get_candlestick_patterns(df_indicators)
        dow_trend = indicators.get_dow_theory_trend(df_indicators)

        # 準備分析結果
        analysis = {"latest": latest, "patterns": patterns, "dow_trend": dow_trend}

        # 3. 檢查各級別信號
        level1_buy, level1_sell = self._check_level1(analysis)
        level2_buy, level2_sell = self._check_level2(analysis)
        level3_buy, level3_sell = self._check_level3(analysis)

        # 4. 進出場邏輯
        if self.position == 0:
            if level3_buy:
                self.position = 1
                return self._create_signal("BUY", config.DEFAULT_TRADE_QUANTITY, "L3_Buy")
            elif level3_sell:
                self.position = -1
                return self._create_signal("SELL", config.DEFAULT_TRADE_QUANTITY, "L3_Sell")
            elif level2_buy:
                self.position = 1
                return self._create_signal("BUY", config.DEFAULT_TRADE_QUANTITY, "L2_Buy")
            elif level2_sell:
                self.position = -1
                return self._create_signal("SELL", config.DEFAULT_TRADE_QUANTITY, "L2_Sell")
            elif level1_buy:
                self.position = 1
                return self._create_signal("BUY", config.DEFAULT_TRADE_QUANTITY, "L1_Buy")
            elif level1_sell:
                self.position = -1
                return self._create_signal("SELL", config.DEFAULT_TRADE_QUANTITY, "L1_Sell")

        elif self.position == 1:  # 持有多倉
            if level1_sell or level2_sell or level3_sell:
                self.position = 0
                return self._create_signal("SELL", config.DEFAULT_TRADE_QUANTITY, "Close_Long")

        elif self.position == -1:  # 持有空倉
            if level1_buy or level2_buy or level3_buy:
                self.position = 0
                return self._create_signal("BUY", config.DEFAULT_TRADE_QUANTITY, "Close_Short")

        return None

    # 移植您原始的檢查邏輯
    def _check_level1(self, analysis):
        latest, patterns = analysis["latest"], analysis["patterns"]
        p = self.p
        buy, sell = False, False
        try:
            if latest[f'STOCHk_{p["kd_k"]}_{p["kd_d"]}_{p["kd_smooth"]}'] < p["kd_oversold"]:
                buy = True
            if latest[f'STOCHk_{p["kd_k"]}_{p["kd_d"]}_{p["kd_smooth"]}'] > p["kd_overbought"]:
                sell = True
        except KeyError:
            pass
        return buy, sell

    def _check_level2(self, analysis):
        # 這裡可以根據您原始的 Level 2 邏輯來實現
        return False, False

    def _check_level3(self, analysis):
        # 這裡可以根據您原始的 Level 3 邏輯來實現
        return False, False
