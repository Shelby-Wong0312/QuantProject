# 檔案位置: adapters/backtesting_adapter.py

from backtesting import Strategy as BacktestingStrategy
from backtesting.lib import crossover
import pandas as pd

# 根據您的路徑，使用 strategy (單數)
from strategy.abstract_strategy import AbstractStrategyBase, Signal

class BacktestingPyAdapter(BacktestingStrategy):
    """
    一個適配器，用於將繼承自 AbstractStrategyBase 的策略連接到 backtesting.py 框架。
    """
    
    rsi_period = 14
    short_ma_period = 10
    long_ma_period = 30
    
    abstract_strategy_class = None
    strategy_params = {}

    def init(self):
        """
        由 backtesting.py 呼叫的初始化方法。
        """
        params_for_strategy = self.strategy_params.copy()
        params_for_strategy.update({
            'rsi_period': self.rsi_period,
            'short_ma_period': self.short_ma_period,
            'long_ma_period': self.long_ma_period,
        })
        
        if not self.abstract_strategy_class:
            raise ValueError("必須提供 abstract_strategy_class 參數")
            
        self.strategy = self.abstract_strategy_class(parameters=params_for_strategy)

        # --- 以下是修正的部分 ---
        # (可選) 註冊指標以供 backtesting.py 繪圖。
        # 這個部分比較敏感，我們先將其註解掉，以確保核心回測能夠順利運行。
        # 策略的核心邏輯仍然會在 next() 中正確計算指標並執行。
        
        # self.indicators_def = self.strategy.get_indicator_definitions()
        # for name, (func, params) in self.indicators_def.items():
        #     indicator_series = self.data.Close
        #     self.I(func, indicator_series, **params, name=name)
        # --- 修正結束 ---


    def next(self):
        """
        由 backtesting.py 在每根 K 棒上呼叫的核心方法。
        """
        current_bar_index = len(self.data.Close) - 1
        data_slice = self.data.df.iloc[:current_bar_index + 1]

        signals = self.strategy.on_data(data_slice)

        if not signals:
            return

        for signal in signals:
            if signal.action == 'BUY_ENTRY':
                self.buy(sl=signal.sl, tp=signal.tp)
            
            elif signal.action == 'CLOSE_LONG_CONDITION':
                self.position.close()
                