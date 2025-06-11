# adapters/backtesting_adapter.py

from backtesting import Strategy
import pandas as pd

class BacktestingPyAdapter(Strategy):
    """
    一個轉接器類別，用於將我們自訂的策略（繼承自 AbstractStrategyBase）
    與 backtesting.py 函式庫進行對接。
    """
    # 這些類別變數將由 bt.run() 方法在執行時動態傳入
    abstract_strategy_class = None
    strategy_params = None

    def init(self):
        """
        初始化轉接器和內部的抽象策略。
        """
        if self.abstract_strategy_class is None:
            raise ValueError("abstract_strategy_class must be provided to Backtest.run()")

        # 1. 實例化我們自己的策略
        # 只傳入參數，不傳入 data
        self.strategy = self.abstract_strategy_class(
            parameters=self.strategy_params or {}
        )
        
        # 2. 執行我們策略的內部初始化（已在構造函數中完成）
        # self.strategy._initialize_parameters()  # 這行可以移除，因為已在 __init__ 中調用
        
        # 3. 為了能在圖表上繪製指標，我們從策略中獲取指標定義
        indicator_defs = self.strategy.get_indicator_definitions()
        for name, (func, params) in indicator_defs.items():
            # 創建一個包裝函數來處理 pandas_ta 的返回值
            def indicator_wrapper(data, func=func, **kwargs):
                result = func(pd.Series(data), **kwargs)
                if result is None:
                    return data  # 如果計算失敗，返回原始數據
                if isinstance(result, pd.Series):
                    return result.values  # 轉換為 numpy array
                return result
            
            # 使用 backtesting.py 的 I() 方法來計算並準備繪製指標
            indicator_series = self.I(indicator_wrapper, self.data.Close, **params)
            # 將指標儲存在 self 中，以便在 next() 中可能使用（雖然目前沒用到）
            setattr(self, name, indicator_series)

    def next(self):
        """
        backtesting.py 在每個時間點（每根K棒）都會呼叫此方法。
        """
        # 1. 準備策略所需的數據切片 (DataFrame)
        # len(self.data) 會給出當前 K 棒的索引+1
        current_data_slice = self.data.df.iloc[:len(self.data)]

        # 2. 呼叫我們策略的核心邏輯
        signals = self.strategy.on_data(current_data_slice)
        
        # 3. 處理策略返回的信號，並將其轉換為 backtesting.py 的操作
        for signal in signals:
            if signal.action == 'BUY_ENTRY':
                # 使用從信號中傳來的止損和止盈價格
                self.buy(sl=signal.sl, tp=signal.tp)
            
            elif signal.action == 'SELL_ENTRY':
                self.sell(sl=signal.sl, tp=signal.tp)

            elif signal.action == 'CLOSE_POSITION':
                # 平掉所有現有倉位
                self.position.close()