# In src/strategies.py
from backtesting import Strategy
from backtesting.lib import crossover
import pandas_ta as ta # For calculating EMA
import pandas as pd

def ema(series, length):
    return pd.Series(series).ewm(span=length, adjust=False).mean().values

class MaCrossoverSlTpPercent(Strategy):
    # --- 策略參數 ---
    # MA 週期
    ma_short_len = 5  # 短期均線週期 (您可以後續在 Backtest 初始化時覆寫這些值)
    ma_long_len = 20   # 長期均線週期
    
    # 退出參數
    # 將百分比轉換為小數 (例如 1.5% -> 0.015)
    exit_percentage = 1.5 / 100 

    def init(self):
        # --- 指標計算 ---
        # 移動平均線 (使用 pandas_ta 計算 EMA)
        self.ma_short = self.I(ema, self.data.Close, self.ma_short_len)
        self.ma_long = self.I(ema, self.data.Close, self.ma_long_len)

    def next(self):
        # --- 信號條件 ---
        # MA 信號
        # crossover(series1, series2) -> series1 上穿 series2
        ma_golden_cross = crossover(self.ma_short, self.ma_long)
        # ta.crossunder in Pine Script is equivalent to crossover(series2, series1)
        ma_death_cross = crossover(self.ma_long, self.ma_short) 

        # --- 進場邏輯 ---
        # 黃金交叉做多
        if ma_golden_cross:
            if self.position.is_short: # 如果目前有空倉，先平倉
                self.position.close()
            # 使用幾乎所有可用資金 (Backtesting.py 中 size < 1 代表權益百分比)
            # Pine Script 中的 default_qty_value=100 (strategy.percent_of_equity)
            # 在 Backtesting.py 中可以用略小於1的 size 模擬，例如 0.99 (99% 的權益)
            # 或者，如果您想更精確控制，需要根據可用資金和價格計算單位數
            # 為了簡化，我們先假設一個較大的權益比例
            entry_price = self.data.Close[-1] # 以當前收盤價作為參考計算 SL/TP
            sl_price_long = entry_price * (1 - self.exit_percentage)
            tp_price_long = entry_price * (1 + self.exit_percentage)
            self.buy(size=0.99, sl=sl_price_long, tp=tp_price_long) # 買入，並設置止損止盈

        # 死亡交叉做空
        elif ma_death_cross: # 使用 elif 避免在同一根 K 棒上同時出現金叉又死叉的進場 (雖然不太可能)
            if self.position.is_long: # 如果目前有多倉，先平倉
                self.position.close()
            entry_price = self.data.Close[-1]
            sl_price_short = entry_price * (1 + self.exit_percentage)
            tp_price_short = entry_price * (1 - self.exit_percentage)
            self.sell(size=0.99, sl=sl_price_short, tp=tp_price_short) # 賣出，並設置止損止盈

        # --- 退場邏輯 (固定百分比的止盈/止損) ---
        # 在 Backtesting.py 中，止盈 (tp) 和止損 (sl) 參數是直接在 self.buy() 或 self.sell() 中設定的，
        # 它們是絕對價格水平。一旦設定，回測引擎會自動處理。
        # 上面的進場邏輯已經包含了止盈止損的設定。
        # 因此，不需要像 Pine Script 中那樣在持有倉位後再用 strategy.exit() 單獨設定。