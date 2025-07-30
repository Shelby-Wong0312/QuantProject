# quant_project/main_backtest.py
# FINAL FIX - Adjusted end_date for free Alpaca plan

import logging
from datetime import datetime, timedelta  # <--- 新增導入 timedelta
from backtesting import Backtest, Strategy
from alpaca_trade_api.rest import TimeFrame
import pandas as pd

# 匯入我們自己的模組
import config
from data_pipeline.history_loader import HistoryLoader
from strategies import indicators

# --- 日誌設定 ---
if __name__ == "__main__":
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format='%(asctime)s - %(name)-15s - %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
logger = logging.getLogger(__name__)


# --- 將我們的策略與 backtesting.py 函式庫對接 ---
class ComprehensiveStrategyAdapter(Strategy):
    """轉接器，將我們的完整策略對接到 backtesting.py 引擎。"""
    
    strategy_params = config.STRATEGY_PARAMS['Comprehensive_v1']
    
    def init(self):
        self.p = self.strategy_params
        price_df = self.data.df.copy()
        self.indicator_df = indicators.add_all_indicators(price_df, self.p)

    def next(self):
        if self.data.index[-1] not in self.indicator_df.index:
            return
        
        latest = self.indicator_df.loc[self.data.index[-1]]
        patterns = indicators.get_candlestick_patterns(self.data.df.iloc[:len(self.data)])
        buy_signals, sell_signals = self._check_level1(latest, patterns)
        
        if buy_signals and not self.position:
            self.buy()
        elif sell_signals and not self.position:
            self.sell()

    def _check_level1(self, latest: pd.Series, patterns: dict):
        buy, sell = [], []
        p = self.p
        
        if latest.get(f'STOCHk_{p["kd_k"]}_{p["kd_d"]}_{p["kd_smooth"]}'):
            if latest[f'STOCHk_{p["kd_k"]}_{p["kd_d"]}_{p["kd_smooth"]}'] < p['kd_oversold']: buy.append(1)
            if latest[f'STOCHk_{p["kd_k"]}_{p["kd_d"]}_{p["kd_smooth"]}'] > p['kd_overbought']: sell.append(1)
        if 'rsi' in latest and latest['rsi'] < p['rsi_oversold']: buy.append(1)
        if 'rsi' in latest and latest['rsi'] > p['rsi_overbought']: sell.append(1)
        if f'BBL_{p["bb_period"]}_{p["bb_std"]}' in latest and latest['Close'] < latest[f'BBL_{p["bb_period"]}_{p["bb_std"]}']: buy.append(1)
        if f'BBU_{p["bb_period"]}_{p["bb_std"]}' in latest and latest['Close'] > latest[f'BBU_{p["bb_period"]}_{p["bb_std"]}']: sell.append(1)
        if patterns.get('bullish'): buy.append(1)
        if patterns.get('bearish'): sell.append(1)
            
        return buy, sell


# --- 主回測程序 ---
def run_backtest():
    """執行回測的主函式。"""
    logger.info("--- 開始執行完整策略回測 ---")
    
    loader = HistoryLoader()
    
    symbol_to_backtest = "AAPL"
    start_date = "2020-01-01"
    
    # --- 日期修改處 ---
    # 將結束日期設定為昨天，以符合Alpaca免費數據源的權限
    end_date_dt = datetime.now() - timedelta(days=1)
    end_date = end_date_dt.strftime("%Y-%m-%d")
    # --- 修改結束 ---
    
    logger.info(f"設定回測期間: {start_date} 至 {end_date}")
    
    data = loader.get_bars(symbol_to_backtest, TimeFrame.Day, start_date, end_date)
    if data.empty: return

    bt = Backtest(data, ComprehensiveStrategyAdapter, cash=100_000, commission=.001)
    stats = bt.run()
    
    print(stats)
    
    report_filename = f'backtest_report_{symbol_to_backtest}_{start_date}_to_{end_date}.html'
    bt.plot(filename=report_filename, open_browser=True)
    logger.info(f"已生成詳細回測報告: {report_filename}")


if __name__ == "__main__":
    run_backtest()