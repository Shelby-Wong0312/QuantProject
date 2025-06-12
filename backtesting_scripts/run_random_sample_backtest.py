import sys
import os
import argparse
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from backtesting import Backtest
import yfinance as yf
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

from adapters.backtesting_adapter import BacktestingPyAdapter
from strategy.concrete_strategies.three_level_strategy import ThreeLevelStrategy
from backtesting_scripts.report_generator import HTMLReportGenerator

STRATEGY_PARAMS = {
    'short_ma_period': 5,
    'long_ma_period': 20,
    'bias_period': 20,
    'bias_upper': 7.0,
    'bias_lower': -8.0,
    'kd_k': 14,
    'kd_d': 3,
    'kd_smooth': 3,
    'kd_overbought': 80,
    'kd_oversold': 20,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'bb_period': 20,
    'bb_std': 2.0,
    'ichi_tenkan': 9,
    'ichi_kijun': 26,
    'ichi_senkou_b': 52,
    'vol_ma_period': 20,
    'vol_multiplier': 1.5,
    'atr_period': 14,
    'atr_multiplier_sl': 2.0,
    'atr_multiplier_tp': 4.0,
    'risk_per_trade': 0.01,
}

BACKTEST_CONFIG = {
    'start_date': '2023-01-01',
    'end_date': datetime.today().strftime('%Y-%m-%d'),
    'initial_cash': 1000,
    'commission': 0.002,
    'margin': 1.0,
}

def load_random_symbols(filepath="valid_tickers.txt", sample_size=100):
    if not os.path.exists(filepath):
        print(f"找不到 {filepath}，使用預設股票列表")
        return random.sample(["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"], min(sample_size, 5))
    with open(filepath, 'r') as f:
        symbols = [line.strip().upper() for line in f if line.strip()]
    if len(symbols) <= sample_size:
        return symbols
    return random.sample(symbols, sample_size)

def run_single_backtest(ticker, level, show_plot=False, verbose=True):
    if verbose:
        print(f"\n{'='*60}")
        print(f"開始回測 {ticker} - Level {level}")
        print(f"{'='*60}")
    try:
        if verbose:
            print(f"正在下載 {ticker} 的歷史數據...")
        data = yf.download(
            ticker,
            start=BACKTEST_CONFIG['start_date'],
            end=BACKTEST_CONFIG['end_date'],
            interval='1d',
            progress=False
        )
        if data.empty or len(data) < 100:
            if verbose:
                print(f"❌ {ticker} 數據不足，跳過")
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.columns = [col.capitalize() for col in data.columns]
        if verbose:
            print(f"✓ 成功載入 {len(data)} 筆數據")
        strategy_params = STRATEGY_PARAMS.copy()
        strategy_params['symbol'] = ticker
        strategy_params['level'] = level
        bt = Backtest(
            data,
            BacktestingPyAdapter,
            cash=BACKTEST_CONFIG['initial_cash'],
            commission=BACKTEST_CONFIG['commission'],
            margin=BACKTEST_CONFIG['margin'],
            trade_on_close=True
        )
        stats = bt.run(
            abstract_strategy_class=ThreeLevelStrategy,
            strategy_params=strategy_params
        )
        if verbose:
            print(f"\n{'－'*30} Level {level} 回測結果 {'－'*30}")
            print(f"總報酬率: {stats['Return [%]']:.2f}%")
            print(f"年化報酬率: {stats['Return (Ann.) [%]']:.2f}%")
            print(f"夏普比率: {stats['Sharpe Ratio']:.2f}")
            print(f"最大回撤: {stats['Max. Drawdown [%]']:.2f}%")
            print(f"勝率: {stats['Win Rate [%]']:.2f}%")
            print(f"交易次數: {stats['# Trades']}")
        return {
            'Symbol': ticker,
            'Level': level,
            'Return %': stats['Return [%]'],
            'Annual Return %': stats['Return (Ann.) [%]'],
            'Sharpe Ratio': stats['Sharpe Ratio'],
            'Max Drawdown %': stats['Max. Drawdown [%]'],
            'Win Rate %': stats['Win Rate [%]'],
            'Trades': stats['# Trades'],
            'Avg Trade %': stats['Avg. Trade [%]'] if stats['# Trades'] > 0 else 0,
            'Buy & Hold %': stats['Buy & Hold Return [%]'],
            'Outperformance %': stats['Return [%]'] - stats['Buy & Hold Return [%]']
        }
    except Exception as e:
        if verbose:
            print(f"❌ 回測 {ticker} Level {level} 時發生錯誤: {str(e)}")
        return None

def run_random_sample_backtest(level_choice, sample_size=100):
    symbols = load_random_symbols(sample_size=sample_size)
    print(f"\n隨機抽取 {len(symbols)} 檔股票進行回測: {symbols}")
    results = []
    successful = 0
    failed = 0
    start_time = time.time()
    if level_choice == 'all':
        levels = [1, 2, 3]
    else:
        levels = [int(level_choice)]
    for i, ticker in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] 正在處理 {ticker}...")
        for level in levels:
            result = run_single_backtest(ticker, level, show_plot=False, verbose=True)
            if result:
                results.append(result)
                successful += 1
                print(f" ✓ Level {level} 成功 (報酬率: {result['Return %']:.2f}%)")
            else:
                failed += 1
                print(f" ✗ Level {level} 失敗")
        if i < len(symbols):
            time.sleep(0.2)
    total_time = time.time() - start_time
    print(f"\n批量回測完成！總耗時: {total_time/60:.1f} 分鐘 成功: {successful} 失敗: {failed}")
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(['Symbol', 'Level'])
        results_df.to_csv('random_sample_backtest_results.csv', index=False)
        print(f"\n詳細結果已保存至 random_sample_backtest_results.csv")
        report_generator = HTMLReportGenerator()
        levels_to_report = [int(level_choice)] if level_choice != 'all' else [1, 2, 3]
        for lvl in levels_to_report:
            df_lvl = results_df[results_df['Level'] == lvl]
            if df_lvl.empty:
                continue
            report_data = {
                'strategy_level': lvl,
                'start_date': BACKTEST_CONFIG['start_date'],
                'end_date': BACKTEST_CONFIG['end_date'],
                'initial_capital': BACKTEST_CONFIG['initial_cash'],
                'final_equity': None,
                'total_return': df_lvl['Return %'].mean() / 100,
                'sharpe_ratio': df_lvl['Sharpe Ratio'].mean(),
                'max_drawdown': df_lvl['Max Drawdown %'].mean() / 100,
                'win_rate': df_lvl['Win Rate %'].mean() / 100,
                'total_trades': int(df_lvl['Trades'].sum()),
                'equity_curve': None,
                'trades': df_lvl.to_dict('records'),
                'stock_universe': f"隨機抽樣{len(df_lvl)}檔股票",
                'strategy_name': f"Level {lvl} 策略隨機抽樣回測",
                'date_range': f"{BACKTEST_CONFIG['start_date']} ~ {BACKTEST_CONFIG['end_date']}"
            }
            filename = f"RandomSample_Level{lvl}_report.html" if level_choice != 'all' else f"RandomSample_Level{lvl}_report.html"
            report_generator.generate_report(report_data)
            import shutil
            shutil.move('backtest_report.html', filename)
            print(f"✓ 已產生 HTML 報告: {filename}")

def main():
    parser = argparse.ArgumentParser(description="隨機抽樣三級交易策略回測系統")
    parser.add_argument('--level', type=str, default='all', help='指定要回測的級別: 1、2、3 或 all (預設: all)')
    parser.add_argument('--sample', type=int, default=100, help='隨機抽樣股票數量 (預設: 100)')
    args = parser.parse_args()
    print("隨機抽樣三級交易策略回測系統 v1.0")
    print("="*80)
    run_random_sample_backtest(args.level, args.sample)

if __name__ == "__main__":
    main() 