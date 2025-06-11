import sys
import os
import argparse
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

def load_symbols_from_file(filepath="valid_tickers.txt"):
    if not os.path.exists(filepath):
        print(f"找不到 {filepath}，使用預設股票列表")
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    with open(filepath, 'r') as f:
        symbols = [line.strip().upper() for line in f if line.strip()]
    return symbols

# 策略參數配置
STRATEGY_PARAMS = {
    # MA 參數
    'short_ma_period': 5,
    'long_ma_period': 20,
    
    # BIAS 參數
    'bias_period': 20,
    'bias_upper': 7.0,
    'bias_lower': -8.0,
    
    # KD 參數 (Slow KD)
    'kd_k': 14,
    'kd_d': 3,
    'kd_smooth': 3,
    'kd_overbought': 80,
    'kd_oversold': 20,
    
    # MACD 參數
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    
    # RSI 參數
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    
    # Bollinger Bands 參數
    'bb_period': 20,
    'bb_std': 2.0,
    
    # Ichimoku 參數
    'ichi_tenkan': 9,
    'ichi_kijun': 26,
    'ichi_senkou_b': 52,
    
    # Volume 參數
    'vol_ma_period': 20,
    'vol_multiplier': 1.5,
    
    # ATR 風險管理參數
    'atr_period': 14,
    'atr_multiplier_sl': 2.0,
    'atr_multiplier_tp': 4.0,
    
    # 資金管理
    'risk_per_trade': 0.01,  # 每次交易使用1%的資金
}

# 回測參數
BACKTEST_CONFIG = {
    'start_date': '2023-01-01',
    'end_date': datetime.today().strftime('%Y-%m-%d'),
    'initial_cash': 1000,  # 1000美金初始資金
    'commission': 0.002,   # 0.2%手續費
    'margin': 1.0,        # 不使用槓桿
}

def run_single_backtest(ticker, level, show_plot=False, verbose=True):
    """對單一股票執行特定級別的回測"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"開始回測 {ticker} - Level {level}")
        print(f"{'='*60}")
    
    try:
        # 下載數據
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
            
        # 處理列名
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.columns = [col.capitalize() for col in data.columns]
        
        if verbose:
            print(f"✓ 成功載入 {len(data)} 筆數據")
        
        # 設置策略參數
        strategy_params = STRATEGY_PARAMS.copy()
        strategy_params['symbol'] = ticker
        strategy_params['level'] = level  # 設置策略級別
        
        # 創建並運行回測
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

def run_all_levels_backtest(symbols, level_choice, limit=None):
    if limit:
        symbols = symbols[:limit]
    print(f"\n{'='*80}")
    print(f"三級交易策略批量回測系統")
    print(f"{'='*80}")
    print(f"策略參數:")
    print(f"  - 每次交易風險: {STRATEGY_PARAMS['risk_per_trade']*100}%")
    print(f"  - Level 1: 單一指標信號")
    print(f"  - Level 2: 雙指標共振")
    print(f"  - Level 3: 三指標以上共振")
    print(f"\n回測配置:")
    print(f"  - 初始資金: ${BACKTEST_CONFIG['initial_cash']}")
    print(f"  - 手續費: {BACKTEST_CONFIG['commission']*100}%")
    print(f"  - 期間: {BACKTEST_CONFIG['start_date']} ~ {BACKTEST_CONFIG['end_date']}")
    print(f"\n已載入 {len(symbols)} 個股票代碼")
    user_input = input(f"\n即將開始回測所有 {len(symbols)} 個股票的指定級別，這可能需要較長時間。是否繼續? (Y/n): ")
    if user_input.strip().lower() == 'n':
        print("已取消回測")
        return
    print(f"\n開始批量回測...")
    print("="*80)
    results = []
    successful = 0
    failed = 0
    start_time = time.time()
    # 決定要跑哪些level
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
        if i % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = avg_time * (len(symbols) - i)
            print(f"\n進度: {i}/{len(symbols)} ({i/len(symbols)*100:.1f}%) - 預計剩餘時間: {remaining/60:.1f} 分鐘")
        if i < len(symbols):
            time.sleep(0.2)
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"批量回測完成！")
    print(f"總耗時: {total_time/60:.1f} 分鐘")
    print(f"成功: {successful} 個回測, 失敗: {failed} 個回測")
    print(f"{'='*80}")
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(['Symbol', 'Level'])
        results_df.to_csv('all_levels_backtest_results.csv', index=False)
        print(f"\n詳細結果已保存至 all_levels_backtest_results.csv")
        print("\n各級別平均表現:")
        level_stats = results_df.groupby('Level').agg({
            'Return %': 'mean',
            'Sharpe Ratio': 'mean',
            'Win Rate %': 'mean',
            'Trades': 'mean'
        }).round(2)
        print(level_stats)

def main():
    parser = argparse.ArgumentParser(description="三級交易策略回測系統")
    parser.add_argument('--level', type=str, default='all', help='指定要回測的級別: 1、2、3 或 all (預設: all)')
    args = parser.parse_args()
    print("三級交易策略回測系統 v1.0")
    print("="*80)
    symbols = load_symbols_from_file()
    run_all_levels_backtest(symbols, args.level)

if __name__ == "__main__":
    main() 