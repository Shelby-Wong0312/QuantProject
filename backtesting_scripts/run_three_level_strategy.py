# backtesting_scripts/run_three_level_strategy.py

import sys
import os
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

# --- 1. 載入股票列表 ---
def load_symbols_from_file(filepath="tickers.txt"):
    if not os.path.exists(filepath):
        print(f"找不到 {filepath}，使用預設股票列表")
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "WMT"]
    with open(filepath, 'r') as f:
        symbols = [line.strip().upper() for line in f if line.strip()]
    return symbols

# --- 2. 策略參數配置（完全按照您的要求）---
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
    'start_date': '2021-01-01',
    'end_date': datetime.today().strftime('%Y-%m-%d'),
    'initial_cash': 1000,  # 1000美金初始資金
    'commission': 0.002,   # 0.2%手續費
    'margin': 1.0,        # 不使用槓桿
}

def run_single_backtest(ticker, show_plot=False, verbose=True):
    """對單一股票執行回測"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"開始回測 {ticker}")
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
        
        # 顯示結果
        if verbose:
            print(f"\n{'－'*30} 回測結果 {'－'*30}")
            print(f"總報酬率: {stats['Return [%]']:.2f}%")
            print(f"年化報酬率: {stats['Return (Ann.) [%]']:.2f}%")
            print(f"夏普比率: {stats['Sharpe Ratio']:.2f}")
            print(f"最大回撤: {stats['Max. Drawdown [%]']:.2f}%")
            print(f"勝率: {stats['Win Rate [%]']:.2f}%")
            print(f"交易次數: {stats['# Trades']}")
            
            # 顯示交易級別分布
            if stats['_trades'] is not None and not stats['_trades'].empty:
                trades_df = stats['_trades']
                if 'Tag' in trades_df.columns:
                    level_counts = {}
                    for tag in trades_df['Tag']:
                        if tag and 'L1' in str(tag):
                            level_counts['Level 1'] = level_counts.get('Level 1', 0) + 1
                        elif tag and 'L2' in str(tag):
                            level_counts['Level 2'] = level_counts.get('Level 2', 0) + 1
                        elif tag and 'L3' in str(tag):
                            level_counts['Level 3'] = level_counts.get('Level 3', 0) + 1
                    
                    if level_counts:
                        print(f"\n交易級別分布:")
                        for level, count in sorted(level_counts.items()):
                            print(f"  {level}: {count} 筆")
        
        # 繪製圖表
        if show_plot:
            print("\n正在生成圖表...")
            bt.plot(open_browser=True)
            
        return {
            'Symbol': ticker,
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
            print(f"❌ 回測 {ticker} 時發生錯誤: {str(e)}")
        return None

def run_batch_backtest_all_stocks():
    """批量回測所有股票 - 不偷懶版本"""
    symbols = load_symbols_from_file()
    
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
    
    # 確認是否要繼續
    user_input = input(f"\n即將開始回測所有 {len(symbols)} 個股票，這可能需要較長時間。是否繼續? (Y/n): ")
    if user_input.strip().lower() == 'n':
        print("已取消回測")
        return
    
    print(f"\n開始批量回測...")
    print("="*80)
    
    results = []
    successful = 0
    failed = 0
    start_time = time.time()
    
    # 回測所有股票
    for i, ticker in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] 正在處理 {ticker}...", end='', flush=True)
        
        result = run_single_backtest(ticker, show_plot=False, verbose=False)
        
        if result:
            results.append(result)
            successful += 1
            print(f" ✓ 成功 (報酬率: {result['Return %']:.2f}%)")
        else:
            failed += 1
            print(f" ✗ 失敗")
        
        # 每處理10個股票顯示進度
        if i % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = avg_time * (len(symbols) - i)
            print(f"\n進度: {i}/{len(symbols)} ({i/len(symbols)*100:.1f}%) - 預計剩餘時間: {remaining/60:.1f} 分鐘")
        
        # 避免請求過快
        if i < len(symbols):
            time.sleep(0.2)
    
    # 計算總耗時
    total_time = time.time() - start_time
    
    # 顯示匯總結果
    print(f"\n{'='*80}")
    print(f"批量回測完成！")
    print(f"總耗時: {total_time/60:.1f} 分鐘")
    print(f"成功: {successful} 個股票, 失敗: {failed} 個股票")
    print(f"{'='*80}")
    
    if results:
        # 轉換為DataFrame並排序
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Return %', ascending=False)
        
        # 顯示表現最佳的股票
        print(f"\n{'－'*40} 表現最佳的20個股票 {'－'*40}")
        print(results_df.head(20)[['Symbol', 'Return %', 'Annual Return %', 'Sharpe Ratio', 
                                   'Max Drawdown %', 'Trades', 'Outperformance %']].to_string(index=False))
        
        # 顯示表現最差的股票
        print(f"\n{'－'*40} 表現最差的20個股票 {'－'*40}")
        print(results_df.tail(20)[['Symbol', 'Return %', 'Annual Return %', 'Sharpe Ratio', 
                                   'Max Drawdown %', 'Trades', 'Outperformance %']].to_string(index=False))
        
        # 統計摘要
        print(f"\n{'－'*40} 整體統計摘要 {'－'*40}")
        print(f"測試股票總數: {len(results)}")
        print(f"平均報酬率: {results_df['Return %'].mean():.2f}%")
        print(f"中位數報酬率: {results_df['Return %'].median():.2f}%")
        print(f"最高報酬率: {results_df['Return %'].max():.2f}% ({results_df.iloc[0]['Symbol']})")
        print(f"最低報酬率: {results_df['Return %'].min():.2f}% ({results_df.iloc[-1]['Symbol']})")
        print(f"正報酬股票數: {len(results_df[results_df['Return %'] > 0])} ({len(results_df[results_df['Return %'] > 0])/len(results_df)*100:.1f}%)")
        print(f"負報酬股票數: {len(results_df[results_df['Return %'] < 0])} ({len(results_df[results_df['Return %'] < 0])/len(results_df)*100:.1f}%)")
        print(f"平均交易次數: {results_df['Trades'].mean():.1f}")
        print(f"平均夏普比率: {results_df['Sharpe Ratio'].mean():.2f}")
        print(f"平均最大回撤: {results_df['Max Drawdown %'].mean():.2f}%")
        
        # 打敗買入持有策略的統計
        outperform_count = len(results_df[results_df['Outperformance %'] > 0])
        print(f"\n打敗買入持有策略的股票數: {outperform_count} ({outperform_count/len(results_df)*100:.1f}%)")
        print(f"平均超額報酬: {results_df['Outperformance %'].mean():.2f}%")
        
        # 按交易次數分組統計
        print(f"\n{'－'*40} 交易活躍度分析 {'－'*40}")
        no_trades = len(results_df[results_df['Trades'] == 0])
        low_trades = len(results_df[(results_df['Trades'] > 0) & (results_df['Trades'] <= 10)])
        medium_trades = len(results_df[(results_df['Trades'] > 10) & (results_df['Trades'] <= 50)])
        high_trades = len(results_df[results_df['Trades'] > 50])
        
        print(f"無交易: {no_trades} 個股票")
        print(f"低頻交易 (1-10次): {low_trades} 個股票")
        print(f"中頻交易 (11-50次): {medium_trades} 個股票")
        print(f"高頻交易 (>50次): {high_trades} 個股票")
        
        # 儲存完整結果
        output_file = f'three_level_strategy_full_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\n✓ 完整結果已儲存至: {output_file}")
        
        # 儲存優秀股票列表
        top_stocks = results_df[
            (results_df['Return %'] > 20) & 
            (results_df['Sharpe Ratio'] > 0.5) & 
            (results_df['Max Drawdown %'] > -30)
        ]
        
        if not top_stocks.empty:
            top_file = f'three_level_top_stocks_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            top_stocks.to_csv(top_file, index=False)
            print(f"✓ 優秀股票列表已儲存至: {top_file}")
            print(f"  共 {len(top_stocks)} 個股票符合條件 (報酬>20%, 夏普>0.5, 回撤<30%)")
        
        return results_df
    
    return None

def main():
    """主程式 - 執行完整的批量回測"""
    print("三級交易策略回測系統 v1.0")
    print("="*80)
    
    # 直接執行批量回測所有股票
    run_batch_backtest_all_stocks()

if __name__ == "__main__":
    main() 