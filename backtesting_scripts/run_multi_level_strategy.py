# backtesting_scripts/run_multi_level_strategy.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from backtesting import Backtest
import yfinance as yf
from datetime import datetime
import time

from adapters.backtesting_adapter import BacktestingPyAdapter
from strategy.concrete_strategies.multi_level_confluence_strategy import MultiLevelConfluenceStrategy

# --- 1. 載入股票列表 ---
def load_symbols_from_file(filepath="valid_tickers.txt"):
    if not os.path.exists(filepath):
        print(f"找不到 {filepath}，使用預設股票列表")
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    with open(filepath, 'r') as f:
        symbols = [line.strip().upper() for line in f if line.strip()]
    return symbols

# --- 2. 主要配置 ---
# 交易模式選擇
TEST_MODE = "BATCH"  # "SINGLE" 或 "BATCH"
SINGLE_TICKER = "AAPL"  # 單一測試時使用的股票
BATCH_LIMIT = 50  # 批量測試的股票數量限制

# 策略參數（完全按照您的要求）
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

def run_single_backtest(ticker, show_plot=True):
    """對單一股票執行回測"""
    print(f"\n{'='*60}")
    print(f"開始回測 {ticker}")
    print(f"{'='*60}")
    
    try:
        # 下載數據
        print(f"正在下載 {ticker} 的歷史數據...")
        data = yf.download(
            ticker, 
            start=BACKTEST_CONFIG['start_date'],
            end=BACKTEST_CONFIG['end_date'],
            interval='1d',
            progress=False
        )
        
        if data.empty or len(data) < 100:
            print(f"❌ {ticker} 數據不足，跳過")
            return None
            
        # 處理列名
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.columns = [col.capitalize() for col in data.columns]
        
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
            abstract_strategy_class=MultiLevelConfluenceStrategy,
            strategy_params=strategy_params
        )
        
        # 顯示結果
        print(f"\n{'－'*30} 回測結果 {'－'*30}")
        print(f"總報酬率: {stats['Return [%]']:.2f}%")
        print(f"年化報酬率: {stats['Return (Ann.) [%]']:.2f}%")
        print(f"夏普比率: {stats['Sharpe Ratio']:.2f}")
        print(f"最大回撤: {stats['Max. Drawdown [%]']:.2f}%")
        print(f"勝率: {stats['Win Rate [%]']:.2f}%")
        print(f"交易次數: {stats['# Trades']}")
        print(f"平均交易報酬: {stats['Avg. Trade [%]']:.2f}%")
        
        # 顯示交易明細
        if stats['_trades'] is not None and not stats['_trades'].empty:
            print(f"\n{'－'*30} 交易明細 {'－'*30}")
            trades_df = stats['_trades'][['Size', 'EntryPrice', 'ExitPrice', 'ReturnPct', 'Duration']]
            print(trades_df.head(10))  # 顯示前10筆交易
            
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
            'Avg Trade %': stats['Avg. Trade [%]'] if stats['# Trades'] > 0 else 0
        }
        
    except Exception as e:
        print(f"❌ 回測 {ticker} 時發生錯誤: {str(e)}")
        return None

def run_batch_backtest(symbols, limit=None):
    """批量回測多個股票"""
    if limit:
        symbols = symbols[:limit]
        
    print(f"\n{'='*60}")
    print(f"開始批量回測 {len(symbols)} 個股票")
    print(f"{'='*60}")
    
    results = []
    successful = 0
    failed = 0
    
    for i, ticker in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] 正在處理 {ticker}...")
        
        result = run_single_backtest(ticker, show_plot=False)
        
        if result:
            results.append(result)
            successful += 1
        else:
            failed += 1
            
        # 避免請求過快
        if i < len(symbols):
            time.sleep(0.5)
    
    # 顯示匯總結果
    print(f"\n{'='*60}")
    print(f"批量回測完成！成功: {successful}, 失敗: {failed}")
    print(f"{'='*60}")
    
    if results:
        # 轉換為DataFrame並排序
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Return %', ascending=False)
        
        # 顯示表現最好的股票
        print(f"\n{'－'*30} 表現最佳的10個股票 {'－'*30}")
        print(results_df.head(10).to_string(index=False))
        
        # 顯示表現最差的股票
        print(f"\n{'－'*30} 表現最差的10個股票 {'－'*30}")
        print(results_df.tail(10).to_string(index=False))
        
        # 統計摘要
        print(f"\n{'－'*30} 整體統計 {'－'*30}")
        print(f"平均報酬率: {results_df['Return %'].mean():.2f}%")
        print(f"中位數報酬率: {results_df['Return %'].median():.2f}%")
        print(f"最高報酬率: {results_df['Return %'].max():.2f}%")
        print(f"最低報酬率: {results_df['Return %'].min():.2f}%")
        print(f"正報酬股票數: {len(results_df[results_df['Return %'] > 0])}")
        print(f"負報酬股票數: {len(results_df[results_df['Return %'] < 0])}")
        
        # 儲存結果
        output_file = f'multi_level_strategy_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\n✓ 結果已儲存至: {output_file}")
        
        return results_df
    
    return None

def main():
    """主程式"""
    print(f"多層級共振策略回測系統")
    print(f"使用參數: {BACKTEST_CONFIG['initial_cash']}美金初始資金, {BACKTEST_CONFIG['commission']*100}%手續費")
    print(f"回測期間: {BACKTEST_CONFIG['start_date']} 至 {BACKTEST_CONFIG['end_date']}")
    
    if TEST_MODE == "SINGLE":
        # 單一股票測試
        run_single_backtest(SINGLE_TICKER, show_plot=True)
        
    elif TEST_MODE == "BATCH":
        # 批量測試
        symbols = load_symbols_from_file()
        
        if not symbols:
            print("無法載入股票列表")
            return
            
        print(f"已載入 {len(symbols)} 個股票代碼")
        
        # 詢問用戶是否要限制測試數量
        if len(symbols) > BATCH_LIMIT:
            print(f"\n注意: 股票數量較多，建議先測試前 {BATCH_LIMIT} 個")
            user_input = input(f"是否只測試前 {BATCH_LIMIT} 個股票? (Y/n): ").strip().lower()
            
            if user_input != 'n':
                symbols = symbols[:BATCH_LIMIT]
        
        run_batch_backtest(symbols)
    
    else:
        print(f"未知的測試模式: {TEST_MODE}")

if __name__ == "__main__":
    main() 