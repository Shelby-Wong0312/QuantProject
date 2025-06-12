import sys
import os
import argparse
import random
import json
from datetime import datetime
import time
import pandas as pd
from backtesting import Backtest
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Correctly add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.backtesting_adapter import BacktestingPyAdapter
from strategy.concrete_strategies.three_level_strategy import ThreeLevelStrategy
from backtesting_scripts.report_generator import ReportGenerator

# --- Default Strategy Parameters ---
STRATEGY_PARAMS = {
    'short_ma_period': 5, 'long_ma_period': 20, 'bias_period': 20, 'bias_upper': 7.0, 'bias_lower': -8.0,
    'kd_k': 14, 'kd_d': 3, 'kd_smooth': 3, 'kd_overbought': 80, 'kd_oversold': 20,
    'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9, 'rsi_period': 14, 'rsi_overbought': 70, 'rsi_oversold': 30,
    'bb_period': 20, 'bb_std': 2.0, 'ichi_tenkan': 9, 'ichi_kijun': 26, 'ichi_senkou_b': 52,
    'vol_ma_period': 20, 'vol_multiplier': 1.5, 'atr_period': 14, 'atr_multiplier_sl': 2.0, 'atr_multiplier_tp': 4.0,
    'risk_per_trade': 0.01,
}

# --- Backtest Configuration ---
BACKTEST_CONFIG = {
    'start_date': '2023-01-01',
    'end_date': datetime.today().strftime('%Y-%m-%d'),
    'initial_cash': 10000,
    'commission': 0.002,
    'margin': 1.0,
}

# 策略類型映射
STRATEGY_TYPES = {
    # Level 1 策略
    '1-1': 'KD指標策略',
    '1-2': 'RSI指標策略', 
    '1-3': 'MACD指標策略',
    '1-4': 'BIAS乖離率策略',
    '1-5': '布林通道策略',
    '1-6': '移動平均線策略',
    '1-7': 'K線形態策略',
    
    # Level 2 策略
    '2-1': '趨勢+RSI組合',
    '2-2': '雲帶+MACD組合',
    '2-3': 'BIAS+K線形態組合',
    '2-4': '布林擠壓突破組合',
    '2-5': '斐波那契+K線組合',
    
    # Level 3 策略
    '3-1': '雲帶+RSI+布林組合',
    '3-2': 'MA+MACD+量能組合',
    '3-3': '道氏+MA+KD組合',
    '3-4': '雲帶+斐波那契+K線組合'
}

def load_valid_tickers(filepath="valid_tickers.txt"):
    """Loads a list of tickers from a file."""
    try:
        with open(filepath, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Warning: Ticker file not found at {filepath}. Using a default list.")
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

def download_data_cached(tickers, start_date, end_date, cache):
    """Downloads data for a list of tickers, using a cache to avoid re-downloads."""
    print("Pre-loading historical data for selected tickers...")
    for ticker in tickers:
        if ticker not in cache:
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not data.empty and len(data) > 50:
                    # Handle both tuple and string column names
                    if hasattr(data.columns, 'levels'):  # MultiIndex columns
                        data.columns = data.columns.get_level_values(0)
                    # Now safely capitalize
                    data.columns = [str(col).capitalize() for col in data.columns]
                    cache[ticker] = data
                    print(f"  - Loaded data for {ticker}")
                else:
                    print(f"  - No data for {ticker}")
            except Exception as e:
                print(f"  - Failed to load data for {ticker}: {e}")
            time.sleep(0.1) # Avoid getting rate-limited
    return cache

def run_backtests(tickers, level, initial_cash, commission_rate, data_cache, strategy_type=None):
    """Runs backtests for a list of tickers and returns the results."""
    results = []
    strategy_class = BacktestingPyAdapter
    
    for ticker in tickers:
        if ticker not in data_cache:
            continue
            
        data = data_cache[ticker]
        strategy_params = STRATEGY_PARAMS.copy()
        strategy_params.update({'symbol': ticker, 'level': level})
        
        # 如果指定了具體策略類型，添加到參數中
        if strategy_type:
            strategy_params.update({'strategy_type': strategy_type})

        bt = Backtest(data, strategy_class, cash=initial_cash, commission=commission_rate, trade_on_close=True)
        
        try:
            stats = bt.run(abstract_strategy_class=ThreeLevelStrategy, strategy_params=strategy_params)
            stats.name = ticker  # Assign ticker name to the result series
            # Make sure to include the raw data needed for the report
            stats['_equity_curve'] = stats['_equity_curve'][['Equity']]
            
            # 确保交易记录中包含股票代码
            trades_df = stats['_trades'].copy()
            if not trades_df.empty:
                trades_df['Symbol'] = ticker  # 添加股票代码列
            stats['_trades'] = trades_df
            
            stats['Symbol'] = ticker # Add ticker symbol for easier access later
            results.append(stats)
            print(f"✓ Backtest for {ticker} successful.")
        except Exception as e:
            print(f"✗ Backtest for {ticker} failed: {e}")
            results.append(None) # Add a placeholder for failed backtests
            
    return results

def get_price_data_json(tickers, data_cache):
    """Fetches historical data from the cache and returns it as a JSON object."""
    price_data = {}
    # print("Preparing price data for the report...")
    for ticker in tickers:
        if ticker in data_cache:
            df = data_cache[ticker].copy()
            df.index = df.index.strftime('%Y-%m-%d')
            price_data[ticker] = {
                'dates': df.index.tolist(),
                'open': df['Open'].tolist(), 'high': df['High'].tolist(),
                'low': df['Low'].tolist(), 'close': df['Close'].tolist(),
                'volume': df['Volume'].tolist()
            }
    return json.dumps(price_data)

# --- Main Execution Logic ---

def main(level: int, sample_size: int, strategy: str = None):
    """Main function to run the backtesting and reporting process."""
    
    # --- 1. Setup ---
    start_time = time.time()
    all_tickers = load_valid_tickers()
    # Ensure sample size is not larger than the number of available tickers
    sample_size = min(sample_size, len(all_tickers))
    selected_tickers = random.sample(all_tickers, sample_size)
    
    if strategy:
        if strategy in STRATEGY_TYPES:
            strategy_name = STRATEGY_TYPES[strategy]
            print(f"Running {strategy} ({strategy_name}) backtest on {len(selected_tickers)} randomly selected tickers.")
        else:
            print(f"Error: Strategy '{strategy}' not found. Available strategies:")
            for key, value in STRATEGY_TYPES.items():
                print(f"  {key}: {value}")
            return
    else:
        print(f"Running Level {level} backtest on {len(selected_tickers)} randomly selected tickers.")

    # --- 2. Data Loading ---
    data_cache = {}
    download_data_cached(selected_tickers, BACKTEST_CONFIG['start_date'], BACKTEST_CONFIG['end_date'], data_cache)
    
    # Filter out tickers for which data loading failed
    runnable_tickers = [ticker for ticker in selected_tickers if ticker in data_cache]
    if not runnable_tickers:
        print("Could not load data for any of the selected tickers. Aborting.")
        return

    # --- 3. Run Backtests ---
    print("\n--- Running Backtests ---")
    backtest_results = run_backtests(
        runnable_tickers,
        level,
        BACKTEST_CONFIG['initial_cash'],
        BACKTEST_CONFIG['commission'],
        data_cache,
        strategy
    )
    
    # --- 4. Portfolio Analysis ---
    print("\n--- Generating Portfolio Analysis ---")
    successful_results = [res for res in backtest_results if res is not None]
    if not successful_results:
        print("No backtests were successful. Cannot generate a report.")
        return

    # Use a common date index from the first successful backtest's data
    first_successful_ticker = successful_results[0].name
    all_dates = data_cache[first_successful_ticker].index
    
    # --- Combine Equity Curves ---
    total_equity_change = pd.Series(0.0, index=all_dates)
    for result in successful_results:
        equity_df = result['_equity_curve']
        # Reindex to common date range and forward-fill missing values
        reindexed_equity = equity_df['Equity'].reindex(all_dates).ffill()
        # Calculate equity change relative to initial cash for this specific backtest
        equity_change = reindexed_equity - BACKTEST_CONFIG['initial_cash']
        total_equity_change = total_equity_change.add(equity_change, fill_value=0)

    # Final portfolio equity is the base capital plus the sum of all changes
    portfolio_equity_curve = pd.Series(BACKTEST_CONFIG['initial_cash'], index=all_dates) + total_equity_change
    portfolio_equity_curve.name = 'Equity'
    
    # --- Combine Trades ---
    all_trades_list = [res['_trades'] for res in successful_results if not res['_trades'].empty]
    all_trades_df = pd.concat(all_trades_list, ignore_index=True) if all_trades_list else pd.DataFrame()

    # --- Calculate Combined Stats ---
    # Calculate basic portfolio stats manually
    total_return = (portfolio_equity_curve.iloc[-1] - BACKTEST_CONFIG['initial_cash']) / BACKTEST_CONFIG['initial_cash'] * 100
    
    # Calculate max drawdown
    peak = portfolio_equity_curve.expanding().max()
    drawdown = (portfolio_equity_curve - peak) / peak * 100
    max_drawdown = drawdown.min()
    
    # Calculate win rate from trades
    win_rate = 0
    total_trades = len(all_trades_df)
    if total_trades > 0:
        winning_trades = len(all_trades_df[all_trades_df['PnL'] > 0])
        win_rate = winning_trades / total_trades * 100
    
    # Create portfolio stats dictionary
    portfolio_stats = {
        'Equity Final [$]': portfolio_equity_curve.iloc[-1],
        'Return [%]': total_return,
        'Sharpe Ratio': 0,  # Would need daily returns to calculate properly
        'Max. Drawdown [%]': abs(max_drawdown),
        'Win Rate [%]': win_rate,
        '# Trades': total_trades
    }
    
    # --- 5. Display Terminal Results ---
    print("\n" + "="*80)
    print("📊 回測結果摘要")
    print("="*80)
    
    # 显示基本信息
    strategy_label = f"{strategy} ({STRATEGY_TYPES[strategy]})" if strategy else f"Level {level}"
    print(f"🎯 策略類型: {strategy_label}")
    print(f"📈 測試股票: {len(successful_results)} 支 ({', '.join([r.name for r in successful_results])})")
    print(f"📅 回測期間: {BACKTEST_CONFIG['start_date']} ~ {BACKTEST_CONFIG['end_date']}")
    print(f"💰 初始資金: ${BACKTEST_CONFIG['initial_cash']:,.2f}")
    
    # 显示组合绩效
    final_equity = portfolio_stats['Equity Final [$]']
    total_return = portfolio_stats['Return [%]']
    max_drawdown = portfolio_stats['Max. Drawdown [%]']
    sharpe_ratio = portfolio_stats['Sharpe Ratio']
    win_rate = portfolio_stats['Win Rate [%]']
    total_trades = portfolio_stats['# Trades']
    
    print(f"\n💼 組合績效:")
    print(f"   💵 最終資金: ${final_equity:,.2f}")
    print(f"   📈 總報酬率: {total_return:+.2f}%")
    color = "🟢" if total_return > 0 else "🔴" if total_return < 0 else "🟡"
    performance_rating = "優秀" if total_return > 15 else "良好" if total_return > 5 else "一般" if total_return > -5 else "需改進"
    print(f"   {color} 績效評級: {performance_rating}")
    print(f"   📉 最大回撤: {max_drawdown:.2f}%")
    print(f"   ⚡ 夏普比率: {sharpe_ratio:.3f}")
    print(f"   🎯 勝率: {win_rate:.1f}%")
    print(f"   🔄 總交易數: {total_trades}")
    
    # 显示个股表现
    print(f"\n📋 個股表現:")
    # 计算年化回报率（假设回测期间约2.5年）
    years = (pd.to_datetime(BACKTEST_CONFIG['end_date']) - pd.to_datetime(BACKTEST_CONFIG['start_date'])).days / 365.25
    annual_return = ((final_equity / BACKTEST_CONFIG['initial_cash']) ** (1/years) - 1) * 100 if years > 0 else 0
    
    print(f"{'股票代碼':<8} {'總報酬率':<12} {'交易次數':<8} {'勝率':<8} {'最終淨值':<12}")
    print("-" * 60)
    
    # 按個股結果排序顯示
    for i, result in enumerate(sorted(successful_results, key=lambda x: x['Return [%]'], reverse=True)):
        symbol = result.name
        stock_return = result['Return [%]']
        stock_trades = result['# Trades']
        stock_winrate = result.get('Win Rate [%]', 0)
        final_value = result['Equity Final [$]']
        
        status = "🟢" if stock_return > 0 else "🔴" if stock_return < 0 else "🟡"
        print(f"{symbol:<8} {status} {stock_return:>8.2f}%   {stock_trades:>6}   {stock_winrate:>6.1f}%   ${final_value:>9,.0f}")
    
    # 显示交易统计
    if not all_trades_df.empty:
        print(f"\n💼 交易統計:")
        total_pnl = all_trades_df['PnL'].sum()
        winning_trades = len(all_trades_df[all_trades_df['PnL'] > 0])
        losing_trades = len(all_trades_df[all_trades_df['PnL'] <= 0])
        
        avg_win = all_trades_df[all_trades_df['PnL'] > 0]['PnL'].mean() if winning_trades > 0 else 0
        avg_loss = all_trades_df[all_trades_df['PnL'] <= 0]['PnL'].mean() if losing_trades > 0 else 0
        
        print(f"   💰 總盈虧: ${total_pnl:,.2f}")
        print(f"   ✅ 獲利交易: {winning_trades} 筆 (平均: ${avg_win:,.2f})")
        print(f"   ❌ 虧損交易: {losing_trades} 筆 (平均: ${avg_loss:,.2f})")
        if losing_trades > 0 and avg_loss != 0:
            profit_factor = abs((avg_win * winning_trades) / (avg_loss * losing_trades))
            print(f"   ⚖️  獲利因子: {profit_factor:.2f}")
        
        # 显示年化回报率
        print(f"   📊 年化報酬率: {annual_return:.2f}%")
    
    print("="*80)
    
    # --- 6. Generate Report ---
    print("\n--- Generating HTML Report ---")
    
    # The report generator expects a dictionary, not a class instance or series
    backtest_results_dict = {
        'equity_curve': portfolio_equity_curve.to_frame(),
        'trades': all_trades_df.to_dict('records'),
        'start_date': BACKTEST_CONFIG['start_date'],
        'end_date': BACKTEST_CONFIG['end_date'],
        'initial_capital': BACKTEST_CONFIG['initial_cash'],
        'final_equity': portfolio_stats['Equity Final [$]'],
        'total_return': portfolio_stats['Return [%]'] / 100,
        'sharpe_ratio': portfolio_stats['Sharpe Ratio'],
        'max_drawdown': portfolio_stats['Max. Drawdown [%]'] / 100,
        'win_rate': portfolio_stats['Win Rate [%]'] / 100,
        'total_trades': portfolio_stats['# Trades'],
        'stock_universe': f"隨機抽樣 {len(successful_results)} 檔股票",
        'individual_results': {result.name: result for result in successful_results}
    }
    
    # Define filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if strategy:
        report_filename = f"Backtest_{strategy}_report_{timestamp}.html"
        results_filename = f"Backtest_{strategy}_results_{timestamp}.csv"
        strategy_label = f"{strategy} ({STRATEGY_TYPES[strategy]})"
    else:
        report_filename = f"Backtest_Level{level}_report_{timestamp}.html"
        results_filename = f"Backtest_Level{level}_results_{timestamp}.csv"
        strategy_label = f"Level {level}"

    report_generator = ReportGenerator()
    report_generator.generate_report(
        backtest_results=backtest_results_dict,
        output_path=report_filename,
        strategy_level=strategy_label
    )
    
    # --- 7. Save Raw Results ---
    raw_results_df = pd.DataFrame([res for res in successful_results])
    # Drop non-serializable columns before saving
    if not raw_results_df.empty:
        raw_results_df = raw_results_df.drop(columns=['_equity_curve', '_trades'], errors='ignore')
        raw_results_df.to_csv(results_filename, index=False)
        print(f"📄 CSV結果已保存: {results_filename}")

    print(f"🌐 HTML報告已生成: {report_filename}")
    print(f"\n⏱️  處理完成，用時 {time.time() - start_time:.2f} 秒")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="量化交易策略回測系統")
    
    # 創建互斥群組：可以選擇 level 或 strategy，但不能同時選擇
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--level', type=int, choices=[1, 2, 3], help='策略級別 (1, 2, 3)')
    group.add_argument('--strategy', type=str, help='具體策略類型 (例如: 1-1, 2-1, 3-1)')
    
    parser.add_argument('--stocks', type=int, default=5, help='隨機抽樣股票數量')
    parser.add_argument('--list-strategies', action='store_true', help='列出所有可用策略')
    
    args = parser.parse_args()
    
    # 列出所有策略
    if args.list_strategies:
        print("可用策略列表：")
        print("\nLevel 1 基礎策略：")
        for key, value in STRATEGY_TYPES.items():
            if key.startswith('1-'):
                print(f"  {key}: {value}")
        print("\nLevel 2 組合策略：")
        for key, value in STRATEGY_TYPES.items():
            if key.startswith('2-'):
                print(f"  {key}: {value}")
        print("\nLevel 3 高級策略：")
        for key, value in STRATEGY_TYPES.items():
            if key.startswith('3-'):
                print(f"  {key}: {value}")
        exit(0)
    
    # 設定默認值
    if not args.level and not args.strategy:
        args.level = 1  # 默認使用 Level 1
    
    # 從 strategy 推導 level
    if args.strategy:
        if args.strategy in STRATEGY_TYPES:
            level = int(args.strategy.split('-')[0])
        else:
            print(f"錯誤：策略 '{args.strategy}' 不存在。使用 --list-strategies 查看所有可用策略。")
            exit(1)
    else:
        level = args.level
    
    main(level=level, sample_size=args.stocks, strategy=args.strategy) 