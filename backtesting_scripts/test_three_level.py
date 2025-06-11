# backtesting_scripts/test_three_level.py
"""
三級策略快速測試腳本
直接運行: python backtesting_scripts/test_three_level.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting_scripts.run_three_level_strategy import run_single_backtest, STRATEGY_PARAMS, BACKTEST_CONFIG

# 測試幾個知名股票
TEST_STOCKS = ["AAPL", "MSFT", "GOOGL", "TSLA", "JPM"]

if __name__ == "__main__":
    print("="*80)
    print("三級交易策略 - 快速測試")
    print("="*80)
    print(f"\n策略特點:")
    print(f"- Level 1: 單一指標信號（低信賴度）")
    print(f"- Level 2: 雙指標共振（中等信賴度）")
    print(f"- Level 3: 三指標以上共振（高信賴度）")
    print(f"\n風險管理:")
    print(f"- 每次交易風險: {STRATEGY_PARAMS['risk_per_trade']*100}%")
    print(f"- 止損: {STRATEGY_PARAMS['atr_multiplier_sl']} ATR")
    print(f"- 止盈: {STRATEGY_PARAMS['atr_multiplier_tp']} ATR")
    
    print(f"\n測試股票: {', '.join(TEST_STOCKS)}")
    
    # 測試每個股票
    for i, ticker in enumerate(TEST_STOCKS, 1):
        print(f"\n[{i}/{len(TEST_STOCKS)}] 測試 {ticker}")
        result = run_single_backtest(ticker, show_plot=(i==1), verbose=True)
        
        if result:
            print(f"\n{ticker} 快速摘要:")
            print(f"  報酬率: {result['Return %']:.2f}%")
            print(f"  交易次數: {result['Trades']}")
            print(f"  超越買入持有: {result['Outperformance %']:.2f}%")
    
    print("\n" + "="*80)
    print("快速測試完成！")
    print("如需測試所有股票，請運行: python backtesting_scripts/run_three_level_strategy.py") 