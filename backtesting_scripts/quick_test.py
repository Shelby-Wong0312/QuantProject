# backtesting_scripts/quick_test.py
"""
快速測試腳本 - 用於測試多層級共振策略
直接運行: python backtesting_scripts/quick_test.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting_scripts.run_multi_level_strategy import run_single_backtest, STRATEGY_PARAMS, BACKTEST_CONFIG

# 測試一個流動性好的知名股票
TEST_TICKER = "AAPL"  # 可以改成任何你想測試的股票

if __name__ == "__main__":
    print("="*60)
    print("多層級共振策略 - 快速測試")
    print("="*60)
    print(f"\n策略參數:")
    print(f"- MA週期: {STRATEGY_PARAMS['short_ma_period']}/{STRATEGY_PARAMS['long_ma_period']}")
    print(f"- BIAS: {STRATEGY_PARAMS['bias_lower']}% ~ {STRATEGY_PARAMS['bias_upper']}%")
    print(f"- KD: {STRATEGY_PARAMS['kd_oversold']}/{STRATEGY_PARAMS['kd_overbought']}")
    print(f"- RSI: {STRATEGY_PARAMS['rsi_oversold']}/{STRATEGY_PARAMS['rsi_overbought']}")
    print(f"- 每次交易風險: {STRATEGY_PARAMS['risk_per_trade']*100}%")
    print(f"- 止損倍數: {STRATEGY_PARAMS['atr_multiplier_sl']} ATR")
    print(f"- 止盈倍數: {STRATEGY_PARAMS['atr_multiplier_tp']} ATR")
    
    print(f"\n回測配置:")
    print(f"- 初始資金: ${BACKTEST_CONFIG['initial_cash']}")
    print(f"- 手續費: {BACKTEST_CONFIG['commission']*100}%")
    print(f"- 期間: {BACKTEST_CONFIG['start_date']} ~ {BACKTEST_CONFIG['end_date']}")
    
    # 執行回測
    run_single_backtest(TEST_TICKER, show_plot=True) 