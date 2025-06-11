# backtesting_scripts/run_portfolio_backtest.py
"""
投資組合回測執行腳本
直接運行: python backtesting_scripts/run_portfolio_backtest.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting_scripts.portfolio_backtest import PortfolioBacktest

def main():
    """主程式"""
    print("投資組合三級策略回測系統")
    print("="*80)
    
    # 直接執行Level 3策略回測
    print("\n執行 Level 3 (三指標及以上共振) 策略回測...")
    
    # 創建回測引擎
    backtest = PortfolioBacktest(
        initial_capital=1000,
        commission=0.002,
        position_size_pct=0.01
    )
    
    # 執行回測
    success = backtest.run_backtest(
        strategy_level=3,  # 使用Level 3策略
        start_date='2023-01-01',
        end_date=None  # 使用當前日期
    )
    
    if success:
        # 生成報告
        backtest.generate_report(strategy_level=3)
        print("\n✓ 回測完成！請查看 backtest_report.html")
    else:
        print("\n✗ 回測失敗")

if __name__ == "__main__":
    main() 