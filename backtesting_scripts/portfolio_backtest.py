# backtesting_scripts/portfolio_backtest.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from strategy.concrete_strategies.portfolio_three_level_strategy import PortfolioThreeLevelStrategy
from backtesting_scripts.report_generator import HTMLReportGenerator

class PortfolioBacktest:
    """投資組合回測引擎"""
    
    def __init__(self, initial_capital=1000, commission=0.002, position_size_pct=0.01):
        """
        初始化回測引擎
        
        Args:
            initial_capital: 初始資金（美元）
            commission: 手續費率
            position_size_pct: 每次交易佔總資產的比例
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.position_size_pct = position_size_pct
        
        # 帳戶狀態
        self.cash = initial_capital
        self.positions = {}  # {symbol: {'shares': x, 'avg_price': y}}
        self.trades = []  # 交易記錄
        self.equity_curve = []  # 資金曲線
        self.daily_returns = []  # 日收益率
        
    def load_symbols(self, filepath="valid_tickers.txt"):
        """載入股票列表"""
        if not os.path.exists(filepath):
            print(f"找不到 {filepath}，使用預設股票列表")
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "WMT"]
        
        with open(filepath, 'r') as f:
            symbols = [line.strip().upper() for line in f if line.strip()]
        return symbols
    
    def download_data(self, symbol, start_date, end_date):
        """下載股票數據"""
        try:
            data = yf.download(symbol, start=start_date, end=end_date, interval='1d', progress=False)
            if not data.empty:
                return data
        except:
            pass
        return None
    
    def calculate_position_size(self, price):
        """計算倉位大小"""
        total_equity = self.get_total_equity()
        position_value = total_equity * self.position_size_pct
        shares = int(position_value / price)
        return shares
    
    def get_total_equity(self, current_prices=None):
        """計算總資產"""
        equity = self.cash
        
        if self.positions and current_prices:
            for symbol, pos in self.positions.items():
                if symbol in current_prices:
                    equity += pos['shares'] * current_prices[symbol]
                    
        return equity
    
    def execute_buy(self, symbol, price, date, reason):
        """執行買入"""
        shares = self.calculate_position_size(price)
        if shares == 0:
            return False
            
        cost = shares * price * (1 + self.commission)
        
        if cost > self.cash:
            # 資金不足，調整股數
            shares = int(self.cash / (price * (1 + self.commission)))
            if shares == 0:
                return False
            cost = shares * price * (1 + self.commission)
        
        # 更新現金和倉位
        self.cash -= cost
        
        if symbol in self.positions:
            # 加倉
            old_shares = self.positions[symbol]['shares']
            old_avg_price = self.positions[symbol]['avg_price']
            new_shares = old_shares + shares
            new_avg_price = (old_shares * old_avg_price + shares * price) / new_shares
            
            self.positions[symbol] = {
                'shares': new_shares,
                'avg_price': new_avg_price
            }
        else:
            # 新倉位
            self.positions[symbol] = {
                'shares': shares,
                'avg_price': price
            }
        
        # 記錄交易
        self.trades.append({
            'date': date,
            'symbol': symbol,
            'action': 'BUY',
            'shares': shares,
            'price': price,
            'commission': shares * price * self.commission,
            'reason': reason
        })
        
        return True
    
    def execute_sell(self, symbol, price, date, reason):
        """執行賣出（平倉）"""
        if symbol not in self.positions:
            return False
            
        pos = self.positions[symbol]
        shares = pos['shares']
        
        # 計算收益
        revenue = shares * price * (1 - self.commission)
        cost_basis = shares * pos['avg_price']
        pnl = revenue - cost_basis
        
        # 更新現金
        self.cash += revenue
        
        # 移除倉位
        del self.positions[symbol]
        
        # 記錄交易
        self.trades.append({
            'date': date,
            'symbol': symbol,
            'action': 'SELL',
            'shares': shares,
            'price': price,
            'commission': shares * price * self.commission,
            'pnl': pnl,
            'reason': reason
        })
        
        return True
    
    def run_backtest(self, strategy_level=3, start_date='2023-01-01', end_date=None):
        """
        執行回測
        
        Args:
            strategy_level: 策略級別 (1, 2, 或 3)
            start_date: 開始日期
            end_date: 結束日期
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"\n{'='*80}")
        print(f"投資組合回測系統")
        print(f"{'='*80}")
        print(f"策略級別: Level {strategy_level}")
        print(f"回測期間: {start_date} ~ {end_date}")
        print(f"初始資金: ${self.initial_capital}")
        print(f"手續費率: {self.commission*100}%")
        print(f"倉位大小: 總資產的{self.position_size_pct*100}%")
        
        # 載入股票列表
        symbols = self.load_symbols()
        print(f"股票數量: {len(symbols)}")
        
        # 初始化策略
        strategy = PortfolioThreeLevelStrategy(strategy_level=strategy_level)
        
        # 生成交易日期列表
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        trading_days = pd.bdate_range(start=start, end=end)
        
        print(f"\n開始回測...")
        print("="*80)
        
        # 下載所有股票數據
        print("正在下載股票數據...")
        all_data = {}
        valid_symbols = []
        
        for i, symbol in enumerate(symbols):
            if i % 100 == 0:
                print(f"  進度: {i}/{len(symbols)}")
                
            data = self.download_data(symbol, start_date, end_date)
            if data is not None and len(data) > 60:  # 需要足夠的數據計算指標
                all_data[symbol] = data
                valid_symbols.append(symbol)
                
        print(f"有效股票數量: {len(valid_symbols)}")
        
        # 主回測循環
        print("\n執行交易模擬...")
        
        for i, date in enumerate(trading_days):
            if i % 20 == 0:  # 每月顯示一次進度
                print(f"  日期: {date.strftime('%Y-%m-%d')}")
            
            # 獲取當日價格
            current_prices = {}
            for symbol in valid_symbols:
                if date in all_data[symbol].index:
                    current_prices[symbol] = all_data[symbol].loc[date, 'Close']
            
            # 更新資金曲線
            total_equity = self.get_total_equity(current_prices)
            self.equity_curve.append({
                'date': date,
                'equity': total_equity,
                'cash': self.cash,
                'positions_value': total_equity - self.cash
            })
            
            # 計算日收益率
            if len(self.equity_curve) > 1:
                daily_return = (total_equity - self.equity_curve[-2]['equity']) / self.equity_curve[-2]['equity']
                self.daily_returns.append(daily_return)
            
            # === 多標的同時持有邏輯 ===
            # 1. 先檢查所有持倉的出場信號
            for symbol in list(self.positions.keys()):
                if symbol in all_data and date in all_data[symbol].index:
                    hist_data = all_data[symbol].loc[:date]
                    signal = strategy.check_signals(hist_data)
                    if signal['action'] == 'sell':
                        price = current_prices.get(symbol)
                        if price:
                            self.execute_sell(symbol, price, date, '|'.join(signal['reasons']))
            # 2. 檢查所有未持倉標的的進場信號
            for symbol in valid_symbols:
                if symbol in self.positions:
                    continue  # 已持有
                if symbol in all_data and date in all_data[symbol].index:
                    hist_data = all_data[symbol].loc[:date]
                    signal = strategy.check_signals(hist_data)
                    if signal['action'] == 'buy':
                        price = current_prices.get(symbol)
                        if price:
                            self.execute_buy(symbol, price, date, '|'.join(signal['reasons']))
        
        # 回測結束，平掉所有倉位
        print("\n回測結束，清算所有倉位...")
        final_date = trading_days[-1]
        for symbol in list(self.positions.keys()):
            if symbol in current_prices:
                self.execute_sell(symbol, current_prices[symbol], final_date, '回測結束')
        
        print("\n回測完成！")
        
        # 計算績效指標
        self.calculate_performance_metrics()
        
        return True
    
    def calculate_performance_metrics(self):
        """計算績效指標"""
        if not self.equity_curve:
            return
            
        # 最終資產
        self.final_equity = self.equity_curve[-1]['equity']
        
        # 總報酬率
        self.total_return = (self.final_equity - self.initial_capital) / self.initial_capital
        
        # 夏普比率
        if self.daily_returns:
            returns_array = np.array(self.daily_returns)
            self.sharpe_ratio = np.sqrt(252) * returns_array.mean() / (returns_array.std() + 1e-6)
        else:
            self.sharpe_ratio = 0
        
        # 最大回撤
        equity_values = [e['equity'] for e in self.equity_curve]
        peak = equity_values[0]
        max_dd = 0
        
        for value in equity_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
                
        self.max_drawdown = max_dd
        
        # 勝率
        winning_trades = [t for t in self.trades if t['action'] == 'SELL' and t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trades if t['action'] == 'SELL' and t.get('pnl', 0) <= 0]
        total_closed_trades = len(winning_trades) + len(losing_trades)
        
        if total_closed_trades > 0:
            self.win_rate = len(winning_trades) / total_closed_trades
        else:
            self.win_rate = 0
        
        # 總交易次數
        self.total_trades = len([t for t in self.trades if t['action'] == 'BUY'])
    
    def generate_report(self, strategy_level):
        """生成HTML報告"""
        report_generator = HTMLReportGenerator()
        
        # 準備數據
        report_data = {
            'strategy_level': strategy_level,
            'start_date': self.equity_curve[0]['date'].strftime('%Y-%m-%d'),
            'end_date': self.equity_curve[-1]['date'].strftime('%Y-%m-%d'),
            'initial_capital': self.initial_capital,
            'final_equity': self.final_equity,
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades,
            'equity_curve': self.equity_curve,
            'trades': self.trades
        }
        
        # 生成報告
        report_generator.generate_report(report_data)
        print("\n✓ HTML報告已生成: backtest_report.html")

def main():
    """主程式"""
    print("投資組合三級策略回測系統 v1.0")
    print("="*80)
    
    # 選擇策略級別
    print("\n請選擇策略級別:")
    print("1 - Level 1: 單一指標信號")
    print("2 - Level 2: 雙指標共振")
    print("3 - Level 3: 三指標及以上共振")
    
    while True:
        try:
            level = int(input("\n請輸入策略級別 (1-3): "))
            if level in [1, 2, 3]:
                break
            else:
                print("請輸入 1, 2 或 3")
        except:
            print("請輸入有效的數字")
    
    # 創建回測引擎
    backtest = PortfolioBacktest(
        initial_capital=1000,
        commission=0.002,
        position_size_pct=0.01
    )
    
    # 執行回測
    backtest.run_backtest(
        strategy_level=level,
        start_date='2023-01-01',
        end_date=None  # 使用當前日期
    )
    
    # 生成報告
    backtest.generate_report(strategy_level=level)

if __name__ == "__main__":
    main() 