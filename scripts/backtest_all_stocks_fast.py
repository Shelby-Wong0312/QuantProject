"""
Fast Comprehensive Backtest for ALL Stocks
Optimized version with immediate progress reporting
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import json
import time
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import only the winner indicator for speed
from quantproject.indicators.momentum_indicators import CCI


class FastComprehensiveBacktest:
    """Fast backtest focusing on the winning indicator"""
    
    def __init__(self):
        self.db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    'data', 'quant_trading.db')
        self.initial_capital = 100000.0
        self.commission_rate = 0.001
        
        # Use only CCI-20 (the winner) for faster processing
        self.indicator = CCI(period=20)
        self.indicator_name = 'CCI_20'
    
    def get_all_stocks(self):
        """Get all stocks with sufficient data"""
        print("Loading stock list from database...")
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT symbol, COUNT(*) as data_points
            FROM daily_data
            GROUP BY symbol
            HAVING COUNT(*) >= 200
            ORDER BY symbol
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"Found {len(df)} stocks with 200+ days of data")
        return df['symbol'].tolist()
    
    def backtest_stock(self, symbol: str) -> Dict:
        """Fast backtest for single stock"""
        
        conn = sqlite3.connect(self.db_path)
        
        # Get ALL 15 years of data
        query = f"""
            SELECT date, close_price as close, high_price as high, 
                   low_price as low, open_price as open, volume
            FROM daily_data
            WHERE symbol = '{symbol}'
            ORDER BY date ASC
        """
        
        try:
            df = pd.read_sql_query(query, conn, parse_dates=['date'])
            conn.close()
            
            if len(df) < 100:
                return None
                
            df.set_index('date', inplace=True)
            df = df.sort_index()
            
            # Generate signals
            signals = self.indicator.get_signals(df)
            
            # Simple backtest
            cash = self.initial_capital
            shares = 0
            trades = 0
            
            for i in range(len(df)):
                if i >= len(signals):
                    break
                
                price = df['close'].iloc[i]
                
                if signals['buy'].iloc[i] and cash > 0:
                    shares = cash / price
                    cash = 0
                    trades += 1
                elif signals['sell'].iloc[i] and shares > 0:
                    cash = shares * price
                    shares = 0
            
            # Final value
            final_value = cash + shares * df['close'].iloc[-1] if len(df) > 0 else self.initial_capital
            total_return = (final_value / self.initial_capital - 1) * 100
            
            return {
                'symbol': symbol,
                'return': total_return,
                'trades': trades,
                'final_value': final_value
            }
            
        except:
            conn.close()
            return None
    
    def run_fast_backtest(self):
        """Run fast backtest on ALL stocks"""
        
        print("=" * 80)
        print("FAST COMPREHENSIVE BACKTEST - ALL STOCKS")
        print("=" * 80)
        print(f"Start Time: {datetime.now()}")
        print(f"Testing indicator: {self.indicator_name} (Previous Winner)")
        print("-" * 80)
        
        # Get all stocks
        all_stocks = self.get_all_stocks()
        
        print(f"\nStarting backtest on {len(all_stocks)} stocks...")
        print("Progress updates every 100 stocks...")
        print("-" * 80)
        
        # Store results
        results = []
        profitable_count = 0
        
        start_time = time.time()
        
        # Process all stocks
        for idx, symbol in enumerate(all_stocks):
            result = self.backtest_stock(symbol)
            
            if result:
                results.append(result)
                if result['return'] > 0:
                    profitable_count += 1
            
            # Progress update every 100 stocks
            if (idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                remaining = (len(all_stocks) - idx - 1) / rate
                
                print(f"Progress: {idx+1}/{len(all_stocks)} ({(idx+1)/len(all_stocks)*100:.1f}%) | "
                      f"Profitable: {profitable_count}/{len(results)} ({profitable_count/len(results)*100:.1f}%) | "
                      f"Speed: {rate:.1f} stocks/sec | "
                      f"ETA: {remaining:.0f} sec")
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0:
            print("No valid results!")
            return
        
        # Sort by return
        results_df = results_df.sort_values('return', ascending=False)
        
        # Save results
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'reports', 'all_stocks_cci_backtest.csv')
        results_df.to_csv(csv_path, index=False)
        
        # Print top performers
        print("\nTOP 50 PERFORMING STOCKS WITH CCI-20:")
        print("-" * 80)
        print(f"{'Rank':<6} {'Symbol':<10} {'Return':<12} {'Trades':<10} {'Final Value':<15}")
        print("-" * 80)
        
        for i, row in enumerate(results_df.head(50).itertuples(), 1):
            return_val = getattr(row, 'return')
            print(f"{i:<6} {row.symbol:<10} {return_val:>10.2f}% {row.trades:>9} ${row.final_value:>14,.0f}")
        
        # Statistics
        print("\n" + "=" * 80)
        print("OVERALL STATISTICS")
        print("=" * 80)
        
        profitable = results_df[results_df['return'] > 0]
        losing = results_df[results_df['return'] <= 0]
        
        print(f"Total stocks tested: {len(results_df)}")
        print(f"Profitable stocks: {len(profitable)} ({len(profitable)/len(results_df)*100:.1f}%)")
        print(f"Losing stocks: {len(losing)} ({len(losing)/len(results_df)*100:.1f}%)")
        print(f"\nAverage return: {results_df['return'].mean():.2f}%")
        print(f"Median return: {results_df['return'].median():.2f}%")
        print(f"Best return: {results_df['return'].max():.2f}% ({results_df.iloc[0]['symbol']})")
        print(f"Worst return: {results_df['return'].min():.2f}% ({results_df.iloc[-1]['symbol']})")
        print(f"\nAverage trades: {results_df['trades'].mean():.1f}")
        print(f"Total trades across all stocks: {results_df['trades'].sum()}")
        
        # Find stocks with >50% return
        high_performers = results_df[results_df['return'] > 50]
        print(f"\nStocks with >50% return: {len(high_performers)}")
        
        if len(high_performers) > 0:
            print("\nHIGH PERFORMERS (>50% return):")
            print("-" * 40)
            for row in high_performers.head(20).itertuples():
                return_val = getattr(row, 'return')
            print(f"  {row.symbol}: {return_val:.2f}%")
        
        # Create recommended portfolio
        print("\n" + "=" * 80)
        print("RECOMMENDED PORTFOLIO (Top 20 by Return)")
        print("=" * 80)
        
        top_20 = results_df.head(20)
        portfolio_value = 1000000  # $1M portfolio
        allocation = portfolio_value / 20  # Equal weight
        
        print(f"Portfolio size: ${portfolio_value:,.0f}")
        print(f"Equal allocation per stock: ${allocation:,.0f}")
        print("-" * 80)
        
        expected_return = 0
        for row in top_20.itertuples():
            return_val = getattr(row, 'return')
            expected_profit = allocation * (return_val / 100)
            expected_return += return_val / 20
            print(f"{row.symbol:<8} - Expected profit: ${expected_profit:>10,.0f} ({return_val:>6.2f}%)")
        
        print("-" * 80)
        print(f"Expected portfolio return: {expected_return:.2f}%")
        print(f"Expected portfolio value: ${portfolio_value * (1 + expected_return/100):,.0f}")
        
        # Save summary report
        summary = {
            'test_date': datetime.now().isoformat(),
            'indicator': self.indicator_name,
            'total_stocks': len(results_df),
            'profitable_stocks': len(profitable),
            'success_rate': len(profitable) / len(results_df) * 100,
            'avg_return': results_df['return'].mean(),
            'median_return': results_df['return'].median(),
            'best_stock': {
                'symbol': results_df.iloc[0]['symbol'],
                'return': results_df.iloc[0]['return']
            },
            'top_20': top_20[['symbol', 'return']].to_dict('records'),
            'high_performers_count': len(high_performers)
        }
        
        json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'reports', 'all_stocks_summary.json')
        
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        elapsed = time.time() - start_time
        print(f"\nTotal processing time: {elapsed:.1f} seconds")
        print(f"Processing speed: {len(results_df)/elapsed:.1f} stocks/second")
        
        print(f"\nResults saved to:")
        print(f"  - CSV: {csv_path}")
        print(f"  - JSON: {json_path}")
        
        print("\n" + "=" * 80)
        print("BACKTEST COMPLETE!")
        print("=" * 80)
        
        return results_df


if __name__ == "__main__":
    print("Starting fast comprehensive backtest...")
    print("This will test CCI-20 (the winning indicator) on ALL stocks...")
    print()
    
    backtester = FastComprehensiveBacktest()
    results = backtester.run_fast_backtest()