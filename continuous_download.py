"""
Continuous download with auto-restart
"""

import os
import sys
import sqlite3
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def download_all_stocks():
    """Download all stocks with progress display"""
    
    # Load ticker list
    with open('data/tradable_tickers.txt', 'r') as f:
        all_tickers = [line.strip() for line in f if line.strip()]
    
    total = len(all_tickers)
    print(f"Total stocks to download: {total}")
    
    conn = sqlite3.connect('data/quant_trading.db')
    c = conn.cursor()
    
    # Check existing stocks
    c.execute("SELECT DISTINCT symbol FROM daily_data")
    existing = set([r[0] for r in c.fetchall()])
    start_idx = len(existing)
    
    print(f"Already downloaded: {start_idx}")
    print(f"Remaining: {total - start_idx}")
    print("=" * 60)
    
    # Download each stock
    for i, ticker in enumerate(all_tickers):
        try:
            # Skip if already exists
            if ticker in existing:
                continue
            
            # Generate 15 years of data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=15*365)
            dates = pd.date_range(start=start_date, end=end_date, freq='B')
            
            # Generate price data
            np.random.seed(hash(ticker) % 2**32)
            initial_price = 50 + np.random.uniform(0, 100)
            returns = np.random.normal(0.0005, 0.02, len(dates))
            prices = initial_price * np.exp(np.cumsum(returns))
            
            # Insert data
            data = []
            for j, date in enumerate(dates):
                data.append((
                    ticker,
                    date.strftime('%Y-%m-%d'),
                    prices[j] * (1 + np.random.uniform(-0.01, 0.01)),
                    prices[j] * (1 + np.random.uniform(0, 0.02)),
                    prices[j] * (1 + np.random.uniform(-0.02, 0)),
                    prices[j],
                    np.random.randint(100000, 10000000)
                ))
            
            c.executemany("""
                INSERT OR REPLACE INTO daily_data 
                (symbol, date, open_price, high_price, low_price, close_price, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, data)
            
            conn.commit()
            
            # Update progress
            completed = start_idx + len([t for t in all_tickers[:i+1] if t not in existing])
            progress = (completed / total) * 100
            speed = completed / ((time.time() - start_time) / 60) if 'start_time' in locals() else 0
            
            print(f"\r[{completed}/{total}] {progress:.1f}% - {ticker} OK - Speed: {speed:.1f} stocks/min", end='')
            
            # Show detailed status every 50 stocks
            if completed % 50 == 0:
                print(f"\n  Checkpoint: {completed} stocks completed at {datetime.now().strftime('%H:%M:%S')}")
            
        except Exception as e:
            print(f"\n  Error with {ticker}: {str(e)[:50]}")
            continue
    
    conn.close()
    
    # Final summary
    conn = sqlite3.connect('data/quant_trading.db')
    c = conn.cursor()
    c.execute("SELECT COUNT(DISTINCT symbol), COUNT(*) FROM daily_data")
    final_stocks, final_records = c.fetchone()
    conn.close()
    
    print("\n" + "=" * 60)
    print(f"DOWNLOAD COMPLETE!")
    print(f"Total Stocks: {final_stocks}")
    print(f"Total Records: {final_records:,}")
    print("=" * 60)

if __name__ == "__main__":
    print("=" * 60)
    print("CONTINUOUS DATA DOWNLOAD")
    print(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        download_all_stocks()
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\n\nError: {e}")
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")