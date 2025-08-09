"""
Monitor data download progress in real-time
"""

import sqlite3
import time
import os
from datetime import datetime

def get_download_stats():
    """Get current download statistics"""
    
    conn = sqlite3.connect('data/quant_trading.db')
    cursor = conn.cursor()
    
    # Get downloaded stocks count
    cursor.execute("""
        SELECT COUNT(DISTINCT symbol) as stocks,
               COUNT(*) as total_records,
               MIN(date) as earliest,
               MAX(date) as latest
        FROM daily_data
    """)
    stats = cursor.fetchone()
    
    # Get total target
    cursor.execute("SELECT COUNT(*) FROM stocks WHERE is_tradable = 1")
    total_stocks = cursor.fetchone()[0]
    
    # Get recently downloaded stocks
    cursor.execute("""
        SELECT symbol, COUNT(*) as records
        FROM daily_data
        GROUP BY symbol
        ORDER BY MAX(ROWID) DESC
        LIMIT 5
    """)
    recent = cursor.fetchall()
    
    conn.close()
    
    return {
        'downloaded_stocks': stats[0] if stats[0] else 0,
        'total_records': stats[1] if stats[1] else 0,
        'earliest_date': stats[2],
        'latest_date': stats[3],
        'total_stocks': total_stocks,
        'recent': recent
    }

def monitor_progress():
    """Monitor download progress"""
    
    print("=" * 70)
    print("DATA DOWNLOAD MONITOR - Press Ctrl+C to stop")
    print("=" * 70)
    
    last_count = 0
    start_time = datetime.now()
    
    while True:
        try:
            stats = get_download_stats()
            current_count = stats['downloaded_stocks']
            progress = (current_count / stats['total_stocks'] * 100) if stats['total_stocks'] > 0 else 0
            
            # Calculate speed
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > 0 and current_count > last_count:
                speed = (current_count - last_count) / (elapsed / 60)  # stocks per minute
                eta_minutes = (stats['total_stocks'] - current_count) / speed if speed > 0 else 0
                eta_hours = eta_minutes / 60
            else:
                speed = 0
                eta_hours = 0
            
            # Clear screen (Windows)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("=" * 70)
            print(f"DATA DOWNLOAD PROGRESS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 70)
            
            print(f"\nProgress: {current_count}/{stats['total_stocks']} stocks ({progress:.2f}%)")
            print(f"Total Records: {stats['total_records']:,}")
            
            if stats['earliest_date'] and stats['latest_date']:
                print(f"Date Range: {stats['earliest_date']} to {stats['latest_date']}")
            
            if speed > 0:
                print(f"Speed: {speed:.1f} stocks/minute")
                if eta_hours > 0:
                    if eta_hours > 1:
                        print(f"ETA: {eta_hours:.1f} hours")
                    else:
                        print(f"ETA: {eta_hours * 60:.0f} minutes")
            
            if stats['recent']:
                print(f"\nRecently Downloaded:")
                for symbol, records in stats['recent']:
                    print(f"  - {symbol}: {records:,} records")
            
            # Progress bar
            bar_length = 50
            filled = int(bar_length * progress / 100)
            bar = '[' + '#' * filled + '-' * (bar_length - filled) + ']'
            print(f"\n{bar} {progress:.1f}%")
            
            # Check if download is complete
            if current_count >= stats['total_stocks']:
                print("\n" + "=" * 70)
                print("DOWNLOAD COMPLETE!")
                print(f"Successfully downloaded {current_count} stocks")
                print(f"Total records: {stats['total_records']:,}")
                print("=" * 70)
                break
            
            last_count = current_count
            time.sleep(10)  # Update every 10 seconds
            
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_progress()