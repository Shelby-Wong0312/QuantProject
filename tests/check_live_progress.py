"""
Real-time download progress checker
"""

import sqlite3
import time
from datetime import datetime

def check_progress():
    """Check and display real-time progress"""
    
    print("=" * 60)
    print("REAL-TIME DATA DOWNLOAD MONITOR")
    print("=" * 60)
    
    # 初始檢查
    conn = sqlite3.connect('data/quant_trading.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(DISTINCT symbol) FROM daily_data")
    initial_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM daily_data")
    initial_records = cursor.fetchone()[0]
    
    print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Initial Status: {initial_count} stocks, {initial_records:,} records")
    
    # Monitor for 30 seconds
    print("\nMonitoring... (30 seconds)")
    print("-" * 40)
    
    for i in range(6):  # Check every 5 seconds for 30 seconds
        time.sleep(5)
        
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM daily_data")
        current_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM daily_data")
        current_records = cursor.fetchone()[0]
        
        new_stocks = current_count - initial_count
        new_records = current_records - initial_records
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        if new_stocks > 0 or new_records > 0:
            print(f"[{timestamp}] [NEW] Added: {new_stocks} stocks, {new_records:,} records")
            
            # Show recently downloaded stocks
            cursor.execute("""
                SELECT symbol, COUNT(*) as cnt 
                FROM daily_data 
                GROUP BY symbol 
                ORDER BY MAX(ROWID) DESC 
                LIMIT 3
            """)
            recent = cursor.fetchall()
            for symbol, cnt in recent:
                print(f"           -> {symbol}: {cnt} records")
        else:
            print(f"[{timestamp}] - No new data")
    
    # Final statistics
    cursor.execute("SELECT COUNT(DISTINCT symbol) FROM daily_data")
    final_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM daily_data")
    final_records = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM stocks WHERE is_tradable = 1")
    total_stocks = cursor.fetchone()[0]
    
    conn.close()
    
    print("-" * 40)
    print(f"\nMonitoring Results:")
    print(f"  New Stocks Added: {final_count - initial_count}")
    print(f"  New Records Added: {final_records - initial_records:,}")
    print(f"  Current Progress: {final_count}/{total_stocks} ({final_count/total_stocks*100:.2f}%)")
    
    if final_count - initial_count > 0:
        print(f"\n[SUCCESS] System is downloading data!")
        speed = (final_count - initial_count) / 0.5  # stocks per minute
        eta = (total_stocks - final_count) / speed / 60 if speed > 0 else 0
        print(f"   Speed: {speed:.1f} stocks/minute")
        print(f"   ETA: {eta:.1f} hours")
    else:
        print(f"\n[WARNING] No new data detected")
        print(f"   Possible reasons:")
        print(f"   1. Download process not running")
        print(f"   2. Network connection issues")
        print(f"   3. API rate limits")

if __name__ == "__main__":
    check_progress()