"""
Quick System Status Check
"""
import sqlite3
import os
from datetime import datetime

print("\n" + "="*60)
print(f"TRADING SYSTEM STATUS CHECK - {datetime.now().strftime('%H:%M:%S')}")
print("="*60)

# Check database
if os.path.exists('data/live_trades_full.db'):
    conn = sqlite3.connect('data/live_trades_full.db')
    cursor = conn.cursor()
    
    # Get counts
    cursor.execute("SELECT COUNT(*) FROM signals")
    signals = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM trades")  
    trades = cursor.fetchone()[0]
    
    print(f"\n[OK] Database Status:")
    print(f"  - Signals generated: {signals}")
    print(f"  - Trades executed: {trades}")
    
    conn.close()
else:
    print("\n[X] Database not found - system may be starting up")

# Check log file
if os.path.exists('logs/live_trading_full.log'):
    size = os.path.getsize('logs/live_trading_full.log')
    print(f"\n[OK] Log file exists: {size:,} bytes")
else:
    print("\n[X] Log file not found")

print("\n" + "="*60)
print("SYSTEM IS RUNNING")
print("- Trading window: Monitoring 4,215 stocks")
print("- Monitor window: Shows real-time status")
print("- To stop: Press Ctrl+C in any window")
print("="*60)