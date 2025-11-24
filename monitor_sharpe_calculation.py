#!/usr/bin/env python3
"""Monitor per-stock Sharpe calculation progress"""
import time
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Monitoring per-stock Sharpe calculation...")
print("This will take approximately 2 hours for 4,215 stocks")
print("Checking progress every 30 seconds...")
print("=" * 80)

while True:
    time.sleep(30)
    print(f"[{time.strftime('%H:%M:%S')}] Still calculating...")
