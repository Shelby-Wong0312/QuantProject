#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
檢查自動化系統狀態
"""

import os
import json
from datetime import datetime

print("\n" + "="*60)
print(" Automation System Status Check")
print("="*60)
print(f" Check Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)

# Check log file
if os.path.exists('TASK_EXECUTION_LOG.md'):
    print("\n[LOG FILE] TASK_EXECUTION_LOG.md exists")
    with open('TASK_EXECUTION_LOG.md', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if 'Last update' in line or '2025-08-07' in line[-30:]:
                try:
                    print(f"  Last line: {line.strip()[-30:]}")
                except:
                    print("  [Contains non-ASCII characters]")
                break
else:
    print("\n[WARNING] Log file not found")

# Check diagnosis report
if os.path.exists('diagnosis_report.json'):
    print("\n[DIAGNOSIS] diagnosis_report.json found")
    with open('diagnosis_report.json', 'r') as f:
        data = json.load(f)
        print(f"  Status: {data.get('status', 'unknown')}")
        print(f"  Time: {data.get('timestamp', 'unknown')}")
        if 'checks' in data:
            for check, result in data['checks'].items():
                print(f"  {check}: {result}")

# Check if automation scripts exist
scripts = [
    'automation_core.py',
    'automation_with_logging.py',
    'mt4_diagnosis.py',
    'collect_btc_markets.py',
    'qa_trading_test.py'
]

print("\n[SCRIPTS] Checking required scripts:")
for script in scripts:
    if os.path.exists(script):
        print(f"  [OK] {script}")
    else:
        print(f"  [MISSING] {script}")

# Check for recent data files
data_files = ['market_data.json', 'automation_status.json']
print("\n[DATA FILES] Checking data files:")
for file in data_files:
    if os.path.exists(file):
        mod_time = os.path.getmtime(file)
        mod_datetime = datetime.fromtimestamp(mod_time)
        print(f"  [OK] {file} - Last modified: {mod_datetime.strftime('%H:%M:%S')}")
    else:
        print(f"  [NOT FOUND] {file}")

print("\n" + "="*60)
print(" Status Check Complete")
print("="*60)