#!/usr/bin/env python3
"""
Monitor full backtest progress and report when complete
"""

import time
import os
import sys
from datetime import datetime

def check_backtest_status():
    """Check if backtest is complete"""
    # Check if output file exists and contains completion marker
    output_file = "backtest_full_output.txt"

    if not os.path.exists(output_file):
        return False, "No output file found"

    try:
        with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

            # Check for completion marker
            if "[SUCCESS] FULL BACKTEST COMPLETE!" in content:
                return True, "Completed"

            # Check for error
            if "Traceback" in content and "[3/6] Running backtest" not in content:
                return True, "Error occurred"

            # Extract progress
            if "Backtesting stocks:" in content:
                lines = content.split('\n')
                for line in reversed(lines):
                    if "Backtesting stocks:" in line and '%' in line:
                        # Extract percentage
                        try:
                            pct_start = line.find('|') + 1
                            pct_end = line.find('%', pct_start)
                            if pct_start > 0 and pct_end > 0:
                                pct_str = line[pct_start:pct_end].strip()
                                # Clean up percentage string
                                pct_str = ''.join(c for c in pct_str if c.isdigit() or c == '.')
                                if pct_str:
                                    pct = float(pct_str)
                                    return False, f"In progress: {pct:.1f}% complete"
                        except:
                            pass

            return False, "Running (checking...)"

    except Exception as e:
        return False, f"Error reading file: {e}"

def main():
    """Monitor backtest progress"""
    print("=" * 80)
    print("BACKTEST PROGRESS MONITOR")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Monitoring backtest_full_output.txt for completion...")
    print("=" * 80)

    check_interval = 60  # Check every 60 seconds
    last_status = ""

    while True:
        is_complete, status = check_backtest_status()

        current_time = datetime.now().strftime('%H:%M:%S')

        if status != last_status:
            print(f"[{current_time}] Status: {status}")
            last_status = status

        if is_complete:
            print("\n" + "=" * 80)
            print(f"BACKTEST FINISHED AT {current_time}")
            print(f"Final Status: {status}")
            print("=" * 80)

            if "Error" in status:
                print("⚠️ Backtest ended with error - check backtest_full_output.txt")
                sys.exit(1)
            else:
                print("✓ Backtest completed successfully!")
                print("Results should be in:")
                print("  - reports/backtest/local_ppo_oos_full_4215_2023_2025.md")
                print("  - reports/backtest/local_ppo_oos_full_4215_2023_2025_metrics.json")
                print("  - reports/backtest/visualizations/")
                sys.exit(0)

        time.sleep(check_interval)

if __name__ == "__main__":
    main()
