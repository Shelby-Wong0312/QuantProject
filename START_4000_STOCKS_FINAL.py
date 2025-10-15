"""
Final Working Version - 4000+ Stock Monitoring System
Ready for production use
"""

import sys
import os
from pathlib import Path
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


def main():
    """Main program: Start monitoring system"""

    print("\n" + "=" * 80)
    print("4000+ STOCK MONITORING SYSTEM - PRODUCTION VERSION")
    print("=" * 80)

    try:
        # 1. Import modules
        print("\n[1/4] Loading modules...")
        from monitoring.tiered_monitor import TieredMonitor, TierLevel
        from data_pipeline.free_data_client import FreeDataClient

        print("   [OK] Modules loaded")

        # 2. Load stock list
        print("\n[2/4] Loading stocks...")
        all_symbols_file = Path("data/all_symbols.txt")
        if all_symbols_file.exists():
            with open(all_symbols_file, "r") as f:
                all_symbols = [line.strip() for line in f.readlines() if line.strip()]
                # You can adjust the number of stocks here
                all_symbols = all_symbols[:4000]  # Use up to 4000 stocks
        else:
            # Demo list if no file
            all_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM"]

        print(f"   [OK] Loaded {len(all_symbols)} stocks")

        # 3. Initialize monitor
        print("\n[3/4] Setting up monitoring system...")

        # Ensure config exists
        config_path = Path("monitoring/config.yaml")
        if not config_path.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)
            # Create minimal config
            import yaml

            config = {
                "monitoring": {
                    "tiers": {
                        "S": {"max_symbols": 40, "update_interval": 1},
                        "A": {"max_symbols": 100, "update_interval": 60},
                        "B": {"max_symbols": 4000, "update_interval": 300},
                    }
                }
            }
            with open(config_path, "w") as f:
                yaml.dump(config, f)

        monitor = TieredMonitor(str(config_path))

        # Add stocks to tiers
        print("   Adding stocks to tiers...")
        for i, symbol in enumerate(all_symbols):
            if i < 40:
                monitor._add_stock_to_tier(symbol, TierLevel.S_TIER)
            elif i < 140:
                monitor._add_stock_to_tier(symbol, TierLevel.A_TIER)
            else:
                monitor._add_stock_to_tier(symbol, TierLevel.B_TIER)

        print(f"   [OK] Monitoring system ready")
        print(f"        S-tier: 40 stocks (real-time)")
        print(f"        A-tier: 100 stocks (1-min updates)")
        print(f"        B-tier: {len(all_symbols)-140} stocks (5-min scans)")

        # 4. Start monitoring
        print("\n[4/4] Starting monitoring...")
        monitor.start_monitoring()

        print("\n" + "=" * 80)
        print("SYSTEM RUNNING - Monitoring", len(all_symbols), "stocks")
        print("Press Ctrl+C to stop")
        print("=" * 80 + "\n")

        # Main loop
        start_time = time.time()
        while True:
            elapsed = int(time.time() - start_time)

            # Show status every 30 seconds
            if elapsed % 30 == 0 and elapsed > 0:
                status = monitor.get_monitoring_status()
                print(
                    f"\n[STATUS] Runtime: {elapsed}s | Stocks: {status.get('total_stocks', len(all_symbols))} | Active: {status.get('is_running', False)}"
                )

                # Show some tier details
                details = monitor.get_tier_details()
                if details and "S_tier" in details:
                    s_tier = details["S_tier"]
                    if "symbol_count" in s_tier:
                        print(f"         S-tier: {s_tier['symbol_count']} stocks monitored")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
        if "monitor" in locals():
            monitor.stop_monitoring()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("\nSystem stopped")


if __name__ == "__main__":
    # Check dependencies
    try:
        import pandas
        import numpy
        import yfinance
        import yaml
    except ImportError as e:
        print(f"Missing package: {e}")
        print("Install with: pip install pandas numpy yfinance pyyaml")
        sys.exit(1)

    # Run
    main()
