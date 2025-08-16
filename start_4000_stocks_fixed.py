"""
4000+ Stock Large-Scale Monitoring System - Fixed Version
"""

import asyncio
import sys
import os
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

async def main():
    """Main program: Start 4000+ stock monitoring system"""
    
    print("\n" + "="*80)
    print("[STARTING] 4000+ Stock Large-Scale Monitoring System")
    print("="*80)
    
    try:
        # 1. Import necessary modules
        print("\n[1/6] Loading system modules...")
        from monitoring.tiered_monitor import TieredMonitor, TierLevel
        from data_pipeline.free_data_client import FreeDataClient
        from src.indicators.indicator_calculator import IndicatorCalculator
        from monitoring.signal_scanner import SignalScanner
        
        print("   [OK] Core modules loaded successfully")
        
        # 2. Initialize data client
        print("\n[2/6] Initializing data pipeline...")
        data_client = FreeDataClient()
        print("   [OK] Data pipeline ready (Yahoo Finance + Alpha Vantage)")
        
        # 3. Load stock list
        print("\n[3/6] Loading stock list...")
        
        # Get S&P 500 stock list as starting point
        sp500_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B', 'NVDA', 'JPM', 'JNJ',
            'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC', 'ADBE', 'CRM', 'NFLX',
            'PFE', 'CMCSA', 'KO', 'PEP', 'TMO', 'CSCO', 'ABT', 'NKE', 'CVX', 'WMT',
            'ACN', 'MRK', 'COST', 'WFC', 'VZ', 'DHR', 'TXN', 'INTC', 'T', 'MS',
            'UNP', 'BMY', 'MDT', 'LIN', 'QCOM', 'LOW', 'HON', 'PM', 'AMGN', 'IBM'
        ]  # First 50 stocks for demo
        
        # Load from file if available
        all_symbols_file = Path('data/all_symbols.txt')
        if all_symbols_file.exists():
            with open(all_symbols_file, 'r') as f:
                all_symbols = [line.strip() for line in f.readlines() if line.strip()]
                # Limit to first 100 for testing
                all_symbols = all_symbols[:100]
                print(f"   [OK] Loaded {len(all_symbols)} stocks from file")
        else:
            # Use demo list
            all_symbols = sp500_symbols
            print(f"   [OK] Using demo list with {len(all_symbols)} stocks")
        
        # 4. Initialize tiered monitoring system
        print("\n[4/6] Initializing tiered monitoring system...")
        
        # Create or check config file
        config_path = Path('monitoring/config.yaml')
        if not config_path.exists():
            # Create default config if not exists
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                'monitoring': {
                    'tiers': {
                        'S': {
                            'max_symbols': 40,
                            'update_interval': 1,
                            'indicators': ['RSI', 'MACD', 'BB'],
                            'timeframes': ['1m', '5m', '1h']
                        },
                        'A': {
                            'max_symbols': 100,
                            'update_interval': 60,
                            'indicators': ['RSI', 'MACD'],
                            'timeframes': ['5m', '1h']
                        },
                        'B': {
                            'max_symbols': 4000,
                            'update_interval': 300,
                            'indicators': ['RSI'],
                            'timeframes': ['1d']
                        }
                    }
                }
            }
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            print("   [OK] Created default config file")
        
        # Initialize monitor with config file path
        monitor = TieredMonitor(str(config_path))
        
        # Add stocks to tiers
        for i, symbol in enumerate(all_symbols):
            if i < 10:
                # First 10 stocks to S tier
                monitor._add_stock_to_tier(symbol, TierLevel.S_TIER)
            elif i < 30:
                # Next 20 stocks to A tier
                monitor._add_stock_to_tier(symbol, TierLevel.A_TIER)
            else:
                # Rest to B tier
                monitor._add_stock_to_tier(symbol, TierLevel.B_TIER)
        
        # Get monitoring status
        print(f"   [OK] Tiered monitoring system ready")
        print(f"      S-tier: 10 stocks (real-time)")
        print(f"      A-tier: 20 stocks (high-freq)")
        print(f"      B-tier: {len(all_symbols) - 30} stocks (market scan)")
        
        # 5. Initialize signal scanner
        print("\n[5/6] Initializing signal scanning system...")
        signal_scanner = SignalScanner(str(config_path))
        print("   [OK] Signal scanner ready")
        
        # 6. Start monitoring loop
        print("\n[6/6] Starting monitoring system...")
        print("\n" + "="*80)
        print("[RUNNING] System monitoring {} stocks".format(len(all_symbols)))
        print("Press Ctrl+C to stop")
        print("="*80 + "\n")
        
        # Start monitoring system
        monitor.start_monitoring()
        
        # Main loop - display status
        scan_counter = 0
        while True:
            scan_counter += 1
            
            # Display status every 10 seconds
            if scan_counter % 10 == 0:
                status = monitor.get_monitoring_status()
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Monitoring Status:")
                print(f"  Total stocks: {status.get('total_stocks', len(all_symbols))} | "
                      f"Running: {'Yes' if status.get('is_running', False) else 'No'}")
                
                # Get tier details
                tier_details = monitor.get_tier_details()
                if 'S_tier' in tier_details and tier_details['S_tier']['stocks']:
                    print(f"\n  S-tier hot stocks: {', '.join(tier_details['S_tier']['stocks'][:5])}")
            
            # Wait 1 second
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n[!] User interrupted, shutting down system...")
    except Exception as e:
        print(f"\n[ERROR] System error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop monitoring
        if 'monitor' in locals():
            monitor.stop_monitoring()
        
        print("\n" + "="*80)
        print("System stopped")
        print("="*80)


if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 7):
        print("ERROR: Python 3.7+ required")
        sys.exit(1)
    
    # Check required packages
    try:
        import pandas
        import numpy
        import yfinance
        import yaml
    except ImportError as e:
        print(f"ERROR: Missing required package: {e}")
        print("Please run: pip install pandas numpy yfinance pyyaml python-dotenv")
        sys.exit(1)
    
    # Run main program
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSystem shutdown complete")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)