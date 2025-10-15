#!/usr/bin/env python3
"""
Enhanced Data Pipeline Demo - Simple Version
Demonstrates 4000+ stock monitoring system capabilities
"""

import time
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data_pipeline.free_data_client import FreeDataClient


def demo_enhanced_system():
    """Demonstrate enhanced data pipeline functionality"""

    print("=" * 70)
    print("ENHANCED DATA PIPELINE SYSTEM DEMO")
    print("Support for 4000+ Large-Scale Stock Monitoring")
    print("=" * 70)

    # Initialize client
    print("\nInitializing data client...")
    client = FreeDataClient()
    print(f"Database location: {client.db_path}")
    print(f"Batch size: {client.batch_size}")
    print(f"Max worker threads: {client.max_workers}")

    # Demo 1: Batch quotes
    print("\n" + "=" * 50)
    print("DEMO 1: Batch Quote System")
    print("=" * 50)

    test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    print(f"Test symbols: {', '.join(test_symbols)}")

    start_time = time.time()
    quotes = client.get_batch_quotes(test_symbols, show_progress=True)
    duration = time.time() - start_time

    print("\nBatch quote results:")
    print(f"   Successfully retrieved: {len(quotes)}/{len(test_symbols)} stocks")
    print(f"   Processing time: {duration:.2f} seconds")
    print(f"   Average speed: {len(quotes)/duration:.1f} stocks/sec")

    # Show quote details
    print("\nReal-time quotes:")
    for symbol, data in list(quotes.items())[:3]:
        print(f"   {symbol}: ${data['price']:.2f} (Volume: {data['volume']:,})")

    # Demo 2: Market overview
    print("\n" + "=" * 50)
    print("DEMO 2: Market Overview")
    print("=" * 50)

    overview = client.get_market_overview()
    market_status = "OPEN" if overview.get("is_open") else "CLOSED"
    print(f"Market status: {market_status}")
    print(f"Trading session: {overview.get('session_type')}")

    if "indices" in overview:
        print("\nMajor indices:")
        for index, data in overview["indices"].items():
            if data:
                print(f"   {index}: ${data.get('price', 0):.2f}")

    # Demo 3: Cache performance
    print("\n" + "=" * 50)
    print("DEMO 3: Cache System Performance")
    print("=" * 50)

    # First request (build cache)
    print("First request (building cache)...")
    start_time = time.time()
    quotes1 = client.get_batch_quotes(test_symbols[:3], use_cache=False, show_progress=False)
    first_time = time.time() - start_time

    # Second request (use cache)
    print("Second request (using cache)...")
    start_time = time.time()
    quotes2 = client.get_batch_quotes(test_symbols[:3], use_cache=True, show_progress=False)
    cached_time = time.time() - start_time

    speedup = first_time / cached_time if cached_time > 0 else float("inf")
    print("\nCache performance:")
    print(f"   First request: {first_time:.2f} seconds")
    print(f"   Cached request: {cached_time:.2f} seconds")
    print(f"   Speedup factor: {speedup:.1f}x")

    # Demo 4: Large scale simulation
    print("\n" + "=" * 50)
    print("DEMO 4: Large Scale Processing Capability")
    print("=" * 50)

    # Simulate large symbol list
    large_symbols = test_symbols * 10  # 50 symbols
    print(f"Simulating processing of {len(large_symbols)} stocks...")

    start_time = time.time()
    large_quotes = client.get_batch_quotes(large_symbols, show_progress=True)
    large_duration = time.time() - start_time

    print("\nLarge scale processing results:")
    print(f"   Processed stocks: {len(large_symbols)}")
    print(f"   Successfully retrieved: {len(large_quotes)}")
    print(f"   Processing time: {large_duration:.2f} seconds")
    print(f"   Throughput: {len(large_quotes)/large_duration:.1f} stocks/sec")

    # System status summary
    print("\n" + "=" * 70)
    print("SYSTEM STATUS SUMMARY")
    print("=" * 70)

    print(f"Database: {client.db_path}")
    print(f"Cache duration: {client.cache_duration} seconds")
    print(f"Alpha Vantage API: {'Configured' if client.alpha_vantage_key else 'Not configured'}")
    print(f"Batch processing: {client.batch_size} stocks/batch")
    print(f"Concurrent threads: {client.max_workers} threads")
    print("Support scale: 4000+ stocks")

    print("\nDemo completed! System is ready for large-scale stock monitoring")
    print("Data has been saved to local database for reuse")


if __name__ == "__main__":
    demo_enhanced_system()
