# -*- coding: utf-8 -*-
"""
Simple MT4 Data Collection System Test
Tests compatibility with existing EventLoop and system components
"""

import asyncio
import sys
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))


def test_imports():
    """Test if all modules can be imported successfully"""
    print("Testing imports...")

    try:
        # Test core event system imports
        from core.event import EventType, MarketEvent
        from core.event_loop import EventLoop

        print("✅ Core event system imported successfully")

        # Test MT4 data collection imports
        from mt4_bridge.data_collection import (
            TickCollector,
            TickData,
            OHLCAggregator,
            OHLCBar,
            TimeFrame,
            MT4DataFeed,
            DataStorage,
        )

        print("✅ MT4 data collection modules imported successfully")

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_tick_data_structure():
    """Test TickData structure"""
    print("Testing TickData structure...")

    try:
        from mt4_bridge.data_collection import TickData

        # Create a test tick
        tick = TickData(
            symbol="EURUSD", timestamp=datetime.now(timezone.utc), bid=1.1000, ask=1.1002
        )

        # Test conversion methods
        tick_dict = tick.to_dict()
        recreated_tick = TickData.from_dict(tick_dict)

        assert tick.symbol == recreated_tick.symbol
        assert tick.bid == recreated_tick.bid
        assert tick.ask == recreated_tick.ask

        print("✅ TickData structure test passed")
        return True

    except Exception as e:
        print(f"❌ TickData test failed: {e}")
        return False


def test_ohlc_structure():
    """Test OHLCBar structure"""
    print("Testing OHLCBar structure...")

    try:
        from mt4_bridge.data_collection import OHLCBar, TimeFrame

        # Create a test OHLC bar
        bar = OHLCBar(
            symbol="EURUSD",
            timeframe=TimeFrame.M1,
            timestamp=datetime.now(timezone.utc),
            open=1.1000,
            high=1.1005,
            low=1.0995,
            close=1.1002,
            volume=1000,
            tick_count=50,
        )

        # Test conversion methods
        bar_dict = bar.to_dict()
        recreated_bar = OHLCBar.from_dict(bar_dict)

        assert bar.symbol == recreated_bar.symbol
        assert bar.timeframe == recreated_bar.timeframe
        assert bar.open == recreated_bar.open

        print("✅ OHLCBar structure test passed")
        return True

    except Exception as e:
        print(f"❌ OHLCBar test failed: {e}")
        return False


async def test_event_compatibility():
    """Test compatibility with existing event system"""
    print("Testing event system compatibility...")

    try:
        from core.event import EventType, MarketEvent
        from core.event_loop import EventLoop
        import pandas as pd

        # Create event loop
        event_loop = EventLoop()

        # Track received events
        received_events = []

        async def market_event_handler(event):
            received_events.append(event)

        # Register handler
        event_loop.add_handler(EventType.MARKET, market_event_handler)

        # Create test DataFrame
        dates = [datetime.now(timezone.utc) - timedelta(minutes=i) for i in range(10, 0, -1)]
        data = []

        for i, date in enumerate(dates):
            data.append(
                {
                    "Date": date,
                    "Open": 1.1000 + i * 0.0001,
                    "High": 1.1005 + i * 0.0001,
                    "Low": 1.0995 + i * 0.0001,
                    "Close": 1.1002 + i * 0.0001,
                    "Volume": 1000 + i * 10,
                }
            )

        df = pd.DataFrame(data)
        df.set_index("Date", inplace=True)

        # Create test event
        test_event = MarketEvent(
            symbol="EURUSD", timestamp=datetime.now(timezone.utc), ohlcv_data=df
        )

        # Start event loop
        event_task = asyncio.create_task(event_loop.run())

        # Send test event
        await event_loop.put_event(test_event)

        # Wait for processing
        await asyncio.sleep(0.1)

        # Stop event loop
        event_loop.stop()
        event_task.cancel()

        try:
            await event_task
        except asyncio.CancelledError:
            pass

        # Check results
        success = len(received_events) > 0

        if success:
            print("✅ Event system compatibility test passed")
        else:
            print("❌ Event system compatibility test failed")

        return success

    except Exception as e:
        print(f"❌ Event compatibility test failed: {e}")
        return False


def test_configuration():
    """Test module configuration and defaults"""
    print("Testing module configuration...")

    try:
        from mt4_bridge.data_collection import get_default_config, get_version_info

        # Test default configuration
        config = get_default_config()
        assert isinstance(config, dict)
        assert "tick_collection" in config
        assert "ohlc_aggregation" in config
        assert "data_storage" in config

        # Test version info
        version_info = get_version_info()
        assert isinstance(version_info, dict)
        assert "version" in version_info
        assert "components" in version_info

        print("✅ Configuration test passed")
        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_timeframe_enum():
    """Test TimeFrame enumeration"""
    print("Testing TimeFrame enumeration...")

    try:
        from mt4_bridge.data_collection import TimeFrame

        # Test basic timeframes
        assert TimeFrame.M1.minutes == 1
        assert TimeFrame.M5.minutes == 5
        assert TimeFrame.M15.minutes == 15
        assert TimeFrame.H1.minutes == 60

        # Test string values
        assert TimeFrame.M1.value == "M1"
        assert TimeFrame.H1.value == "H1"

        print("✅ TimeFrame enumeration test passed")
        return True

    except Exception as e:
        print(f"❌ TimeFrame test failed: {e}")
        return False


async def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("MT4 Data Collection System - Simple Test Suite")
    print("=" * 60)

    tests = [
        ("Import Test", test_imports),
        ("TickData Structure", test_tick_data_structure),
        ("OHLCBar Structure", test_ohlc_structure),
        ("TimeFrame Enum", test_timeframe_enum),
        ("Configuration", test_configuration),
        ("Event Compatibility", test_event_compatibility),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)

        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")

    print("\nDetailed Results:")
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        emoji = "✅" if result else "❌"
        print(f"  {emoji} {test_name}: {status}")

    print("=" * 60)

    if passed == total:
        print("🎉 All tests passed! The MT4 Data Collection System is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")

    return passed == total


if __name__ == "__main__":
    asyncio.run(run_all_tests())
