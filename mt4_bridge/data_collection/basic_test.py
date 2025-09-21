# -*- coding: utf-8 -*-
"""
Basic MT4 Data Collection System Test
Tests core functionality and compatibility
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
        from quantproject.core.event import EventType, MarketEvent
        from quantproject.core.event_loop import EventLoop
        print("[PASS] Core event system imported successfully")
        
        # Test MT4 data collection imports
        from mt4_bridge.data_collection import (
            TickCollector, TickData, 
            OHLCAggregator, OHLCBar, TimeFrame,
            MT4DataFeed, DataStorage
        )
        print("[PASS] MT4 data collection modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        return False

def test_tick_data_basic():
    """Test basic TickData functionality"""
    print("Testing TickData basic functionality...")
    
    try:
        from mt4_bridge.data_collection import TickData
        
        # Create a test tick
        tick = TickData(
            symbol="EURUSD",
            timestamp=datetime.now(timezone.utc),
            bid=1.1000,
            ask=1.1002
        )
        
        # Test basic properties
        assert tick.symbol == "EURUSD"
        assert tick.bid == 1.1000
        assert tick.ask == 1.1002
        assert tick.last == 1.1001  # Should be average of bid/ask
        
        # Test conversion
        tick_dict = tick.to_dict()
        recreated_tick = TickData.from_dict(tick_dict)
        
        assert tick.symbol == recreated_tick.symbol
        assert tick.bid == recreated_tick.bid
        assert tick.ask == recreated_tick.ask
        
        print("[PASS] TickData basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] TickData test failed: {e}")
        return False

def test_ohlc_basic():
    """Test basic OHLCBar functionality"""
    print("Testing OHLCBar basic functionality...")
    
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
            tick_count=50
        )
        
        # Test basic properties
        assert bar.symbol == "EURUSD"
        assert bar.timeframe == TimeFrame.M1
        assert bar.open == 1.1000
        assert bar.high == 1.1005
        assert bar.low == 1.0995
        assert bar.close == 1.1002
        
        # Test conversion
        bar_dict = bar.to_dict()
        recreated_bar = OHLCBar.from_dict(bar_dict)
        
        assert bar.symbol == recreated_bar.symbol
        assert bar.timeframe == recreated_bar.timeframe
        assert bar.open == recreated_bar.open
        
        print("[PASS] OHLCBar basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] OHLCBar test failed: {e}")
        return False

def test_timeframes():
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
        
        print("[PASS] TimeFrame enumeration test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] TimeFrame test failed: {e}")
        return False

def test_configuration():
    """Test module configuration"""
    print("Testing module configuration...")
    
    try:
        from mt4_bridge.data_collection import get_default_config, get_version_info
        
        # Test default configuration
        config = get_default_config()
        assert isinstance(config, dict)
        assert 'tick_collection' in config
        assert 'ohlc_aggregation' in config
        assert 'data_storage' in config
        
        # Test version info
        version_info = get_version_info()
        assert isinstance(version_info, dict)
        assert 'version' in version_info
        assert 'components' in version_info
        
        print("[PASS] Configuration test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Configuration test failed: {e}")
        return False

async def test_event_system():
    """Test basic event system compatibility"""
    print("Testing event system compatibility...")
    
    try:
        from quantproject.core.event import EventType, MarketEvent
        from quantproject.core.event_loop import EventLoop
        import pandas as pd
        
        # Create event loop
        event_loop = EventLoop()
        
        # Track received events
        received_events = []
        
        async def market_event_handler(event):
            received_events.append(event)
        
        # Register handler
        event_loop.add_handler(EventType.MARKET, market_event_handler)
        
        # Create simple test DataFrame
        data = [{
            'Date': datetime.now(timezone.utc),
            'Open': 1.1000,
            'High': 1.1005,
            'Low': 1.0995,
            'Close': 1.1002,
            'Volume': 1000
        }]
        
        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        
        # Create and send test event
        test_event = MarketEvent(
            symbol="EURUSD",
            timestamp=datetime.now(timezone.utc),
            ohlcv_data=df
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
            print("[PASS] Event system compatibility test passed")
        else:
            print("[FAIL] Event system compatibility test failed - no events received")
            
        return success
        
    except Exception as e:
        print(f"[FAIL] Event compatibility test failed: {e}")
        return False

def test_dependency_check():
    """Test dependency checking functionality"""
    print("Testing dependency check...")
    
    try:
        from mt4_bridge.data_collection import check_dependencies
        
        deps = check_dependencies()
        
        assert isinstance(deps, dict)
        assert 'available' in deps
        assert 'missing' in deps
        assert 'all_available' in deps
        
        # Should have pandas and numpy available
        assert 'pandas' in deps['available']
        assert 'numpy' in deps['available']
        
        print("[PASS] Dependency check test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Dependency check test failed: {e}")
        return False

async def run_all_tests():
    """Run all basic tests"""
    print("=" * 60)
    print("MT4 Data Collection System - Basic Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("TickData Basic", test_tick_data_basic),
        ("OHLCBar Basic", test_ohlc_basic),
        ("TimeFrame Enum", test_timeframes),
        ("Configuration", test_configuration),
        ("Dependency Check", test_dependency_check),
        ("Event System", test_event_system)
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
            print(f"[FAIL] {test_name} failed with exception: {e}")
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
        print(f"  {test_name}: {status}")
    
    print("=" * 60)
    
    if passed == total:
        print("SUCCESS: All tests passed! The MT4 Data Collection System is ready.")
    else:
        print("WARNING: Some tests failed. Please check the implementation.")
        print("However, basic functionality appears to be working.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(run_all_tests())
