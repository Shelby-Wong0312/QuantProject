#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for DWX ZeroMQ Connector
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mt4_bridge.DWX_ZeroMQ_Connector_v2_0_1_RC8 import DWX_ZeroMQ_Connector
import time
from datetime import datetime

def test_dwx_connection():
    """Test DWX ZeroMQ Connector"""
    
    print("\n" + "="*60)
    print(" DWX ZeroMQ Connector Test ")
    print("="*60)
    
    print(f"\nCurrent time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize the connector with default ports
    print("\nInitializing DWX Connector...")
    print("Default Ports:")
    print("  PUSH: 32768 (Python -> MT4)")
    print("  PULL: 32769 (MT4 -> Python)")
    print("  SUB:  32770 (Market Data)")
    
    try:
        # Create connector instance
        dwx = DWX_ZeroMQ_Connector(
            _ClientID='PythonTradingClient',
            _host='localhost',
            _protocol='tcp',
            _PUSH_PORT=32768,
            _PULL_PORT=32769,
            _SUB_PORT=32770,
            _verbose=True
        )
        
        print("\n[SUCCESS] Connector initialized!")
        
        # Give it a moment to connect
        time.sleep(2)
        
        # Test 1: Get account info
        print("\n1. Testing account info request...")
        dwx._DWX_MTX_GET_ACCOUNT_INFO_()
        time.sleep(2)
        
        # Check if we have account info
        if hasattr(dwx, '_AccountInfo'):
            print(f"   Account info received: {dwx._AccountInfo}")
        else:
            print("   No account info received yet")
        
        # Test 2: Subscribe to a symbol
        print("\n2. Subscribing to EURUSD...")
        dwx._DWX_MTX_SUBSCRIBE_MARKETDATA_('EURUSD')
        time.sleep(2)
        
        # Test 3: Get open orders
        print("\n3. Getting open orders...")
        dwx._DWX_MTX_GET_ALL_OPEN_TRADES_()
        time.sleep(2)
        
        # DWX stores responses in different attributes
        print("   Checking for open trades...")
        
        # Test 4: Check market data
        print("\n4. Checking market data...")
        if hasattr(dwx, '_zmq_Market_Data_DB'):
            print(f"   Market data: {dwx._zmq_Market_Data_DB}")
        else:
            print("   No market data received yet")
        
        # Keep listening for 5 seconds
        print("\n5. Listening for data (5 seconds)...")
        for i in range(5):
            time.sleep(1)
            print(f"   {i+1}/5 seconds...")
            
            # Check for new data
            if hasattr(dwx, '_zmq_Market_Data_DB') and dwx._zmq_Market_Data_DB:
                for symbol, data in dwx._zmq_Market_Data_DB.items():
                    print(f"   {symbol}: Bid={data[0]}, Ask={data[1]}")
                break
        
        print("\n[SUCCESS] Connection test completed!")
        
        # Clean up
        print("\nClosing connection...")
        dwx._DWX_MTX_CLOSE_()
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False

def check_mt4_setup():
    """Display MT4 setup requirements for DWX"""
    print("\n" + "="*60)
    print(" MT4 Setup Requirements for DWX ")
    print("="*60)
    
    print("\n1. Install DWX EA in MT4:")
    print("   - Copy DWX_ZeroMQ_Server_v2.0.1_RC8.mq4 to MT4/MQL4/Experts/")
    print("   - Compile in MetaEditor (F7)")
    
    print("\n2. Load EA on chart:")
    print("   - Drag EA to any chart (preferably EURUSD)")
    print("   - In EA settings, ensure these ports:")
    print("     * PUSH_PORT: 32768")
    print("     * PULL_PORT: 32769")
    print("     * PUB_PORT:  32770")
    
    print("\n3. Enable AutoTrading:")
    print("   - Click AutoTrading button (should be green)")
    print("   - EA should show smiley face")
    
    print("\n4. Check Expert tab in MT4:")
    print("   - Should show 'DWX Server initialized'")
    print("   - Should show port bindings")

def main():
    # Test the connection
    success = test_dwx_connection()
    
    if not success:
        print("\n[INFO] Connection failed. Showing setup requirements...")
        check_mt4_setup()
    
    print("\n" + "="*60)
    print(" Test Complete ")
    print("="*60)

if __name__ == "__main__":
    main()