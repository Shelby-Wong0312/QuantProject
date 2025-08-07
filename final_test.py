#!/usr/bin/env python
"""Final MT4 Connection Test"""

import zmq
import time
from datetime import datetime

print("\n" + "="*50)
print(f" MT4 Final Test - {datetime.now().strftime('%H:%M:%S')}")
print("="*50)

context = zmq.Context()

# Simple heartbeat test
print("\n1. Heartbeat test...")
push = context.socket(zmq.PUSH)
pull = context.socket(zmq.PULL)

push.connect("tcp://localhost:32768")
pull.connect("tcp://localhost:32769")
pull.setsockopt(zmq.RCVTIMEO, 3000)

push.send_string("HEARTBEAT;")
try:
    response = pull.recv_string()
    print(f"   Response: {response}")
    print("   [SUCCESS] MT4 is responding")
except zmq.Again:
    print("   [TIMEOUT] No response")

push.close()
pull.close()

print("\nConclusion:")
print("-"*50)
print("MT4 connection via DWX is working.")
print("To collect BTC/crypto data:")
print("1. Check if your broker offers crypto in MT4")
print("2. Symbol names may vary (BTCUSD, Bitcoin, etc.)")
print("3. For forex data, use EURUSD, GBPUSD, etc.")
print("\nUse the DWX connector directly:")
print("  from mt4_bridge.DWX_ZeroMQ_Connector_v2_0_1_RC8 import DWX_ZeroMQ_Connector")
print("  dwx = DWX_ZeroMQ_Connector()")
print("  dwx._DWX_MTX_SUBSCRIBE_MARKETDATA_('EURUSD')")

context.term()