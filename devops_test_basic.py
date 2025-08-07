#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DevOps 基本連接測試
"""

import zmq
import time
from datetime import datetime

print("\n" + "="*50)
print(" DevOps Basic Connection Test ")
print("="*50)
print(f" Time: {datetime.now().strftime('%H:%M:%S')}")

# Test 1: ZeroMQ ports
print("\n[Test 1] ZeroMQ Ports")
context = zmq.Context()

try:
    # Test PUSH
    push = context.socket(zmq.PUSH)
    push.connect("tcp://localhost:32768")
    print("  PUSH (32768): Connected")
    
    # Test PULL
    pull = context.socket(zmq.PULL)
    pull.connect("tcp://localhost:32769")
    pull.setsockopt(zmq.RCVTIMEO, 500)
    print("  PULL (32769): Connected")
    
    # Test SUB
    sub = context.socket(zmq.SUB)
    sub.connect("tcp://localhost:32770")
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    print("  SUB (32770): Connected")
    
    # Quick heartbeat
    print("\n[Test 2] Quick Heartbeat")
    push.send_string("HEARTBEAT;")
    print("  Sent: HEARTBEAT")
    
    try:
        response = pull.recv_string()
        print(f"  Received: {response[:100]}")
        print("\n[RESULT] Connection OK ✓")
    except zmq.Again:
        print("  No response (timeout)")
        print("\n[RESULT] EA may not be running ✗")
    
    # Cleanup
    push.close()
    pull.close()
    sub.close()
    
except Exception as e:
    print(f"\n[ERROR] {e}")
    print("[RESULT] Connection failed ✗")

finally:
    context.term()

print("\n" + "="*50)
print(" Test Complete ")
print("="*50)