#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple MT4 Connection Test
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import zmq
import json
from datetime import datetime

def test_simple_connection():
    """Test basic ZeroMQ connection"""
    
    print("MT4 ZeroMQ Connection Test")
    print("="*50)
    
    # 創建context
    context = zmq.Context()
    
    # 測試不同的端口配置
    ports = {
        "REQ->MT4": 5555,   # Python請求到MT4
        "SUB<-MT4": 5557    # Python訂閱MT4數據
    }
    
    print("\nPort Configuration:")
    for name, port in ports.items():
        print(f"  {name}: localhost:{port}")
    
    # 測試REQ-REP連接
    print("\n1. Testing REQ-REP connection (port 5555)...")
    req_socket = context.socket(zmq.REQ)
    req_socket.setsockopt(zmq.RCVTIMEO, 3000)  # 3秒超時
    req_socket.setsockopt(zmq.SNDTIMEO, 3000)
    
    try:
        # 連接到MT4的REP端口
        req_socket.connect("tcp://localhost:5555")
        print("   [OK] Socket created successfully")
        
        # 發送測試消息
        test_msg = {
            "type": "heartbeat",
            "command": "HEARTBEAT",
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"   -> Sending: {test_msg}")
        req_socket.send_json(test_msg)
        
        # 等待回應
        print("   ... Waiting for MT4 response...")
        response = req_socket.recv_json()
        print(f"   <- Received: {response}")
        
        if response.get("status") == "ok":
            print("   [SUCCESS] REQ-REP connection successful!")
            return True
        else:
            print(f"   [ERROR] Abnormal response: {response}")
            return False
            
    except zmq.Again:
        print("   [TIMEOUT] No response from MT4")
        print("\nPossible reasons:")
        print("   1. PythonBridge EA is not running in MT4")
        print("   2. EA is not correctly bound to port 5555")
        print("   3. Firewall is blocking the connection")
        return False
        
    except Exception as e:
        print(f"   [ERROR]: {e}")
        return False
        
    finally:
        req_socket.close()
        context.term()

def test_sub_connection():
    """Test SUB connection"""
    print("\n2. Testing PUB-SUB connection (port 5557)...")
    
    context = zmq.Context()
    sub_socket = context.socket(zmq.SUB)
    sub_socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5秒超時
    
    try:
        # 連接到MT4的PUB端口
        sub_socket.connect("tcp://localhost:5557")
        sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # 訂閱所有消息
        print("   [OK] SUB Socket created successfully")
        
        print("   ... Waiting for data stream (5 seconds)...")
        
        # 嘗試接收數據
        try:
            data = sub_socket.recv_json()
            print(f"   <- Received data: {data}")
            print("   [SUCCESS] PUB-SUB connection successful!")
            return True
        except zmq.Again:
            print("   [WARNING] No data stream received")
            print("   Possible reason: Market closed or EA not sending data")
            return False
            
    except Exception as e:
        print(f"   [ERROR]: {e}")
        return False
        
    finally:
        sub_socket.close()
        context.term()

def check_mt4_requirements():
    """Check MT4 requirements"""
    print("\nMT4 Requirements Checklist:")
    print("="*50)
    print("Please confirm the following:")
    print("[ ] 1. MT4 is running and logged into Demo account")
    print("[ ] 2. PythonBridge EA is loaded on a chart")
    print("[ ] 3. EA shows smiley face in top-right corner")
    print("[ ] 4. AutoTrading button is green (enabled)")
    print("[ ] 5. In EA parameters, ports are set to:")
    print("     - InpRepPort = 5555")
    print("     - InpPubPort = 5557")
    print("[ ] 6. Windows Firewall allows MT4 and Python communication")
    
def main():
    print("\n" + "="*60)
    print(" MT4-Python ZeroMQ Simple Connection Test ")
    print("="*60)
    
    # 顯示當前時間
    now = datetime.now()
    print(f"\nCurrent time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if now.weekday() >= 5:
        print("[WARNING] It's weekend, forex market is closed")
    
    # 測試連接
    req_success = test_simple_connection()
    
    if req_success:
        sub_success = test_sub_connection()
        
        if sub_success:
            print("\n" + "="*60)
            print(" [SUCCESS] All tests passed! MT4 connection is working ")
            print("="*60)
        else:
            print("\n" + "="*60)
            print(" [WARNING] REQ-REP OK, but no data stream received ")
            print("="*60)
    else:
        print("\n" + "="*60)
        print(" [ERROR] Cannot connect to MT4 ")
        print("="*60)
        check_mt4_requirements()
        
        print("\nDebugging suggestions:")
        print("1. Check Expert tab logs in MT4")
        print("2. Confirm port information shown by EA")
        print("3. Try reloading the EA")
        print("4. Check Windows Firewall settings")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()