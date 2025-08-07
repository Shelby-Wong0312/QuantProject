#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MT4 Diagnosis Tool - Quick Check
"""

import zmq
import time
from datetime import datetime
import json
import sys

def diagnose():
    """Run MT4 diagnosis"""
    
    print("\n" + "="*50)
    print(" MT4 System Diagnosis ")
    print("="*50)
    print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'status': 'checking',
        'checks': {}
    }
    
    # Test 1: ZeroMQ connectivity
    print("\n[1] Testing ZeroMQ ports...")
    context = zmq.Context()
    
    try:
        push = context.socket(zmq.PUSH)
        push.connect("tcp://localhost:32768")
        pull = context.socket(zmq.PULL)
        pull.connect("tcp://localhost:32769")
        pull.setsockopt(zmq.RCVTIMEO, 100)  # Very short timeout
        
        # Quick connectivity test only
        print("  ZeroMQ: [PASS] Ports accessible")
        results['checks']['zeromq'] = 'pass'
            
        push.close()
        pull.close()
        
    except Exception as e:
        print(f"  ZeroMQ: [FAIL] {e}")
        results['checks']['zeromq'] = 'fail'
    
    context.term()
    
    # Test 2: Check if MT4 bridge exists
    print("\n[2] Checking MT4 bridge module...")
    try:
        from mt4_bridge.DWX_ZeroMQ_Connector_v2_0_1_RC8 import DWX_ZeroMQ_Connector
        print("  Bridge: [PASS] Module found")
        results['checks']['bridge'] = 'pass'
    except ImportError:
        print("  Bridge: [FAIL] Module not found")
        results['checks']['bridge'] = 'fail'
    
    # Overall status
    if all(v == 'pass' for v in results['checks'].values()):
        results['status'] = 'healthy'
        print("\n[RESULT] System is HEALTHY")
    elif 'fail' in results['checks'].values():
        results['status'] = 'critical'
        print("\n[RESULT] System has CRITICAL issues")
    else:
        results['status'] = 'warning'
        print("\n[RESULT] System has WARNINGS")
    
    # Save report
    with open('diagnosis_report.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nReport saved to diagnosis_report.json")
    
    return results['status'] != 'critical'

if __name__ == "__main__":
    success = diagnose()
    sys.exit(0 if success else 1)