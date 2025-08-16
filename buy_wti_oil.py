"""
Buy WTI Crude Oil - Direct Purchase
"""

import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set API credentials
# os.environ['CAPITAL_API_KEY'] removed - use .env file
# os.environ['CAPITAL_IDENTIFIER'] removed - use .env file
# os.environ['CAPITAL_API_PASSWORD'] removed - use .env file
os.environ['CAPITAL_DEMO_MODE'] = 'True'

from src.connectors.capital_com_api import CapitalComAPI

def buy_wti_oil():
    """Execute WTI oil purchase"""
    print("\n" + "="*60)
    print("     WTI CRUDE OIL PURCHASE - 1000 UNITS")
    print("="*60)
    
    # Initialize API
    print("\n[STEP 1] Initializing API...")
    api = CapitalComAPI()
    
    # Authenticate
    print("[STEP 2] Authenticating...")
    if not api.authenticate():
        print("[ERROR] Authentication failed")
        return False
    print("[SUCCESS] Authenticated")
    
    # Get account info
    print("\n[STEP 3] Getting account information...")
    accounts = api.get_accounts()
    if accounts:
        acc = accounts[0] if isinstance(accounts, list) else accounts
        balance = acc.get('balance', {})
        print(f"Account Balance: ${balance.get('balance', 0):,.2f}")
        print(f"Available Funds: ${balance.get('available', 0):,.2f}")
    
    # Common WTI/Oil epic codes on Capital.com
    # These are the typical formats used by Capital.com for commodities
    possible_epics = [
        'OIL_CRUDE',       # Standard WTI code
        'CC.D.CL.UNC.IP',  # WTI Continuous
        'CC.D.CL.UMP.IP',  # WTI Month+1
        'CC.D.CL.UFM.IP',  # WTI Front Month
        'OIL',             # Simple oil code
        'USCRUDE',         # US Crude
    ]
    
    print("\n[STEP 4] Attempting to place order for WTI Crude Oil...")
    print("Testing different epic codes...")
    
    order_placed = False
    successful_epic = None
    
    for epic in possible_epics:
        print(f"\nTrying epic: {epic}")
        
        # Try to place the order
        try:
            result = api.place_order(
                symbol=epic,
                direction='BUY',
                size=1000,
                order_type='MARKET'
            )
            
            if result:
                print(f"[SUCCESS] Order placed with epic: {epic}")
                print(f"Deal Reference: {result}")
                order_placed = True
                successful_epic = epic
                break
            else:
                print(f"  Failed with {epic}")
                
        except Exception as e:
            print(f"  Error with {epic}: {str(e)[:50]}")
            continue
    
    if not order_placed:
        print("\n[ALTERNATIVE] Trying with direct API call...")
        
        # Try direct API call with more specific parameters
        import requests
        import json
        
        headers = {
            'X-CAP-API-KEY': os.environ['CAPITAL_API_KEY'],
            'Content-Type': 'application/json',
            'CST': api.cst if hasattr(api, 'cst') else '',
            'X-SECURITY-TOKEN': api.security_token if hasattr(api, 'security_token') else ''
        }
        
        # Try with CFD format
        order_data = {
            "epic": "OIL_CRUDE",
            "direction": "BUY", 
            "size": 1000,
            "orderType": "MARKET",
            "currencyCode": "USD",
            "forceOpen": True,
            "guaranteedStop": False
        }
        
        try:
            response = requests.post(
                f"{api.base_url}/api/v1/positions",
                headers=headers,
                json=order_data
            )
            
            print(f"\nDirect API Response: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print("[SUCCESS] Order placed via direct API")
                print(f"Response: {json.dumps(data, indent=2)}")
                order_placed = True
            else:
                print(f"Response: {response.text[:200]}")
                
        except Exception as e:
            print(f"Direct API error: {e}")
    
    # Check positions
    if order_placed:
        print("\n[STEP 5] Verifying position...")
        time.sleep(2)  # Wait for position to register
        
        positions = api.get_positions()
        if positions:
            print(f"\nOpen positions: {len(positions)}")
            for pos in positions[:5]:  # Show first 5 positions
                # Position object has attributes, not dict
                if hasattr(pos, 'symbol'):
                    print(f"\nPosition:")
                    print(f"  Symbol: {pos.symbol if hasattr(pos, 'symbol') else 'N/A'}")
                    print(f"  Direction: {pos.direction if hasattr(pos, 'direction') else 'N/A'}")
                    print(f"  Size: {pos.size if hasattr(pos, 'size') else 0}")
                    print(f"  Entry Price: ${pos.entry_price if hasattr(pos, 'entry_price') else 0:.2f}")
                    print(f"  Current P&L: ${pos.unrealized_pnl if hasattr(pos, 'unrealized_pnl') else 0:.2f}")
                else:
                    # If it's a dict
                    print(f"\nPosition: {pos}")
        
        return True
    else:
        print("\n[FAILED] Could not place order for WTI Crude Oil")
        print("\nPossible reasons:")
        print("1. Market may be closed")
        print("2. Epic code may be different on your account")
        print("3. Insufficient funds")
        print("4. API restrictions")
        
        print("\n[SUGGESTION] Try searching for available oil markets:")
        print("Use the search_markets() function with 'oil' or 'commodity' keywords")
        
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("     CAPITAL.COM WTI CRUDE OIL PURCHASE")
    print("     Demo Account - 1000 Units")
    print("="*60)
    
    success = buy_wti_oil()
    
    if success:
        print("\n" + "="*60)
        print("     ✓ TRADE EXECUTED SUCCESSFULLY")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("     ✗ TRADE FAILED - SEE DETAILS ABOVE")
        print("="*60)