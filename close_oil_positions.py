"""
Close Oil Positions - Sell WTI Oil
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

def close_oil_positions():
    """Close oil positions by placing SELL orders"""
    print("\n" + "="*60)
    print("     CLOSE OIL POSITIONS")
    print("="*60)
    
    # Initialize API
    print("\n[1] Initializing API...")
    api = CapitalComAPI()
    
    # Authenticate
    print("[2] Authenticating...")
    if not api.authenticate():
        print("[ERROR] Authentication failed")
        return False
    print("[OK] Authenticated")
    
    # Get account info before
    print("\n[3] Account Status BEFORE:")
    accounts = api.get_accounts()
    if accounts:
        acc = accounts[0] if isinstance(accounts, list) else accounts
        balance = acc.get('balance', {})
        print(f"  Balance: ${balance.get('balance', 0):,.2f}")
        print(f"  Available: ${balance.get('available', 0):,.2f}")
        print(f"  P&L: ${balance.get('profitLoss', 0):,.2f}")
    
    # Since we bought 2x 1000 units of OIL_CRUDE, we need to sell them
    print("\n[4] Placing SELL orders to close positions...")
    
    # We know we have 2 open BUY positions of 1000 units each
    # To close them, we need to SELL the same amounts
    
    orders_placed = []
    
    # First SELL order
    print("\nSelling 1000 units of OIL_CRUDE (Position 1)...")
    result1 = api.place_order(
        symbol='OIL_CRUDE',
        direction='SELL',
        size=1000,
        order_type='MARKET'
    )
    
    if result1:
        print(f"[SUCCESS] SELL Order 1 placed: {result1}")
        orders_placed.append(result1)
    else:
        print("[FAILED] Could not place SELL Order 1")
    
    # Wait a bit between orders (API rate limit)
    time.sleep(0.5)
    
    # Second SELL order
    print("\nSelling 1000 units of OIL_CRUDE (Position 2)...")
    result2 = api.place_order(
        symbol='OIL_CRUDE',
        direction='SELL',
        size=1000,
        order_type='MARKET'
    )
    
    if result2:
        print(f"[SUCCESS] SELL Order 2 placed: {result2}")
        orders_placed.append(result2)
    else:
        print("[FAILED] Could not place SELL Order 2")
    
    # Wait for orders to execute
    if orders_placed:
        print("\n[5] Waiting for orders to execute...")
        time.sleep(3)
        
        # Get updated account info
        print("\n[6] Account Status AFTER:")
        accounts = api.get_accounts()
        if accounts:
            acc = accounts[0] if isinstance(accounts, list) else accounts
            balance = acc.get('balance', {})
            print(f"  Balance: ${balance.get('balance', 0):,.2f}")
            print(f"  Available: ${balance.get('available', 0):,.2f}")
            print(f"  P&L: ${balance.get('profitLoss', 0):,.2f}")
        
        # Check positions
        print("\n[7] Checking remaining positions...")
        positions = api.get_positions()
        if positions:
            print(f"  Remaining positions: {len(positions)}")
        else:
            print("  All positions closed!")
        
        return True
    else:
        print("\n[FAILED] No orders were placed")
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("     CAPITAL.COM - CLOSE OIL POSITIONS")
    print("="*60)
    
    success = close_oil_positions()
    
    if success:
        print("\n" + "="*60)
        print("     SUCCESS - OIL POSITIONS CLOSED")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("     FAILED TO CLOSE POSITIONS")
        print("="*60)