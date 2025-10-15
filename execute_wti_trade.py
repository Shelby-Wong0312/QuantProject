"""
Execute WTI Crude Oil Trade
執行WTI原油交易
"""

import os
import sys
import json
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set API credentials
# os.environ['CAPITAL_API_KEY'] removed - use .env file
# os.environ['CAPITAL_IDENTIFIER'] removed - use .env file
# os.environ['CAPITAL_API_PASSWORD'] removed - use .env file
os.environ["CAPITAL_DEMO_MODE"] = "True"

from src.connectors.capital_com_api import CapitalComAPI


def execute_wti_trade():
    """執行WTI原油交易"""
    print("\n" + "=" * 60)
    print("WTI CRUDE OIL TRADING")
    print("=" * 60)

    # Initialize API
    print("\n[1/4] Initializing Capital.com API...")
    api = CapitalComAPI()

    # Authenticate
    print("[2/4] Authenticating...")
    if not api.authenticate():
        print("[ERROR] Failed to authenticate with Capital.com")
        return False

    print("[OK] Successfully authenticated")

    # Get account info
    account = api.get_accounts()
    if account and "accounts" in account and len(account["accounts"]) > 0:
        acc = account["accounts"][0]
        print(f"\n[ACCOUNT INFO]")
        print(f"Balance: ${acc.get('balance', {}).get('balance', 0):,.2f}")
        print(f"Available: ${acc.get('balance', {}).get('available', 0):,.2f}")
        print(f"P&L: ${acc.get('balance', {}).get('profitLoss', 0):,.2f}")

    # Search for WTI Crude Oil
    print("\n[3/4] Searching for WTI Crude Oil...")

    # Common WTI symbols on Capital.com
    possible_symbols = [
        "OIL_CRUDE",  # WTI Crude Oil
        "CRUDE_OIL",  # Alternative name
        "OIL",  # Short name
        "CL",  # Futures symbol
        "WTI",  # Direct WTI
        "USCRUDE",  # US Crude
        "OIL.CRUDE",  # With dot
        "OIL-CRUDE",  # With dash
    ]

    found_symbol = None
    market_info = None

    for symbol in possible_symbols:
        print(f"  Trying: {symbol}...")
        result = api.search_markets(symbol)

        if result and "markets" in result:
            markets = result["markets"]
            # Look for WTI or Crude Oil in results
            for market in markets:
                epic = market.get("epic", "")
                name = market.get("instrumentName", "").upper()

                # Check if this is WTI Crude Oil
                if "WTI" in name or "CRUDE" in name or "OIL" in epic.upper():
                    found_symbol = epic
                    market_info = market
                    print(f"  [FOUND] {name} - Epic: {epic}")
                    break

            if found_symbol:
                break

    if not found_symbol:
        print("[ERROR] Could not find WTI Crude Oil symbol")
        print("\n[INFO] Attempting to list all available commodities...")

        # Try to search for oil markets
        oil_search = api.search_markets("oil")
        if oil_search and "markets" in oil_search:
            print("\nAvailable oil-related markets:")
            count = 0
            for market in oil_search["markets"]:
                print(f"  - {market.get('instrumentName')}: {market.get('epic')}")
                count += 1
                if count >= 10:  # Show first 10 oil markets
                    break
        return False

    # Get current price
    print(f"\n[MARKET INFO]")
    print(f"Symbol: {found_symbol}")
    print(f"Name: {market_info.get('instrumentName', 'N/A')}")

    price_info = api.get_market_price(found_symbol)
    if price_info:
        bid = price_info.get("bid", 0)
        offer = price_info.get("offer", 0)
        print(f"Bid: ${bid:.2f}")
        print(f"Ask: ${offer:.2f}")
        print(f"Spread: ${(offer-bid):.2f}")
    else:
        # Use a default price if we can't get current price
        offer = 75.0  # Approximate WTI price
        print(f"[WARNING] Could not get current price, using estimate: ${offer:.2f}")

    # Execute trade
    print(f"\n[4/4] Placing BUY order for 1000 units of WTI Crude Oil...")
    print(f"Estimated cost: ${offer * 1000:,.2f}")

    # Place the order
    order_result = api.place_order(
        epic=found_symbol, direction="BUY", size=1000, currency_code="USD", force_open=True
    )

    if order_result:
        print("\n[SUCCESS] Order placed successfully!")
        print(f"\nOrder Details:")
        print(f"Deal Reference: {order_result.get('dealReference', 'N/A')}")
        print(f"Status: {order_result.get('dealStatus', 'N/A')}")

        if "affectedDeals" in order_result:
            for deal in order_result["affectedDeals"]:
                print(f"Deal ID: {deal.get('dealId')}")
                print(f"Status: {deal.get('status')}")

        # Get updated positions
        print("\n[POSITIONS] Checking open positions...")
        positions = api.get_open_positions()
        if positions and "positions" in positions:
            print(f"\nTotal open positions: {len(positions['positions'])}")

            # Find our WTI position
            for pos in positions["positions"]:
                if found_symbol in pos.get("market", {}).get("epic", ""):
                    print(f"\nWTI Position:")
                    print(f"  Size: {pos.get('position', {}).get('size', 0)}")
                    print(f"  Direction: {pos.get('position', {}).get('direction', 'N/A')}")
                    print(f"  Entry Price: ${pos.get('position', {}).get('level', 0):.2f}")
                    print(f"  Current P&L: ${pos.get('position', {}).get('profit', 0):.2f}")

        return True
    else:
        print("[ERROR] Failed to place order")
        print("Please check:")
        print("1. Account has sufficient funds")
        print("2. Market is open")
        print("3. Symbol is correct")
        return False


if __name__ == "__main__":
    try:
        print("\n" + "=" * 60)
        print("     WTI CRUDE OIL PURCHASE")
        print("     Demo Account Trading")
        print("=" * 60)

        # Execute trade directly without confirmation
        print("\n[INFO] Executing trade for 1000 units of WTI Crude Oil...")

        # Execute trade
        success = execute_wti_trade()

        if success:
            print("\n" + "=" * 60)
            print("     TRADE COMPLETED SUCCESSFULLY")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("     TRADE FAILED - SEE ERRORS ABOVE")
            print("=" * 60)

    except KeyboardInterrupt:
        print("\n[CANCELLED] Trade cancelled by user")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback

        traceback.print_exc()
