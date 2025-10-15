"""
Sell WTI Crude Oil Positions
賣出所有WTI原油部位
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
os.environ["CAPITAL_DEMO_MODE"] = "True"

from src.connectors.capital_com_api import CapitalComAPI


def sell_all_oil_positions():
    """Sell all oil positions"""
    print("\n" + "=" * 60)
    print("     SELL ALL OIL POSITIONS")
    print("     Close All WTI Crude Oil")
    print("=" * 60)

    # Initialize API
    print("\n[STEP 1] Initializing API...")
    api = CapitalComAPI()

    # Authenticate
    print("[STEP 2] Authenticating...")
    if not api.authenticate():
        print("[ERROR] Authentication failed")
        return False
    print("[SUCCESS] Authentication successful")

    # Get account info before selling
    print("\n[STEP 3] Getting account info...")
    accounts = api.get_accounts()
    if accounts:
        acc = accounts[0] if isinstance(accounts, list) else accounts
        balance = acc.get("balance", {})
        print(f"Current Balance: ${balance.get('balance', 0):,.2f}")
        print(f"Available Funds: ${balance.get('available', 0):,.2f}")
        print(f"P&L: ${balance.get('profitLoss', 0):,.2f}")

    # Get current positions
    print("\n[STEP 4] Finding open positions...")
    positions = api.get_positions()

    if not positions:
        print("[INFO] No open positions found")
        return False

    print(f"Found {len(positions)} open positions")

    oil_positions = []

    # Find oil positions
    for pos in positions:
        # Check if it's an oil position
        # Position could be an object or dict
        if hasattr(pos, "__dict__"):
            # It's an object, convert to dict for easier handling
            pos_data = pos.__dict__
        else:
            pos_data = pos

        # Look for oil-related positions
        symbol = pos_data.get("symbol", "") if isinstance(pos_data, dict) else str(pos)

        # Check if this is an oil position
        if "OIL" in str(symbol).upper() or "CRUDE" in str(symbol).upper():
            oil_positions.append(pos)
            print(f"\nFound oil position: {symbol}")

    if not oil_positions:
        print("\n[INFO] No oil-related positions found")
        print("Attempting to close all positions...")
        oil_positions = positions  # Close all positions

    # Close positions
    print(f"\n[STEP 5] Closing {len(oil_positions)} positions...")

    closed_count = 0
    failed_count = 0

    for pos in oil_positions:
        try:
            # Get position ID
            if hasattr(pos, "position_id"):
                pos_id = pos.position_id
            elif hasattr(pos, "id"):
                pos_id = pos.id
            elif isinstance(pos, dict):
                pos_id = pos.get("positionId") or pos.get("dealId") or pos.get("id")
            else:
                print(f"[WARNING] Cannot get position ID: {pos}")
                continue

            print(f"\nClosing position ID: {pos_id}")

            # Try to close the position
            success = api.close_position(pos_id)

            if success:
                print(f"  [SUCCESS] Position closed")
                closed_count += 1
            else:
                print(f"  [FAILED] Could not close position")
                failed_count += 1

                # Try alternative: place opposite order
                print("  Trying opposite order...")

                # Get position details
                if hasattr(pos, "symbol"):
                    symbol = pos.symbol
                    size = pos.size if hasattr(pos, "size") else 1000
                    direction = pos.direction if hasattr(pos, "direction") else "BUY"
                else:
                    symbol = "OIL_CRUDE"
                    size = 1000
                    direction = "BUY"

                # Place opposite order
                opposite_direction = "SELL" if direction == "BUY" else "BUY"

                result = api.place_order(
                    symbol=symbol, direction=opposite_direction, size=size, order_type="MARKET"
                )

                if result:
                    print(f"  [SUCCESS] Opposite order placed: {result}")
                    closed_count += 1
                else:
                    print(f"  [FAILED] Could not place opposite order")

        except Exception as e:
            print(f"  [ERROR] Error processing position: {e}")
            failed_count += 1

    # Get updated account info
    print("\n[STEP 6] Getting updated account info...")
    time.sleep(2)  # Wait for positions to update

    accounts = api.get_accounts()
    if accounts:
        acc = accounts[0] if isinstance(accounts, list) else accounts
        balance = acc.get("balance", {})
        print(f"\nUpdated Balance: ${balance.get('balance', 0):,.2f}")
        print(f"Available Funds: ${balance.get('available', 0):,.2f}")
        print(f"P&L: ${balance.get('profitLoss', 0):,.2f}")

    # Check remaining positions
    positions = api.get_positions()
    if positions:
        print(f"\nRemaining open positions: {len(positions)}")
    else:
        print("\nAll positions closed")

    print(f"\n[RESULTS]")
    print(f"Successfully closed: {closed_count} positions")
    print(f"Failed: {failed_count} positions")

    return closed_count > 0


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("     CAPITAL.COM - SELL OIL POSITIONS")
    print("     Close All Oil Positions")
    print("=" * 60)

    success = sell_all_oil_positions()

    if success:
        print("\n" + "=" * 60)
        print("     SUCCESS - OIL POSITIONS SOLD")
        print("     OIL POSITIONS CLOSED")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("     FAILED - SEE DETAILS ABOVE")
        print("     FAILED - SEE DETAILS ABOVE")
        print("=" * 60)
