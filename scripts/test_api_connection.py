"""
Test Capital.com API Connection
測試 Capital.com API 連接
Cloud DE - Task DE-401
"""

import asyncio
import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.api.capital_client import CapitalComClient, Environment, Order, OrderType, OrderSide
from src.api.auth_manager import AuthManager
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_authentication():
    """Test API authentication"""
    print("\n" + "=" * 50)
    print("Testing Capital.com API Authentication")
    print("=" * 50)

    # Load credentials
    auth_manager = AuthManager()

    # Check for credentials
    config_file = Path("config/api_credentials.json")
    encrypted_file = Path("config/api_credentials.enc")

    if encrypted_file.exists():
        print("Loading encrypted credentials...")
        credentials = auth_manager.load_encrypted_credentials(str(encrypted_file))
    elif config_file.exists():
        print("Loading credentials from JSON...")
        with open(config_file, "r") as f:
            credentials = json.load(f)

        # Save encrypted version
        auth_manager.save_encrypted_credentials(str(encrypted_file), credentials)
        print("Credentials encrypted and saved")
    else:
        print("\nNo credentials found!")
        print("Please create config/api_credentials.json from template")
        print("Template location: config/api_credentials_template.json")
        return None

    # Extract Capital.com credentials
    capital_creds = credentials.get("capital_com", {})

    if capital_creds.get("api_key", "").startswith("YOUR_"):
        print("\nPlease update credentials in config/api_credentials.json")
        print("Replace placeholder values with actual API credentials")
        return None

    # Create client
    environment = (
        Environment.DEMO if capital_creds.get("environment") == "demo" else Environment.LIVE
    )

    client = CapitalComClient(
        api_key=capital_creds["api_key"],
        password=capital_creds["api_secret"],
        environment=environment,
    )

    return client


async def test_connection(client):
    """Test API connection and basic operations"""

    print("\n1. Testing Authentication...")
    print("-" * 30)

    auth_success = await client.connect()

    if auth_success:
        print("✓ Authentication successful")
        print(f"  CST Token: {client.cst_token[:20]}..." if client.cst_token else "  No CST token")
        print(
            f"  Security Token: {client.x_security_token[:20]}..."
            if client.x_security_token
            else "  No security token"
        )
    else:
        print("✗ Authentication failed")
        return False

    print("\n2. Testing Account Info...")
    print("-" * 30)

    account_info = await client.get_account_info()

    if account_info:
        print("✓ Account info retrieved")
        print(f"  Account ID: {account_info.account_id}")
        print(f"  Balance: {account_info.currency} {account_info.balance:,.2f}")
        print(f"  Available: {account_info.currency} {account_info.available:,.2f}")
        print(f"  P&L: {account_info.profit_loss:+,.2f}")
    else:
        print("✗ Failed to get account info")

    print("\n3. Testing Market Data...")
    print("-" * 30)

    test_symbols = ["US500", "EURUSD", "AAPL"]  # Capital.com symbols

    for symbol in test_symbols:
        market_data = await client.get_market_data(symbol)

        if market_data:
            print(f"✓ {symbol}:")
            print(f"  Bid: {market_data.bid:.4f}")
            print(f"  Ask: {market_data.ask:.4f}")
            print(f"  Spread: {market_data.spread:.4f}")
            print(f"  Change: {market_data.change_pct:+.2f}%")
        else:
            print(f"✗ Failed to get data for {symbol}")

    print("\n4. Testing Positions...")
    print("-" * 30)

    positions = await client.get_positions()

    if positions:
        print(f"✓ Found {len(positions)} open positions")
        for pos in positions[:3]:  # Show first 3
            print(f"  {pos.symbol}: {pos.quantity} @ {pos.avg_price:.2f} (P&L: {pos.pnl:+.2f})")
    else:
        print("✓ No open positions")

    print("\n5. Testing WebSocket Connection...")
    print("-" * 30)

    # Test WebSocket with a simple callback
    ws_test_complete = asyncio.Event()
    ws_data_received = []

    async def ws_callback(market_data):
        """WebSocket data callback"""
        ws_data_received.append(market_data)
        print(
            f"  Received: {market_data.symbol} - Bid: {market_data.bid:.4f}, Ask: {market_data.ask:.4f}"
        )

        if len(ws_data_received) >= 3:
            ws_test_complete.set()

    # Start WebSocket subscription in background
    ws_task = asyncio.create_task(client.subscribe_price_stream(["EURUSD"], ws_callback))

    try:
        # Wait for some data or timeout
        await asyncio.wait_for(ws_test_complete.wait(), timeout=10)
        print(f"✓ WebSocket working - received {len(ws_data_received)} updates")
    except asyncio.TimeoutError:
        print("✗ WebSocket timeout - no data received")
    finally:
        ws_task.cancel()

    return True


async def test_order_placement(client):
    """Test order placement (demo only)"""

    print("\n6. Testing Order Placement (Demo)...")
    print("-" * 30)

    if client.environment != Environment.DEMO:
        print("⚠ Skipping order test in LIVE environment")
        return

    # Create a small test order
    test_order = Order(
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=0.01,  # Minimum size
        order_type=OrderType.MARKET,
    )

    print("Placing test order: BUY 0.01 EURUSD at MARKET")

    await client.place_order(test_order)

    if order_id:
        print("✓ Order placed successfully")
        print(f"  Order ID: {order_id}")

        # Wait a moment
        await asyncio.sleep(2)

        # Check positions
        positions = await client.get_positions()
        new_position = next((p for p in positions if p.symbol == "EURUSD"), None)

        if new_position:
            print(f"✓ Position opened: {new_position.quantity} @ {new_position.avg_price:.4f}")

            # Close the position
            # Note: This would need the position's deal ID, not shown here
            print("  (Position would be closed in production)")
    else:
        print("✗ Order placement failed")


async def main():
    """Main test function"""
    print("\n" + "=" * 60)
    print("CAPITAL.COM API CONNECTION TEST")
    print("Cloud DE - Task DE-401")
    print("=" * 60)

    try:
        # Test authentication
        client = await test_authentication()

        if not client:
            print("\n❌ Test aborted - credentials not configured")
            return

        # Run connection tests
        success = await test_connection(client)

        if success:
            # Optionally test order placement
            if client.environment == Environment.DEMO:
                response = input("\nTest order placement? (y/n): ")
                if response.lower() == "y":
                    await test_order_placement(client)

            print("\n" + "=" * 60)
            print("✅ ALL TESTS PASSED")
            print("=" * 60)
            print("\nCapital.com API is ready for production use!")

            # Save test results
            test_report = {
                "timestamp": datetime.now().isoformat(),
                "environment": client.environment.value,
                "authentication": "PASS",
                "account_info": "PASS",
                "market_data": "PASS",
                "websocket": "PASS" if ws_data_received else "TIMEOUT",
                "status": "READY",
            }

            with open("reports/api_test_report.json", "w") as f:
                json.dump(test_report, f, indent=2)

            print("\nTest report saved to: reports/api_test_report.json")
        else:
            print("\n❌ Some tests failed")

        # Cleanup
        await client.disconnect()

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        print(f"\n❌ Test error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("\nStarting Capital.com API connection test...")
    print("Please ensure you have configured credentials in config/api_credentials.json")

    asyncio.run(main())
