# Test Capital.com API - Simple Version
import os
import sys
import logging
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Capital.com API configuration
CAPITAL_API_KEY = os.getenv("CAPITAL_API_KEY")
CAPITAL_IDENTIFIER = os.getenv("CAPITAL_IDENTIFIER")
CAPITAL_API_PASSWORD = os.getenv("CAPITAL_API_PASSWORD")
CAPITAL_BASE_URL = "https://demo-api-capital.backend-capital.com/api/v1"

print("Capital.com API Test")
print("=" * 60)
print(f"API Key: {CAPITAL_API_KEY[:10]}..." if CAPITAL_API_KEY else "API Key: NOT FOUND")
print(f"Identifier: {CAPITAL_IDENTIFIER}" if CAPITAL_IDENTIFIER else "Identifier: NOT FOUND")
print(f"Password: {'*' * len(CAPITAL_API_PASSWORD) if CAPITAL_API_PASSWORD else 'NOT FOUND'}")
print("=" * 60)


class SimpleCapitalTest:
    def __init__(self):
        self.session = requests.Session()
        self.cst = None
        self.x_security_token = None

    def login(self):
        """Login to Capital.com API"""
        login_url = f"{CAPITAL_BASE_URL}/session"
        headers = {"X-CAP-API-KEY": CAPITAL_API_KEY, "Content-Type": "application/json"}
        payload = {"identifier": CAPITAL_IDENTIFIER, "password": CAPITAL_API_PASSWORD}

        try:
            response = self.session.post(login_url, headers=headers, json=payload, timeout=15)
            if response.status_code == 200:
                self.cst = response.headers.get("CST")
                self.x_security_token = response.headers.get("X-SECURITY-TOKEN")
                print("✅ Successfully logged in to Capital.com")
                return True
            else:
                print(f"❌ Login failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Login error: {e}")
            return False

    def get_markets(self):
        """Get available markets"""
        if not self.cst:
            print("❌ Not logged in")
            return

        url = f"{CAPITAL_BASE_URL}/markets"
        headers = {"CST": self.cst, "X-SECURITY-TOKEN": self.x_security_token}

        try:
            response = self.session.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                data = response.json()
                markets = data.get("markets", [])
                print(f"✅ Found {len(markets)} available markets")

                # Show first 10 US stocks
                us_stocks = [m for m in markets if ".US" in m.get("epic", "")][:10]
                print("\nFirst 10 US stocks:")
                for stock in us_stocks:
                    print(f"  - {stock['epic']}: {stock['instrumentName']}")
                return markets
            else:
                print(f"❌ Failed to get markets: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Error getting markets: {e}")
            return []

    def get_historical_data(self, symbol="AAPL.US", resolution="DAY", days_back=30):
        """Get historical price data"""
        if not self.cst:
            print("❌ Not logged in")
            return

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        url = f"{CAPITAL_BASE_URL}/prices/{symbol}"
        headers = {"CST": self.cst, "X-SECURITY-TOKEN": self.x_security_token}
        params = {
            "resolution": resolution,
            "from": start_date.strftime("%Y-%m-%dT00:00:00"),
            "to": end_date.strftime("%Y-%m-%dT23:59:59"),
            "max": 1000,
        }

        try:
            response = self.session.get(url, headers=headers, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                prices = data.get("prices", [])
                print(f"✅ Got {len(prices)} price records for {symbol}")

                if prices:
                    # Show last 5 prices
                    print(f"\nLast 5 prices for {symbol}:")
                    for price in prices[-5:]:
                        date = price["snapshotTime"]
                        close = price["closePrice"]["ask"]
                        print(f"  {date}: ${close}")
                return prices
            else:
                print(f"❌ Failed to get prices: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Error getting prices: {e}")
            return []

    def get_market_info(self, symbol="AAPL.US"):
        """Get current market info and price"""
        if not self.cst:
            print("❌ Not logged in")
            return

        url = f"{CAPITAL_BASE_URL}/markets/{symbol}"
        headers = {"CST": self.cst, "X-SECURITY-TOKEN": self.x_security_token}

        try:
            response = self.session.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                data = response.json()
                snapshot = data.get("snapshot", {})
                if snapshot.get("offer") and snapshot.get("bid"):
                    mid_price = (snapshot["offer"] + snapshot["bid"]) / 2
                    print(f"✅ {symbol} current price: ${mid_price:.2f}")
                    print(f"   Bid: ${snapshot['bid']}, Ask: ${snapshot['offer']}")
                    print(f"   Market status: {data.get('marketStatus', 'Unknown')}")
                return data
            else:
                print(f"❌ Failed to get market info: {response.status_code}")
                return {}
        except Exception as e:
            print(f"❌ Error getting market info: {e}")
            return {}


def main():
    """Run the tests"""
    tester = SimpleCapitalTest()

    # Test 1: Login
    print("\nTest 1: Login to Capital.com API")
    print("-" * 40)
    if not tester.login():
        print("Cannot proceed without login")
        return

    # Test 2: Get available markets
    print("\nTest 2: Get available markets")
    print("-" * 40)
    tester.get_markets()

    # Test 3: Get historical data
    print("\nTest 3: Get historical data for AAPL")
    print("-" * 40)
    tester.get_historical_data("AAPL.US", "DAY", 30)

    # Test 4: Get current prices for multiple symbols
    print("\nTest 4: Get current prices")
    print("-" * 40)
    symbols = ["AAPL.US", "MSFT.US", "GOOGL.US", "TSLA.US"]
    for symbol in symbols:
        tester.get_market_info(symbol)
        print()

    print("\nAll tests completed!")


if __name__ == "__main__":
    main()
