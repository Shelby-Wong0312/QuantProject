"""
Test Alpaca Paper Trading Connection
測試Alpaca Paper Trading連接
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_alpaca_direct_api():
    """Test Alpaca using direct REST API calls"""
    print("\n" + "="*60)
    print("ALPACA PAPER TRADING CONNECTION TEST")
    print("="*60)
    
    # Get credentials
    api_key = os.getenv('ALPACA_API_KEY_ID')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    base_url = os.getenv('ALPACA_END_POINT', 'https://paper-api.alpaca.markets/v2')
    
    print(f"API Key: {api_key[:10]}..." if api_key else "No API Key")
    print(f"Secret: {secret_key[:10]}..." if secret_key else "No Secret")
    print(f"Endpoint: {base_url}")
    
    # Test 1: Direct API call to account endpoint
    print("\n[1] Testing account endpoint...")
    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': secret_key
    }
    
    try:
        response = requests.get(f"{base_url}/account", headers=headers)
        print(f"Response Status: {response.status_code}")
        
        if response.status_code == 200:
            account = response.json()
            print("[SUCCESS] Connected to Alpaca Paper Trading!")
            print(f"Account Number: {account.get('account_number', 'N/A')}")
            print(f"Status: {account.get('status', 'N/A')}")
            print(f"Cash: ${float(account.get('cash', 0)):,.2f}")
            print(f"Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
            return True
        else:
            print(f"[ERROR] {response.status_code}: {response.text}")
            
            if response.status_code == 403:
                print("\n[DIAGNOSIS] 403 Forbidden - Possible causes:")
                print("1. API keys might be for Live trading, not Paper")
                print("2. Keys might be expired or inactive")
                print("3. Account might not be approved yet")
                
    except Exception as e:
        print(f"[ERROR] Request failed: {e}")
    
    # Test 2: Try data API (free tier)
    print("\n[2] Testing free market data...")
    data_url = "https://data.alpaca.markets/v2"
    
    try:
        # Latest trade endpoint (requires authentication)
        response = requests.get(
            f"{data_url}/stocks/AAPL/trades/latest",
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            print("[SUCCESS] Market data access working!")
            if 'trade' in data:
                trade = data['trade']
                print(f"AAPL Latest Trade: ${trade.get('p', 0):.2f}")
        else:
            print(f"[INFO] Market data response: {response.status_code}")
            
    except Exception as e:
        print(f"[INFO] Market data test: {e}")
    
    # Test 3: Alternative connection method
    print("\n[3] Testing with alpaca-trade-api library...")
    try:
        from alpaca_trade_api import REST
        
        # Use the endpoint from .env
        api = REST(
            api_key,
            secret_key,
            base_url=base_url.replace('/v2', ''),  # Remove /v2 for the library
            api_version='v2'
        )
        
        account = api.get_account()
        print(f"[SUCCESS] Library connection working!")
        print(f"Account Status: {account.status}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Library connection: {e}")
    
    return False

def test_alternative_data():
    """Test alternative free data sources"""
    print("\n" + "="*60)
    print("TESTING ALTERNATIVE DATA SOURCES")
    print("="*60)
    
    # Test 1: Twelve Data (free tier)
    print("\n[1] Twelve Data (free alternative)...")
    try:
        response = requests.get(
            "https://api.twelvedata.com/price",
            params={
                "symbol": "AAPL",
                "apikey": "demo"  # Demo key for testing
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print("[OK] Twelve Data available (free tier)")
            print(f"     AAPL: ${float(data.get('price', 0)):.2f}")
            print("     Sign up at: https://twelvedata.com/")
    except:
        pass
    
    # Test 2: Finnhub (free tier)
    print("\n[2] Finnhub (free alternative)...")
    print("     Free tier: 60 calls/minute")
    print("     Sign up at: https://finnhub.io/")
    
    # Test 3: IEX Cloud (free tier)
    print("\n[3] IEX Cloud (free alternative)...")
    print("     Free tier: 50,000 messages/month")
    print("     Sign up at: https://iexcloud.io/")

def main():
    print("\n" + "="*80)
    print("ALPACA PAPER TRADING DIAGNOSTIC")
    print("="*80)
    
    # Test Alpaca
    alpaca_ok = test_alpaca_direct_api()
    
    if not alpaca_ok:
        print("\n" + "="*60)
        print("TROUBLESHOOTING STEPS")
        print("="*60)
        print("\n1. Verify your Alpaca account:")
        print("   - Go to https://app.alpaca.markets/")
        print("   - Make sure you're viewing Paper Trading (not Live)")
        print("   - Check if your account is active")
        
        print("\n2. Regenerate Paper Trading keys:")
        print("   - Click on 'API Keys' in Paper Trading mode")
        print("   - Delete old keys and generate new ones")
        print("   - Make sure to copy both Key ID and Secret Key")
        
        print("\n3. Check account status:")
        print("   - Your account might be pending approval")
        print("   - Paper trading should work immediately after signup")
        
        # Show alternatives
        test_alternative_data()
        
        print("\n" + "="*60)
        print("RECOMMENDED ACTION")
        print("="*60)
        print("\nWhile fixing Alpaca, you can use:")
        print("1. Yahoo Finance for historical data (working)")
        print("2. Alpha Vantage for indicators (working)")
        print("3. Capital.com for execution (working)")
        print("\nOr sign up for free alternatives above")
    else:
        print("\n[SUCCESS] Alpaca Paper Trading is working!")
        print("You can proceed with Plan B implementation")

if __name__ == "__main__":
    main()