"""
Debug and fix Alpaca connection
"""

import os
from dotenv import load_dotenv

load_dotenv()

def test_alpaca_detailed():
    """Detailed Alpaca test with debugging"""
    print("\n" + "="*60)
    print("ALPACA MARKETS DETAILED TEST")
    print("="*60)
    
    # Check environment variables
    api_key = os.getenv('ALPACA_API_KEY_ID')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    print(f"API Key ID: {api_key[:10]}..." if api_key else "API Key ID: NOT FOUND")
    print(f"Secret Key: {secret_key[:10]}..." if secret_key else "Secret Key: NOT FOUND")
    
    if not api_key or not secret_key:
        print("\n[ERROR] Missing credentials in .env file")
        return False
    
    try:
        # Try different connection methods
        print("\n[1] Testing with alpaca-py (new library)...")
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
            
            # Trading client (no auth needed for paper trading)
            trading_client = TradingClient(api_key, secret_key, paper=True)
            
            # Get account
            account = trading_client.get_account()
            print(f"[OK] Connected with alpaca-py")
            print(f"     Account ID: {account.id}")
            print(f"     Status: {account.status}")
            print(f"     Cash: ${float(account.cash):,.2f}")
            
            # Data client (no auth needed for free data)
            data_client = StockHistoricalDataClient(api_key, secret_key)
            print(f"[OK] Data client ready")
            
            return True
            
        except ImportError:
            print("[INFO] alpaca-py not installed")
            print("       Run: pip install alpaca-py")
        except Exception as e:
            print(f"[ERROR] alpaca-py failed: {e}")
    
        print("\n[2] Testing with alpaca-trade-api (legacy)...")
        try:
            from alpaca_trade_api import REST
            
            # Try paper trading endpoint
            api = REST(
                key_id=api_key,
                secret_key=secret_key,
                base_url='https://paper-api.alpaca.markets',
                api_version='v2'
            )
            
            account = api.get_account()
            print(f"[OK] Connected with alpaca-trade-api")
            print(f"     Status: {account.status}")
            print(f"     Buying Power: ${float(account.buying_power):,.2f}")
            
            return True
            
        except ImportError:
            print("[INFO] alpaca-trade-api not installed")
        except Exception as e:
            print(f"[ERROR] alpaca-trade-api failed: {e}")
            
            # Try to understand the error
            if "forbidden" in str(e).lower():
                print("\n[DIAGNOSIS] 'Forbidden' error usually means:")
                print("  1. API keys are for live trading (not paper)")
                print("  2. Account not approved for API access")
                print("  3. Wrong endpoint URL")
                print("\n[SOLUTION] Please check:")
                print("  1. Go to https://app.alpaca.markets/")
                print("  2. Make sure you're in PAPER trading mode")
                print("  3. Generate new API keys for PAPER trading")
                print("  4. Update .env file with paper trading keys")
    
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
    
    return False

def test_yahoo_detailed():
    """Test Yahoo Finance with alternative methods"""
    print("\n" + "="*60)
    print("YAHOO FINANCE DETAILED TEST")
    print("="*60)
    
    try:
        import yfinance as yf
        
        # Method 1: Direct download
        print("[1] Testing direct download...")
        try:
            data = yf.download("AAPL", period="1d", progress=False)
            if not data.empty:
                print(f"[OK] Direct download working")
                print(f"     AAPL Close: ${data['Close'].iloc[-1]:.2f}")
                return True
        except Exception as e:
            print(f"[ERROR] Direct download failed: {e}")
        
        # Method 2: Ticker with different approach
        print("\n[2] Testing Ticker method...")
        try:
            ticker = yf.Ticker("AAPL")
            hist = ticker.history(period="1d")
            if not hist.empty:
                print(f"[OK] Ticker history working")
                print(f"     AAPL Close: ${hist['Close'].iloc[-1]:.2f}")
                return True
        except Exception as e:
            print(f"[ERROR] Ticker failed: {e}")
            
        print("\n[INFO] Yahoo Finance may have temporary issues")
        print("       This is free data, occasional failures are normal")
        
    except ImportError:
        print("[ERROR] yfinance not installed")
        print("        Run: pip install yfinance --upgrade")
    
    return False

def main():
    print("\n" + "="*80)
    print("DEBUGGING DATA SOURCE CONNECTIONS")
    print("="*80)
    
    # Test Alpaca
    alpaca_ok = test_alpaca_detailed()
    
    # Test Yahoo
    yahoo_ok = test_yahoo_detailed()
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if not alpaca_ok:
        print("\nFor Alpaca:")
        print("1. Visit https://app.alpaca.markets/")
        print("2. Switch to PAPER trading mode")
        print("3. Generate new API keys for paper trading")
        print("4. Update .env file with new keys")
        print("\nAlternatively, install alpaca-py:")
        print("pip install alpaca-py")
    
    if not yahoo_ok:
        print("\nFor Yahoo Finance:")
        print("1. Update yfinance: pip install --upgrade yfinance")
        print("2. Try again later (temporary issues)")
        print("3. Use Alpha Vantage as backup")
    
    if alpaca_ok and yahoo_ok:
        print("\n[SUCCESS] All issues resolved!")
    else:
        print("\n[INFO] You can still proceed with:")
        print("  - Alpha Vantage (working)")
        print("  - Capital.com (working)")

if __name__ == "__main__":
    main()