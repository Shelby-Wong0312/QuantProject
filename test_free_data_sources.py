"""
測試免費數據源連接 - 方案B
Test all free data sources connectivity
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_alpaca_connection():
    """測試Alpaca Markets連接"""
    print("\n" + "="*60)
    print("📊 Testing Alpaca Markets Connection")
    print("="*60)
    
    try:
        from alpaca_trade_api import REST
        
        # Get credentials
        api_key = os.getenv('ALPACA_API_KEY_ID')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not api_key or not secret_key:
            print("❌ Missing Alpaca credentials in .env")
            return False
            
        # Initialize client (paper trading)
        base_url = 'https://paper-api.alpaca.markets'
        api = REST(api_key, secret_key, base_url, api_version='v2')
        
        # Test account access
        account = api.get_account()
        print(f"✅ Connected to Alpaca Markets")
        print(f"   Account Status: {account.status}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Cash: ${float(account.cash):,.2f}")
        
        # Test market data
        bars = api.get_bars("AAPL", "1Min", limit=5).df
        if not bars.empty:
            print(f"✅ Real-time data working")
            print(f"   Latest AAPL price: ${bars['close'].iloc[-1]:.2f}")
        
        return True
        
    except ImportError:
        print("⚠️ alpaca-trade-api not installed")
        print("   Run: pip install alpaca-trade-api")
        return False
    except Exception as e:
        print(f"❌ Alpaca connection failed: {e}")
        return False

def test_yahoo_finance():
    """測試Yahoo Finance數據"""
    print("\n" + "="*60)
    print("📈 Testing Yahoo Finance")
    print("="*60)
    
    try:
        import yfinance as yf
        
        # Test single stock
        ticker = yf.Ticker("AAPL")
        info = ticker.info
        
        print(f"✅ Yahoo Finance working")
        print(f"   Company: {info.get('longName', 'Apple Inc.')}")
        print(f"   Current Price: ${info.get('currentPrice', 0):.2f}")
        print(f"   Market Cap: ${info.get('marketCap', 0):,.0f}")
        
        # Test historical data
        hist = ticker.history(period="5d")
        if not hist.empty:
            print(f"✅ Historical data available")
            print(f"   5-day data points: {len(hist)}")
            print(f"   Latest close: ${hist['Close'].iloc[-1]:.2f}")
        
        # Test bulk download
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        data = yf.download(symbols, period="1d", interval="1m", progress=False)
        
        if not data.empty:
            print(f"✅ Bulk download working")
            print(f"   Downloaded {len(symbols)} stocks")
        
        return True
        
    except ImportError:
        print("⚠️ yfinance not installed")
        print("   Run: pip install yfinance")
        return False
    except Exception as e:
        print(f"❌ Yahoo Finance failed: {e}")
        return False

def test_alpha_vantage():
    """測試Alpha Vantage免費API"""
    print("\n" + "="*60)
    print("📉 Testing Alpha Vantage Free API")
    print("="*60)
    
    try:
        import requests
        
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            print("❌ Missing Alpha Vantage API key in .env")
            return False
        
        # Test technical indicator (RSI)
        url = f"https://www.alphavantage.co/query"
        params = {
            'function': 'RSI',
            'symbol': 'AAPL',
            'interval': 'daily',
            'time_period': 14,
            'series_type': 'close',
            'apikey': api_key
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'Technical Analysis: RSI' in data:
            print(f"✅ Alpha Vantage connected")
            rsi_data = data['Technical Analysis: RSI']
            latest_date = list(rsi_data.keys())[0]
            latest_rsi = rsi_data[latest_date]['RSI']
            print(f"   Latest RSI for AAPL: {float(latest_rsi):.2f}")
            print(f"   Date: {latest_date}")
            
            # Show API limits
            print(f"⚠️ Rate limits:")
            print(f"   - 5 calls per minute")
            print(f"   - 500 calls per day")
            
            return True
        elif 'Note' in data:
            print(f"⚠️ API rate limit reached")
            print(f"   {data['Note']}")
            return True  # API works but rate limited
        else:
            print(f"❌ Unexpected response: {data}")
            return False
            
    except Exception as e:
        print(f"❌ Alpha Vantage failed: {e}")
        return False

def test_capital_com():
    """測試Capital.com連接（執行驗證）"""
    print("\n" + "="*60)
    print("💰 Testing Capital.com Connection")
    print("="*60)
    
    try:
        # Add project root to path
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from src.connectors.capital_com_api import CapitalComAPI
        
        api = CapitalComAPI()
        if api.authenticate():
            print(f"✅ Capital.com connected")
            
            # Get account info
            accounts = api.get_accounts()
            if accounts:
                acc = accounts[0] if isinstance(accounts, list) else accounts
                balance = acc.get('balance', {})
                print(f"   Balance: ${balance.get('balance', 0):,.2f}")
                print(f"   Available: ${balance.get('available', 0):,.2f}")
            
            return True
        else:
            print(f"❌ Capital.com authentication failed")
            return False
            
    except Exception as e:
        print(f"❌ Capital.com failed: {e}")
        return False

def check_packages():
    """檢查必要套件"""
    print("\n" + "="*60)
    print("📦 Checking Required Packages")
    print("="*60)
    
    packages = {
        'alpaca-trade-api': False,
        'yfinance': False,
        'requests': False,
        'python-dotenv': False,
        'pandas': False,
        'numpy': False
    }
    
    for package in packages:
        try:
            if package == 'alpaca-trade-api':
                import alpaca_trade_api
            elif package == 'yfinance':
                import yfinance
            elif package == 'requests':
                import requests
            elif package == 'python-dotenv':
                import dotenv
            elif package == 'pandas':
                import pandas
            elif package == 'numpy':
                import numpy
            packages[package] = True
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Run: pip install {package}")
    
    return all(packages.values())

def main():
    """主測試流程"""
    print("\n" + "="*80)
    print("FREE DATA SOURCES CONNECTIVITY TEST - PLAN B")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check packages first
    packages_ok = check_packages()
    
    if not packages_ok:
        print("\n⚠️ Please install missing packages first:")
        print("pip install alpaca-trade-api yfinance python-dotenv")
        return
    
    # Test each data source
    results = {
        'Alpaca Markets': test_alpaca_connection(),
        'Yahoo Finance': test_yahoo_finance(),
        'Alpha Vantage': test_alpha_vantage(),
        'Capital.com': test_capital_com()
    }
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    for source, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {source}: {'Connected' if status else 'Failed'}")
    
    success_rate = sum(results.values()) / len(results) * 100
    print(f"\nSuccess Rate: {success_rate:.0f}%")
    
    if success_rate == 100:
        print("\n🎉 All data sources are working! Ready to start Plan B.")
        print("\n📝 Next Steps:")
        print("1. Build data pipeline clients")
        print("2. Implement unified interface")
        print("3. Start collecting data")
    elif success_rate >= 75:
        print("\n⚠️ Most sources working. Fix the failed ones before proceeding.")
    else:
        print("\n❌ Multiple failures. Please check your configuration.")
    
    # Save results
    results_file = 'data_source_test_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'plan': 'B (Zero Cost)',
            'results': results,
            'success_rate': success_rate
        }, f, indent=2)
    print(f"\n💾 Results saved to {results_file}")

if __name__ == "__main__":
    main()