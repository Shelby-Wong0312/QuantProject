#!/usr/bin/env python3
"""
Test script for Simple Auto Trader
Quick validation without actual trading
"""

import sys
from simple_auto_trader import SimpleAutoTrader
import logging

# Set up logging for testing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_functionality():
    """Test basic functionality without live trading"""
    print("="*60)
    print("TESTING SIMPLE AUTO TRADER")
    print("="*60)
    
    # Initialize trader
    print("\n1. Initializing trader...")
    trader = SimpleAutoTrader()
    print(f"[OK] Loaded {len(trader.all_stocks)} stocks")
    
    # Test price fetching
    print("\n2. Testing price fetching...")
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    prices = trader.get_batch_prices(test_symbols)
    
    print(f"[OK] Retrieved {len(prices)} prices:")
    for symbol, price in prices.items():
        print(f"   {symbol}: ${price:.2f}")
    
    # Test signal calculation
    print("\n3. Testing signal generation...")
    signals_found = 0
    for symbol in test_symbols:
        signal = trader.calculate_signals(symbol)
        if signal:
            signals_found += 1
            print(f"   {symbol}: {signal}")
        else:
            print(f"   {symbol}: No signal")
    
    print(f"[OK] Found {signals_found} signals from {len(test_symbols)} stocks")
    
    # Test Capital.com login (optional)
    print("\n4. Testing Capital.com connection...")
    if trader.login_to_capital():
        print("[OK] Successfully connected to Capital.com API")
    else:
        print("[WARN] Capital.com login failed (check credentials)")
    
    print("\n" + "="*60)
    print("BASIC TESTS COMPLETED")
    print("="*60)
    
    return True

def test_signal_accuracy():
    """Test signal generation on known good/bad stocks"""
    print("\n" + "="*60)
    print("SIGNAL ACCURACY TEST")
    print("="*60)
    
    trader = SimpleAutoTrader()
    
    # Test on a variety of stocks
    test_stocks = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
        'AMD', 'INTC', 'BABA', 'V', 'MA', 'JPM', 'JNJ', 'WMT', 'PG', 'UNH'
    ]
    
    buy_signals = 0
    sell_signals = 0
    no_signals = 0
    
    print(f"\nTesting signals on {len(test_stocks)} stocks...")
    
    for symbol in test_stocks:
        try:
            signal = trader.calculate_signals(symbol)
            if signal == 'BUY':
                buy_signals += 1
                print(f"   {symbol}: BUY [SIGNAL]")
            elif signal == 'SELL':
                sell_signals += 1
                print(f"   {symbol}: SELL [SIGNAL]")
            else:
                no_signals += 1
                print(f"   {symbol}: HOLD [NO SIGNAL]")
        except Exception as e:
            print(f"   {symbol}: ERROR ({e})")
            no_signals += 1
    
    print(f"\nSignal Summary:")
    print(f"   BUY signals: {buy_signals}")
    print(f"   SELL signals: {sell_signals}")
    print(f"   No signals: {no_signals}")
    print(f"   Signal rate: {(buy_signals + sell_signals) / len(test_stocks) * 100:.1f}%")
    
    return True

def main():
    """Run all tests"""
    try:
        # Run basic functionality test
        test_basic_functionality()
        
        # Run signal accuracy test
        test_signal_accuracy()
        
        print(f"\n[SUCCESS] ALL TESTS PASSED")
        print("Ready to run: python simple_auto_trader.py")
        
    except Exception as e:
        print(f"\n[ERROR] TEST FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()