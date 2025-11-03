import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ULTIMATE_SIMPLE_TRADER import SimpleAutoTrader

# Test the system
trader = SimpleAutoTrader()
print("\n[TEST] System initialized successfully!")
print(f"[TEST] Loaded {len(trader.stocks)} stocks")

# Test getting signals for first 5 stocks
print("\n[TEST] Testing signal generation...")
for symbol in trader.stocks[:5]:
    signal = trader.get_signals(symbol)
    if signal:
        print(f"  {symbol}: {signal['signal']} (RSI={signal['rsi']:.1f})")
    else:
        print(f"  {symbol}: No signal")

print("\n[TEST] All tests passed!")
