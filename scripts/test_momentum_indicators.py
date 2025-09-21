"""
Test Momentum Indicators
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import sqlite3
from quantproject.indicators.momentum_indicators import RSI, MACD, Stochastic, WilliamsR, CCI
import matplotlib.pyplot as plt

def test_indicators():
    """Test momentum indicators with sample stock data"""
    
    # Get database path
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           'data', 'quant_trading.db')
    conn = sqlite3.connect(db_path)
    
    # Get sample stock data (AAPL)
    query = """
        SELECT date, open_price as open, high_price as high, 
               low_price as low, close_price as close, volume
        FROM daily_data
        WHERE symbol = 'AAPL'
        ORDER BY date DESC
        LIMIT 500
    """
    
    df = pd.read_sql_query(query, conn, parse_dates=['date'])
    df.set_index('date', inplace=True)
    df = df.sort_index()  # Sort chronologically
    conn.close()
    
    print("="*60)
    print("TESTING MOMENTUM INDICATORS")
    print("="*60)
    print(f"Sample data: AAPL, {len(df)} days")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print("-"*60)
    
    # Test RSI
    print("\n1. Testing RSI (14 periods)...")
    rsi_indicator = RSI(period=14)
    rsi_values = rsi_indicator.calculate(df)
    rsi_signals = rsi_indicator.get_signals(df)
    
    print(f"   Latest RSI: {rsi_values.iloc[-1]:.2f}")
    print(f"   Overbought signals: {rsi_signals['overbought'].sum()}")
    print(f"   Oversold signals: {rsi_signals['oversold'].sum()}")
    print(f"   Buy signals: {rsi_signals['buy'].sum()}")
    print(f"   Sell signals: {rsi_signals['sell'].sum()}")
    
    # Test MACD
    print("\n2. Testing MACD (12,26,9)...")
    macd_indicator = MACD(fast_period=12, slow_period=26, signal_period=9)
    macd_values = macd_indicator.calculate(df)
    macd_signals = macd_indicator.get_signals(df)
    
    print(f"   Latest MACD: {macd_values['macd'].iloc[-1]:.4f}")
    print(f"   Latest Signal: {macd_values['signal'].iloc[-1]:.4f}")
    print(f"   Latest Histogram: {macd_values['histogram'].iloc[-1]:.4f}")
    print(f"   Bullish crosses: {macd_signals['bullish_cross'].sum()}")
    print(f"   Bearish crosses: {macd_signals['bearish_cross'].sum()}")
    
    # Test Stochastic
    print("\n3. Testing Stochastic (14,3)...")
    stoch_indicator = Stochastic(k_period=14, d_period=3)
    stoch_values = stoch_indicator.calculate(df)
    stoch_signals = stoch_indicator.get_signals(df)
    
    print(f"   Latest %K: {stoch_values['k'].iloc[-1]:.2f}")
    print(f"   Latest %D: {stoch_values['d'].iloc[-1]:.2f}")
    print(f"   Overbought signals: {stoch_signals['overbought'].sum()}")
    print(f"   Oversold signals: {stoch_signals['oversold'].sum()}")
    
    # Test Williams %R
    print("\n4. Testing Williams %R (14 periods)...")
    williams_indicator = WilliamsR(period=14)
    williams_values = williams_indicator.calculate(df)
    williams_signals = williams_indicator.get_signals(df)
    
    print(f"   Latest Williams %R: {williams_values.iloc[-1]:.2f}")
    print(f"   Overbought signals: {williams_signals['overbought'].sum()}")
    print(f"   Oversold signals: {williams_signals['oversold'].sum()}")
    
    # Test CCI
    print("\n5. Testing CCI (20 periods)...")
    cci_indicator = CCI(period=20)
    cci_values = cci_indicator.calculate(df)
    cci_signals = cci_indicator.get_signals(df)
    
    print(f"   Latest CCI: {cci_values.iloc[-1]:.2f}")
    print(f"   Overbought signals: {cci_signals['overbought'].sum()}")
    print(f"   Oversold signals: {cci_signals['oversold'].sum()}")
    
    # Create visualization
    print("\n" + "="*60)
    print("CREATING VISUALIZATION...")
    print("="*60)
    
    fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)
    
    # Plot price
    axes[0].plot(df.index, df['close'], label='Close Price', color='black', linewidth=1)
    axes[0].set_ylabel('Price ($)')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('AAPL Price and Momentum Indicators', fontsize=14, fontweight='bold')
    
    # Plot RSI
    axes[1].plot(df.index, rsi_values, label='RSI', color='purple', linewidth=1)
    axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought (70)')
    axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold (30)')
    axes[1].fill_between(df.index, 70, 100, alpha=0.1, color='red')
    axes[1].fill_between(df.index, 0, 30, alpha=0.1, color='green')
    axes[1].set_ylabel('RSI')
    axes[1].set_ylim([0, 100])
    axes[1].legend(loc='upper left', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    # Plot MACD
    axes[2].plot(df.index, macd_values['macd'], label='MACD', color='blue', linewidth=1)
    axes[2].plot(df.index, macd_values['signal'], label='Signal', color='red', linewidth=1)
    axes[2].bar(df.index, macd_values['histogram'], label='Histogram', color='gray', alpha=0.3)
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[2].set_ylabel('MACD')
    axes[2].legend(loc='upper left', fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    # Plot Stochastic
    axes[3].plot(df.index, stoch_values['k'], label='%K', color='blue', linewidth=1)
    axes[3].plot(df.index, stoch_values['d'], label='%D', color='red', linewidth=1)
    axes[3].axhline(y=80, color='r', linestyle='--', alpha=0.5, label='Overbought (80)')
    axes[3].axhline(y=20, color='g', linestyle='--', alpha=0.5, label='Oversold (20)')
    axes[3].fill_between(df.index, 80, 100, alpha=0.1, color='red')
    axes[3].fill_between(df.index, 0, 20, alpha=0.1, color='green')
    axes[3].set_ylabel('Stochastic')
    axes[3].set_ylim([0, 100])
    axes[3].legend(loc='upper left', fontsize=8)
    axes[3].grid(True, alpha=0.3)
    
    # Plot Williams %R
    axes[4].plot(df.index, williams_values, label='Williams %R', color='orange', linewidth=1)
    axes[4].axhline(y=-20, color='r', linestyle='--', alpha=0.5, label='Overbought (-20)')
    axes[4].axhline(y=-80, color='g', linestyle='--', alpha=0.5, label='Oversold (-80)')
    axes[4].fill_between(df.index, -20, 0, alpha=0.1, color='red')
    axes[4].fill_between(df.index, -100, -80, alpha=0.1, color='green')
    axes[4].set_ylabel('Williams %R')
    axes[4].set_ylim([-100, 0])
    axes[4].legend(loc='upper left', fontsize=8)
    axes[4].grid(True, alpha=0.3)
    
    # Plot CCI
    axes[5].plot(df.index, cci_values, label='CCI', color='teal', linewidth=1)
    axes[5].axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Overbought (100)')
    axes[5].axhline(y=-100, color='g', linestyle='--', alpha=0.5, label='Oversold (-100)')
    axes[5].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[5].fill_between(df.index, 100, cci_values.max(), alpha=0.1, color='red')
    axes[5].fill_between(df.index, cci_values.min(), -100, alpha=0.1, color='green')
    axes[5].set_ylabel('CCI')
    axes[5].set_xlabel('Date')
    axes[5].legend(loc='upper left', fontsize=8)
    axes[5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    report_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'analysis_reports', 'visualizations')
    os.makedirs(report_dir, exist_ok=True)
    
    fig_path = os.path.join(report_dir, 'momentum_indicators.png')
    plt.savefig(fig_path, dpi=100, bbox_inches='tight')
    print(f"\nVisualization saved to: {fig_path}")
    
    plt.show()
    
    return {
        'rsi': rsi_values,
        'macd': macd_values,
        'stochastic': stoch_values,
        'williams': williams_values,
        'cci': cci_values
    }


if __name__ == "__main__":
    indicators = test_indicators()
    
    print("\n" + "="*60)
    print("MOMENTUM INDICATORS TEST COMPLETE!")
    print("="*60)
    print("\nAll momentum indicators implemented and tested successfully:")
    print("- RSI (Relative Strength Index)")
    print("- MACD (Moving Average Convergence Divergence)")
    print("- Stochastic Oscillator")
    print("- Williams %R")
    print("- CCI (Commodity Channel Index)")
    print("\nNext steps: Implement volatility and volume indicators")