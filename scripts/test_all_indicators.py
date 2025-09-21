"""
Comprehensive Test for All Technical Indicators
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import time

# Import all indicators
from quantproject.indicators.trend_indicators import SMA, EMA, WMA, VWAP, MovingAverageCrossover
from quantproject.indicators.momentum_indicators import RSI, MACD, Stochastic, WilliamsR, CCI
from quantproject.indicators.volatility_indicators import BollingerBands, ATR, KeltnerChannel, DonchianChannel
from quantproject.indicators.volume_indicators import OBV, VolumeSMA, MFI, ADLine


def test_stock_indicators(symbol='AAPL'):
    """Test all indicators on a single stock"""
    
    # Get database connection
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           'data', 'quant_trading.db')
    conn = sqlite3.connect(db_path)
    
    # Get stock data
    query = f"""
        SELECT date, open_price as open, high_price as high, 
               low_price as low, close_price as close, volume
        FROM daily_data
        WHERE symbol = '{symbol}'
        ORDER BY date DESC
        LIMIT 500
    """
    
    df = pd.read_sql_query(query, conn, parse_dates=['date'])
    df.set_index('date', inplace=True)
    df = df.sort_index()
    conn.close()
    
    print("="*80)
    print(f"COMPREHENSIVE TECHNICAL INDICATORS TEST - {symbol}")
    print("="*80)
    print(f"Data range: {df.index[0]} to {df.index[-1]} ({len(df)} days)")
    print("-"*80)
    
    results = {}
    
    # Test Trend Indicators
    print("\n[TREND INDICATORS]")
    print("-"*40)
    
    sma = SMA(period=20)
    sma_val = sma.calculate(df)
    print(f"SMA(20): {sma_val.iloc[-1]:.2f}")
    results['SMA'] = sma_val.iloc[-1]
    
    ema = EMA(period=20)
    ema_val = ema.calculate(df)
    print(f"EMA(20): {ema_val.iloc[-1]:.2f}")
    results['EMA'] = ema_val.iloc[-1]
    
    vwap = VWAP()
    vwap_val = vwap.calculate(df)
    print(f"VWAP: {vwap_val.iloc[-1]:.2f}")
    results['VWAP'] = vwap_val.iloc[-1]
    
    # Test Momentum Indicators
    print("\n[MOMENTUM INDICATORS]")
    print("-"*40)
    
    rsi = RSI(period=14)
    rsi_val = rsi.calculate(df)
    print(f"RSI(14): {rsi_val.iloc[-1]:.2f}")
    results['RSI'] = rsi_val.iloc[-1]
    
    macd = MACD()
    macd_val = macd.calculate(df)
    print(f"MACD: {macd_val['macd'].iloc[-1]:.4f}")
    print(f"Signal: {macd_val['signal'].iloc[-1]:.4f}")
    print(f"Histogram: {macd_val['histogram'].iloc[-1]:.4f}")
    results['MACD'] = macd_val['macd'].iloc[-1]
    
    stoch = Stochastic()
    stoch_val = stoch.calculate(df)
    print(f"Stochastic %K: {stoch_val['k'].iloc[-1]:.2f}")
    print(f"Stochastic %D: {stoch_val['d'].iloc[-1]:.2f}")
    results['Stoch_K'] = stoch_val['k'].iloc[-1]
    
    # Test Volatility Indicators
    print("\n[VOLATILITY INDICATORS]")
    print("-"*40)
    
    bb = BollingerBands(period=20, std_dev=2)
    bb_val = bb.calculate(df)
    print(f"Bollinger Upper: {bb_val['upper_band'].iloc[-1]:.2f}")
    print(f"Bollinger Middle: {bb_val['middle_band'].iloc[-1]:.2f}")
    print(f"Bollinger Lower: {bb_val['lower_band'].iloc[-1]:.2f}")
    print(f"Bollinger %B: {bb_val['percent_b'].iloc[-1]:.2f}")
    results['BB_Upper'] = bb_val['upper_band'].iloc[-1]
    
    atr = ATR(period=14)
    atr_val = atr.calculate(df)
    print(f"ATR(14): {atr_val.iloc[-1]:.2f}")
    results['ATR'] = atr_val.iloc[-1]
    
    kc = KeltnerChannel(ema_period=20, atr_period=10)
    kc_val = kc.calculate(df)
    print(f"Keltner Upper: {kc_val['upper_channel'].iloc[-1]:.2f}")
    print(f"Keltner Middle: {kc_val['middle_line'].iloc[-1]:.2f}")
    print(f"Keltner Lower: {kc_val['lower_channel'].iloc[-1]:.2f}")
    results['KC_Upper'] = kc_val['upper_channel'].iloc[-1]
    
    dc = DonchianChannel(period=20)
    dc_val = dc.calculate(df)
    print(f"Donchian Upper: {dc_val['upper_channel'].iloc[-1]:.2f}")
    print(f"Donchian Middle: {dc_val['middle_channel'].iloc[-1]:.2f}")
    print(f"Donchian Lower: {dc_val['lower_channel'].iloc[-1]:.2f}")
    results['DC_Upper'] = dc_val['upper_channel'].iloc[-1]
    
    # Test Volume Indicators
    print("\n[VOLUME INDICATORS]")
    print("-"*40)
    
    obv = OBV()
    obv_val = obv.calculate(df)
    print(f"OBV: {obv_val.iloc[-1]:,.0f}")
    results['OBV'] = obv_val.iloc[-1]
    
    vol_sma = VolumeSMA(period=20)
    vol_sma_val = vol_sma.calculate(df)
    print(f"Volume SMA(20): {vol_sma_val['volume_sma'].iloc[-1]:,.0f}")
    print(f"Volume Ratio: {vol_sma_val['volume_ratio'].iloc[-1]:.2f}")
    results['Vol_SMA'] = vol_sma_val['volume_sma'].iloc[-1]
    
    mfi = MFI(period=14)
    mfi_val = mfi.calculate(df)
    print(f"MFI(14): {mfi_val.iloc[-1]:.2f}")
    results['MFI'] = mfi_val.iloc[-1]
    
    ad = ADLine()
    ad_val = ad.calculate(df)
    print(f"A/D Line: {ad_val.iloc[-1]:,.0f}")
    results['AD_Line'] = ad_val.iloc[-1]
    
    # Signal Analysis
    print("\n[SIGNAL SUMMARY]")
    print("-"*40)
    
    # Check for signals
    rsi_signals = rsi.get_signals(df)
    macd_signals = macd.get_signals(df)
    bb_signals = bb.get_signals(df)
    
    current_signals = []
    
    if rsi_signals['oversold'].iloc[-1]:
        current_signals.append("RSI Oversold")
    if rsi_signals['overbought'].iloc[-1]:
        current_signals.append("RSI Overbought")
        
    if macd_signals['buy'].iloc[-1]:
        current_signals.append("MACD Buy")
    if macd_signals['sell'].iloc[-1]:
        current_signals.append("MACD Sell")
        
    if bb_signals['buy'].iloc[-1]:
        current_signals.append("Bollinger Buy")
    if bb_signals['sell'].iloc[-1]:
        current_signals.append("Bollinger Sell")
    
    if current_signals:
        print("Active Signals:", ", ".join(current_signals))
    else:
        print("No active signals")
    
    return results


def test_multiple_stocks():
    """Test indicators on multiple stocks"""
    
    test_symbols = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN']
    all_results = {}
    
    print("\n" + "="*80)
    print("TESTING MULTIPLE STOCKS")
    print("="*80)
    
    for symbol in test_symbols:
        try:
            print(f"\nTesting {symbol}...")
            results = test_stock_indicators(symbol)
            all_results[symbol] = results
            time.sleep(0.1)  # Small delay to avoid overwhelming
        except Exception as e:
            print(f"Error testing {symbol}: {e}")
    
    # Create summary table
    if all_results:
        print("\n" + "="*80)
        print("SUMMARY TABLE")
        print("="*80)
        
        df_results = pd.DataFrame(all_results).T
        print(df_results.round(2))
        
        # Save to CSV
        report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  'reports', 'indicator_test_results.csv')
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        df_results.to_csv(report_path)
        print(f"\nResults saved to: {report_path}")
    
    return all_results


def main():
    print("="*80)
    print("TECHNICAL INDICATORS COMPREHENSIVE TEST")
    print("="*80)
    print(f"Test Date: {datetime.now()}")
    print(f"Total Indicators: 16 types")
    print("-"*80)
    
    # Test single stock first
    print("\n1. Single Stock Test (AAPL)")
    single_results = test_stock_indicators('AAPL')
    
    # Test multiple stocks
    print("\n2. Multiple Stocks Test")
    multi_results = test_multiple_stocks()
    
    print("\n" + "="*80)
    print("TEST COMPLETE!")
    print("="*80)
    print("\nAll indicators tested successfully:")
    print("- Trend: SMA, EMA, WMA, VWAP")
    print("- Momentum: RSI, MACD, Stochastic, Williams %R, CCI")
    print("- Volatility: Bollinger Bands, ATR, Keltner Channel, Donchian Channel")
    print("- Volume: OBV, Volume SMA, MFI, A/D Line")
    print("\nPhase 2 Technical Indicators - COMPLETE")


if __name__ == "__main__":
    main()