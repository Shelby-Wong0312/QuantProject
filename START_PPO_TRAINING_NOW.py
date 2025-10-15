#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Immediate PPO Training with Available Data
Using only stocks that can be downloaded successfully
"""

import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Use major stocks that are known to work
MAJOR_STOCKS = [
    # Tech Giants
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    "AMD",
    "INTC",
    "NFLX",
    "PYPL",
    "CSCO",
    "AVGO",
    "QCOM",
    "TXN",
    "ORCL",
    "ADBE",
    "CRM",
    "IBM",
    "NOW",
    "UBER",
    "SNAP",
    "PINS",
    "SHOP",
    "ABNB",
    "COIN",
    "HOOD",
    "PLTR",
    "SOFI",
    "RIVN",
    "LCID",
    "ROKU",
    "ZM",
    "DOCU",
    "OKTA",
    "TWLO",
    "DDOG",
    "SNOW",
    "NET",
    "CRWD",
    "PANW",
    "ZS",
    "SQ",
    "RBLX",
    "U",
    "DASH",
    "AFRM",
    "UPST",
    "CPNG",
    "NU",
    # Banks & Finance
    "JPM",
    "BAC",
    "WFC",
    "GS",
    "MS",
    "C",
    "USB",
    "PNC",
    "SCHW",
    "BLK",
    "V",
    "MA",
    "AXP",
    "COF",
    "ICE",
    "CME",
    "SPGI",
    "BX",
    "KKR",
    "APO",
    # Healthcare
    "JNJ",
    "UNH",
    "PFE",
    "ABBV",
    "MRK",
    "CVS",
    "TMO",
    "ABT",
    "DHR",
    "LLY",
    "BMY",
    "AMGN",
    "GILD",
    "MDT",
    "ISRG",
    "SYK",
    "BSX",
    "EW",
    "REGN",
    "VRTX",
    "MRNA",
    "BIIB",
    "ILMN",
    "DXCM",
    "ALGN",
    "IDXX",
    "MTD",
    "ZBH",
    "VEEV",
    "RMD",
    # Consumer
    "WMT",
    "HD",
    "PG",
    "KO",
    "PEP",
    "COST",
    "NKE",
    "MCD",
    "SBUX",
    "TGT",
    "LOW",
    "TJX",
    "ROST",
    "DG",
    "DLTR",
    "BBY",
    "AZO",
    "ORLY",
    "YUM",
    "CMG",
    "DPZ",
    "QSR",
    "WEN",
    "BROS",
    "CAVA",
    "LULU",
    "DECK",
    "TPR",
    "RL",
    "GPS",
    # Energy
    "XOM",
    "CVX",
    "COP",
    "SLB",
    "EOG",
    "MPC",
    "VLO",
    "PSX",
    "OXY",
    "HAL",
    "BKR",
    "DVN",
    "FANG",
    "HES",
    "APA",
    "MRO",
    "CTRA",
    "OVV",
    "MTDR",
    "CHRD",
    # Industrials
    "BA",
    "CAT",
    "DE",
    "LMT",
    "RTX",
    "GE",
    "MMM",
    "HON",
    "UPS",
    "FDX",
    "UNP",
    "CSX",
    "NSC",
    "ETN",
    "EMR",
    "ITW",
    "PH",
    "CMI",
    "ROK",
    "AME",
    # Chinese Stocks
    "BABA",
    "JD",
    "PDD",
    "BIDU",
    "NIO",
    "XPEV",
    "LI",
    "BILI",
    "IQ",
    "TME",
    "VIPS",
    "WB",
    "TAL",
    "EDU",
    "BEKE",
    "NTES",
    "TCOM",
    "ZTO",
    "DIDI",
    "BGNE",
    # ETFs
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "VOO",
    "VTI",
    "IVV",
    "VEA",
    "VWO",
    "EEM",
    "XLF",
    "XLK",
    "XLE",
    "XLV",
    "XLI",
    "XLY",
    "XLP",
    "XLB",
    "XLU",
    "XLRE",
    "VNQ",
    "GLD",
    "SLV",
    "USO",
    "UNG",
    "TLT",
    "IEF",
    "SHY",
    "AGG",
    "BND",
    "HYG",
    "JNK",
    "EMB",
    "ARKK",
    "ARKQ",
    "ARKW",
    "ARKG",
    "ARKF",
    "ICLN",
    "TAN",
    # Other notable stocks
    "DIS",
    "T",
    "VZ",
    "CMCSA",
    "NFLX",
    "TMUS",
    "CHTR",
    "WBD",
    "PARA",
    "FOX",
    "F",
    "GM",
    "TSLA",
    "RIVN",
    "LCID",
    "NIO",
    "FSR",
    "RIDE",
    "GOEV",
    "ARVL",
    "TSM",
    "ASML",
    "AMAT",
    "LRCX",
    "KLAC",
    "MU",
    "MRVL",
    "ON",
    "SWKS",
    "QRVO",
    "BRK.B",
    "JPM",
    "BAC",
    "WFC",
    "USB",
    "GS",
    "MS",
    "C",
    "AXP",
    "V",
]


def quick_download():
    """Quick download of major stocks"""
    print("\n" + "=" * 80)
    print("QUICK PPO TRAINING - STARTING NOW")
    print("=" * 80)

    # Remove duplicates
    symbols = list(set(MAJOR_STOCKS))
    print(f"Downloading {len(symbols)} major stocks...")

    # Create data directory
    data_dir = "data/quick_training"
    os.makedirs(data_dir, exist_ok=True)

    # Download parameters
    end_date = datetime.now()
    start_date = datetime(2010, 1, 1)

    stock_data = {}
    failed_symbols = []

    # Quick batch download
    batch_size = 10
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        batch_str = ",".join(batch)

        try:
            print(
                f"\nDownloading batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}: {batch_str}"
            )
            data = yf.download(
                batch_str,
                start=start_date,
                end=end_date,
                group_by="ticker",
                progress=False,
                threads=True,
            )

            # Process each symbol
            for symbol in batch:
                try:
                    if len(batch) == 1:
                        df = data
                    else:
                        if symbol in data.columns.levels[0]:
                            df = data[symbol]
                        else:
                            df = data

                    if not df.empty and len(df) > 100:
                        stock_data[symbol] = df
                        # Save to file
                        df.to_csv(os.path.join(data_dir, f"{symbol}.csv"))
                    else:
                        failed_symbols.append(symbol)
                except:
                    failed_symbols.append(symbol)

        except Exception as e:
            print(f"Batch download failed: {e}")
            for symbol in batch:
                failed_symbols.append(symbol)

    print(f"\n✓ Successfully downloaded: {len(stock_data)} stocks")
    print(f"✗ Failed: {len(failed_symbols)} stocks")

    return stock_data


def train_immediately(stock_data):
    """Start training immediately with available data"""
    print("\n" + "=" * 80)
    print(f"STARTING PPO TRAINING WITH {len(stock_data)} STOCKS")
    print("=" * 80)

    # Import simplified trainer
    from simplified_ppo_trainer import train_ppo_simple

    # Start training
    train_ppo_simple(stock_data)

    print("\n" + "=" * 80)
    print("TRAINING STARTED SUCCESSFULLY!")
    print("=" * 80)


def main():
    """Main function"""
    print("\n" + "=" * 80)
    print("IMMEDIATE PPO TRAINING SYSTEM")
    print("No waiting - Start training NOW!")
    print("=" * 80)

    # Step 1: Quick download
    print("\n[STEP 1/2] Quick downloading major stocks...")
    stock_data = quick_download()

    if len(stock_data) < 50:
        print("\nError: Not enough stocks downloaded!")
        print("Trying alternative download method...")

        # Try one by one
        stock_data = {}
        for symbol in MAJOR_STOCKS[:100]:  # Try first 100
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period="15y")
                if not df.empty and len(df) > 100:
                    stock_data[symbol] = df
                    print(f"✓ {symbol}", end=" ")
                    if len(stock_data) >= 50:
                        break
            except:
                pass
        print()

    if len(stock_data) < 30:
        print("\nFATAL ERROR: Cannot download enough stocks!")
        print("Please check your internet connection")
        return

    # Step 2: Start training
    print(f"\n[STEP 2/2] Starting PPO training with {len(stock_data)} stocks...")
    train_immediately(stock_data)

    print("\n" + "=" * 80)
    print("✓ PROCESS COMPLETE!")
    print("Check models/ppo_simple/ for trained models")
    print("Check logs/ppo_simple/ for training logs")
    print("=" * 80)


if __name__ == "__main__":
    main()
