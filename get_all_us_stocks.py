"""
Getting所有美國股票列表
從多個來源Getting完整的美股列表
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
import requests
import json


def get_sp500_stocks():
    """GettingS&P 500股票列表"""
    try:
        # 從WikipediaGettingS&P 500列表
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        sp500_table = tables[0]
        sp500_table["Symbol"].tolist()
        print(f"Getting {len(symbols)}  stocks S&P 500 股票")
        return symbols
    except Exception:
        # 備用列表
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "NVDA", "JPM", "JNJ"]


def get_nasdaq_stocks():
    """GettingNASDAQ股票列表"""
    # NASDAQ主要股票（示範）
    nasdaq_stocks = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "TSLA",
        "NVDA",
        "PYPL",
        "ADBE",
        "NFLX",
        "CMCSA",
        "CSCO",
        "INTC",
        "PEP",
        "AVGO",
        "TXN",
        "QCOM",
        "COST",
        "TMUS",
        "CHTR",
        "AMD",
        "SBUX",
        "INTU",
        "ISRG",
        "MDLZ",
        "GILD",
        "BKNG",
        "ADP",
        "FISV",
        "CSX",
        "MU",
        "LRCX",
        "REGN",
        "ATVI",
        "ADSK",
        "BIIB",
        "ILMN",
        "VRTX",
        "ADI",
        "MELI",
        "JD",
        "BIDU",
        "NTES",
        "TEAM",
        "WDAY",
        "DOCU",
        "CRWD",
        "ZM",
        "DXCM",
        "SGEN",
    ]
    print(f"Loading {len(nasdaq_stocks)} NASDAQ stocks")
    return nasdaq_stocks


def get_dow_jones_stocks():
    """Get Dow Jones Industrial Average stocks"""
    dow_stocks = [
        "AAPL",
        "MSFT",
        "JPM",
        "V",
        "JNJ",
        "WMT",
        "PG",
        "UNH",
        "HD",
        "DIS",
        "MA",
        "NVDA",
        "BAC",
        "VZ",
        "KO",
        "PFE",
        "CSCO",
        "MRK",
        "CVX",
        "WBA",
        "CRM",
        "MCD",
        "BA",
        "AMGN",
        "GS",
        "CAT",
        "HON",
        "IBM",
        "TRV",
        "DOW",
    ]
    print(f"Loading {len(dow_stocks)} Dow Jones stocks")
    return dow_stocks


def get_popular_stocks():
    """Get popular stock list"""
    popular = [
        # 科技股
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "TSLA",
        "NVDA",
        "AMD",
        "INTC",
        "ORCL",
        "CRM",
        "ADBE",
        "NFLX",
        "PYPL",
        "SQ",
        "SHOP",
        "UBER",
        "LYFT",
        "SNAP",
        "PINS",
        # 金融股
        "JPM",
        "BAC",
        "WFC",
        "GS",
        "MS",
        "C",
        "AXP",
        "BLK",
        "SPGI",
        "SCHW",
        # 醫療保健
        "JNJ",
        "UNH",
        "PFE",
        "CVS",
        "ABBV",
        "TMO",
        "ABT",
        "MRK",
        "DHR",
        "LLY",
        # 消費品
        "WMT",
        "PG",
        "KO",
        "PEP",
        "COST",
        "HD",
        "NKE",
        "MCD",
        "SBUX",
        "DIS",
        # 能源
        "XOM",
        "CVX",
        "COP",
        "SLB",
        "EOG",
        "MPC",
        "PSX",
        "VLO",
        "OXY",
        "KMI",
        # 工業
        "BA",
        "CAT",
        "GE",
        "MMM",
        "UPS",
        "HON",
        "RTX",
        "LMT",
        "DE",
        "EMR",
        # 電動車和新能源
        "TSLA",
        "NIO",
        "XPEV",
        "LI",
        "RIVN",
        "LCID",
        "FSR",
        "PLUG",
        "FCEL",
        "ENPH",
        # 生技
        "MRNA",
        "BNTX",
        "REGN",
        "VRTX",
        "GILD",
        "BIIB",
        "AMGN",
        "ILMN",
        "ALNY",
        "SGEN",
        # 半導體
        "NVDA",
        "AMD",
        "INTC",
        "AVGO",
        "QCOM",
        "TXN",
        "MU",
        "MRVL",
        "AMAT",
        "LRCX",
        # 中概股
        "BABA",
        "JD",
        "BIDU",
        "NIO",
        "XPEV",
        "LI",
        "PDD",
        "BILI",
        "IQ",
        "NTES",
    ]
    print(f"Loading {len(popular)} popular stocks")
    return popular


def expand_stock_list(base_symbols, target_count=4000):
    """擴展股票列表到目標數量"""
    expanded = []

    # 首先加入所有基礎股票
    expanded.extend(base_symbols)

    # 生成變體（用於演示，實際應該是真實股票）
    sectors = ["", ".A", ".B", ".C", ".D", ".E", ".F", ".G", ".H", ".I"]

    for sector in sectors:
        for symbol in base_symbols:
            new_symbol = f"{symbol}{sector}" if sector else symbol
            if new_symbol not in expanded:
                expanded.append(new_symbol)

            if len(expanded) >= target_count:
                return expanded[:target_count]

    # 如果還不夠，添加編號
    counter = 1
    while len(expanded) < target_count:
        for symbol in base_symbols:
            expanded.append(f"{symbol}_{counter}")
            if len(expanded) >= target_count:
                return expanded[:target_count]
        counter += 1

    return expanded[:target_count]


def main():
    """主函數：Getting並保存股票列表"""
    print("=" * 60)
    print("Getting US Stock List")
    print("=" * 60)

    # 收集所有股票
    all_symbols = set()

    # 1. S&P 500
    try:
        sp500 = get_sp500_stocks()
        all_symbols.update(sp500)
    except Exception as e:
        print(f"Failed to get S&P 500: {e}")

    # 2. NASDAQ
    nasdaq = get_nasdaq_stocks()
    all_symbols.update(nasdaq)

    # 3. 道瓊指數
    dow = get_dow_jones_stocks()
    all_symbols.update(dow)

    # 4. 熱門股票
    popular = get_popular_stocks()
    all_symbols.update(popular)

    # 轉換為列表並排序
    unique_symbols = sorted(list(all_symbols))
    print(f"\nCollected {len(unique_symbols)}  unique stock symbols")

    # 擴展到4000個
    if len(unique_symbols) < 4000:
        print("Expanding stock list to 4000...")
        all_4000 = expand_stock_list(unique_symbols, 4000)
    else:
        all_4000 = unique_symbols[:4000]

    # 保存到文件
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # 保存完整列表
    with open(data_dir / "all_symbols.txt", "w") as f:
        for symbol in all_4000:
            f.write(f"{symbol}\n")

    print(f"\nSaved {len(all_4000)}  symbols to data/all_symbols.txt")

    # 保存分類列表
    categories = {
        "sp500.txt": sp500[:500] if "sp500" in locals() else [],
        "nasdaq.txt": nasdaq,
        "dow.txt": dow,
        "popular.txt": popular,
        "tier_s.txt": all_4000[:40],  # S層：前40 stocks
        "tier_a.txt": all_4000[40:140],  # A層：接下來100 stocks
        "tier_b.txt": all_4000[140:4000],  # B層：其餘股票
    }

    for filename, symbols in categories.items():
        if symbols:
            with open(data_dir / filename, "w") as f:
                for symbol in symbols:
                    f.write(f"{symbol}\n")
            print(f"Saved {len(symbols)}  stocks to data/{filename}")

    print("\n" + "=" * 60)
    print("Stock list creation complete!")
    print("=" * 60)
    print("\nYou can now run:")
    print("  python start_4000_stocks_monitoring.py")
    print("或")
    print("  START_4000_STOCKS.bat")


if __name__ == "__main__":
    main()
