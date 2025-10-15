#!/usr/bin/env python3
"""
獲取Capital.com所有可交易股票
Get all tradable stocks from Capital.com
"""

import os
import json
import requests
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class CapitalComStockFetcher:
    def __init__(self):
        self.api_key = os.getenv("CAPITAL_API_KEY")
        self.base_url = "https://api-capital.backend-capital.com/api/v1"
        self.headers = {
            "X-CAP-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

    def get_all_markets(self) -> List[Dict]:
        """獲取所有可交易市場"""
        print("[INFO] Fetching all markets from Capital.com...")

        try:
            # Get all markets
            response = requests.get(
                f"{self.base_url}/markets",
                headers=self.headers,
                params={"limit": 10000},  # Get maximum number
            )

            if response.status_code == 200:
                response.json()
                markets = data.get("markets", [])
                print(f"[SUCCESS] Found {len(markets)} total markets")
                return markets
            else:
                print(f"[ERROR] API responded with status {response.status_code}")
                return []

        except Exception as e:
            print(f"[ERROR] Failed to fetch markets: {e}")
            return []

    def filter_stocks(self, markets: List[Dict]) -> List[str]:
        """篩選出股票類型的市場"""
        stocks = []

        for market in markets:
            # Check if it's a stock (not forex, commodity, index, etc.)
            instrument_type = market.get("instrumentType", "")
            epic = market.get("epic", "")
            name = market.get("instrumentName", "")

            # Filter for stocks/shares
            if (
                instrument_type in ["SHARES", "STOCK", "EQUITY"]
                or "SHARES" in instrument_type.upper()
                or "STOCK" in instrument_type.upper()
            ):

                # Extract ticker symbol from epic
                # Capital.com format is usually like "AAPL" or "US.AAPL"
                ticker = epic.split(".")[-1] if "." in epic else epic

                stocks.append(
                    {
                        "ticker": ticker,
                        "epic": epic,
                        "name": name,
                        "type": instrument_type,
                    }
                )

        print(f"[INFO] Filtered {len(stocks)} stocks from markets")
        return stocks

    def save_stock_list(
        self, stocks: List[Dict], filename: str = "capital_stocks.json"
    ):
        """保存股票列表"""
        # Save detailed JSON
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(stocks, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] Stock list saved to {filename}")

        # Save simple ticker list
        tickers = [s["ticker"] for s in stocks]
        with open("capital_tickers.txt", "w") as f:
            for ticker in sorted(tickers):
                f.write(f"{ticker}\n")
        print(f"[SAVED] {len(tickers)} tickers saved to capital_tickers.txt")

        return tickers


def main():
    print("=" * 80)
    print("FETCHING ALL CAPITAL.COM TRADABLE STOCKS")
    print("=" * 80)

    # Initialize fetcher
    fetcher = CapitalComStockFetcher()

    # Get all markets
    markets = fetcher.get_all_markets()

    if not markets:
        print("[ERROR] No markets fetched. Check API credentials.")

        # Use a comprehensive default list of popular stocks
        print("[INFO] Using default comprehensive stock list...")

        # Major US stocks (S&P 500 + NASDAQ leaders)
        us_stocks = [
            # Tech Giants
            "AAPL",
            "MSFT",
            "GOOGL",
            "GOOG",
            "AMZN",
            "META",
            "NVDA",
            "TSLA",
            "AMD",
            "INTC",
            "CSCO",
            "ORCL",
            "IBM",
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
            "TWTR",
            "ABNB",
            "DASH",
            "COIN",
            "HOOD",
            "RBLX",
            "PLTR",
            "SNOW",
            "NET",
            "DDOG",
            "ZM",
            "DOCU",
            "OKTA",
            "TWLO",
            "TEAM",
            "CRWD",
            "PANW",
            # Finance
            "JPM",
            "BAC",
            "WFC",
            "GS",
            "MS",
            "C",
            "USB",
            "PNC",
            "BK",
            "AXP",
            "V",
            "MA",
            "COF",
            "DFS",
            "BLK",
            "SCHW",
            "ICE",
            "CME",
            "SPGI",
            # Healthcare
            "JNJ",
            "UNH",
            "PFE",
            "ABBV",
            "TMO",
            "ABT",
            "CVS",
            "MRK",
            "LLY",
            "BMY",
            "AMGN",
            "GILD",
            "MDT",
            "DHR",
            "ISRG",
            "SYK",
            "BSX",
            "EW",
            "REGN",
            "VRTX",
            "MRNA",
            "BIIB",
            "ILMN",
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
            "CVX",
            "XOM",
            "DIS",
            "CMCSA",
            "NFLX",
            "T",
            "VZ",
            # Industrial
            "BA",
            "CAT",
            "DE",
            "UPS",
            "FDX",
            "LMT",
            "RTX",
            "GE",
            "MMM",
            "HON",
            "UNP",
            "CSX",
            "NSC",
            "EMR",
            "ETN",
            "ITW",
            "PH",
            "ROK",
            # More tech and growth
            "ROKU",
            "TTD",
            "SPOT",
            "ZS",
            "ESTC",
            "MDB",
            "FSLY",
            "FVRR",
            "UPWK",
            "SE",
            "MELI",
            "CPNG",
            "GRAB",
            "NU",
            "RIVN",
            "LCID",
            "NIO",
            "XPEV",
            "LI",
            "BABA",
            "JD",
            "PDD",
            "BIDU",
            "BILI",
            "TSM",
            "ASML",
            "ARM",
            "SMCI",
            "AVGO",
            "QCOM",
            "MU",
            "LRCX",
            "KLAC",
            "AMAT",
            "ADI",
            "MRVL",
            "NXPI",
            "MCHP",
            "TXN",
            "XLNX",
        ]

        # Chinese ADRs
        china_stocks = [
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
            "BGNE",
            "TAL",
            "EDU",
            "DIDI",
        ]

        # European stocks (if available)
        eu_stocks = [
            "ASML",
            "SAP",
            "NVO",
            "AZN",
            "SHEL",
            "TM",
            "NVS",
            "HSBC",
            "BP",
            "TOT",
            "SNY",
            "GSK",
            "DEO",
            "BUD",
            "UL",
            "RIO",
        ]

        # Combine all unique tickers
        all_tickers = list(set(us_stocks + china_stocks + eu_stocks))

        # Create stock list in Capital.com format
        stocks = []
        for ticker in sorted(all_tickers):
            stocks.append(
                {"ticker": ticker, "epic": ticker, "name": ticker, "type": "SHARES"}
            )

        # Save the list
        with open("capital_stocks.json", "w", encoding="utf-8") as f:
            json.dump(stocks, f, indent=2)

        with open("capital_tickers.txt", "w") as f:
            for ticker in sorted(all_tickers):
                f.write(f"{ticker}\n")

        print(f"[INFO] Created default list with {len(all_tickers)} stocks")
        return all_tickers

    # Filter stocks
    stocks = fetcher.filter_stocks(markets)

    # Save stock list
    tickers = fetcher.save_stock_list(stocks)

    print("\n" + "=" * 80)
    print(f"COMPLETE: Found {len(tickers)} tradable stocks on Capital.com")
    print("=" * 80)

    # Print sample
    print("\nSample stocks:")
    for stock in stocks[:10]:
        print(f"  - {stock['ticker']}: {stock['name']}")

    return tickers


if __name__ == "__main__":
    main()
