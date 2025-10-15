#!/usr/bin/env python3
"""
獲取Capital.com真實可交易股票列表
通過API直接獲取
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()


class CapitalComAPI:
    def __init__(self):
        self.api_key = os.getenv("CAPITAL_API_KEY")
        self.password = os.getenv("CAPITAL_API_PASSWORD")
        self.identifier = os.getenv("CAPITAL_IDENTIFIER")
        self.demo = True  # 使用Demo API

        if self.demo:
            self.base_url = "https://demo-api-capital.backend-capital.com/api/v1"
        else:
            self.base_url = "https://api-capital.backend-capital.com/api/v1"

        self.session = requests.Session()
        self.headers = {
            "X-CAP-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

    def create_session(self):
        """創建交易會話"""
        print("[AUTH] Creating session with Capital.com...")

        endpoint = f"{self.base_url}/session"
        payload = {"identifier": self.identifier, "password": self.password}

        try:
            response = self.session.post(endpoint, json=payload, headers=self.headers)

            if response.status_code == 200:
                response.json()
                self.cst = response.headers.get("CST")
                self.x_security_token = response.headers.get("X-SECURITY-TOKEN")

                # 更新headers
                self.headers["CST"] = self.cst
                self.headers["X-SECURITY-TOKEN"] = self.x_security_token

                print("[AUTH] Session created successfully")
                return True
            else:
                print(f"[AUTH] Failed to create session: {response.status_code}")
                print(f"[AUTH] Response: {response.text}")
                return False

        except Exception as e:
            print(f"[AUTH] Error creating session: {e}")
            return False

    def get_all_markets(self, search_term=""):
        """獲取所有可交易市場"""
        print("[API] Fetching markets...")

        endpoint = f"{self.base_url}/markets"
        params = {}
        if search_term:
            params["searchTerm"] = search_term

        try:
            response = self.session.get(endpoint, headers=self.headers, params=params)

            if response.status_code == 200:
                response.json()
                markets = data.get("markets", [])
                print(f"[API] Found {len(markets)} markets")
                return markets
            else:
                print(f"[API] Failed to get markets: {response.status_code}")
                return []

        except Exception as e:
            print(f"[API] Error fetching markets: {e}")
            return []

    def get_market_navigation(self):
        """獲取市場導航（分類）"""
        endpoint = f"{self.base_url}/marketnavigation"

        try:
            response = self.session.get(endpoint, headers=self.headers)

            if response.status_code == 200:
                return response.json()
            else:
                print(f"[API] Failed to get navigation: {response.status_code}")
                return {}

        except Exception as e:
            print(f"[API] Error: {e}")
            return {}

    def search_us_stocks(self):
        """搜索美國股票"""
        # 搜索常見美股
        all_stocks = []

        # 搜索不同類型的股票
        search_terms = [
            "",  # 獲取所有
            "US",  # 美國股票
            "NYSE",  # 紐交所
            "NASDAQ",  # 納斯達克
            "S&P",  # 標普
            "DOW",  # 道瓊斯
        ]

        for term in search_terms:
            print(f"[SEARCH] Searching for: {term}")
            markets = self.get_all_markets(term)

            for market in markets:
                instrument_type = market.get("instrumentType", "")
                epic = market.get("epic", "")
                name = market.get("instrumentName", "")

                # 篩選股票類型
                if (
                    "SHARES" in instrument_type.upper()
                    or "STOCK" in instrument_type.upper()
                    or "EQUITY" in instrument_type.upper()
                    or (
                        epic
                        and not any(
                            x in epic.upper()
                            for x in ["INDEX", "FX", "COMMODITY", "CRYPTO"]
                        )
                    )
                ):

                    stock_info = {
                        "epic": epic,
                        "name": name,
                        "type": instrument_type,
                        "ticker": epic.split(".")[-1] if "." in epic else epic,
                    }

                    # 避免重複
                    if not any(s["epic"] == epic for s in all_stocks):
                        all_stocks.append(stock_info)

        return all_stocks


def get_comprehensive_stock_list():
    """獲取綜合股票列表"""

    # 美國主要股票（確定可在Capital.com交易的）
    major_us_stocks = [
        # 科技巨頭
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
        "NIO",
        "XPEV",
        "LI",
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
        # 金融
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
        "MCO",
        "MSCI",
        "TROW",
        # 醫療保健
        "JNJ",
        "UNH",
        "PFE",
        "ABBV",
        "CVS",
        "MRK",
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
        "IDXX",
        "ALGN",
        "DXCM",
        "MTD",
        "WST",
        "ZBH",
        # 消費
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
        # 工業
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
        "FTV",
        "AME",
        # 能源
        "XOM",
        "CVX",
        "COP",
        "SLB",
        "EOG",
        "PXD",
        "MPC",
        "VLO",
        "PSX",
        "OXY",
        "KMI",
        "WMB",
        "ET",
        "EPD",
        "ENB",
        "TRP",
        "LNG",
        "FANG",
        "DVN",
        "HES",
        # 材料
        "LIN",
        "APD",
        "SHW",
        "ECL",
        "DD",
        "NEM",
        "FCX",
        "DOW",
        "PPG",
        "ALB",
        # 房地產
        "AMT",
        "PLD",
        "CCI",
        "EQIX",
        "PSA",
        "DLR",
        "O",
        "WELL",
        "SPG",
        "AVB",
        # 公用事業
        "NEE",
        "DUK",
        "SO",
        "D",
        "EXC",
        "SRE",
        "AEP",
        "XEL",
        "ED",
        "PEG",
        # 通訊
        "T",
        "VZ",
        "TMUS",
        "CHTR",
        "CMCSA",
        "DIS",
        "NFLX",
        "PARA",
        "WBD",
        "FOX",
    ]

    # 中概股
    china_stocks = [
        "BABA",
        "JD",
        "PDD",
        "BIDU",
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
        "YMM",
        "HUYA",
        "DOYU",
    ]

    # ETF
    etfs = [
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
    ]

    return major_us_stocks + china_stocks + etfs


def main():
    print("=" * 80)
    print("FETCHING REAL CAPITAL.COM TRADABLE STOCKS")
    print("=" * 80)

    # 嘗試從API獲取
    api = CapitalComAPI()

    stocks = []
    if api.create_session():
        # 獲取市場導航
        navigation = api.get_market_navigation()
        if navigation:
            print("[API] Market navigation retrieved")

        # 搜索股票
        stocks = api.search_us_stocks()
        print(f"[API] Found {len(stocks)} stocks from API")

    # 如果API返回的股票太少，使用預定義列表
    if len(stocks) < 100:
        print("[INFO] Using comprehensive predefined stock list")
        predefined = get_comprehensive_stock_list()

        # 轉換為統一格式
        for ticker in predefined:
            stocks.append(
                {"ticker": ticker, "epic": ticker, "name": ticker, "type": "SHARES"}
            )

    # 去重
    unique_stocks = []
    seen_tickers = set()
    for stock in stocks:
        ticker = stock["ticker"]
        if ticker not in seen_tickers:
            unique_stocks.append(stock)
            seen_tickers.add(ticker)

    print(f"\n[RESULT] Total unique stocks: {len(unique_stocks)}")

    # 保存列表
    with open("capital_real_stocks.json", "w", encoding="utf-8") as f:
        json.dump(unique_stocks, f, indent=2, ensure_ascii=False)

    # 保存ticker列表
    with open("capital_real_tickers.txt", "w") as f:
        for stock in unique_stocks:
            f.write(f"{stock['ticker']}\n")

    print("[SAVED] Stock list saved to capital_real_stocks.json")
    print("[SAVED] Ticker list saved to capital_real_tickers.txt")

    # 顯示樣本
    print("\nSample stocks:")
    for stock in unique_stocks[:20]:
        print(f"  {stock['ticker']}: {stock['name']}")

    return unique_stocks


if __name__ == "__main__":
    main()
