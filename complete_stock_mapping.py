#!/usr/bin/env python3
"""
完整映射所有Capital.com股票到Yahoo Finance
逐個驗證4446個股票
"""

import json
import os
import yfinance as yf
from datetime import datetime
from tqdm import tqdm
import time
import re
import pickle


class CapitalToYahooMapper:
    def __init__(self):
        self.capital_stocks = []
        self.mapped_stocks = []
        self.unmapped_stocks = []
        self.cache_file = "mapping_cache.pkl"
        self.load_cache()

    def load_cache(self):
        """載入緩存的映射結果"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    cache = pickle.load(f)
                    self.mapped_stocks = cache.get("mapped", [])
                    self.unmapped_stocks = cache.get("unmapped", [])
                    print(
                        f"[CACHE] Loaded {len(self.mapped_stocks)} mapped stocks from cache"
                    )
            except Exception:
                pass

    def save_cache(self):
        """保存映射結果到緩存"""
        cache = {
            "mapped": self.mapped_stocks,
            "unmapped": self.unmapped_stocks,
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.cache_file, "wb") as f:
            pickle.dump(cache, f)

    def load_capital_stocks(self):
        """載入所有Capital.com股票"""
        if os.path.exists("capital_real_stocks.json"):
            with open("capital_real_stocks.json", "r", encoding="utf-8") as f:
                self.capital_stocks = json.load(f)
                print(f"[LOAD] Loaded {len(self.capital_stocks)} Capital.com stocks")
        else:
            print("[ERROR] capital_real_stocks.json not found")
            return False
        return True

    def clean_symbol(self, symbol):
        """清理符號"""
        if not symbol:
            return ""

        cleaned = symbol.upper().strip()

        # 移除貨幣後綴
        currency_suffixes = [
            "USD",
            "GBP",
            "EUR",
            "JPY",
            "CHF",
            "CAD",
            "AUD",
            "SGD",
            "HKD",
            "CNH",
            "CNY",
            "DKK",
            "SEK",
            "NOK",
        ]
        for suffix in currency_suffixes:
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)]

        # 移除下劃線和之後的內容
        if "_" in cleaned:
            cleaned = cleaned.split("_")[0]

        # 移除年份（期貨）
        cleaned = re.sub(r"\d{4}$", "", cleaned)

        return cleaned

    def generate_yahoo_candidates(self, capital_stock):
        """生成可能的Yahoo Finance符號"""
        ticker = capital_stock.get("ticker", "")
        epic = capital_stock.get("epic", "")
        name = capital_stock.get("name", "")

        candidates = []
        seen = set()

        def add_candidate(symbol):
            if symbol and symbol not in seen:
                candidates.append(symbol)
                seen.add(symbol)

        # 1. 原始ticker
        add_candidate(ticker)

        # 2. 清理後的ticker
        cleaned = self.clean_symbol(ticker)
        add_candidate(cleaned)

        # 3. Epic
        if epic != ticker:
            add_candidate(epic)
            add_candidate(self.clean_symbol(epic))

        # 4. 加密貨幣處理
        crypto_keywords = [
            "BTC",
            "ETH",
            "DOGE",
            "ADA",
            "DOT",
            "LINK",
            "UNI",
            "MATIC",
            "SOL",
            "AVAX",
            "SHIB",
            "XRP",
            "BNB",
            "ATOM",
            "LTC",
            "TRX",
            "ALGO",
            "XLM",
            "VET",
            "FIL",
            "EOS",
            "AAVE",
            "MKR",
            "COMP",
            "SNX",
            "YFI",
            "SUSHI",
        ]

        for crypto in crypto_keywords:
            if crypto in ticker.upper():
                add_candidate(f"{crypto}-USD")
                break

        # 5. 外匯處理
        forex_pairs = {
            "EURUSD": "EURUSD=X",
            "GBPUSD": "GBPUSD=X",
            "USDJPY": "USDJPY=X",
            "USDCHF": "USDCHF=X",
            "AUDUSD": "AUDUSD=X",
            "USDCAD": "USDCAD=X",
            "NZDUSD": "NZDUSD=X",
            "EURGBP": "EURGBP=X",
            "EURJPY": "EURJPY=X",
            "GBPJPY": "GBPJPY=X",
        }

        ticker_clean = ticker.upper().replace("_", "")
        for fx_pair, yahoo_symbol in forex_pairs.items():
            if fx_pair in ticker_clean:
                add_candidate(yahoo_symbol)

        # 6. 商品期貨
        commodities = {
            "GOLD": "GC=F",
            "SILVER": "SI=F",
            "CRUDE": "CL=F",
            "OIL": "CL=F",
            "BRENT": "BZ=F",
            "NATGAS": "NG=F",
            "COPPER": "HG=F",
            "WHEAT": "ZW=F",
            "CORN": "ZC=F",
            "SOYBEAN": "ZS=F",
            "COTTON": "CT=F",
            "SUGAR": "SB=F",
            "COFFEE": "KC=F",
            "COCOA": "CC=F",
        }

        for commodity, yahoo_symbol in commodities.items():
            if commodity in ticker.upper() or commodity in name.upper():
                add_candidate(yahoo_symbol)

        # 7. 國際市場後綴
        # 英國
        if any(
            x in str(capital_stock).upper()
            for x in ["GB", "GBP", "UK", "LONDON", "LSE"]
        ):
            add_candidate(f"{cleaned}.L")

        # 德國
        if any(
            x in str(capital_stock).upper()
            for x in ["DE", "GERMANY", "FRANKFURT", "XETRA"]
        ):
            add_candidate(f"{cleaned}.DE")

        # 法國
        if any(x in str(capital_stock).upper() for x in ["FR", "FRANCE", "PARIS"]):
            add_candidate(f"{cleaned}.PA")

        # 香港
        if any(x in str(capital_stock).upper() for x in ["HK", "HKD", "HONG KONG"]):
            add_candidate(f"{cleaned}.HK")

        # 日本
        if any(
            x in str(capital_stock).upper() for x in ["JP", "JAPAN", "TOKYO", "JPY"]
        ):
            add_candidate(f"{cleaned}.T")

        # 加拿大
        if any(
            x in str(capital_stock).upper() for x in ["CA", "CANADA", "TORONTO", "TSX"]
        ):
            add_candidate(f"{cleaned}.TO")

        # 澳大利亞
        if any(
            x in str(capital_stock).upper()
            for x in ["AU", "AUSTRALIA", "ASX", "SYDNEY"]
        ):
            add_candidate(f"{cleaned}.AX")

        # 新加坡
        if any(x in str(capital_stock).upper() for x in ["SG", "SINGAPORE", "SGX"]):
            add_candidate(f"{cleaned}.SI")

        # 瑞士
        if any(
            x in str(capital_stock).upper() for x in ["CH", "SWISS", "ZURICH", "CHF"]
        ):
            add_candidate(f"{cleaned}.SW")

        # 荷蘭
        if any(
            x in str(capital_stock).upper() for x in ["NL", "NETHERLANDS", "AMSTERDAM"]
        ):
            add_candidate(f"{cleaned}.AS")

        # 8. 特殊處理某些已知的符號
        special_mappings = {
            "BRK.B": "BRK-B",
            "BF.B": "BF-B",
            "BRK.A": "BRK-A",
            "BF.A": "BF-A",
        }

        for old, new in special_mappings.items():
            if old in ticker:
                add_candidate(new)

        return candidates

    def validate_yahoo_symbol(self, symbol):
        """驗證Yahoo Finance符號"""
        if not symbol:
            return False, None

        try:
            ticker = yf.Ticker(symbol)

            # 嘗試獲取信息
            info = ticker.info
            if info and len(info) > 3:  # 有基本信息
                return True, info.get("longName", info.get("shortName", symbol))

            # 備用方法：嘗試獲取歷史數據
            hist = ticker.history(period="5d")
            if not hist.empty:
                return True, symbol

        except Exception:
            pass

        return False, None

    def map_single_stock(self, capital_stock):
        """映射單個股票"""
        candidates = self.generate_yahoo_candidates(capital_stock)

        for candidate in candidates:
            is_valid, name = self.validate_yahoo_symbol(candidate)
            if is_valid:
                return {
                    "capital_ticker": capital_stock.get("ticker", ""),
                    "capital_epic": capital_stock.get("epic", ""),
                    "capital_name": capital_stock.get("name", ""),
                    "yahoo_symbol": candidate,
                    "yahoo_name": name,
                    "verified": True,
                    "verified_at": datetime.now().isoformat(),
                }

        return None

    def map_all_stocks(self):
        """映射所有股票"""
        print("\n[START] Mapping all Capital.com stocks to Yahoo Finance")
        print(f"[INFO] Total stocks to process: {len(self.capital_stocks)}")

        # 檢查已處理的股票
        processed_tickers = set()
        for stock in self.mapped_stocks:
            processed_tickers.add(stock["capital_ticker"])
        for stock in self.unmapped_stocks:
            processed_tickers.add(stock["capital_ticker"])

        # 篩選未處理的股票
        stocks_to_process = []
        for stock in self.capital_stocks:
            ticker = stock.get("ticker", "")
            if ticker not in processed_tickers:
                stocks_to_process.append(stock)

        print(f"[INFO] Already processed: {len(processed_tickers)}")
        print(f"[INFO] Remaining to process: {len(stocks_to_process)}")

        if not stocks_to_process:
            print("[INFO] All stocks already processed!")
            return

        # 處理剩餘股票
        batch_size = 10
        for i in tqdm(
            range(0, len(stocks_to_process), batch_size), desc="Processing batches"
        ):
            batch = stocks_to_process[i : i + batch_size]

            for stock in batch:
                mapping = self.map_single_stock(stock)

                if mapping:
                    self.mapped_stocks.append(mapping)
                else:
                    self.unmapped_stocks.append(
                        {
                            "capital_ticker": stock.get("ticker", ""),
                            "capital_epic": stock.get("epic", ""),
                            "capital_name": stock.get("name", ""),
                            "yahoo_symbol": None,
                            "verified": False,
                        }
                    )

            # 定期保存進度
            if (i + batch_size) % 100 == 0:
                self.save_cache()
                print(
                    f"\n[PROGRESS] Processed {min(i+batch_size, len(stocks_to_process))}/{len(stocks_to_process)}"
                )
                print(
                    f"[STATS] Mapped: {len(self.mapped_stocks)}, Failed: {len(self.unmapped_stocks)}"
                )
                time.sleep(1)  # 避免請求過快

        # 最終保存
        self.save_cache()

    def save_final_results(self):
        """保存最終結果"""
        print("\n[SAVE] Saving final mapping results...")

        # 1. 完整映射結果
        full_results = {
            "mapped": self.mapped_stocks,
            "unmapped": self.unmapped_stocks,
            "statistics": {
                "total_capital_stocks": len(self.capital_stocks),
                "successfully_mapped": len(self.mapped_stocks),
                "failed_to_map": len(self.unmapped_stocks),
                "success_rate": (
                    f"{len(self.mapped_stocks) / len(self.capital_stocks) * 100:.1f}%"
                    if self.capital_stocks
                    else "0%"
                ),
            },
            "created_at": datetime.now().isoformat(),
        }

        with open("capital_yahoo_full_mapping.json", "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)

        # 2. Yahoo符號列表
        with open("yahoo_symbols_all.txt", "w") as f:
            for stock in self.mapped_stocks:
                f.write(f"{stock['yahoo_symbol']}\n")

        # 3. Capital符號列表
        with open("capital_symbols_all.txt", "w") as f:
            for stock in self.mapped_stocks:
                f.write(f"{stock['capital_ticker']}\n")

        # 4. 簡單映射表
        simple_map = {}
        for stock in self.mapped_stocks:
            simple_map[stock["capital_ticker"]] = stock["yahoo_symbol"]

        with open("capital_yahoo_simple_map.json", "w", encoding="utf-8") as f:
            json.dump(simple_map, f, indent=2)

        print("[SAVED] Files created:")
        print("  - capital_yahoo_full_mapping.json")
        print(f"  - yahoo_symbols_all.txt ({len(self.mapped_stocks)} symbols)")
        print(f"  - capital_symbols_all.txt ({len(self.mapped_stocks)} symbols)")
        print("  - capital_yahoo_simple_map.json")


def main():
    print("=" * 80)
    print("COMPLETE CAPITAL.COM TO YAHOO FINANCE MAPPING")
    print("Processing ALL 4446 stocks")
    print("=" * 80)

    mapper = CapitalToYahooMapper()

    # 載入Capital.com股票
    if not mapper.load_capital_stocks():
        return

    # 執行映射
    mapper.map_all_stocks()

    # 保存結果
    mapper.save_final_results()

    # 顯示統計
    print("\n" + "=" * 80)
    print("MAPPING COMPLETE!")
    print("=" * 80)
    print(f"Total Capital.com stocks: {len(mapper.capital_stocks)}")
    print(f"Successfully mapped: {len(mapper.mapped_stocks)}")
    print(f"Failed to map: {len(mapper.unmapped_stocks)}")

    if mapper.mapped_stocks:
        success_rate = len(mapper.mapped_stocks) / len(mapper.capital_stocks) * 100
        print(f"Success rate: {success_rate:.1f}%")

    # 顯示樣本
    print("\nSample successful mappings:")
    for stock in mapper.mapped_stocks[:10]:
        print(
            f"  {stock['capital_ticker']:10} -> {stock['yahoo_symbol']:10} : {stock['capital_name'][:30]}"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()
