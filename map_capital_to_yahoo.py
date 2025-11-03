#!/usr/bin/env python3
"""
將Capital.com的所有股票映射到Yahoo Finance
以Capital.com為核心，找出對應的Yahoo Finance代碼
"""

import json
import yfinance as yf
from datetime import datetime, timedelta
from tqdm import tqdm
import time
import re


def load_capital_stocks():
    """載入所有Capital.com股票"""

    if os.path.exists("capital_real_stocks.json"):
        with open("capital_real_stocks.json", "r", encoding="utf-8") as f:
            json.load(f)
            print(f"[LOAD] Found {len(data)} Capital.com stocks")
            return data
    else:
        print("[ERROR] capital_real_stocks.json not found")
        return []


def clean_symbol(symbol):
    """清理符號，移除特殊字符"""
    # 移除常見的後綴和前綴
    cleaned = symbol.upper()

    # 移除貨幣後綴
    for suffix in [
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
    ]:
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]

    # 移除下劃線後的內容（如_W表示週期）
    if "_" in cleaned:
        cleaned = cleaned.split("_")[0]

    # 移除數字後綴（如2025表示期貨）
    cleaned = re.sub(r"\d{4}$", "", cleaned)

    return cleaned


def try_yahoo_symbols(capital_stock):
    """嘗試多種Yahoo Finance符號格式"""
    ticker = capital_stock.get("ticker", "")
    epic = capital_stock.get("epic", "")
    name = capital_stock.get("name", "")

    # 生成可能的Yahoo符號
    candidates = []

    # 1. 原始ticker
    if ticker:
        candidates.append(ticker)

    # 2. 清理後的ticker
    cleaned = clean_symbol(ticker)
    if cleaned and cleaned != ticker:
        candidates.append(cleaned)

    # 3. Epic格式
    if epic and epic != ticker:
        candidates.append(epic)
        cleaned_epic = clean_symbol(epic)
        if cleaned_epic and cleaned_epic not in candidates:
            candidates.append(cleaned_epic)

    # 4. 特殊處理
    # 加密貨幣
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
    ]
    for keyword in crypto_keywords:
        if keyword in ticker.upper():
            candidates.append(f"{keyword}-USD")
            break

    # 外匯對
    forex_pairs = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
    for pair in forex_pairs:
        if pair in ticker.upper().replace("_", ""):
            candidates.append(f"{pair}=X")
            break

    # 商品期貨
    commodities = {
        "GOLD": "GC=F",
        "SILVER": "SI=F",
        "OIL": "CL=F",
        "CRUDE": "CL=F",
        "NATGAS": "NG=F",
        "COPPER": "HG=F",
        "WHEAT": "ZW=F",
        "CORN": "ZC=F",
        "SOYBEAN": "ZS=F",
    }
    for commodity, yahoo_symbol in commodities.items():
        if commodity in ticker.upper() or commodity in name.upper():
            candidates.append(yahoo_symbol)
            break

    # 英國股票（添加.L後綴）
    uk_keywords = ["GB", "GBP", "UK", "London", "British"]
    if any(keyword in str(capital_stock) for keyword in uk_keywords):
        candidates.append(f"{cleaned}.L")

    # 德國股票（添加.DE後綴）
    de_keywords = ["DE", "EUR", "Germany", "German", "Frankfurt"]
    if any(keyword in str(capital_stock) for keyword in de_keywords):
        candidates.append(f"{cleaned}.DE")

    # 香港股票（添加.HK後綴）
    hk_keywords = ["HK", "HKD", "Hong Kong"]
    if any(keyword in str(capital_stock) for keyword in hk_keywords):
        candidates.append(f"{cleaned}.HK")

    # 去重並返回
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    return unique_candidates


def validate_yahoo_symbol(symbol):
    """驗證Yahoo Finance符號是否有效"""
    try:
        # 快速檢查 - 只下載1天數據
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # 檢查是否有基本信息
        if info and (
            info.get("regularMarketPrice") or info.get("bid") or info.get("ask")
        ):
            return True, info.get("longName", info.get("shortName", symbol))

        # 備用方法：嘗試下載最近數據
        yf.download(
            symbol,
            start=(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
            end=datetime.now().strftime("%Y-%m-%d"),
            progress=False,
            threads=False,
        )

        if not data.empty:
            return True, symbol

    except Exception:
        pass

    return False, None


def map_all_stocks():
    """映射所有Capital.com股票到Yahoo Finance"""

    # 載入Capital.com股票
    capital_stocks = load_capital_stocks()

    if not capital_stocks:
        print("[ERROR] No Capital.com stocks to process")
        return []

    print(
        f"\n[START] Mapping {len(capital_stocks)} Capital.com stocks to Yahoo Finance"
    )
    print("[INFO] This will take some time...\n")

    # 映射結果
    mapped_stocks = []
    unmapped_stocks = []

    # 處理每個股票
    for i, capital_stock in enumerate(tqdm(capital_stocks, desc="Mapping stocks")):
        # 生成候選Yahoo符號
        candidates = try_yahoo_symbols(capital_stock)

        # 測試每個候選符號
        found = False
        for yahoo_symbol in candidates:
            is_valid, name = validate_yahoo_symbol(yahoo_symbol)

            if is_valid:
                mapped_stocks.append(
                    {
                        "capital_ticker": capital_stock.get("ticker", ""),
                        "capital_epic": capital_stock.get("epic", ""),
                        "capital_name": capital_stock.get("name", ""),
                        "yahoo_symbol": yahoo_symbol,
                        "yahoo_name": name,
                        "verified": True,
                    }
                )
                found = True
                break

        if not found:
            unmapped_stocks.append(
                {
                    "capital_ticker": capital_stock.get("ticker", ""),
                    "capital_epic": capital_stock.get("epic", ""),
                    "capital_name": capital_stock.get("name", ""),
                    "yahoo_symbol": None,
                    "verified": False,
                }
            )

        # 每處理100個股票暫停一下
        if (i + 1) % 100 == 0:
            time.sleep(1)
            print(f"\n[PROGRESS] Processed {i+1}/{len(capital_stocks)} stocks")
            print(
                f"[STATS] Mapped: {len(mapped_stocks)}, Unmapped: {len(unmapped_stocks)}"
            )

    return mapped_stocks, unmapped_stocks


def save_mapping_results(mapped_stocks, unmapped_stocks):
    """保存映射結果"""

    # 1. 保存完整映射結果
    all_mappings = {
        "mapped": mapped_stocks,
        "unmapped": unmapped_stocks,
        "statistics": {
            "total_capital_stocks": len(mapped_stocks) + len(unmapped_stocks),
            "successfully_mapped": len(mapped_stocks),
            "failed_to_map": len(unmapped_stocks),
            "success_rate": f"{len(mapped_stocks) / (len(mapped_stocks) + len(unmapped_stocks)) * 100:.1f}%",
        },
    }

    with open("capital_to_yahoo_mapping.json", "w", encoding="utf-8") as f:
        json.dump(all_mappings, f, indent=2, ensure_ascii=False)

    # 2. 保存成功映射的Yahoo符號列表（用於下載數據）
    with open("mapped_yahoo_symbols.txt", "w") as f:
        for stock in mapped_stocks:
            f.write(f"{stock['yahoo_symbol']}\n")

    # 3. 保存成功映射的Capital符號列表（用於交易）
    with open("mapped_capital_symbols.txt", "w") as f:
        for stock in mapped_stocks:
            f.write(f"{stock['capital_ticker']}\n")

    # 4. 保存簡化的映射表（ticker對照）
    simple_mapping = {}
    for stock in mapped_stocks:
        simple_mapping[stock["capital_ticker"]] = stock["yahoo_symbol"]

    with open("capital_yahoo_simple_map.json", "w", encoding="utf-8") as f:
        json.dump(simple_mapping, f, indent=2)

    print("\n[SAVED] Mapping results saved to:")
    print("  - capital_to_yahoo_mapping.json (complete mapping)")
    print(f"  - mapped_yahoo_symbols.txt ({len(mapped_stocks)} Yahoo symbols)")
    print(f"  - mapped_capital_symbols.txt ({len(mapped_stocks)} Capital symbols)")
    print("  - capital_yahoo_simple_map.json (simple ticker mapping)")


def main():
    print("=" * 80)
    print("CAPITAL.COM TO YAHOO FINANCE MAPPING")
    print("=" * 80)

    # 執行映射
    mapped, unmapped = map_all_stocks()

    # 保存結果
    save_mapping_results(mapped, unmapped)

    # 顯示統計
    print("\n" + "=" * 80)
    print("MAPPING COMPLETE!")
    print("=" * 80)
    print(f"Total Capital.com stocks: {len(mapped) + len(unmapped)}")
    print(
        f"Successfully mapped: {len(mapped)} ({len(mapped)/(len(mapped)+len(unmapped))*100:.1f}%)"
    )
    print(f"Failed to map: {len(unmapped)}")

    # 顯示樣本
    if mapped:
        print("\nSample successful mappings:")
        for stock in mapped[:10]:
            print(
                f"  {stock['capital_ticker']:10} -> {stock['yahoo_symbol']:10} : {stock['capital_name'][:30]}"
            )

    if unmapped:
        print(f"\nSample unmapped stocks (first 10 of {len(unmapped)}):")
        for stock in unmapped[:10]:
            print(f"  {stock['capital_ticker']:10} : {stock['capital_name'][:40]}")

    print("=" * 80)


if __name__ == "__main__":
    import os

    main()
