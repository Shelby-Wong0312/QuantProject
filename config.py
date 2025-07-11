# quant_project/config.py
# FINAL VERSION - with .env debugging

import os
import sys
from dotenv import load_dotenv, find_dotenv

# --- .env 檔案加載與偵錯 ---
print("--- 開始加載 .env 檔案 ---")
# 尋找 .env 檔案的路徑
env_file_path = find_dotenv()

if env_file_path:
    print(f"✅ 找到 .env 檔案，路徑為: {env_file_path}")
    # verbose=True 會打印出詳細的加載過程
    load_dotenv(dotenv_path=env_file_path, verbose=True)
else:
    print("❌ 警告：在專案目錄中找不到 .env 檔案！")

print("--- .env 檔案加載完畢 ---")
print("\n--- 開始讀取環境變數 ---")

# --- Capital.com API 設定 ---
CAPITAL_API_KEY = os.getenv("CAPITAL_API_KEY")
CAPITAL_IDENTIFIER = os.getenv("CAPITAL_IDENTIFIER")
CAPITAL_API_PASSWORD = os.getenv("CAPITAL_API_PASSWORD")
CAPITAL_BASE_URL = "https://demo-api-capital.backend-capital.com/api/v1"

# --- 偵錯輸出：打印讀取到的值 ---
# 我們將每個值都打印出來，看看程式到底讀到了什麼
print(f"CAPITAL_API_KEY 的值: '{CAPITAL_API_KEY}'")
print(f"CAPITAL_IDENTIFIER 的值: '{CAPITAL_IDENTIFIER}'")
print(f"CAPITAL_API_PASSWORD 的值: '{CAPITAL_API_PASSWORD}'")
print("--- 環境變數讀取完畢 ---\n")


# --- 啟動安全檢查 ---
# 如果任何一個 Capital.com 憑證為空，則中止程式
if not all([CAPITAL_API_KEY, CAPITAL_IDENTIFIER, CAPITAL_API_PASSWORD]):
    print("="*60)
    print("❌ 致命錯誤：Capital.com API 憑證未完整設定或讀取失敗。")
    print("請檢查您的 .env 檔案，確保所有變數都已正確填寫。")
    print("="*60)
    sys.exit(1) # 中止程式


# --- 交易標的設定 ---
def _load_tickers_from_file(filepath: str, suffix: str = ".US") -> list[str]:
    # ... (此函式維持不變)
    try:
        with open(filepath, 'r') as f:
            tickers = [line.strip() + suffix for line in f if line.strip() and not line.strip().startswith('#')]
        # print(f"成功從 {filepath} 加載 {len(tickers)} 個股票代碼。") # 移到主程式中打印
        return tickers
    except FileNotFoundError:
        return []

ALL_VALID_SYMBOLS = _load_tickers_from_file('valid_tickers.txt')

SYMBOLS_TO_TRADE = [
    "AAPL.US", "MSFT.US", "GOOGL.US", "TSLA.US", "AMZN.US", "NVDA.US", "META.US"
]

# --- 資金與風險管理 ---
INITIAL_CAPITAL = 100_000.0
DEFAULT_TRADE_QUANTITY = 10

# --- 策略參數 ---
STRATEGY_PARAMS = {
    'Comprehensive_v1': {
        'ma_short_period': 5, 'ma_long_period': 20,
        'bias_period': 20, 'bias_upper': 7.0, 'bias_lower': -8.0,
        'kd_k': 14, 'kd_d': 3, 'kd_smooth': 3, 'kd_overbought': 80, 'kd_oversold': 20,
        'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
        'rsi_period': 14, 'rsi_overbought': 70, 'rsi_oversold': 30,
        'bb_period': 20, 'bb_std': 2.0,
        'ichi_tenkan': 9, 'ichi_kijun': 26, 'ichi_senkou_b': 52,
        'vol_ma_period': 20, 'vol_multiplier': 1.5,
        'atr_period': 14, 'atr_multiplier_sl': 2.0, 'atr_multiplier_tp': 4.0,
    }
}

# --- 系統設定 ---
LOG_LEVEL = "INFO"