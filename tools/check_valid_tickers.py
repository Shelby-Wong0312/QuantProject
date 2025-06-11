import aiohttp
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CAPITAL_API_KEY")
IDENTIFIER = os.getenv("CAPITAL_IDENTIFIER")
PASSWORD = os.getenv("CAPITAL_API_PASSWORD")
BASE_URL = os.getenv("CAPITAL_BASE_API_URL", "https://demo-api-capital.backend-capital.com/api/v1")

TICKERS_FILE = "tickers.txt"
VALID_FILE = "valid_tickers.txt"
INVALID_FILE = "invalid_tickers.txt"

async def login():
    url = f"{BASE_URL}/session"
    headers = {"X-CAP-API-KEY": API_KEY, "Content-Type": "application/json"}
    payload = {"identifier": IDENTIFIER, "password": PASSWORD, "encryptedPassword": False}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status == 200:
                cst = resp.headers.get("CST")
                x_security_token = resp.headers.get("X-SECURITY-TOKEN")
                return cst, x_security_token
            else:
                print(f"登入失敗: {resp.status}")
                return None, None

def read_tickers():
    with open(TICKERS_FILE, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

async def check_symbol(session, cst, x_security_token, symbol):
    url = f"{BASE_URL}/markets/{symbol}"
    headers = {
        "X-CAP-API-KEY": API_KEY,
        "CST": cst,
        "X-SECURITY-TOKEN": x_security_token,
        "Content-Type": "application/json"
    }
    try:
        async with session.get(url, headers=headers) as resp:
            if resp.status == 200:
                return True
            else:
                return False
    except Exception as e:
        print(f"查詢{symbol}時出錯: {e}")
        return False

async def main():
    cst, x_security_token = await login()
    if not cst or not x_security_token:
        print("API登入失敗，請檢查環境變數")
        return
    tickers = read_tickers()
    valid = []
    invalid = []
    async with aiohttp.ClientSession() as session:
        for i, symbol in enumerate(tickers, 1):
            ok = await check_symbol(session, cst, x_security_token, symbol)
            if ok:
                print(f"✓ {symbol}")
                valid.append(symbol)
            else:
                print(f"✗ {symbol}")
                invalid.append(symbol)
            await asyncio.sleep(0.2)
    with open(VALID_FILE, 'w') as f:
        for s in valid:
            f.write(s + '\n')
    with open(INVALID_FILE, 'w') as f:
        for s in invalid:
            f.write(s + '\n')
    print(f"\n檢查完成！有效: {len(valid)}，無效: {len(invalid)}")

if __name__ == "__main__":
    asyncio.run(main()) 