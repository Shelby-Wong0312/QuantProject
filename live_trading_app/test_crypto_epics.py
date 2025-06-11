import asyncio
import aiohttp
import os
from dotenv import load_dotenv

load_dotenv()

async def test_crypto_epics():
    """測試虛擬貨幣的 EPIC 格式"""
    api_key = os.getenv("CAPITAL_API_KEY")
    identifier = os.getenv("CAPITAL_IDENTIFIER")
    password = os.getenv("CAPITAL_API_PASSWORD")
    base_url = os.getenv("CAPITAL_BASE_API_URL", "https://demo-api-capital.backend-capital.com/api/v1")
    
    # 登錄
    login_url = f"{base_url}/session"
    headers = {
        "X-CAP-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "identifier": identifier,
        "password": password,
        "encryptedPassword": False
    }
    
    async with aiohttp.ClientSession() as session:
        # 登錄
        async with session.post(login_url, headers=headers, json=payload) as response:
            if response.status != 200:
                print(f"登錄失敗: {response.status}")
                return
            
            cst = response.headers.get("CST")
            x_security_token = response.headers.get("X-SECURITY-TOKEN")
            print("登錄成功")
        
        headers = {
            "X-CAP-API-KEY": api_key,
            "CST": cst,
            "X-SECURITY-TOKEN": x_security_token,
            "Content-Type": "application/json"
        }
        
        # 搜索比特幣相關市場
        print("\n搜索 Bitcoin 相關市場:")
        print("-" * 50)
        
        search_url = f"{base_url}/markets"
        params = {"searchTerm": "Bitcoin"}
        
        try:
            async with session.get(search_url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    markets = data.get('markets', [])
                    for market in markets[:10]:
                        print(f"EPIC: {market.get('epic')} - 名稱: {market.get('instrumentName')}")
                else:
                    print(f"搜索失敗: {response.status}")
        except Exception as e:
            print(f"搜索錯誤: {e}")
        
        # 搜索以太坊
        print("\n\n搜索 Ethereum 相關市場:")
        print("-" * 50)
        
        params = {"searchTerm": "Ethereum"}
        try:
            async with session.get(search_url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    markets = data.get('markets', [])
                    for market in markets[:10]:
                        print(f"EPIC: {market.get('epic')} - 名稱: {market.get('instrumentName')}")
                else:
                    print(f"搜索失敗: {response.status}")
        except Exception as e:
            print(f"搜索錯誤: {e}")
        
        # 測試一些可能的 EPIC 格式
        print("\n\n測試具體的 EPIC:")
        print("-" * 50)
        
        test_epics = [
            "BITCOIN",
            "Bitcoin",
            "BTCUSD",
            "BTC-USD",
            "CRYPTO.BITCOIN",
            "ETHEREUM",
            "Ethereum",
            "ETHUSD",
            "ETH-USD",
            "CRYPTO.ETHEREUM"
        ]
        
        for epic in test_epics:
            url = f"{base_url}/markets/{epic}"
            try:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        instrument = data.get('instrument', {})
                        print(f"✓ {epic} - 成功! 名稱: {instrument.get('name', 'N/A')}")
                    else:
                        print(f"✗ {epic} - 失敗 ({response.status})")
            except Exception as e:
                print(f"✗ {epic} - 錯誤: {e}")

if __name__ == "__main__":
    asyncio.run(test_crypto_epics()) 