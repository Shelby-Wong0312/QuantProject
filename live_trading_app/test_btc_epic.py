import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import aiohttp
from dotenv import load_dotenv

load_dotenv()

async def test_btc_epics():
    """測試不同的 BTC EPIC 格式"""
    # 更新 API Key
    api_key = "oVGhAub8ezuC9Zo1"  # 使用新的 API Key
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
                text = await response.text()
                print(f"錯誤信息: {text}")
                return
            
            cst = response.headers.get("CST")
            x_security_token = response.headers.get("X-SECURITY-TOKEN")
            print("✅ 登錄成功")
        
        headers = {
            "X-CAP-API-KEY": api_key,
            "CST": cst,
            "X-SECURITY-TOKEN": x_security_token,
            "Content-Type": "application/json"
        }
        
        # 測試不同的 BTC EPIC 格式
        test_epics = [
            "BTCUSD",
            "Bitcoin",
            "BTC",
            "BITCOIN",
            "CRYPTO.BTCUSD",
            "CC.D.BTC.USS.IP",
            "Bitcoin vs US Dollar",
        ]
        
        print("\n測試不同的 BTC EPIC 格式:")
        print("-" * 50)
        
        for epic in test_epics:
            url = f"{base_url}/markets/{epic}"
            try:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ {epic} - 成功!")
                        if 'snapshot' in data:
                            snapshot = data['snapshot']
                            print(f"   買價: ${snapshot.get('bid', 0):,.2f}")
                            print(f"   賣價: ${snapshot.get('offer', 0):,.2f}")
                        print()
                    else:
                        print(f"❌ {epic} - 失敗 ({response.status})")
            except Exception as e:
                print(f"❌ {epic} - 錯誤: {e}")
        
        # 搜索 Bitcoin 相關市場
        print("\n搜索 Bitcoin 相關市場:")
        print("-" * 50)
        
        search_url = f"{base_url}/markets"
        params = {"searchTerm": "Bitcoin"}
        
        async with session.get(search_url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                markets = data.get('markets', [])
                print(f"找到 {len(markets)} 個相關市場:")
                for market in markets[:5]:  # 只顯示前5個
                    print(f"- EPIC: {market.get('epic')}")
                    print(f"  名稱: {market.get('instrumentName')}")
                    print()

if __name__ == "__main__":
    asyncio.run(test_btc_epics()) 