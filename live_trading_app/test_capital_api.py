import asyncio
import aiohttp
import os
from dotenv import load_dotenv

load_dotenv()

async def test_epics():
    """测试不同的 EPIC 格式"""
    api_key = os.getenv("CAPITAL_API_KEY")
    identifier = os.getenv("CAPITAL_IDENTIFIER")
    password = os.getenv("CAPITAL_API_PASSWORD")
    base_url = os.getenv("CAPITAL_BASE_API_URL", "https://demo-api-capital.backend-capital.com/api/v1")
    
    # 登录
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
        # 登录
        async with session.post(login_url, headers=headers, json=payload) as response:
            if response.status != 200:
                print(f"登录失败: {response.status}")
                return
            
            cst = response.headers.get("CST")
            x_security_token = response.headers.get("X-SECURITY-TOKEN")
            print("登录成功")
        
        # 测试不同的 EPIC 格式
        test_symbols = [
            "AAPL",          # Apple
            "AAPL.US",       
            "US.AAPL",
            "APPLE.US_ALL",
            "MSFT",          # Microsoft
            "MSFT.US",
            "US.MSFT",
            "MICROSOFT.US_ALL",
            "TSLA",          # Tesla
            "TSLA.US",
            "US.TSLA",
            "TESLA.US_ALL",
            "US500",         # S&P 500
            "GOLD",          # 黄金
            "EURUSD",        # 欧元/美元
        ]
        
        headers = {
            "X-CAP-API-KEY": api_key,
            "CST": cst,
            "X-SECURITY-TOKEN": x_security_token,
            "Content-Type": "application/json"
        }
        
        print("\n测试不同的 EPIC 格式:")
        print("-" * 50)
        
        for epic in test_symbols:
            url = f"{base_url}/markets/{epic}"
            try:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        instrument = data.get('instrument', {})
                        print(f"✓ {epic} - 成功! 名称: {instrument.get('name', 'N/A')}")
                    else:
                        print(f"✗ {epic} - 失败 ({response.status})")
            except Exception as e:
                print(f"✗ {epic} - 错误: {e}")
        
        # 搜索市场
        print("\n\n搜索 Apple 相关市场:")
        print("-" * 50)
        
        search_url = f"{base_url}/markets"
        params = {"searchTerm": "Apple"}
        
        try:
            async with session.get(search_url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    markets = data.get('markets', [])
                    for market in markets[:5]:  # 只显示前5个结果
                        print(f"EPIC: {market.get('epic')} - 名称: {market.get('instrumentName')}")
                else:
                    print(f"搜索失败: {response.status}")
        except Exception as e:
            print(f"搜索错误: {e}")

if __name__ == "__main__":
    asyncio.run(test_epics()) 