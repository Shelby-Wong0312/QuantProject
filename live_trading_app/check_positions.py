import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import aiohttp
from dotenv import load_dotenv
import json

load_dotenv()

async def check_account():
    """檢查 Capital.com 賬戶狀態"""
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
                return
            
            cst = response.headers.get("CST")
            x_security_token = response.headers.get("X-SECURITY-TOKEN")
            print("✅ 登錄成功")
        
        # 更新 headers
        headers.update({
            "CST": cst,
            "X-SECURITY-TOKEN": x_security_token
        })
        
        # 1. 獲取賬戶信息
        print("\n📊 賬戶信息:")
        print("-" * 50)
        accounts_url = f"{base_url}/accounts"
        async with session.get(accounts_url, headers=headers) as response:
            if response.status == 200:
                accounts = await response.json()
                for account in accounts.get('accounts', []):
                    print(f"賬戶ID: {account.get('accountId')}")
                    print(f"賬戶類型: {account.get('accountType')}")
                    balance = account.get('balance', {})
                    print(f"餘額: ${balance.get('balance', 0):,.2f}")
                    print(f"可用資金: ${balance.get('available', 0):,.2f}")
                    print(f"盈虧: ${balance.get('profitLoss', 0):,.2f}")
                    print()
        
        # 2. 獲取持倉
        print("\n📈 當前持倉:")
        print("-" * 50)
        positions_url = f"{base_url}/positions"
        async with session.get(positions_url, headers=headers) as response:
            if response.status == 200:
                positions_data = await response.json()
                positions = positions_data.get('positions', [])
                if positions:
                    for pos in positions:
                        print(f"品種: {pos.get('market', {}).get('epic')}")
                        print(f"方向: {pos.get('position', {}).get('direction')}")
                        print(f"數量: {pos.get('position', {}).get('size')}")
                        print(f"開倉價: {pos.get('position', {}).get('level')}")
                        print(f"當前價: {pos.get('market', {}).get('bid')}")
                        print(f"盈虧: ${pos.get('position', {}).get('profit', 0):,.2f}")
                        print("-" * 30)
                else:
                    print("沒有持倉")
        
        # 3. 獲取 BTCUSD 當前價格
        print("\n💰 BTCUSD 當前價格:")
        print("-" * 50)
        btc_url = f"{base_url}/markets/BTCUSD"
        async with session.get(btc_url, headers=headers) as response:
            if response.status == 200:
                btc_data = await response.json()
                snapshot = btc_data.get('snapshot', {})
                print(f"買價: ${snapshot.get('bid', 0):,.2f}")
                print(f"賣價: ${snapshot.get('offer', 0):,.2f}")
                print(f"最高: ${snapshot.get('high', 0):,.2f}")
                print(f"最低: ${snapshot.get('low', 0):,.2f}")

if __name__ == "__main__":
    asyncio.run(check_account()) 