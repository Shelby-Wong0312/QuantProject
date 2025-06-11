import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import aiohttp
from dotenv import load_dotenv
from datetime import datetime
import time

load_dotenv()

async def monitor_system():
    """監控系統運行狀態"""
    api_key = "oVGhAub8ezuC9Zo1"
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
            print("✅ 系統監控已啟動")
            print("=" * 60)
        
        headers.update({
            "CST": cst,
            "X-SECURITY-TOKEN": x_security_token
        })
        
        last_price = 0
        position_count = 0
        
        while True:
            try:
                # 清屏（Windows）
                os.system('cls' if os.name == 'nt' else 'clear')
                
                print(f"🔄 系統監控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 60)
                
                # 1. 獲取 BTC 價格
                btc_url = f"{base_url}/markets/BTCUSD"
                async with session.get(btc_url, headers=headers) as response:
                    if response.status == 200:
                        btc_data = await response.json()
                        snapshot = btc_data.get('snapshot', {})
                        bid = float(snapshot.get('bid', 0))
                        ask = float(snapshot.get('offer', 0))
                        
                        print(f"\n💰 BTCUSD 實時價格:")
                        print(f"   買價: ${bid:,.2f}")
                        print(f"   賣價: ${ask:,.2f}")
                        
                        # 計算價格變化
                        if last_price > 0:
                            change = bid - last_price
                            change_pct = (change / last_price) * 100
                            if change > 0:
                                print(f"   變化: ↑ ${change:,.2f} ({change_pct:+.2f}%)")
                            elif change < 0:
                                print(f"   變化: ↓ ${change:,.2f} ({change_pct:+.2f}%)")
                            else:
                                print(f"   變化: → $0.00 (0.00%)")
                        
                        last_price = bid
                
                # 2. 獲取賬戶信息
                accounts_url = f"{base_url}/accounts"
                async with session.get(accounts_url, headers=headers) as response:
                    if response.status == 200:
                        accounts = await response.json()
                        for account in accounts.get('accounts', []):
                            balance = account.get('balance', {})
                            print(f"\n📊 賬戶狀態:")
                            print(f"   餘額: ${balance.get('balance', 0):,.2f}")
                            print(f"   可用: ${balance.get('available', 0):,.2f}")
                            print(f"   盈虧: ${balance.get('profitLoss', 0):,.2f}")
                
                # 3. 獲取持倉
                positions_url = f"{base_url}/positions"
                async with session.get(positions_url, headers=headers) as response:
                    if response.status == 200:
                        positions_data = await response.json()
                        positions = positions_data.get('positions', [])
                        
                        print(f"\n📈 持倉狀態:")
                        if positions:
                            position_count = len(positions)
                            print(f"   持倉數量: {position_count}")
                            for pos in positions:
                                market = pos.get('market', {})
                                position = pos.get('position', {})
                                print(f"\n   品種: {market.get('epic')}")
                                print(f"   方向: {position.get('direction')}")
                                print(f"   數量: {position.get('size')}")
                                print(f"   開倉價: ${position.get('level'):,.2f}")
                                print(f"   當前盈虧: ${position.get('profit', 0):,.2f}")
                        else:
                            print("   暫無持倉")
                
                print("\n" + "=" * 60)
                print("按 Ctrl+C 停止監控")
                
                # 等待5秒再更新
                await asyncio.sleep(5)
                
            except KeyboardInterrupt:
                print("\n監控已停止")
                break
            except Exception as e:
                print(f"\n錯誤: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(monitor_system())
    except KeyboardInterrupt:
        print("\n監控已停止") 