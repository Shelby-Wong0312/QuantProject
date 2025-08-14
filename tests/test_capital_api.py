"""
Capital.com API Connection Test
檢查 Capital.com API 串接是否正常
"""

import requests
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import time

class CapitalComAPITester:
    """Capital.com API 連接測試器"""
    
    def __init__(self):
        # Capital.com API endpoints
        self.demo_api_url = "https://demo-api-capital.backend-capital.com"
        self.live_api_url = "https://api-capital.backend-capital.com"
        
        # 使用 demo 環境進行測試
        self.base_url = self.demo_api_url
        
        # API credentials (需要替換為實際的認證資訊)
        self.api_key = os.environ.get('CAPITAL_API_KEY', 'YOUR_API_KEY')
        self.password = os.environ.get('CAPITAL_PASSWORD', 'YOUR_PASSWORD')
        self.identifier = os.environ.get('CAPITAL_IDENTIFIER', 'YOUR_EMAIL')
        
        self.session_token = None
        self.cst = None
        self.headers = {
            'Content-Type': 'application/json',
            'X-CAP-API-KEY': self.api_key
        }
        
    def test_connection(self) -> Dict:
        """測試基本連接"""
        print("\n" + "="*60)
        print("CAPITAL.COM API 連接測試")
        print("="*60)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'summary': {}
        }
        
        # Test 1: 檢查 API 端點可達性
        print("\n[測試 1] 檢查 API 端點可達性...")
        try:
            response = requests.get(f"{self.base_url}/api/v1/ping", timeout=10)
            if response.status_code == 200:
                print("✅ API 端點可達")
                results['tests']['endpoint_reachable'] = True
            else:
                print(f"❌ API 端點回應異常: {response.status_code}")
                results['tests']['endpoint_reachable'] = False
        except Exception as e:
            print(f"❌ 無法連接到 API: {e}")
            results['tests']['endpoint_reachable'] = False
            
        # Test 2: 測試公開市場數據端點
        print("\n[測試 2] 測試公開市場數據...")
        try:
            # 獲取市場列表（某些端點可能不需要認證）
            response = requests.get(
                f"{self.base_url}/api/v1/markets",
                headers={'X-CAP-API-KEY': self.api_key},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 成功獲取市場數據")
                results['tests']['public_data'] = True
            elif response.status_code == 401:
                print("⚠️ 需要認證才能訪問市場數據")
                results['tests']['public_data'] = False
            else:
                print(f"❌ 市場數據請求失敗: {response.status_code}")
                results['tests']['public_data'] = False
        except Exception as e:
            print(f"❌ 市場數據請求錯誤: {e}")
            results['tests']['public_data'] = False
            
        # Test 3: 測試認證（如果有認證資訊）
        print("\n[測試 3] 測試 API 認證...")
        if self.api_key != 'YOUR_API_KEY':
            success = self.authenticate()
            results['tests']['authentication'] = success
            
            if success:
                print("✅ API 認證成功")
                
                # Test 4: 測試獲取帳戶資訊
                print("\n[測試 4] 測試獲取帳戶資訊...")
                account_info = self.get_account_info()
                if account_info:
                    print("✅ 成功獲取帳戶資訊")
                    results['tests']['account_info'] = True
                else:
                    print("❌ 無法獲取帳戶資訊")
                    results['tests']['account_info'] = False
                    
                # Test 5: 測試獲取即時價格
                print("\n[測試 5] 測試獲取即時價格...")
                price_data = self.get_market_price("AAPL")
                if price_data:
                    print(f"✅ 成功獲取 AAPL 價格")
                    results['tests']['market_price'] = True
                else:
                    print("❌ 無法獲取市場價格")
                    results['tests']['market_price'] = False
            else:
                print("❌ API 認證失敗")
        else:
            print("⚠️ 未設置 API 認證資訊")
            print("請設置環境變數:")
            print("  - CAPITAL_API_KEY")
            print("  - CAPITAL_PASSWORD")
            print("  - CAPITAL_IDENTIFIER")
            results['tests']['authentication'] = False
            
        # 生成測試摘要
        total_tests = len(results['tests'])
        passed_tests = sum(1 for v in results['tests'].values() if v)
        
        results['summary'] = {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': total_tests - passed_tests,
            'success_rate': f"{(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%"
        }
        
        # 顯示測試摘要
        print("\n" + "="*60)
        print("測試摘要")
        print("="*60)
        print(f"總測試數: {results['summary']['total_tests']}")
        print(f"通過: {results['summary']['passed']}")
        print(f"失敗: {results['summary']['failed']}")
        print(f"成功率: {results['summary']['success_rate']}")
        
        # 保存測試結果
        with open('capital_api_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\n測試結果已保存至: capital_api_test_results.json")
        
        return results
        
    def authenticate(self) -> bool:
        """進行 API 認證"""
        try:
            auth_data = {
                "identifier": self.identifier,
                "password": self.password
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/session",
                headers=self.headers,
                json=auth_data,
                timeout=10
            )
            
            if response.status_code == 200:
                self.cst = response.headers.get('CST')
                self.session_token = response.headers.get('X-SECURITY-TOKEN')
                
                # 更新 headers
                self.headers.update({
                    'CST': self.cst,
                    'X-SECURITY-TOKEN': self.session_token
                })
                return True
            else:
                print(f"認證失敗: {response.status_code}")
                if response.text:
                    print(f"錯誤訊息: {response.text}")
                return False
                
        except Exception as e:
            print(f"認證錯誤: {e}")
            return False
            
    def get_account_info(self) -> Optional[Dict]:
        """獲取帳戶資訊"""
        if not self.session_token:
            return None
            
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/accounts",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except Exception as e:
            print(f"獲取帳戶資訊錯誤: {e}")
            return None
            
    def get_market_price(self, symbol: str) -> Optional[Dict]:
        """獲取市場價格"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/markets/{symbol}",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except Exception as e:
            print(f"獲取價格錯誤: {e}")
            return None
            
    def test_websocket_connection(self) -> bool:
        """測試 WebSocket 連接（用於即時數據）"""
        print("\n[測試 6] 測試 WebSocket 連接...")
        
        # Capital.com WebSocket endpoint
        ws_url = "wss://demo-api-streaming.backend-capital.com"
        
        try:
            import websocket
            
            def on_open(ws):
                print("✅ WebSocket 連接成功")
                ws.close()
                
            def on_error(ws, error):
                print(f"❌ WebSocket 錯誤: {error}")
                
            def on_close(ws):
                print("WebSocket 連接已關閉")
                
            ws = websocket.WebSocketApp(
                ws_url,
                on_open=on_open,
                on_error=on_error,
                on_close=on_close
            )
            
            # 設置超時
            ws.run_forever(timeout=5)
            return True
            
        except ImportError:
            print("⚠️ 需要安裝 websocket-client 來測試 WebSocket")
            print("執行: pip install websocket-client")
            return False
        except Exception as e:
            print(f"❌ WebSocket 測試失敗: {e}")
            return False

def check_api_documentation():
    """檢查 API 文檔連結"""
    print("\n" + "="*60)
    print("Capital.com API 資源")
    print("="*60)
    
    resources = {
        "API 文檔": "https://open-api.capital.com/",
        "開發者入口": "https://capital.com/trading-api",
        "API 狀態": "https://status.capital.com/",
        "技術支援": "https://help.capital.com/hc/en-gb/sections/360004351917-API"
    }
    
    for name, url in resources.items():
        print(f"{name}: {url}")
        
    print("\n註冊 API 步驟:")
    print("1. 訪問 https://capital.com")
    print("2. 註冊/登入帳戶")
    print("3. 前往 Settings > API")
    print("4. 生成 API Key")
    print("5. 設置環境變數或更新配置檔案")

def main():
    """主測試函數"""
    print("\n" + "="*80)
    print("CAPITAL.COM API 連接狀態檢查")
    print("="*80)
    print(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 顯示 API 資源
    check_api_documentation()
    
    # 執行連接測試
    tester = CapitalComAPITester()
    results = tester.test_connection()
    
    # 測試 WebSocket
    tester.test_websocket_connection()
    
    print("\n" + "="*80)
    print("測試完成")
    print("="*80)
    
    # 建議
    print("\n建議:")
    if not results['tests'].get('authentication', False):
        print("1. 請先在 Capital.com 註冊並獲取 API 認證資訊")
        print("2. 設置環境變數:")
        print("   set CAPITAL_API_KEY=your_api_key")
        print("   set CAPITAL_PASSWORD=your_password")
        print("   set CAPITAL_IDENTIFIER=your_email")
    else:
        print("✅ API 連接正常，可以開始使用交易功能")
        
    return results

if __name__ == "__main__":
    main()