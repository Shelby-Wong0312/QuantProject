#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Verify Capital.com API Setup
驗證 Capital.com API 設定
"""

import os
import sys
import io
import requests
from dotenv import load_dotenv

# Force UTF-8 encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Load environment variables
load_dotenv()

print("Capital.com API Setup Verification")
print("=" * 60)

# Get credentials
API_KEY = os.getenv('CAPITAL_API_KEY', '').strip('"')
IDENTIFIER = os.getenv('CAPITAL_IDENTIFIER', '').strip('"')
API_PASSWORD = os.getenv('CAPITAL_API_PASSWORD', '').strip('"')

print("\n重要提醒：")
print("1. 您需要從 Capital.com Demo 帳戶生成 API 金鑰")
print("2. Demo 帳戶網址：https://demo.capital.com/")
print("3. 生成 API 金鑰時，您需要設定一個專用密碼")
print("4. 這個專用密碼不是您的帳戶登入密碼")
print("\n" + "=" * 60)

print("\n當前設定：")
print(f"API Key: {API_KEY}")
print(f"Identifier: {IDENTIFIER}")
print(f"API Password: {'*' * len(API_PASSWORD) if API_PASSWORD else 'NOT SET'}")

print("\n檢查清單：")
print("[ ] 1. 您是否從 Demo 帳戶（不是 Live 帳戶）生成的 API 金鑰？")
print("[ ] 2. 您是否已在 Demo 帳戶中啟用 2FA？")
print("[ ] 3. API 密碼是您在生成金鑰時設定的專用密碼嗎？")
print("[ ] 4. API 金鑰是否在有效期內（檢查到期日）？")

print("\n" + "=" * 60)
print("\n測試最小認證請求...")

# Try minimal authentication
url = "https://demo-api-capital.backend-capital.com/session"
headers = {
    "X-CAP-API-KEY": API_KEY,
    "Content-Type": "application/json"
}
payload = {
    "identifier": IDENTIFIER,
    "password": API_PASSWORD
}

print(f"請求 URL: {url}")
print(f"請求頭: X-CAP-API-KEY={API_KEY}")

try:
    response = requests.post(url, headers=headers, json=payload, timeout=10)
    print(f"\n回應狀態碼: {response.status_code}")
    
    if response.status_code == 200:
        print("✓ 成功！API 認證通過")
        print(f"CST Token: {response.headers.get('CST', 'Not found')}")
        print(f"X-SECURITY-TOKEN: {response.headers.get('X-SECURITY-TOKEN', 'Not found')}")
    else:
        print("✗ 失敗！")
        error_data = response.json()
        print(f"錯誤代碼: {error_data.get('errorCode', 'Unknown')}")
        
        if error_data.get('errorCode') == 'error.null.client.token':
            print("\n可能的原因：")
            print("1. API 金鑰無效或已過期")
            print("2. API 金鑰是從 Live 帳戶生成的，但正在使用 Demo API")
            print("3. API 密碼不正確")
            print("4. 2FA 未在生成 API 金鑰前啟用")
            
except Exception as e:
    print(f"錯誤: {e}")

print("\n" + "=" * 60)
print("\n下一步：")
print("1. 登入您的 Capital.com Demo 帳戶")
print("2. 前往 Settings > API integrations")
print("3. 如果有舊的 API 金鑰，先刪除它")
print("4. 生成新的 API 金鑰並設定一個您記得的專用密碼")
print("5. 更新 .env 檔案中的 CAPITAL_API_PASSWORD")