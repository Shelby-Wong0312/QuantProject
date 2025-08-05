#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check Capital.com API Key
檢查 Capital.com API 金鑰
"""

import os
import sys
import io
import requests
import json
from dotenv import load_dotenv

# Force UTF-8 encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Load environment variables
load_dotenv()

print("Capital.com API Key Verification")
print("=" * 50)
print("\nIMPORTANT NOTES:")
print("1. The API password is NOT your account password")
print("2. It's a custom password you set when creating the API key")
print("3. You need 2FA enabled on your account")
print("4. API keys can expire based on the date you set")
print("\n" + "=" * 50)

# Get credentials
API_KEY = os.getenv('CAPITAL_API_KEY')
IDENTIFIER = os.getenv('CAPITAL_IDENTIFIER')
API_PASSWORD = os.getenv('CAPITAL_API_PASSWORD')

print(f"\nCredentials from .env:")
print(f"API Key: {API_KEY[:4]}...{API_KEY[-4:] if API_KEY else 'NOT FOUND'}")
print(f"Identifier: {IDENTIFIER}")
print(f"API Password: {'Set' if API_PASSWORD else 'NOT FOUND'}")

# Try a minimal test with demo endpoint
print("\n" + "=" * 50)
print("Testing minimal authentication...")

url = "https://demo-api-capital.backend-capital.com/session"
headers = {
    "X-CAP-API-KEY": API_KEY,
    "Content-Type": "application/json"
}
payload = {
    "identifier": IDENTIFIER,
    "password": API_PASSWORD
}

print(f"\nRequest URL: {url}")
print(f"Request Method: POST")

try:
    response = requests.post(url, headers=headers, json=payload, timeout=10)
    
    print(f"\nResponse Status: {response.status_code}")
    
    if response.status_code == 200:
        print("SUCCESS! Authentication successful")
        print(f"CST Token: {response.headers.get('CST', 'Not found')}")
        print(f"X-SECURITY-TOKEN: {response.headers.get('X-SECURITY-TOKEN', 'Not found')}")
    else:
        print("FAILED! Authentication failed")
        try:
            error_data = response.json()
            print(f"Error: {json.dumps(error_data, indent=2)}")
        except:
            print(f"Raw response: {response.text}")
            
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 50)
print("\nTroubleshooting steps:")
print("1. Log in to your Capital.com account")
print("2. Go to Settings > API integrations")
print("3. Check if your API key is active and not expired")
print("4. If needed, generate a new API key with a custom password")
print("5. Make sure to save the API key and custom password")
print("6. Update the .env file with the new credentials")
print("\nNote: The password for API authentication is the custom")
print("password you set when creating the API key, NOT your")
print("Capital.com account password.")