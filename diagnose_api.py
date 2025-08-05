#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnose Capital.com API Issues
診斷 Capital.com API 問題
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

print("Capital.com API Detailed Diagnosis")
print("=" * 60)

# Get credentials
API_KEY = os.getenv('CAPITAL_API_KEY', '').strip('"')  # Remove quotes if present
IDENTIFIER = os.getenv('CAPITAL_IDENTIFIER', '').strip('"')
API_PASSWORD = os.getenv('CAPITAL_API_PASSWORD', '').strip('"')
DEMO_MODE = os.getenv('CAPITAL_DEMO_MODE', 'True').strip('"').lower() == 'true'

print(f"Environment Settings:")
print(f"  Demo Mode: {DEMO_MODE}")
print(f"  API Key: {API_KEY[:4]}...{API_KEY[-4:] if API_KEY else 'NOT FOUND'}")
print(f"  API Key Length: {len(API_KEY)}")
print(f"  Identifier: {IDENTIFIER}")
print(f"  Password Length: {len(API_PASSWORD) if API_PASSWORD else 0}")

print("\n" + "=" * 60)
print("Testing Different Authentication Methods...")

# Method 1: Standard authentication
print("\n1. Testing standard authentication (Demo):")
url = "https://demo-api-capital.backend-capital.com/session"
headers = {
    "X-CAP-API-KEY": API_KEY,
    "Content-Type": "application/json"
}
payload = {
    "identifier": IDENTIFIER,
    "password": API_PASSWORD
}

print(f"   URL: {url}")
print(f"   Headers: X-CAP-API-KEY={API_KEY[:8]}...")
print(f"   Payload: identifier={IDENTIFIER}, password=***")

try:
    response = requests.post(url, headers=headers, json=payload, timeout=10)
    print(f"   Status: {response.status_code}")
    if response.status_code != 200:
        print(f"   Error: {response.text}")
    else:
        print(f"   Success! CST: {response.headers.get('CST', 'Not found')}")
except Exception as e:
    print(f"   Exception: {e}")

# Method 2: Try without quotes in API key
print("\n2. Testing with cleaned API key:")
clean_api_key = API_KEY.replace('"', '').replace("'", '')
headers["X-CAP-API-KEY"] = clean_api_key
print(f"   Cleaned API Key: {clean_api_key[:8]}...")

try:
    response = requests.post(url, headers=headers, json=payload, timeout=10)
    print(f"   Status: {response.status_code}")
    if response.status_code != 200:
        print(f"   Error: {response.text}")
    else:
        print(f"   Success! CST: {response.headers.get('CST', 'Not found')}")
except Exception as e:
    print(f"   Exception: {e}")

# Method 3: Test encryption key endpoint
print("\n3. Testing encryption key endpoint:")
enc_url = "https://demo-api-capital.backend-capital.com/session/encryptionKey"
enc_headers = {
    "X-CAP-API-KEY": API_KEY,
    "Content-Type": "application/json"
}

try:
    response = requests.get(enc_url, headers=enc_headers, timeout=10)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.text[:100]}...")
except Exception as e:
    print(f"   Exception: {e}")

print("\n" + "=" * 60)
print("Diagnostic Summary:")
print("\nIf you're getting 'error.null.client.token', it usually means:")
print("1. The API key is invalid or expired")
print("2. The API key was created for live account but you're using demo URL")
print("3. The password is not the custom API password set during key creation")
print("4. 2FA was not enabled when creating the API key")
print("\nDemo Account Requirements:")
print("- You need a separate demo account (not just a live account)")
print("- Enable 2FA on the demo account")
print("- Generate API key specifically from the demo account")
print("- Use the custom password you set when creating the API key")