#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Capital.com API connection
測試 Capital.com API 連接
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

# Get credentials
API_KEY = os.getenv('CAPITAL_API_KEY')
IDENTIFIER = os.getenv('CAPITAL_IDENTIFIER')
PASSWORD = os.getenv('CAPITAL_API_PASSWORD')

print("Testing Capital.com API Connection")
print("=" * 50)
print(f"API Key: {API_KEY[:4]}...{API_KEY[-4:] if API_KEY else 'NOT FOUND'}")
print(f"Identifier: {IDENTIFIER}")
print(f"Password: {'*' * len(PASSWORD) if PASSWORD else 'NOT FOUND'}")
print("=" * 50)

# Test both demo and live endpoints
endpoints = [
    ("Demo", "https://demo-api-capital.backend-capital.com"),
    ("Live", "https://api-capital.backend-capital.com")
]

for env_name, base_url in endpoints:
    print(f"\nTesting {env_name} environment...")
    print(f"URL: {base_url}/session")
    
    headers = {
        "X-CAP-API-KEY": API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "identifier": IDENTIFIER,
        "password": PASSWORD
    }
    
    print("Request Headers:")
    print(f"  X-CAP-API-KEY: {API_KEY[:8]}...")
    print(f"  Content-Type: application/json")
    print("Request Payload:")
    print(f"  identifier: {IDENTIFIER}")
    print(f"  password: {'*' * 8}")
    
    try:
        response = requests.post(
            f"{base_url}/session",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Body: {response.text}")
        
        if response.status_code == 200:
            print(f"SUCCESS! CST Token: {response.headers.get('CST', 'Not found')}")
            print(f"X-SECURITY-TOKEN: {response.headers.get('X-SECURITY-TOKEN', 'Not found')}")
        else:
            print(f"FAILED!")
            # Try to parse error details
            try:
                error_json = response.json()
                print(f"Error Code: {error_json.get('errorCode', 'Unknown')}")
                print(f"Error Message: {error_json.get('message', 'No message')}")
            except:
                pass
            
    except Exception as e:
        print(f"ERROR: {e}")

print("\n" + "=" * 50)
print("Additional API Information:")
print("1. Make sure your API key is for the correct environment (demo/live)")
print("2. Check if your API key is active and not expired")
print("3. Verify that the identifier is your email or account number")
print("4. For demo accounts, use demo-api-capital.backend-capital.com")