#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Capital.com API for BTC data
"""

import requests
import os
from dotenv import load_dotenv
from datetime import datetime
import json

# Load environment variables
load_dotenv()

print("\n" + "="*50)
print(" Capital.com BTC/Crypto Test ")
print("="*50)
print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# API credentials
API_KEY = os.getenv('CAPITAL_API_KEY')
IDENTIFIER = os.getenv('CAPITAL_IDENTIFIER')
PASSWORD = os.getenv('CAPITAL_PASSWORD')
DEMO_MODE = os.getenv('CAPITAL_DEMO_MODE', 'True').lower() == 'true'

# API URL
base_url = 'https://demo-api-capital.backend-capital.com' if DEMO_MODE else 'https://api-capital.backend-capital.com'

print(f"\nUsing {'DEMO' if DEMO_MODE else 'LIVE'} API")

# Login
print("\n1. Logging in...")
session = requests.Session()

login_data = {
    'identifier': IDENTIFIER,
    'password': PASSWORD
}

headers = {
    'X-CAP-API-KEY': API_KEY,
    'Content-Type': 'application/json'
}

response = session.post(
    f'{base_url}/api/v1/session',
    json=login_data,
    headers=headers
)

if response.status_code == 200:
    print("   [SUCCESS] Logged in")
    cst = response.headers.get('CST')
    x_security_token = response.headers.get('X-SECURITY-TOKEN')
    
    # Update headers for future requests
    session.headers.update({
        'X-SECURITY-TOKEN': x_security_token,
        'CST': cst
    })
else:
    print(f"   [ERROR] Login failed: {response.status_code}")
    print(response.text)
    exit(1)

# Search for crypto markets
print("\n2. Searching for crypto markets...")

search_terms = ['Bitcoin', 'BTC', 'Ethereum', 'ETH', 'Crypto']
crypto_markets = []

for term in search_terms:
    response = session.get(
        f'{base_url}/api/v1/markets',
        params={'searchTerm': term}
    )
    
    if response.status_code == 200:
        data = response.json()
        markets = data.get('markets', [])
        
        for market in markets:
            if market not in crypto_markets:
                crypto_markets.append(market)
                
print(f"   Found {len(crypto_markets)} crypto markets")

# Display crypto markets
if crypto_markets:
    print("\n3. Available crypto markets:")
    for market in crypto_markets[:10]:  # Show first 10
        epic = market.get('epic', '')
        name = market.get('instrumentName', '')
        print(f"   {epic}: {name}")
    
    # Get prices for first crypto
    if crypto_markets:
        first_crypto = crypto_markets[0]
        epic = first_crypto['epic']
        
        print(f"\n4. Getting price for {first_crypto['instrumentName']}...")
        
        response = session.get(
            f'{base_url}/api/v1/markets/{epic}'
        )
        
        if response.status_code == 200:
            data = response.json()
            snapshot = data.get('snapshot', {})
            
            print(f"   Bid: {snapshot.get('bid')}")
            print(f"   Ask: {snapshot.get('offer')}")
            print(f"   Spread: {snapshot.get('spread')}")
            print(f"   Change: {snapshot.get('percentageChange')}%")
else:
    print("\n3. No crypto markets found, trying forex...")
    
    # Try forex
    response = session.get(
        f'{base_url}/api/v1/markets',
        params={'searchTerm': 'EUR/USD'}
    )
    
    if response.status_code == 200:
        data = response.json()
        markets = data.get('markets', [])
        
        if markets:
            market = markets[0]
            epic = market['epic']
            
            print(f"   Found: {market['instrumentName']}")
            
            # Get price
            response = session.get(
                f'{base_url}/api/v1/markets/{epic}'
            )
            
            if response.status_code == 200:
                data = response.json()
                snapshot = data.get('snapshot', {})
                
                print(f"   Bid: {snapshot.get('bid')}")
                print(f"   Ask: {snapshot.get('offer')}")

# Logout
print("\n5. Logging out...")
session.delete(f'{base_url}/api/v1/session')

print("\n" + "="*50)
print(" Test Complete ")
print("="*50)