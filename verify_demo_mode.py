#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Verify Capital.com Demo Mode Configuration
驗證 Capital.com Demo 模式設定
"""

import os
import sys
import io
from dotenv import load_dotenv

# Force UTF-8 encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print("=== Capital.com Demo Mode Verification ===\n")

# Step 1: Check .env file
print("1. Checking .env file...")
load_dotenv(verbose=True)

# Step 2: Check environment variables
print("\n2. Environment Variables:")
demo_mode_env = os.getenv('CAPITAL_DEMO_MODE')
print(f"   CAPITAL_DEMO_MODE from .env: '{demo_mode_env}'")

# Step 3: Import config and check parsed values
print("\n3. Importing config.py...")
import config

print(f"\n4. Config.py Values:")
print(f"   CAPITAL_DEMO_MODE: {config.CAPITAL_DEMO_MODE} (type: {type(config.CAPITAL_DEMO_MODE)})")
print(f"   CAPITAL_API_URL: {config.CAPITAL_API_URL}")

# Step 4: Expected values
print(f"\n5. Verification:")
expected_demo_url = "https://demo-api-capital.backend-capital.com"
expected_live_url = "https://api-capital.backend-capital.com"

if config.CAPITAL_DEMO_MODE:
    print("   ✓ Demo Mode is ENABLED")
    if config.CAPITAL_API_URL == expected_demo_url:
        print(f"   ✓ Using correct Demo URL: {expected_demo_url}")
    else:
        print(f"   ✗ ERROR: Wrong URL! Expected: {expected_demo_url}")
        print(f"            But got: {config.CAPITAL_API_URL}")
else:
    print("   ✗ Demo Mode is DISABLED (Live mode)")
    if config.CAPITAL_API_URL == expected_live_url:
        print(f"   ✓ Using correct Live URL: {expected_live_url}")
    else:
        print(f"   ✗ ERROR: Wrong URL! Expected: {expected_live_url}")
        print(f"            But got: {config.CAPITAL_API_URL}")

# Step 5: Check data feed configuration
print("\n6. Checking LiveDataFeed configuration...")
from src.data_pipeline.live_feed import LiveDataFeed

# Create a dummy event queue to test
class DummyEventQueue:
    async def put_event(self, event):
        pass

feed = LiveDataFeed(symbols=['TEST'], event_queue=DummyEventQueue())
print(f"   LiveDataFeed base_url: {feed.base_url}")
print(f"   LiveDataFeed API key: {feed.api_key[:4]}...{feed.api_key[-4:] if feed.api_key else 'None'}")

# Step 6: Test actual request URL
print("\n7. Testing actual request URL that will be used:")
login_url = f"{feed.base_url}/session"
print(f"   Login URL: {login_url}")

if "demo-api-capital" in login_url:
    print("   ✓ Will use DEMO API endpoint")
elif "api-capital" in login_url:
    print("   ⚠ Will use LIVE API endpoint")
else:
    print("   ✗ Unknown API endpoint!")

print("\n=== Verification Complete ===")
print("\nSummary:")
if config.CAPITAL_DEMO_MODE and config.CAPITAL_API_URL == expected_demo_url:
    print("✓ System is correctly configured for DEMO mode")
else:
    print("✗ Configuration issue detected - please check settings")