#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check Demo Mode Configuration
"""

import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Check raw environment variable
demo_mode_raw = os.getenv('CAPITAL_DEMO_MODE')
print(f"CAPITAL_DEMO_MODE from .env: '{demo_mode_raw}'")

# Import config to see processed value
import config

print(f"\nProcessed values in config.py:")
print(f"CAPITAL_DEMO_MODE: {config.CAPITAL_DEMO_MODE} (type: {type(config.CAPITAL_DEMO_MODE).__name__})")
print(f"CAPITAL_API_URL: {config.CAPITAL_API_URL}")

# Check if using demo URL
if config.CAPITAL_DEMO_MODE:
    print("\nDemo Mode: ENABLED")
    if "demo-api-capital" in config.CAPITAL_API_URL:
        print("✓ Using Demo API URL correctly")
    else:
        print("✗ ERROR: Not using Demo URL!")
else:
    print("\nDemo Mode: DISABLED (Live mode)")

# Check LiveDataFeed
print("\nChecking LiveDataFeed...")
from src.data_pipeline.live_feed import LiveDataFeed

class DummyQueue:
    async def put_event(self, event):
        pass

feed = LiveDataFeed(symbols=['TEST'], event_queue=DummyQueue())
print(f"LiveDataFeed will use: {feed.base_url}")

if config.CAPITAL_DEMO_MODE and "demo-api-capital" in feed.base_url:
    print("\n✓ CONFIRMED: System will use Demo API")
else:
    print("\n✗ WARNING: System NOT configured for Demo API")