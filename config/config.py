# -*- coding: utf-8 -*-
# quant_project/config.py
# FINAL VERSION - with .env debugging

import os
import sys
from dotenv import load_dotenv, find_dotenv
import logging

# Setup logger
logger = logging.getLogger(__name__)


# Force UTF-8 encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# --- .env file loading and debugging ---
print("--- Starting to load .env file ---")
# Find .env file path
env_file_path = find_dotenv()

if env_file_path:
    logger.info(f"✓ Found .env file at: {env_file_path}")
    # verbose=True will print detailed loading process
    load_dotenv(dotenv_path=env_file_path, verbose=True)
else:
    logger.warning(".env file not found in project directory!")

print("--- .env file loading completed ---")

# --- API Configuration ---
CAPITAL_API_KEY = os.getenv('CAPITAL_API_KEY')
CAPITAL_IDENTIFIER = os.getenv('CAPITAL_IDENTIFIER')
CAPITAL_API_PASSWORD = os.getenv('CAPITAL_API_PASSWORD')
CAPITAL_PASSWORD = CAPITAL_API_PASSWORD  # Backward compatibility
CAPITAL_DEMO_MODE = os.getenv('CAPITAL_DEMO_MODE', 'True').lower() == 'true'

# Use demo or live API URL based on mode
if CAPITAL_DEMO_MODE:
    CAPITAL_API_URL = os.getenv('CAPITAL_API_URL', 'https://demo-api-capital.backend-capital.com')
else:
    CAPITAL_API_URL = os.getenv('CAPITAL_API_URL', 'https://api-capital.backend-capital.com')

# Check if API credentials are loaded
if not CAPITAL_API_KEY or not CAPITAL_API_PASSWORD:
    logger.error("API credentials not found!")
    print("Please create a .env file with:")
    print("  CAPITAL_API_KEY=your_api_key")
    print("  CAPITAL_API_PASSWORD=your_password")
    sys.exit(1)
else:
    logger.info(f" API Key loaded: {CAPITAL_API_KEY[:4]}...{CAPITAL_API_KEY[-4:]}")
    logger.info("✓" +  Demo Mode: {CAPITAL_DEMO_MODE}")
    logger.info("✓" +  API URL: {CAPITAL_API_URL}")
    logger.info("✓" +  Identifier: {CAPITAL_IDENTIFIER}")

# --- Trading Configuration ---
SYMBOLS_TO_TRADE = ['EUR/USD', 'Apple', 'Gold']

# --- Logging ---
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# --- Strategy Parameters ---
STRATEGY_PARAMS = {
    'Comprehensive_v1': {
        'lookback_periods': {
            'ema_short': 20,
            'ema_long': 50,
            'atr_period': 14,
            'rsi_period': 14
        },
        'thresholds': {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'volume_spike': 1.5,
            'risk_percent': 0.02
        }
    }
}

# --- Portfolio Configuration ---
INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '100000'))
BASE_POSITION_SIZE = float(os.getenv('BASE_POSITION_SIZE', '0.05'))

print("--- Configuration loaded successfully ---")