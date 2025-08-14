"""
Capital.com API Connector
Complete integration for Capital.com trading API
"""

import requests
import json
import os
import time
import hashlib
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from Crypto.Cipher import PKCS1_v1_5
from Crypto.PublicKey import RSA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    bid: float
    ask: float
    high: float
    low: float
    change: float
    change_pct: float
    update_time: datetime

@dataclass
class Position:
    """Trading position structure"""
    position_id: str
    symbol: str
    direction: str  # BUY or SELL
    size: float
    open_price: float
    current_price: float
    pnl: float
    pnl_pct: float

@dataclass
class Order:
    """Trading order structure"""
    order_id: str
    symbol: str
    direction: str
    size: float
    order_type: str  # MARKET, LIMIT, STOP
    price: Optional[float]
    status: str

class CapitalComAPI:
    """Capital.com API connector"""
    
    def __init__(self, demo: bool = True):
        """
        Initialize Capital.com API connector
        
        Args:
            demo: Use demo environment if True, live if False
        """
        # Select environment
        if demo:
            self.base_url = "https://demo-api-capital.backend-capital.com"
            self.ws_url = "wss://demo-api-streaming.backend-capital.com"
        else:
            self.base_url = "https://api-capital.backend-capital.com"
            self.ws_url = "wss://api-streaming.backend-capital.com"
        
        # Load credentials from environment
        self.api_key = os.environ.get('CAPITAL_API_KEY', '')
        self.identifier = os.environ.get('CAPITAL_IDENTIFIER', '')
        self.password = os.environ.get('CAPITAL_API_PASSWORD', '')  # Fixed: was CAPITAL_PASSWORD
        
        # Session management
        self.session = requests.Session()
        self.cst = None
        self.security_token = None
        self.session_expires = None
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Initialize headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'X-CAP-API-KEY': self.api_key
        })
        
        logger.info(f"Capital.com API initialized ({'Demo' if demo else 'Live'} mode)")
    
    def _rate_limit(self):
        """Implement rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, method: str, endpoint: str, 
                     data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make API request with error handling
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint
            data: Request payload
            
        Returns:
            Response data or None if error
        """
        self._rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == 'GET':
                response = self.session.get(url)
            elif method == 'POST':
                response = self.session.post(url, json=data)
            elif method == 'PUT':
                response = self.session.put(url, json=data)
            elif method == 'DELETE':
                response = self.session.delete(url)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            # Check for successful response
            if response.status_code == 200:
                return response.json() if response.text else {}
            elif response.status_code == 401:
                logger.error("Authentication failed or session expired")
                # Try to re-authenticate
                if self.authenticate():
                    # Retry request
                    return self._make_request(method, endpoint, data)
            else:
                logger.error(f"Request failed: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return None
    
    def encrypt_password(self, password: str, encryption_key: str) -> str:
        """
        Encrypt password using RSA public key
        
        Args:
            password: Plain text password with timestamp
            encryption_key: RSA public key from API
            
        Returns:
            Base64 encoded encrypted password
        """
        try:
            # Add PEM headers if missing
            if not encryption_key.startswith('-----BEGIN'):
                encryption_key = f"-----BEGIN PUBLIC KEY-----\n{encryption_key}\n-----END PUBLIC KEY-----"
            
            # Import the public key
            key = RSA.import_key(encryption_key)
            cipher = PKCS1_v1_5.new(key)
            
            # Encrypt the password
            encrypted = cipher.encrypt(password.encode('utf-8'))
            
            # Return base64 encoded
            return base64.b64encode(encrypted).decode('utf-8')
        except Exception as e:
            logger.error(f"Password encryption failed: {e}")
            # Fallback for demo mode - return plain password
            if self.demo:
                logger.info("Using fallback for demo mode")
                return password
            return ""
    
    def authenticate(self) -> bool:
        """
        Authenticate with Capital.com API
        
        Returns:
            True if authentication successful
        """
        if not all([self.api_key, self.identifier, self.password]):
            logger.error("Missing API credentials")
            return False
        
        try:
            # Step 1: Get encryption key
            response = self.session.get(f"{self.base_url}/api/v1/session/encryptionKey")
            
            if response.status_code != 200:
                logger.error(f"Failed to get encryption key: {response.status_code}")
                return False
            
            encryption_data = response.json()
            encryption_key = encryption_data.get('encryptionKey')
            timestamp = encryption_data.get('timeStamp')
            
            # Step 2: Encrypt password
            encrypted_password = self.encrypt_password(
                f"{self.password}|{timestamp}",
                encryption_key
            )
            
            if not encrypted_password:
                logger.error("Password encryption failed")
                return False
            
            # Step 3: Create session
            auth_data = {
                "identifier": self.identifier,
                "password": encrypted_password,
                "encryptedPassword": True
            }
            
            response = self.session.post(
                f"{self.base_url}/api/v1/session",
                json=auth_data
            )
            
            if response.status_code == 200:
                # Extract session tokens
                self.cst = response.headers.get('CST')
                self.security_token = response.headers.get('X-SECURITY-TOKEN')
                
                # Update session headers
                self.session.headers.update({
                    'CST': self.cst,
                    'X-SECURITY-TOKEN': self.security_token
                })
                
                # Set session expiry (10 minutes)
                self.session_expires = datetime.now() + timedelta(minutes=10)
                
                logger.info("Authentication successful")
                return True
            else:
                logger.error(f"Authentication failed: {response.status_code}")
                if response.text:
                    logger.error(f"Error: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    def is_authenticated(self) -> bool:
        """Check if currently authenticated"""
        if not self.security_token:
            return False
        
        # Check if session expired
        if self.session_expires and datetime.now() > self.session_expires:
            logger.info("Session expired, re-authenticating...")
            return self.authenticate()
        
        return True
    
    def get_accounts(self) -> Optional[List[Dict]]:
        """
        Get trading accounts
        
        Returns:
            List of account information
        """
        if not self.is_authenticated():
            if not self.authenticate():
                return None
        
        data = self._make_request('GET', '/api/v1/accounts')
        if data:
            return data.get('accounts', [])
        return None
    
    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """
        Get current market data for a symbol
        
        Args:
            symbol: Market symbol (e.g., 'AAPL')
            
        Returns:
            MarketData object or None
        """
        if not self.is_authenticated():
            if not self.authenticate():
                return None
        
        data = self._make_request('GET', f'/api/v1/markets/{symbol}')
        
        if data and 'snapshot' in data:
            snapshot = data['snapshot']
            return MarketData(
                symbol=symbol,
                bid=float(snapshot.get('bid', 0)),
                ask=float(snapshot.get('offer', 0)),
                high=float(snapshot.get('high', 0)),
                low=float(snapshot.get('low', 0)),
                change=float(snapshot.get('netChange', 0)),
                change_pct=float(snapshot.get('percentageChange', 0)),
                update_time=datetime.now()
            )
        return None
    
    def get_positions(self) -> Optional[List[Position]]:
        """
        Get all open positions
        
        Returns:
            List of Position objects
        """
        if not self.is_authenticated():
            if not self.authenticate():
                return None
        
        data = self._make_request('GET', '/api/v1/positions')
        
        if data and 'positions' in data:
            positions = []
            for pos in data['positions']:
                positions.append(Position(
                    position_id=pos.get('dealId'),
                    symbol=pos.get('epic'),
                    direction=pos.get('direction'),
                    size=float(pos.get('size', 0)),
                    open_price=float(pos.get('openLevel', 0)),
                    current_price=float(pos.get('level', 0)),
                    pnl=float(pos.get('profit', 0)),
                    pnl_pct=float(pos.get('percentageChange', 0))
                ))
            return positions
        return None
    
    def place_order(self, symbol: str, direction: str, size: float,
                   order_type: str = 'MARKET', 
                   price: Optional[float] = None,
                   stop_loss: Optional[float] = None,
                   take_profit: Optional[float] = None) -> Optional[str]:
        """
        Place a trading order
        
        Args:
            symbol: Market symbol
            direction: BUY or SELL
            size: Position size
            order_type: MARKET, LIMIT, or STOP
            price: Limit/stop price (required for non-market orders)
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Order ID if successful
        """
        if not self.is_authenticated():
            if not self.authenticate():
                return None
        
        order_data = {
            "epic": symbol,
            "direction": direction,
            "size": size,
            "orderType": order_type,
            "guaranteedStop": False,
            "forceOpen": True
        }
        
        # Add price for limit/stop orders
        if order_type != 'MARKET' and price:
            order_data["level"] = price
        
        # Add stop loss
        if stop_loss:
            order_data["stopLevel"] = stop_loss
        
        # Add take profit
        if take_profit:
            order_data["profitLevel"] = take_profit
        
        data = self._make_request('POST', '/api/v1/positions', order_data)
        
        if data and 'dealReference' in data:
            logger.info(f"Order placed: {data['dealReference']}")
            return data['dealReference']
        
        return None
    
    def close_position(self, position_id: str) -> bool:
        """
        Close an open position
        
        Args:
            position_id: Position ID to close
            
        Returns:
            True if successful
        """
        if not self.is_authenticated():
            if not self.authenticate():
                return False
        
        data = self._make_request('DELETE', f'/api/v1/positions/{position_id}')
        return data is not None
    
    def get_historical_prices(self, symbol: str, resolution: str = 'DAY',
                            max_results: int = 100) -> Optional[List[Dict]]:
        """
        Get historical price data
        
        Args:
            symbol: Market symbol
            resolution: MINUTE, MINUTE_5, MINUTE_15, MINUTE_30, HOUR, HOUR_4, DAY, WEEK
            max_results: Maximum number of data points
            
        Returns:
            List of price data
        """
        if not self.is_authenticated():
            if not self.authenticate():
                return None
        
        params = {
            "resolution": resolution,
            "max": max_results
        }
        
        data = self._make_request('GET', f'/api/v1/prices/{symbol}')
        
        if data and 'prices' in data:
            return data['prices']
        return None
    
    def search_markets(self, search_term: str, limit: int = 20) -> Optional[List[Dict]]:
        """
        Search for markets
        
        Args:
            search_term: Search query
            limit: Maximum results
            
        Returns:
            List of matching markets
        """
        if not self.is_authenticated():
            if not self.authenticate():
                return None
        
        params = {
            "searchTerm": search_term,
            "limit": limit
        }
        
        data = self._make_request('GET', '/api/v1/markets')
        
        if data and 'markets' in data:
            return data['markets']
        return None
    
    def keep_alive(self) -> bool:
        """
        Keep session alive by pinging the server
        
        Returns:
            True if session is active
        """
        if not self.is_authenticated():
            return False
        
        data = self._make_request('GET', '/api/v1/ping')
        return data is not None
    
    def logout(self) -> bool:
        """
        Logout and close session
        
        Returns:
            True if successful
        """
        if self.security_token:
            data = self._make_request('DELETE', '/api/v1/session')
            self.cst = None
            self.security_token = None
            self.session_expires = None
            logger.info("Logged out successfully")
            return True
        return False


def test_connection():
    """Test Capital.com API connection"""
    print("\n" + "="*60)
    print("Testing Capital.com API Connection")
    print("="*60)
    
    # Initialize API
    api = CapitalComAPI(demo=True)
    
    # Check credentials
    if not api.api_key:
        print("\n[ERROR] No API credentials found!")
        print("\nPlease set environment variables:")
        print("  set CAPITAL_API_KEY=your_api_key")
        print("  set CAPITAL_API_PASSWORD=your_password")
        print("  set CAPITAL_IDENTIFIER=your_email")
        return False
    
    print(f"\n[INFO] API Key: {api.api_key[:8]}...")
    print(f"[INFO] Identifier: {api.identifier}")
    
    # Test authentication
    print("\n[TEST] Authenticating...")
    if api.authenticate():
        print("[OK] Authentication successful!")
        
        # Get accounts
        print("\n[TEST] Getting accounts...")
        accounts = api.get_accounts()
        if accounts:
            print(f"[OK] Found {len(accounts)} account(s)")
            for acc in accounts:
                print(f"  - {acc.get('accountName')}: {acc.get('balance', {}).get('balance', 0)}")
        
        # Get market data
        print("\n[TEST] Getting market data for AAPL...")
        market_data = api.get_market_data('AAPL')
        if market_data:
            print(f"[OK] AAPL - Bid: {market_data.bid}, Ask: {market_data.ask}")
        
        # Get positions
        print("\n[TEST] Getting positions...")
        positions = api.get_positions()
        if positions is not None:
            print(f"[OK] Found {len(positions)} position(s)")
        
        # Logout
        print("\n[TEST] Logging out...")
        api.logout()
        print("[OK] Logged out successfully")
        
        return True
    else:
        print("[FAIL] Authentication failed!")
        return False


if __name__ == "__main__":
    test_connection()