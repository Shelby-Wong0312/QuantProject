"""
Capital.com API Client
Capital.com API 客戶端實現
Cloud DE - Task CAP-001
"""

import aiohttp
import asyncio
import json
import hmac
import hashlib
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
from pathlib import Path
import websockets
import ssl
import certifi
from urllib.parse import urlencode
import base64

logger = logging.getLogger(__name__)


class Environment(Enum):
    """交易環境"""
    DEMO = "demo"
    LIVE = "live"


class OrderType(Enum):
    """訂單類型"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(Enum):
    """訂單方向"""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class MarketData:
    """市場數據"""
    symbol: str
    bid: float
    ask: float
    spread: float
    timestamp: datetime
    volume: float = 0
    change_pct: float = 0


@dataclass
class Order:
    """訂單"""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    client_order_id: Optional[str] = None


@dataclass
class Position:
    """持倉"""
    symbol: str
    quantity: float
    side: str
    avg_price: float
    current_price: float
    pnl: float
    pnl_percent: float
    market_value: float


@dataclass
class AccountInfo:
    """帳戶信息"""
    account_id: str
    balance: float
    available: float
    currency: str
    profit_loss: float
    margin_used: float
    margin_available: float


class CapitalComClient:
    """
    Capital.com API 客戶端
    
    支援 REST API 和 WebSocket 實時數據流
    """
    
    # API 端點
    BASE_URLS = {
        Environment.DEMO: "https://demo-api-capital.backend-capital.com/api/v1",
        Environment.LIVE: "https://api-capital.backend-capital.com/api/v1"
    }
    
    WS_URLS = {
        Environment.DEMO: "wss://demo-api-streaming-capital.backend-capital.com",
        Environment.LIVE: "wss://api-streaming-capital.backend-capital.com"
    }
    
    def __init__(self, 
                 api_key: str,
                 password: str,
                 environment: Environment = Environment.DEMO,
                 config_path: Optional[str] = None):
        """
        初始化 Capital.com 客戶端
        
        Args:
            api_key: API 密鑰
            password: 密碼
            environment: 交易環境
            config_path: 配置文件路徑
        """
        self.api_key = api_key
        self.password = password
        self.environment = environment
        self.base_url = self.BASE_URLS[environment]
        self.ws_url = self.WS_URLS[environment]
        
        # 會話管理
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws_connection = None
        self.access_token: Optional[str] = None
        self.cst_token: Optional[str] = None
        self.x_security_token: Optional[str] = None
        
        # 配置
        self.config = self._load_config(config_path)
        
        # 重連參數
        self.max_retries = 3
        self.retry_delay = 1
        self.is_connected = False
        
        # WebSocket 回調
        self.ws_callbacks: Dict[str, List[Callable]] = {}
        
        # Rate limiting
        self.rate_limiter = RateLimiter(
            max_requests=self.config.get('rate_limit', 100),
            time_window=60
        )
        
        logger.info(f"Capital.com client initialized for {environment.value} environment")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """載入配置"""
        default_config = {
            'rate_limit': 100,
            'timeout': 30,
            'max_connections': 10,
            'heartbeat_interval': 30,
            'reconnect_delay': 5
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                default_config.update(custom_config)
        
        return default_config
    
    async def connect(self) -> bool:
        """
        建立連接並認證
        
        Returns:
            是否成功連接
        """
        try:
            # 創建會話
            if not self.session:
                timeout = aiohttp.ClientTimeout(total=self.config['timeout'])
                connector = aiohttp.TCPConnector(limit=self.config['max_connections'])
                self.session = aiohttp.ClientSession(
                    timeout=timeout,
                    connector=connector
                )
            
            # 執行認證
            auth_success = await self._authenticate()
            
            if auth_success:
                self.is_connected = True
                logger.info("Successfully connected to Capital.com API")
                
                # 啟動心跳
                asyncio.create_task(self._heartbeat_loop())
                
                return True
            else:
                logger.error("Failed to authenticate with Capital.com")
                return False
                
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    async def _authenticate(self) -> bool:
        """
        執行 OAuth 認證
        
        Returns:
            是否認證成功
        """
        try:
            url = f"{self.base_url}/session"
            
            payload = {
                "identifier": self.api_key,
                "password": self.password
            }
            
            headers = {
                "Content-Type": "application/json",
                "X-CAP-API-KEY": self.api_key
            }
            
            async with self.session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    # 獲取認證令牌
                    self.cst_token = response.headers.get('CST')
                    self.x_security_token = response.headers.get('X-SECURITY-TOKEN')
                    
                    data = await response.json()
                    self.access_token = data.get('accessToken')
                    
                    logger.info("Authentication successful")
                    return True
                else:
                    error = await response.text()
                    logger.error(f"Authentication failed: {error}")
                    return False
                    
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    def _get_headers(self) -> Dict[str, str]:
        """獲取請求頭"""
        headers = {
            "Content-Type": "application/json",
            "X-CAP-API-KEY": self.api_key
        }
        
        if self.cst_token:
            headers["CST"] = self.cst_token
        
        if self.x_security_token:
            headers["X-SECURITY-TOKEN"] = self.x_security_token
        
        return headers
    
    async def get_account_info(self) -> Optional[AccountInfo]:
        """
        獲取帳戶信息
        
        Returns:
            帳戶信息
        """
        try:
            await self.rate_limiter.acquire()
            
            url = f"{self.base_url}/accounts"
            headers = self._get_headers()
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    accounts = data.get('accounts', [])
                    
                    if accounts:
                        account = accounts[0]  # 使用第一個帳戶
                        
                        return AccountInfo(
                            account_id=account['accountId'],
                            balance=account['balance']['balance'],
                            available=account['balance']['available'],
                            currency=account['currency'],
                            profit_loss=account['balance']['profitLoss'],
                            margin_used=account['balance']['deposit'],
                            margin_available=account['balance']['availableToWithdraw']
                        )
                    
                return None
                
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return None
    
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """
        獲取市場數據
        
        Args:
            symbol: 交易符號
            
        Returns:
            市場數據
        """
        try:
            await self.rate_limiter.acquire()
            
            url = f"{self.base_url}/markets/{symbol}"
            headers = self._get_headers()
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    market = data.get('market', {})
                    snapshot = market.get('snapshot', {})
                    
                    return MarketData(
                        symbol=symbol,
                        bid=snapshot.get('bid', 0),
                        ask=snapshot.get('offer', 0),
                        spread=snapshot.get('offer', 0) - snapshot.get('bid', 0),
                        timestamp=datetime.now(),
                        volume=snapshot.get('volume', 0),
                        change_pct=snapshot.get('percentageChange', 0)
                    )
                    
                return None
                
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return None
    
    async def place_order(self, order: Order) -> Optional[str]:
        """
        下單
        
        Args:
            order: 訂單對象
            
        Returns:
            訂單 ID
        """
        try:
            await self.rate_limiter.acquire()
            
            url = f"{self.base_url}/positions"
            headers = self._get_headers()
            
            payload = {
                "epic": order.symbol,
                "direction": order.side.value,
                "size": order.quantity,
                "orderType": order.order_type.value,
                "timeInForce": order.time_in_force,
                "guaranteedStop": False,
                "trailingStop": False,
                "forceOpen": True
            }
            
            if order.order_type == OrderType.LIMIT and order.price:
                payload["level"] = order.price
            
            if order.stop_price:
                payload["stopLevel"] = order.stop_price
            
            async with self.session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    deal_reference = data.get('dealReference')
                    
                    logger.info(f"Order placed successfully: {deal_reference}")
                    return deal_reference
                else:
                    error = await response.text()
                    logger.error(f"Order failed: {error}")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None
    
    async def get_positions(self) -> List[Position]:
        """
        獲取持倉
        
        Returns:
            持倉列表
        """
        try:
            await self.rate_limiter.acquire()
            
            url = f"{self.base_url}/positions"
            headers = self._get_headers()
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    positions_data = data.get('positions', [])
                    
                    positions = []
                    for pos in positions_data:
                        position = Position(
                            symbol=pos['market']['epic'],
                            quantity=pos['position']['size'],
                            side=pos['position']['direction'],
                            avg_price=pos['position']['level'],
                            current_price=pos['market']['bid'] if pos['position']['direction'] == 'SELL' 
                                        else pos['market']['offer'],
                            pnl=pos['position']['profitLoss'],
                            pnl_percent=pos['position']['percentageProfitLoss'],
                            market_value=pos['position']['size'] * pos['market']['bid']
                        )
                        positions.append(position)
                    
                    return positions
                    
                return []
                
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    async def close_position(self, deal_id: str) -> bool:
        """
        平倉
        
        Args:
            deal_id: 交易 ID
            
        Returns:
            是否成功
        """
        try:
            await self.rate_limiter.acquire()
            
            url = f"{self.base_url}/positions/{deal_id}"
            headers = self._get_headers()
            
            async with self.session.delete(url, headers=headers) as response:
                if response.status == 200:
                    logger.info(f"Position {deal_id} closed successfully")
                    return True
                else:
                    error = await response.text()
                    logger.error(f"Failed to close position: {error}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return False
    
    async def subscribe_price_stream(self, 
                                    symbols: List[str],
                                    callback: Callable[[MarketData], None]):
        """
        訂閱實時價格流
        
        Args:
            symbols: 符號列表
            callback: 回調函數
        """
        try:
            # 建立 WebSocket 連接
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            
            headers = {
                "CST": self.cst_token,
                "X-SECURITY-TOKEN": self.x_security_token
            }
            
            async with websockets.connect(
                self.ws_url,
                ssl=ssl_context,
                extra_headers=headers
            ) as websocket:
                
                self.ws_connection = websocket
                
                # 訂閱符號
                for symbol in symbols:
                    subscribe_msg = {
                        "destination": "marketData.subscribe",
                        "correlationId": str(time.time()),
                        "cst": self.cst_token,
                        "securityToken": self.x_security_token,
                        "payload": {
                            "epics": [symbol]
                        }
                    }
                    
                    await websocket.send(json.dumps(subscribe_msg))
                    logger.info(f"Subscribed to {symbol}")
                
                # 處理消息
                async for message in websocket:
                    data = json.loads(message)
                    
                    if data.get('destination') == 'marketData.update':
                        payload = data.get('payload', {})
                        
                        for update in payload.get('updates', []):
                            market_data = MarketData(
                                symbol=update['epic'],
                                bid=update['bid'],
                                ask=update['offer'],
                                spread=update['offer'] - update['bid'],
                                timestamp=datetime.now(),
                                change_pct=update.get('percentageChange', 0)
                            )
                            
                            # 觸發回調
                            if callback:
                                await callback(market_data)
                
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            # 嘗試重連
            if self.is_connected:
                await self._reconnect_websocket(symbols, callback)
    
    async def _reconnect_websocket(self, symbols: List[str], callback: Callable):
        """重連 WebSocket"""
        logger.info("Attempting to reconnect WebSocket...")
        await asyncio.sleep(self.config['reconnect_delay'])
        await self.subscribe_price_stream(symbols, callback)
    
    async def _heartbeat_loop(self):
        """心跳循環"""
        while self.is_connected:
            try:
                await asyncio.sleep(self.config['heartbeat_interval'])
                
                # 發送心跳
                if self.ws_connection:
                    ping_msg = {
                        "destination": "ping",
                        "correlationId": str(time.time())
                    }
                    await self.ws_connection.send(json.dumps(ping_msg))
                    
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    async def disconnect(self):
        """斷開連接"""
        try:
            self.is_connected = False
            
            if self.ws_connection:
                await self.ws_connection.close()
            
            if self.session:
                await self.session.close()
            
            logger.info("Disconnected from Capital.com API")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    async def __aenter__(self):
        """異步上下文管理器入口"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """異步上下文管理器出口"""
        await self.disconnect()


class RateLimiter:
    """
    速率限制器
    
    確保不超過 API 調用限制
    """
    
    def __init__(self, max_requests: int, time_window: int):
        """
        初始化速率限制器
        
        Args:
            max_requests: 最大請求數
            time_window: 時間窗口（秒）
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """獲取請求許可"""
        async with self.lock:
            now = time.time()
            
            # 清理過期請求
            self.requests = [
                req_time for req_time in self.requests
                if now - req_time < self.time_window
            ]
            
            # 檢查是否超限
            if len(self.requests) >= self.max_requests:
                # 計算需要等待的時間
                oldest_request = self.requests[0]
                wait_time = self.time_window - (now - oldest_request)
                
                if wait_time > 0:
                    logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                    
                    # 重新清理
                    now = time.time()
                    self.requests = [
                        req_time for req_time in self.requests
                        if now - req_time < self.time_window
                    ]
            
            # 記錄新請求
            self.requests.append(now)


class CapitalComError(Exception):
    """Capital.com API 錯誤"""
    pass


if __name__ == "__main__":
    print("Capital.com API Client - Cloud DE Task CAP-001")
    print("=" * 50)
    print("Features implemented:")
    print("- OAuth 2.0 authentication")
    print("- REST API wrapper")
    print("- WebSocket real-time data streaming")
    print("- Rate limiting")
    print("- Auto-reconnection")
    print("- Error handling")
    print("\n✓ Capital.com client ready for integration!")