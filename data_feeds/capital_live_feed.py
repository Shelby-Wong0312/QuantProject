import asyncio
import logging
import aiohttp
import json
from datetime import datetime, timezone
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

from core.event_types import MarketDataEvent

load_dotenv()

logger = logging.getLogger(__name__)

class CapitalLiveFeedHandler:
    """
    连接到 Capital.com 的实时数据流处理器
    """
    def __init__(self, event_queue: asyncio.Queue, symbols: List[str]):
        self.event_queue = event_queue
        self.symbols = symbols
        self._running = False
        
        # Capital.com API 配置
        self.api_key = os.getenv("CAPITAL_API_KEY")
        self.identifier = os.getenv("CAPITAL_IDENTIFIER")
        self.password = os.getenv("CAPITAL_API_PASSWORD")
        self.base_url = os.getenv("CAPITAL_BASE_API_URL", "https://demo-api-capital.backend-capital.com/api/v1")
        
        # Session tokens
        self.cst = None
        self.x_security_token = None
        self.session = None
        
    async def _login(self):
        """登录到 Capital.com 获取会话令牌"""
        login_url = f"{self.base_url}/session"
        headers = {
            "X-CAP-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "identifier": self.identifier,
            "password": self.password,
            "encryptedPassword": False
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(login_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    self.cst = response.headers.get("CST")
                    self.x_security_token = response.headers.get("X-SECURITY-TOKEN")
                    logger.info("成功登录到 Capital.com")
                    return True
                else:
                    logger.error(f"登录失败: {response.status}")
                    return False
    
    async def _get_market_data(self, epic: str) -> Dict[str, Any]:
        """获取特定市场的实时数据"""
        url = f"{self.base_url}/markets/{epic}"
        headers = {
            "X-CAP-API-KEY": self.api_key,
            "CST": self.cst,
            "X-SECURITY-TOKEN": self.x_security_token,
            "Content-Type": "application/json"
        }
        
        try:
            async with self.session.get(url, headers=headers) as response:
                response_text = await response.text()
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"获取市场数据失败 {epic}: {response.status} - {response_text}")
                    return None
        except Exception as e:
            logger.error(f"请求市场数据时出错 {epic}: {e}")
            return None
    
    def _map_symbol_to_epic(self, symbol: str) -> str:
        """将股票代码映射到 Capital.com 的 EPIC"""
        # 加密貨幣直接使用原始符號，不需要添加後綴
        # BTCUSD -> BTCUSD (不是 BTCUSD.US)
        return symbol
    
    async def start_feed(self):
        """开始实时数据流"""
        # 首先登录
        if not await self._login():
            logger.error("无法登录到 Capital.com")
            return
        
        self._running = True
        self.session = aiohttp.ClientSession()
        
        logger.info(f"开始监控 {len(self.symbols)} 个股票的实时数据")
        
        # 首先测试一个股票看看能否获取数据
        test_symbol = self.symbols[0] if self.symbols else "AAPL"
        test_epic = self._map_symbol_to_epic(test_symbol)
        logger.info(f"测试获取 {test_symbol} (EPIC: {test_epic}) 的数据...")
        
        test_data = await self._get_market_data(test_epic)
        if test_data:
            logger.info(f"成功获取测试数据: {test_data}")
        else:
            logger.error(f"无法获取测试数据，请检查 EPIC 映射是否正确")
        
        try:
            error_count = 0
            while self._running:
                # 轮询每个股票的数据
                for symbol in self.symbols:
                    if not self._running:
                        break
                        
                    epic = self._map_symbol_to_epic(symbol)
                    logger.debug(f"正在获取 {symbol} (EPIC: {epic}) 的数据...")
                    
                    market_data = await self._get_market_data(epic)
                    
                    if market_data:
                        # 重置错误计数
                        error_count = 0
                        
                        # 检查不同的数据结构
                        if 'snapshot' in market_data:
                            snapshot = market_data['snapshot']
                            logger.debug(f"获取到 snapshot 数据: {snapshot}")
                        elif 'instrument' in market_data:
                            # 可能是不同的数据结构
                            logger.info(f"市场数据结构: {list(market_data.keys())}")
                            snapshot = market_data.get('instrument', {})
                        else:
                            logger.warning(f"未知的数据结构: {list(market_data.keys())}")
                            continue
                        
                        # 创建市场数据事件
                        market_event = MarketDataEvent(
                            symbol=symbol,
                            timestamp=datetime.now(timezone.utc),
                            event_type="LIVE_QUOTE",
                            data={
                                "bid": float(snapshot.get('bid', 0)),
                                "ask": float(snapshot.get('offer', 0)),
                                "last": float(snapshot.get('bid', 0)),  # 使用 bid 作为最新价格
                                "high": float(snapshot.get('high', 0)),
                                "low": float(snapshot.get('low', 0)),
                                "volume": 0,  # Capital.com 可能不提供实时成交量
                                "epic": epic
                            }
                        )
                        
                        # 将事件放入队列
                        self.event_queue.put(market_event)
                        logger.info(f"发送实时数据: {symbol} - Bid: {snapshot.get('bid')}, Ask: {snapshot.get('offer')}")
                    else:
                        error_count += 1
                        if error_count > 10:
                            logger.error(f"连续 {error_count} 次获取数据失败，可能是 API 限制或配置问题")
                
                # 等待一段时间再进行下一轮轮询
                await asyncio.sleep(0)  # 無間隔，盡可能快
                
        except Exception as e:
            logger.error(f"实时数据流出错: {e}", exc_info=True)
        finally:
            if self.session:
                await self.session.close()
    
    def stop(self):
        """停止数据流"""
        self._running = False
        logger.info("CapitalLiveFeedHandler 已停止") 