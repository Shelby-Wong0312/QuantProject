import logging
import aiohttp
import os
from datetime import datetime
from dotenv import load_dotenv

from core.event import SignalEvent, FillEvent, SignalAction

load_dotenv()

logger = logging.getLogger(__name__)

class CapitalExecutionHandler:
    """Capital.com 执行处理器"""
    
    def __init__(self, event_queue):
        self.event_queue = event_queue
        
        # Capital.com API 配置
        self.api_key = os.getenv("CAPITAL_API_KEY")
        self.identifier = os.getenv("CAPITAL_IDENTIFIER")
        self.password = os.getenv("CAPITAL_API_PASSWORD")
        self.base_url = os.getenv("CAPITAL_BASE_API_URL", "https://demo-api-capital.backend-capital.com/api/v1")
        
        # Session tokens
        self.cst = None
        self.x_security_token = None
        
    async def _ensure_logged_in(self):
        """确保已登录"""
        if not self.cst or not self.x_security_token:
            await self._login()
    
    async def _login(self):
        """登录到 Capital.com"""
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
                else:
                    logger.error(f"登录失败: {response.status}")
                    raise Exception("无法登录到 Capital.com")
    
    def _map_symbol_to_epic(self, symbol: str) -> str:
        """将股票代码映射到 Capital.com 的 EPIC"""
        if not symbol.endswith('.US'):
            return f"{symbol}.US"
        return symbol
    
    async def _place_order(self, epic: str, direction: str, size: float, 
                          stop_loss: float = None, take_profit: float = None):
        """下单到 Capital.com"""
        await self._ensure_logged_in()
        
        url = f"{self.base_url}/positions"
        headers = {
            "X-CAP-API-KEY": self.api_key,
            "CST": self.cst,
            "X-SECURITY-TOKEN": self.x_security_token,
            "Content-Type": "application/json"
        }
        
        payload = {
            "epic": epic,
            "direction": direction,
            "size": size,
            "guaranteedStop": False
        }
        
        if stop_loss:
            payload["stopLevel"] = stop_loss
        if take_profit:
            payload["profitLevel"] = take_profit
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                response_data = await response.json()
                
                if response.status == 200 and response_data.get("dealReference"):
                    logger.info(f"订单成功: {epic} {direction} {size} - Deal Ref: {response_data['dealReference']}")
                    return True, response_data
                else:
                    logger.error(f"订单失败: {response.status} - {response_data}")
                    return False, response_data
    
    async def handle_signal_event(self, signal_event: SignalEvent):
        """处理信号事件并执行交易"""
        logger.info(f"收到交易信号: {signal_event.action} for {signal_event.symbol}")
        
        try:
            epic = self._map_symbol_to_epic(signal_event.symbol)
            
            # 根据信号类型确定交易方向
            if signal_event.action == SignalAction.BUY:
                direction = "BUY"
            elif signal_event.action == SignalAction.SELL:
                direction = "SELL"
            else:
                logger.warning(f"未知的信号类型: {signal_event.action}")
                return
            
            # 执行交易
            success, response = await self._place_order(
                epic=epic,
                direction=direction,
                size=signal_event.quantity,
                stop_loss=signal_event.sl_price,
                take_profit=signal_event.tp_price
            )
            
            if success:
                # 创建成交事件
                fill_event = FillEvent(
                    symbol=signal_event.symbol,
                    timestamp=datetime.now(),
                    action=signal_event.action,
                    quantity=signal_event.quantity,
                    fill_price=signal_event.price if signal_event.price else 0,
                    commission=0  # Capital.com 的手续费需要从响应中获取
                )
                
                # 将成交事件放入队列
                self.event_queue.put(fill_event)
                logger.info(f"订单已执行: {signal_event.symbol} {signal_event.action} {signal_event.quantity}")
            
        except Exception as e:
            logger.error(f"执行订单时出错: {e}", exc_info=True) 