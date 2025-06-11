import logging
from core.event import SignalEvent, FillEvent, SignalAction
from execution.capital_client import AsyncCapitalComClient
from datetime import datetime

logger = logging.getLogger(__name__)

class SimpleExecutionHandler:
    """简化版的执行处理器，用于事件驱动系统"""
    
    def __init__(self, event_queue, capital_client: AsyncCapitalComClient):
        self.event_queue = event_queue
        self.capital_client = capital_client
        
    async def handle_signal_event(self, signal_event: SignalEvent):
        """处理信号事件并执行交易"""
        logger.info(f"收到交易信号: {signal_event.action} for {signal_event.symbol}")
        
        try:
            # 这里应该调用 capital_client 的方法来执行交易
            # 由于 AsyncCapitalComClient 是异步的，在同步环境中我们需要特殊处理
            # 现在我们只是模拟执行
            
            # 模拟成交
            fill_price = signal_event.price if signal_event.price else 100.0  # 使用信号价格或默认价格
            
            # 创建成交事件
            fill_event = FillEvent(
                symbol=signal_event.symbol,
                timestamp=datetime.now(),
                action=signal_event.action,
                quantity=signal_event.quantity,
                fill_price=fill_price,
                commission=0.001 * signal_event.quantity * fill_price  # 0.1% 手续费
            )
            
            # 将成交事件放入队列
            self.event_queue.put(fill_event)
            logger.info(f"订单已执行: {signal_event.symbol} {signal_event.action} {signal_event.quantity} @ {fill_price}")
            
        except Exception as e:
            logger.error(f"执行订单时出错: {e}", exc_info=True) 