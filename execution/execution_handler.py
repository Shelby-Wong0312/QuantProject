# 檔案位置: execution/execution_handler.py

import logging
from core.event import SignalEvent, OrderEvent, FillEvent
from execution.capital_client import CapitalComClient

logger = logging.getLogger(__name__)

class ExecutionHandler:
    """
    處理交易執行。它「消費」交易訊號事件，並與券商API互動。
    它也可以「產生」訂單事件和成交事件。
    """
    def __init__(self, event_queue, capital_client: CapitalComClient):
        self.event_queue = event_queue
        self.capital_client = capital_client
        
        # 確保客戶端已登入
        if not self.capital_client._is_session_active():
            logger.info("Capital.com 會話未啟用，正在嘗試登入...")
            if not self.capital_client.login():
                raise ConnectionError("為 ExecutionHandler 登入 Capital.com 失敗。")
            logger.info("成功登入 Capital.com。")

    def handle_signal_event(self, event: SignalEvent):
        """
        處理交易訊號事件，並下達實際訂單。
        """
        logger.info(f"執行處理器收到訊號事件: {event.action} for {event.symbol}")
        
        # 這裡可以擴充更複雜的邏輯，例如將市價單轉為限價單等
        
        api_response = None
        
        if event.action == 'BUY_ENTRY':
            api_response = self.capital_client.place_market_order(
                epic=event.symbol,
                direction="BUY",
                size=event.quantity,
                stop_loss_price=event.sl_price,
                take_profit_price=event.tp_price
            )
        elif event.action == 'CLOSE_LONG_CONDITION':
            # 這裡需要一個更穩健的方法來找到要平倉的 dealId
            # 在一個完整的系統中，這通常由一個 PositionManager 來管理
            # 此處為簡化邏輯
            open_positions = self.capital_client.get_open_positions()
            if open_positions and open_positions.get('positions'):
                for pos in open_positions['positions']:
                    if pos.get('instrument', {}).get('epic') == event.symbol and pos.get('position', {}).get('direction') == 'BUY':
                        deal_id = pos.get('position', {}).get('dealId')
                        logger.info(f"找到多頭倉位 {deal_id}，執行平倉...")
                        api_response = self.capital_client.close_position(deal_id)
                        break 

        if api_response:
            logger.info(f"API 回應: {api_response}")
            # 在此處可以根據 API 回應，產生並發布 OrderEvent 或 FillEvent
            