# quant_project/core/event_loop.py

import asyncio
import logging
from .event import EventType

logger = logging.getLogger(__name__)

class EventLoop:
    def __init__(self):
        self._event_queue = asyncio.Queue()
        self._running = False
        self._handlers = {}

    def add_handler(self, event_type: EventType, handler_coro):
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler_coro)
        logger.info(f"已為事件 '{event_type.name}' 註冊處理器: {handler_coro.__name__}")

    async def put_event(self, event):
        await self._event_queue.put(event)

    async def run(self):
        logger.info("事件循環已啟動...")
        self._running = True
        while self._running:
            try:
                event = await self._event_queue.get()
                if event.type in self._handlers:
                    logger.debug(f"處理事件: {event}")
                    tasks = [handler(event) for handler in self._handlers[event.type]]
                    await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                self._running = False
            except Exception as e:
                logger.error(f"事件循環發生錯誤: {e}", exc_info=True)

    def stop(self):
        logger.info("正在停止事件循環...")
        self._running = False