# core/event_loop.py
import asyncio
import logging

logger = logging.getLogger(__name__)

class AsyncEventLoop:
    def __init__(self):
        self.event_queue = asyncio.Queue()
        self.handlers = {}
        self.running = False
        self._main_loop_task = None

    def register_handler(self, event_type_name: str, handler_callable):
        if not asyncio.iscoroutinefunction(handler_callable):
            raise TypeError(f"Handler {handler_callable.__name__} must be an async function.")
        
        if event_type_name not in self.handlers:
            self.handlers[event_type_name] = []
        
        if handler_callable not in self.handlers[event_type_name]:
            self.handlers[event_type_name].append(handler_callable)
            logger.info(f"Async handler {handler_callable.__name__} registered for {event_type_name}")

    async def post_event(self, event):
        await self.event_queue.put(event)

    async def _dispatch_event(self, handler, event):
        try:
            await handler(event)
        except Exception as e:
            logger.error(f"Error in async handler {handler.__name__} for event {type(event).__name__}: {e}", exc_info=True)

    async def run(self):
        logger.info("Async event loop starting...")
        self.running = True
        while self.running:
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                event_type_name = type(event).__name__
                if event_type_name in self.handlers:
                    for handler in self.handlers[event_type_name]:
                        asyncio.create_task(self._dispatch_event(handler, event))
                self.event_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Critical error in async event loop: {e}", exc_info=True)

        logger.info("Async event loop stopped.")

    def start(self):
        if not self.running:
            self._main_loop_task = asyncio.create_task(self.run())
            logger.info("Async event loop initiated.")

    async def stop(self):
        logger.info("Stop signal received for async event loop.")
        self.running = False
        if self._main_loop_task:
            try:
                await asyncio.wait_for(self._main_loop_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Event loop did not stop gracefully within timeout.")
                self._main_loop_task.cancel()
            except asyncio.CancelledError:
                logger.info("Event loop task was cancelled.")