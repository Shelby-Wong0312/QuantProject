# 檔案位置: core/event_loop.py

import queue
import logging
import time

logger = logging.getLogger(__name__)

class EventLoop:
    """
    事件驅動系統的核心引擎。
    負責管理事件隊列，並將事件分派給已註冊的處理器。
    """
    def __init__(self):
        self.event_queue = queue.Queue()
        self.handlers = {}  # 字典，key為事件類型名稱，value為處理該事件的函式列表
        self.running = True

    def register_handler(self, event_type_name: str, handler_callable):
        """註冊一個事件處理器。"""
        if event_type_name not in self.handlers:
            self.handlers[event_type_name] = []
        if handler_callable not in self.handlers[event_type_name]:
            self.handlers[event_type_name].append(handler_callable)
            logger.info(f"處理器 {handler_callable.__name__} 已註冊處理 {event_type_name} 事件。")

    def post_event(self, event):
        """將一個新的事件放入隊列中。"""
        self.event_queue.put(event)

    def start(self):
        """啟動事件處理循環。"""
        logger.info("事件循環啟動...")
        while self.running:
            try:
                # 等待事件，設置1秒超時以允許檢查 self.running 狀態
                event = self.event_queue.get(block=True, timeout=1.0)
            except queue.Empty:
                # 隊列為空，繼續下一次循環
                continue

            event_type_name = type(event).__name__
            if event_type_name in self.handlers:
                # 將事件分派給所有已註冊的處理器
                for handler in self.handlers[event_type_name]:
                    try:
                        handler(event)
                    except Exception as e:
                        logger.error(f"處理器 {handler.__name__} 在處理 {event_type_name} 事件時發生錯誤: {e}", exc_info=True)
            
            self.event_queue.task_done()

    def stop(self):
        """停止事件循環。"""
        logger.info("收到停止訊號，事件循環即將關閉...")
        self.running = False
        