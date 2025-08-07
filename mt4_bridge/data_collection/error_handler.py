# -*- coding: utf-8 -*-
"""
錯誤處理和日誌配置模組
為 MT4 數據收集系統提供統一的錯誤處理和日誌記錄功能
"""

import logging
import logging.handlers
import sys
import traceback
import functools
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type
from enum import Enum

class LogLevel(Enum):
    """日誌級別枚舉"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

class MT4DataCollectionLogger:
    """MT4 數據收集系統專用日誌器"""
    
    def __init__(self, 
                 name: str = "MT4DataCollection",
                 log_level: LogLevel = LogLevel.INFO,
                 log_file: Optional[str] = None,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 console_output: bool = True):
        """
        初始化日誌器
        
        Args:
            name: 日誌器名稱
            log_level: 日誌級別
            log_file: 日誌文件路徑
            max_file_size: 最大文件大小
            backup_count: 備份文件數量
            console_output: 是否輸出到控制台
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level.value)
        
        # 防止重複添加處理器
        if self.logger.handlers:
            return
        
        # 創建格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        
        # 控制台處理器
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 文件處理器
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def get_logger(self) -> logging.Logger:
        """獲取日誌器實例"""
        return self.logger

class MT4DataCollectionError(Exception):
    """MT4 數據收集系統基礎異常類"""
    
    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.context = context or {}
        self.timestamp = datetime.now()
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "type": self.__class__.__name__
        }

class TickCollectionError(MT4DataCollectionError):
    """Tick 數據收集錯誤"""
    pass

class OHLCAggregationError(MT4DataCollectionError):
    """OHLC 聚合錯誤"""
    pass

class DataStorageError(MT4DataCollectionError):
    """數據存儲錯誤"""
    pass

class MT4ConnectionError(MT4DataCollectionError):
    """MT4 連接錯誤"""
    pass

class DataFeedError(MT4DataCollectionError):
    """數據饋送錯誤"""
    pass

class ErrorHandler:
    """統一錯誤處理器"""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_stats = {
            "total_errors": 0,
            "error_types": {},
            "last_error": None
        }
    
    def handle_error(self, 
                    error: Exception, 
                    context: str = "", 
                    reraise: bool = False,
                    error_code: str = None) -> Optional[MT4DataCollectionError]:
        """
        統一錯誤處理
        
        Args:
            error: 異常對象
            context: 錯誤上下文
            reraise: 是否重新拋出異常
            error_code: 錯誤代碼
        
        Returns:
            處理後的異常對象
        """
        try:
            # 更新統計
            self.error_stats["total_errors"] += 1
            error_type = type(error).__name__
            self.error_stats["error_types"][error_type] = self.error_stats["error_types"].get(error_type, 0) + 1
            
            # 創建標準錯誤對象
            if isinstance(error, MT4DataCollectionError):
                handled_error = error
            else:
                handled_error = MT4DataCollectionError(
                    message=str(error),
                    error_code=error_code or error_type,
                    context={"original_context": context, "traceback": traceback.format_exc()}
                )
            
            self.error_stats["last_error"] = handled_error.to_dict()
            
            # 記錄錯誤
            self.logger.error(
                f"錯誤處理 - {context}: {handled_error.message}",
                extra={
                    "error_code": handled_error.error_code,
                    "context": handled_error.context,
                    "error_type": error_type
                }
            )
            
            # 如果是嚴重錯誤，記錄完整堆疊追蹤
            if isinstance(error, (MT4ConnectionError, DataStorageError)):
                self.logger.critical(f"嚴重錯誤堆疊追蹤:\n{traceback.format_exc()}")
            
            if reraise:
                raise handled_error
            
            return handled_error
            
        except Exception as handling_error:
            # 錯誤處理本身出錯時的後備處理
            self.logger.critical(f"錯誤處理器本身出錯: {handling_error}")
            if reraise:
                raise error  # 拋出原始錯誤
            return None
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """獲取錯誤統計信息"""
        return self.error_stats.copy()
    
    def clear_error_statistics(self):
        """清除錯誤統計"""
        self.error_stats = {
            "total_errors": 0,
            "error_types": {},
            "last_error": None
        }

def error_handler_decorator(error_type: Type[MT4DataCollectionError] = MT4DataCollectionError,
                          logger: logging.Logger = None,
                          reraise: bool = False,
                          context: str = None):
    """
    錯誤處理裝飾器
    
    Args:
        error_type: 要包裝的錯誤類型
        logger: 日誌器
        reraise: 是否重新拋出異常
        context: 錯誤上下文
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler = ErrorHandler(logger)
                func_context = context or f"{func.__module__}.{func.__name__}"
                
                if isinstance(e, MT4DataCollectionError):
                    handled_error = e
                else:
                    handled_error = error_type(
                        message=str(e),
                        error_code=f"{func.__name__}_ERROR",
                        context={"function": func_context, "args": str(args), "kwargs": str(kwargs)}
                    )
                
                error_handler.handle_error(handled_error, func_context, reraise=reraise)
                
                if reraise:
                    raise handled_error
                
                return None
                
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_handler = ErrorHandler(logger)
                func_context = context or f"{func.__module__}.{func.__name__}"
                
                if isinstance(e, MT4DataCollectionError):
                    handled_error = e
                else:
                    handled_error = error_type(
                        message=str(e),
                        error_code=f"{func.__name__}_ERROR",
                        context={"function": func_context, "args": str(args), "kwargs": str(kwargs)}
                    )
                
                error_handler.handle_error(handled_error, func_context, reraise=reraise)
                
                if reraise:
                    raise handled_error
                
                return None
        
        # 根據函數類型返回相應的包裝器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator

def setup_logging(log_level: LogLevel = LogLevel.INFO,
                 log_file: str = None,
                 console_output: bool = True) -> logging.Logger:
    """
    設置 MT4 數據收集系統的日誌配置
    
    Args:
        log_level: 日誌級別
        log_file: 日誌文件路徑
        console_output: 是否輸出到控制台
    
    Returns:
        配置好的日誌器
    """
    # 默認日誌文件路徑
    if log_file is None:
        log_dir = Path("./logs/mt4_data_collection")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"mt4_data_collection_{datetime.now().strftime('%Y%m%d')}.log"
    
    logger_manager = MT4DataCollectionLogger(
        name="MT4DataCollection",
        log_level=log_level,
        log_file=str(log_file),
        console_output=console_output
    )
    
    logger = logger_manager.get_logger()
    logger.info("MT4 數據收集系統日誌已初始化")
    
    return logger

# 導入 asyncio 用於異步函數檢查
import asyncio

# 模組級別的錯誤處理器和日誌器
_default_logger = None
_default_error_handler = None

def get_default_logger() -> logging.Logger:
    """獲取默認日誌器"""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging()
    return _default_logger

def get_default_error_handler() -> ErrorHandler:
    """獲取默認錯誤處理器"""
    global _default_error_handler
    if _default_error_handler is None:
        _default_error_handler = ErrorHandler(get_default_logger())
    return _default_error_handler

# 便捷的錯誤處理函數
def handle_error(error: Exception, 
                context: str = "",
                reraise: bool = False) -> Optional[MT4DataCollectionError]:
    """便捷的錯誤處理函數"""
    return get_default_error_handler().handle_error(error, context, reraise)

# 便捷的日誌記錄函數
def log_info(message: str, **kwargs):
    """記錄信息日誌"""
    get_default_logger().info(message, extra=kwargs)

def log_warning(message: str, **kwargs):
    """記錄警告日誌"""
    get_default_logger().warning(message, extra=kwargs)

def log_error(message: str, **kwargs):
    """記錄錯誤日誌"""
    get_default_logger().error(message, extra=kwargs)

def log_debug(message: str, **kwargs):
    """記錄調試日誌"""
    get_default_logger().debug(message, extra=kwargs)

# 使用示例
if __name__ == "__main__":
    # 設置日誌
    logger = setup_logging(LogLevel.DEBUG, console_output=True)
    
    # 創建錯誤處理器
    error_handler = ErrorHandler(logger)
    
    try:
        # 模擬一些錯誤
        raise TickCollectionError("模擬 Tick 收集錯誤", "TICK_001", {"symbol": "EURUSD"})
        
    except Exception as e:
        handled_error = error_handler.handle_error(e, "測試錯誤處理")
        print(f"處理後的錯誤: {handled_error.to_dict()}")
    
    # 顯示錯誤統計
    stats = error_handler.get_error_statistics()
    print(f"錯誤統計: {stats}")
    
    # 測試裝飾器
    @error_handler_decorator(TickCollectionError, logger, reraise=False)
    def test_function():
        raise ValueError("測試錯誤")
    
    result = test_function()
    print(f"裝飾器測試結果: {result}")