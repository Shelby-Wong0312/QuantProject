# -*- coding: utf-8 -*-
"""
MT4-Python橋接連接器 - ZeroMQ通訊主模組
支援REQ-REP模式的命令通訊和PUB-SUB模式的實時數據流
包含錯誤處理和重連機制
"""

import zmq
import json
import time
import threading
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
from queue import Queue, Empty
from enum import Enum

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MessageType(Enum):
    """訊息類型枚舉"""
    COMMAND = "command"
    RESPONSE = "response"
    DATA_STREAM = "data_stream"
    HEARTBEAT = "heartbeat"
    ERROR = "error"

class ConnectionState(Enum):
    """連接狀態枚舉"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"

class MT4Connector:
    """
    MT4-Python ZeroMQ連接器
    提供雙向通訊：REQ-REP模式用於命令/回應，PUB-SUB模式用於數據流
    """
    
    def __init__(self, 
                 req_port: int = 5555,    # REQ-REP: Python請求端口
                 rep_port: int = 5556,    # REQ-REP: MT4回應端口
                 pub_port: int = 5557,    # PUB-SUB: MT4發布端口
                 sub_port: int = 5558,    # PUB-SUB: Python訂閱端口
                 heartbeat_interval: float = 30.0,  # 心跳間隔(秒)
                 max_retries: int = 3):
        """
        初始化MT4連接器
        
        Args:
            req_port: REQ-REP模式 - Python發送請求的端口
            rep_port: REQ-REP模式 - MT4回應的端口  
            pub_port: PUB-SUB模式 - MT4發布數據的端口
            sub_port: PUB-SUB模式 - Python訂閱的端口
            heartbeat_interval: 心跳間隔
            max_retries: 最大重試次數
        """
        self.req_port = req_port
        self.rep_port = rep_port
        self.pub_port = pub_port
        self.sub_port = sub_port
        self.heartbeat_interval = heartbeat_interval
        self.max_retries = max_retries
        
        # ZeroMQ上下文和套接字
        self.context = None
        self.req_socket = None      # 發送命令
        self.sub_socket = None      # 接收數據流
        
        # 連接狀態
        self.state = ConnectionState.DISCONNECTED
        self.last_heartbeat = None
        
        # 線程控制
        self._running = False
        self._heartbeat_thread = None
        self._data_listener_thread = None
        
        # 回調函數
        self.data_callbacks = {}  # {topic: [callback_functions]}
        self.error_callback = None
        
        # 內部隊列
        self.response_queue = Queue()
        self.data_queue = Queue()
        
        # 重連設置
        self.retry_count = 0
        self.reconnect_delay = 1.0  # 重連延遲(秒)
        
    def connect(self) -> bool:
        """
        建立與MT4的連接
        
        Returns:
            bool: 連接是否成功
        """
        try:
            self.state = ConnectionState.CONNECTING
            logger.info("正在連接到MT4...")
            
            # 創建ZeroMQ上下文
            self.context = zmq.Context()
            
            # 設置REQ套接字(發送命令)
            self.req_socket = self.context.socket(zmq.REQ)
            self.req_socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5秒超時
            self.req_socket.setsockopt(zmq.SNDTIMEO, 5000)
            self.req_socket.connect(f"tcp://localhost:{self.req_port}")
            
            # 設置SUB套接字(接收數據流)
            self.sub_socket = self.context.socket(zmq.SUB)
            self.sub_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1秒超時
            self.sub_socket.connect(f"tcp://localhost:{self.sub_port}")
            self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # 訂閱所有主題
            
            # 測試連接
            if self._test_connection():
                self.state = ConnectionState.CONNECTED
                self.retry_count = 0
                self.last_heartbeat = datetime.now()
                
                # 啟動後台線程
                self._start_threads()
                
                logger.info("成功連接到MT4")
                return True
            else:
                self._cleanup_sockets()
                self.state = ConnectionState.ERROR
                return False
                
        except Exception as e:
            logger.error(f"連接MT4失敗: {e}")
            self.state = ConnectionState.ERROR
            self._cleanup_sockets()
            return False
    
    def disconnect(self):
        """斷開連接"""
        logger.info("正在斷開與MT4的連接...")
        
        self._running = False
        self.state = ConnectionState.DISCONNECTED
        
        # 等待線程結束
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=2.0)
        if self._data_listener_thread and self._data_listener_thread.is_alive():
            self._data_listener_thread.join(timeout=2.0)
        
        # 清理套接字
        self._cleanup_sockets()
        
        logger.info("已斷開與MT4的連接")
    
    def _cleanup_sockets(self):
        """清理套接字資源"""
        try:
            if self.req_socket:
                self.req_socket.close()
                self.req_socket = None
            if self.sub_socket:
                self.sub_socket.close() 
                self.sub_socket = None
            if self.context:
                self.context.term()
                self.context = None
        except Exception as e:
            logger.warning(f"清理套接字時發生錯誤: {e}")
    
    def _test_connection(self) -> bool:
        """測試連接是否正常"""
        try:
            # 發送心跳測試
            test_msg = {
                "type": MessageType.HEARTBEAT.value,
                "timestamp": datetime.now().isoformat()
            }
            
            self.req_socket.send_json(test_msg)
            response = self.req_socket.recv_json()
            
            return response.get("status") == "ok"
        except Exception as e:
            logger.error(f"連接測試失敗: {e}")
            return False
    
    def _start_threads(self):
        """啟動後台線程"""
        self._running = True
        
        # 心跳線程
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_worker, daemon=True)
        self._heartbeat_thread.start()
        
        # 數據監聽線程
        self._data_listener_thread = threading.Thread(target=self._data_listener_worker, daemon=True)
        self._data_listener_thread.start()
    
    def _heartbeat_worker(self):
        """心跳線程工作函數"""
        while self._running:
            try:
                if self.state == ConnectionState.CONNECTED:
                    # 檢查上次心跳時間
                    if self.last_heartbeat:
                        elapsed = (datetime.now() - self.last_heartbeat).total_seconds()
                        if elapsed > self.heartbeat_interval * 2:
                            logger.warning("心跳超時，嘗試重連...")
                            self._attempt_reconnect()
                            continue
                    
                    # 發送心跳
                    self._send_heartbeat()
                
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"心跳線程錯誤: {e}")
                self._attempt_reconnect()
    
    def _send_heartbeat(self):
        """發送心跳信號"""
        try:
            heartbeat_msg = {
                "type": MessageType.HEARTBEAT.value,
                "timestamp": datetime.now().isoformat()
            }
            
            self.req_socket.send_json(heartbeat_msg)
            response = self.req_socket.recv_json()
            
            if response.get("status") == "ok":
                self.last_heartbeat = datetime.now()
            else:
                logger.warning("心跳回應異常")
                
        except zmq.Again:
            logger.warning("心跳超時")
        except Exception as e:
            logger.error(f"發送心跳失敗: {e}")
    
    def _data_listener_worker(self):
        """數據監聽線程工作函數"""
        while self._running:
            try:
                if self.state == ConnectionState.CONNECTED and self.sub_socket:
                    # 非阻塞接收數據
                    try:
                        data = self.sub_socket.recv_json(zmq.NOBLOCK)
                        self._process_received_data(data)
                    except zmq.Again:
                        # 沒有數據，繼續
                        time.sleep(0.01)
                else:
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"數據監聽線程錯誤: {e}")
                time.sleep(1.0)
    
    def _process_received_data(self, data: Dict[str, Any]):
        """處理接收到的數據"""
        try:
            msg_type = data.get("type")
            topic = data.get("topic", "default")
            
            # 觸發回調函數
            if topic in self.data_callbacks:
                for callback in self.data_callbacks[topic]:
                    try:
                        callback(data)
                    except Exception as e:
                        logger.error(f"回調函數執行錯誤: {e}")
            
            # 將數據放入隊列
            self.data_queue.put(data)
            
        except Exception as e:
            logger.error(f"處理接收數據時發生錯誤: {e}")
    
    def _attempt_reconnect(self):
        """嘗試重新連接"""
        if self.retry_count >= self.max_retries:
            logger.error(f"重連次數達到上限({self.max_retries})，停止重連")
            self.state = ConnectionState.ERROR
            return
        
        self.retry_count += 1
        logger.info(f"嘗試重連... ({self.retry_count}/{self.max_retries})")
        
        # 清理現有連接
        self._cleanup_sockets()
        
        # 等待一段時間再重連
        time.sleep(self.reconnect_delay * self.retry_count)
        
        # 嘗試重新連接
        if self.connect():
            logger.info("重連成功")
        else:
            logger.warning("重連失敗")
    
    def send_command(self, command: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        發送命令到MT4並等待回應
        
        Args:
            command: 命令類型
            **kwargs: 命令參數
            
        Returns:
            Optional[Dict]: MT4的回應，失敗時返回None
        """
        if self.state != ConnectionState.CONNECTED:
            logger.error("未連接到MT4，無法發送命令")
            return None
        
        try:
            message = {
                "type": MessageType.COMMAND.value,
                "command": command,
                "timestamp": datetime.now().isoformat(),
                **kwargs
            }
            
            self.req_socket.send_json(message)
            response = self.req_socket.recv_json()
            
            return response
        except zmq.Again:
            logger.error("命令超時")
            return None
        except Exception as e:
            logger.error(f"發送命令失敗: {e}")
            return None
    
    def subscribe_data(self, topic: str, callback: Callable[[Dict[str, Any]], None]):
        """
        訂閱數據主題
        
        Args:
            topic: 數據主題
            callback: 回調函數
        """
        if topic not in self.data_callbacks:
            self.data_callbacks[topic] = []
        
        self.data_callbacks[topic].append(callback)
        logger.info(f"已訂閱數據主題: {topic}")
    
    def unsubscribe_data(self, topic: str, callback: Callable[[Dict[str, Any]], None]):
        """取消訂閱數據主題"""
        if topic in self.data_callbacks:
            try:
                self.data_callbacks[topic].remove(callback)
                if not self.data_callbacks[topic]:
                    del self.data_callbacks[topic]
                logger.info(f"已取消訂閱數據主題: {topic}")
            except ValueError:
                logger.warning(f"回調函數不存在於主題 {topic} 中")
    
    def get_data_nowait(self) -> Optional[Dict[str, Any]]:
        """
        非阻塞獲取數據隊列中的數據
        
        Returns:
            Optional[Dict]: 數據，如果隊列為空則返回None
        """
        try:
            return self.data_queue.get_nowait()
        except Empty:
            return None
    
    def wait_for_data(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        等待數據隊列中的數據
        
        Args:
            timeout: 等待超時時間(秒)
            
        Returns:
            Optional[Dict]: 數據，超時則返回None
        """
        try:
            return self.data_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def set_error_callback(self, callback: Callable[[str], None]):
        """設置錯誤回調函數"""
        self.error_callback = callback
    
    def is_connected(self) -> bool:
        """檢查是否已連接"""
        return self.state == ConnectionState.CONNECTED
    
    def get_connection_state(self) -> ConnectionState:
        """獲取連接狀態"""
        return self.state
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取連接統計信息"""
        return {
            "state": self.state.value,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "retry_count": self.retry_count,
            "subscribed_topics": list(self.data_callbacks.keys()),
            "data_queue_size": self.data_queue.qsize()
        }


class MT4ConnectorManager:
    """
    MT4連接器管理器
    管理多個連接器實例，提供統一的接口
    """
    
    def __init__(self):
        self.connectors = {}  # {name: MT4Connector}
        self.default_connector = None
    
    def add_connector(self, name: str, connector: MT4Connector, set_as_default: bool = False):
        """添加連接器"""
        self.connectors[name] = connector
        if set_as_default or self.default_connector is None:
            self.default_connector = connector
    
    def get_connector(self, name: str = None) -> Optional[MT4Connector]:
        """獲取連接器"""
        if name:
            return self.connectors.get(name)
        else:
            return self.default_connector
    
    def remove_connector(self, name: str):
        """移除連接器"""
        if name in self.connectors:
            connector = self.connectors[name]
            if connector.is_connected():
                connector.disconnect()
            del self.connectors[name]
            
            if self.default_connector == connector:
                self.default_connector = next(iter(self.connectors.values())) if self.connectors else None
    
    def connect_all(self) -> Dict[str, bool]:
        """連接所有連接器"""
        results = {}
        for name, connector in self.connectors.items():
            results[name] = connector.connect()
        return results
    
    def disconnect_all(self):
        """斷開所有連接器"""
        for connector in self.connectors.values():
            if connector.is_connected():
                connector.disconnect()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """獲取所有連接器的統計信息"""
        return {name: connector.get_stats() for name, connector in self.connectors.items()}


# 全局連接器管理器實例
connector_manager = MT4ConnectorManager()

def get_default_connector() -> Optional[MT4Connector]:
    """獲取默認連接器"""
    return connector_manager.get_connector()

def create_default_connector(**kwargs) -> MT4Connector:
    """創建默認連接器"""
    connector = MT4Connector(**kwargs)
    connector_manager.add_connector("default", connector, set_as_default=True)
    return connector