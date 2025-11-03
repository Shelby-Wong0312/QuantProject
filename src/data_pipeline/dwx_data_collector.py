# -*- coding: utf-8 -*-
"""
DWX MT4 數據收集系統
使用DWX ZeroMQ Connector進行MT4整合
"""

import sys
import time
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Callable
from collections import deque
import threading
import logging
from pathlib import Path

# 添加mt4_bridge到路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mt4_bridge.DWX_ZeroMQ_Connector_v2_0_1_RC8 import DWX_ZeroMQ_Connector

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DWXDataCollector:
    """
    基於DWX的MT4數據收集器
    """

    def __init__(
        self,
        client_id: str = "QuantProject",
        host: str = "localhost",
        push_port: int = 32768,
        pull_port: int = 32769,
        sub_port: int = 32770,
        verbose: bool = True,
    ):
        """
        初始化DWX數據收集器

        Args:
            client_id: 客戶端ID
            host: MT4主機地址
            push_port: PUSH端口（發送命令）
            pull_port: PULL端口（接收回應）
            sub_port: SUB端口（市場數據）
            verbose: 是否顯示詳細信息
        """
        self.client_id = client_id
        self.host = host
        self.push_port = push_port
        self.pull_port = pull_port
        self.sub_port = sub_port
        self.verbose = verbose

        # DWX連接器
        self.dwx = None

        # 數據存儲
        self.market_data = {}  # {symbol: MarketData}
        self.tick_history = {}  # {symbol: deque}
        self.account_info = {}
        self.open_trades = {}

        # 訂閱的品種
        self.subscribed_symbols = set()

        # 回調函數
        self.tick_callbacks = []
        self.trade_callbacks = []

        # 統計
        self.stats = {
            "connected": False,
            "tick_count": 0,
            "last_update": None,
            "errors": 0,
        }

        # 線程控制
        self._running = False
        self._monitor_thread = None

    def connect(self) -> bool:
        """連接到MT4"""
        try:
            logger.info(f"正在連接到MT4 {self.host}...")
            logger.info(
                f"端口配置: PUSH={self.push_port}, PULL={self.pull_port}, SUB={self.sub_port}"
            )

            # 創建DWX連接器
            self.dwx = DWX_ZeroMQ_Connector(
                _ClientID=self.client_id,
                _host=self.host,
                _protocol="tcp",
                _PUSH_PORT=self.push_port,
                _PULL_PORT=self.pull_port,
                _SUB_PORT=self.sub_port,
                _verbose=self.verbose,
                _poll_timeout=1000,
                _sleep_delay=0.001,
            )

            # 等待連接建立
            time.sleep(2)

            # 測試連接
            self.dwx._DWX_MTX_GET_ACCOUNT_INFO_()
            time.sleep(2)

            # 檢查是否收到回應
            if hasattr(self.dwx, "_AccountInfo") and self.dwx._AccountInfo:
                self.account_info = self.dwx._AccountInfo
                self.stats["connected"] = True
                logger.info("✓ 成功連接到MT4")
                logger.info(
                    f"  帳戶: {self.account_info.get('_account_number', 'N/A')}"
                )
                logger.info(
                    f"  餘額: ${self.account_info.get('_account_balance', 0):.2f}"
                )
                return True
            else:
                logger.warning("連接已建立但未收到帳戶信息")
                self.stats["connected"] = True
                return True

        except Exception as e:
            logger.error(f"連接失敗: {e}")
            self.stats["errors"] += 1
            return False

    def disconnect(self):
        """斷開連接"""
        try:
            self._running = False

            # 取消所有訂閱
            for symbol in list(self.subscribed_symbols):
                self.unsubscribe(symbol)

            # 關閉DWX連接
            if self.dwx:
                self.dwx._DWX_ZMQ_SHUTDOWN_()

            self.stats["connected"] = False
            logger.info("已斷開MT4連接")

        except Exception as e:
            logger.error(f"斷開連接時發生錯誤: {e}")

    def subscribe(self, symbols: List[str]):
        """訂閱交易品種"""
        if isinstance(symbols, str):
            [symbols]

        for symbol in symbols:
            try:
                # 使用DWX訂閱
                self.dwx._DWX_MTX_SUBSCRIBE_MARKETDATA_(symbol)
                self.subscribed_symbols.add(symbol)

                # 初始化數據存儲
                if symbol not in self.tick_history:
                    self.tick_history[symbol] = deque(maxlen=10000)

                logger.info(f"✓ 已訂閱 {symbol}")

            except Exception as e:
                logger.error(f"訂閱 {symbol} 失敗: {e}")
                self.stats["errors"] += 1

    def unsubscribe(self, symbols: List[str]):
        """取消訂閱"""
        if isinstance(symbols, str):
            [symbols]

        for symbol in symbols:
            try:
                # 使用DWX取消訂閱
                self.dwx._DWX_MTX_UNSUBSCRIBE_MARKETDATA_(symbol)
                self.subscribed_symbols.discard(symbol)
                logger.info(f"已取消訂閱 {symbol}")

            except Exception as e:
                logger.error(f"取消訂閱 {symbol} 失敗: {e}")

    def start_collection(self):
        """開始數據收集"""
        if not self.stats["connected"]:
            logger.error("未連接到MT4，無法開始收集")
            return False

        self._running = True

        # 啟動監控線程
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        logger.info("數據收集已啟動")
        return True

    def stop_collection(self):
        """停止數據收集"""
        self._running = False

        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2)

        logger.info("數據收集已停止")

    def _monitor_loop(self):
        """監控循環 - 處理市場數據"""
        while self._running:
            try:
                # 檢查DWX的市場數據
                if hasattr(self.dwx, "_Market_Data_DB") and self.dwx._Market_Data_DB:
                    for symbol, data in self.dwx._Market_Data_DB.items():
                        if symbol in self.subscribed_symbols:
                            self._process_market_data(symbol, data)

                # 更新統計
                self.stats["last_update"] = datetime.now()

                time.sleep(0.1)  # 100ms更新頻率

            except Exception as e:
                logger.error(f"監控循環錯誤: {e}")
                self.stats["errors"] += 1
                time.sleep(1)

    def _process_market_data(self, symbol: str, data: list):
        """處理市場數據"""
        try:
            # DWX數據格式: [bid, ask, timestamp]
            if len(data) >= 2:
                bid = float(data[0])
                ask = float(data[1])
                timestamp = datetime.now()

                # 更新市場數據
                self.market_data[symbol] = {
                    "symbol": symbol,
                    "bid": bid,
                    "ask": ask,
                    "spread": round((ask - bid) * 10000, 2),  # 點差（點）
                    "timestamp": timestamp,
                }

                # 添加到歷史
                self.tick_history[symbol].append(
                    {"timestamp": timestamp, "bid": bid, "ask": ask}
                )

                # 更新統計
                self.stats["tick_count"] += 1

                # 觸發回調
                for callback in self.tick_callbacks:
                    callback(symbol, self.market_data[symbol])

        except Exception as e:
            logger.error(f"處理市場數據錯誤 {symbol}: {e}")
            self.stats["errors"] += 1

    def get_latest_price(self, symbol: str) -> Optional[Dict]:
        """獲取最新價格"""
        return self.market_data.get(symbol)

    def get_tick_history(self, symbol: str, periods: int = 100) -> pd.DataFrame:
        """獲取Tick歷史數據"""
        if symbol not in self.tick_history:
            return pd.DataFrame()

        ticks = list(self.tick_history[symbol])[-periods:]
        if not ticks:
            return pd.DataFrame()

        df = pd.DataFrame(ticks)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        return df

    def get_account_info(self) -> Dict:
        """獲取帳戶信息"""
        try:
            self.dwx._DWX_MTX_GET_ACCOUNT_INFO_()
            time.sleep(1)

            if hasattr(self.dwx, "_AccountInfo"):
                self.account_info = self.dwx._AccountInfo

        except Exception as e:
            logger.error(f"獲取帳戶信息失敗: {e}")

        return self.account_info

    def get_open_trades(self) -> List[Dict]:
        """獲取開倉交易"""
        try:
            self.dwx._DWX_MTX_GET_ALL_OPEN_TRADES_()
            time.sleep(1)

            # DWX可能將交易存儲在不同的屬性中
            # 需要根據實際測試確定

        except Exception as e:
            logger.error(f"獲取開倉交易失敗: {e}")

        return []

    def place_order(
        self,
        symbol: str,
        order_type: str,
        lots: float,
        sl: float = 0,
        tp: float = 0,
        comment: str = "",
    ) -> bool:
        """下單"""
        try:
            order = {
                "_symbol": symbol,
                "_type": 0 if order_type.upper() == "BUY" else 1,
                "_lots": lots,
                "_SL": sl,
                "_TP": tp,
                "_comment": comment,
                "_magic": 123456,
                "_ticket": 0,
            }

            self.dwx._DWX_MTX_NEW_TRADE_(order)
            logger.info(f"訂單已發送: {symbol} {order_type} {lots} lots")
            return True

        except Exception as e:
            logger.error(f"下單失敗: {e}")
            return False

    def close_position(self, ticket: int) -> bool:
        """平倉"""
        try:
            self.dwx._DWX_MTX_CLOSE_TRADE_BY_TICKET_(ticket)
            logger.info(f"平倉請求已發送: Ticket {ticket}")
            return True

        except Exception as e:
            logger.error(f"平倉失敗: {e}")
            return False

    def add_tick_callback(self, callback: Callable):
        """添加Tick數據回調"""
        self.tick_callbacks.append(callback)

    def get_stats(self) -> Dict:
        """獲取統計信息"""
        return {
            "connected": self.stats["connected"],
            "subscribed_symbols": list(self.subscribed_symbols),
            "tick_count": self.stats["tick_count"],
            "last_update": self.stats["last_update"],
            "errors": self.stats["errors"],
            "current_prices": {
                symbol: {
                    "bid": data["bid"],
                    "ask": data["ask"],
                    "spread": data["spread"],
                }
                for symbol, data in self.market_data.items()
            },
        }


# 便利函數
def create_dwx_collector(**kwargs) -> DWXDataCollector:
    """創建DWX數據收集器"""
    return DWXDataCollector(**kwargs)


def test_dwx_connection():
    """Test DWX connection"""
    print("\n" + "=" * 60)
    print(" DWX MT4 Data Collection Test ")
    print("=" * 60)

    # 創建收集器
    collector = create_dwx_collector(verbose=True)

    # 連接
    if collector.connect():
        print("\n[SUCCESS] Connected")

        # 訂閱EURUSD
        collector.subscribe(["EURUSD", "GBPUSD"])

        # 開始收集
        collector.start_collection()

        # 定義回調函數
        def on_tick(symbol, data):
            print(
                f"{symbol}: Bid={data['bid']:.5f}, Ask={data['ask']:.5f}, Spread={data['spread']}"
            )

        collector.add_tick_callback(on_tick)

        # 運行10秒
        print("\nCollecting data (10 seconds)...")
        for i in range(10):
            time.sleep(1)
            print(f"  {i+1}/10 seconds...")

        # 顯示統計
        stats = collector.get_stats()
        print("\nStatistics:")
        print(f"  Ticks: {stats['tick_count']}")
        print(f"  Errors: {stats['errors']}")

        # 停止
        collector.stop_collection()
        collector.disconnect()

    else:
        print("\n[FAILED] Connection failed")
        print("\nPlease check:")
        print("1. MT4 is running and logged in")
        print("2. DWX Server EA is loaded")
        print("3. Ports are correct (32768, 32769, 32770)")
        print("4. AutoTrading is enabled")


if __name__ == "__main__":
    test_dwx_connection()
