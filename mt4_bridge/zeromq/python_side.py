# -*- coding: utf-8 -*-
"""
ZeroMQ Python-MT4 橋接 - Python 端
這是最推薦的方案，提供低延遲雙向通訊
"""

import zmq
from datetime import datetime
from typing import Dict, Any, Optional


class MT4Bridge:
    def __init__(self, push_port: int = 5555, pull_port: int = 5556):
        """
        初始化 MT4 橋接

        Args:
            push_port: Python 發送命令到 MT4 的端口
            pull_port: Python 接收 MT4 數據的端口
        """
        self.context = zmq.Context()

        # 發送命令到 MT4
        self.push_socket = self.context.socket(zmq.PUSH)
        self.push_socket.connect(f"tcp://localhost:{push_port}")

        # 接收 MT4 數據
        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.connect(f"tcp://localhost:{pull_port}")

        # 設置接收超時
        self.pull_socket.setsockopt(zmq.RCVTIMEO, 1000)

    def send_command(self, command: str, **kwargs) -> None:
        """發送命令到 MT4"""
        message = {
            "command": command,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        self.push_socket.send_json(message)

    def receive_data(self, timeout: int = 1000) -> Optional[Dict[str, Any]]:
        """接收 MT4 數據"""
        try:
            return self.pull_socket.recv_json()
        except zmq.Again:
            return None

    # === 交易功能 ===

    def place_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: float = 0,
        sl: float = 0,
        tp: float = 0,
    ) -> Dict[str, Any]:
        """
        下單

        Args:
            symbol: 交易品種 (如 "EURUSD")
            order_type: "BUY" 或 "SELL"
            volume: 手數
            price: 價格 (市價單為 0)
            sl: 停損價
            tp: 獲利價
        """
        self.send_command(
            "PLACE_ORDER",
            symbol=symbol,
            order_type=order_type,
            volume=volume,
            price=price,
            sl=sl,
            tp=tp,
        )

        # 等待回應
        response = self.receive_data(timeout=5000)
        return response or {"error": "Timeout waiting for order response"}

    def close_order(self, ticket: int) -> Dict[str, Any]:
        """平倉"""
        self.send_command("CLOSE_ORDER", ticket=ticket)
        return self.receive_data(timeout=3000)

    def modify_order(
        self, ticket: int, sl: float = None, tp: float = None
    ) -> Dict[str, Any]:
        """修改訂單"""
        self.send_command("MODIFY_ORDER", ticket=ticket, sl=sl, tp=tp)
        return self.receive_data(timeout=3000)

    # === 數據功能 ===

    def get_quote(self, symbol: str) -> Optional[Dict[str, float]]:
        """獲取即時報價"""
        self.send_command("GET_QUOTE", symbol=symbol)
        return self.receive_data()

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """獲取帳戶資訊"""
        self.send_command("GET_ACCOUNT_INFO")
        return self.receive_data()

    def get_positions(self) -> Optional[list]:
        """獲取持倉"""
        self.send_command("GET_POSITIONS")
        return self.receive_data()

    def get_history(self, symbol: str, timeframe: str, bars: int) -> Optional[list]:
        """
        獲取歷史數據

        Args:
            symbol: 交易品種
            timeframe: 時間框架 ("M1", "M5", "M15", "M30", "H1", "H4", "D1")
            bars: K線數量
        """
        self.send_command("GET_HISTORY", symbol=symbol, timeframe=timeframe, bars=bars)
        return self.receive_data(timeout=5000)

    def close(self):
        """關閉連接"""
        self.push_socket.close()
        self.pull_socket.close()
        self.context.term()


# === 使用範例 ===


def example_usage():
    """示範如何使用 MT4Bridge"""

    # 創建橋接
    bridge = MT4Bridge()

    try:
        # 1. 獲取帳戶資訊
        print("獲取帳戶資訊...")
        account = bridge.get_account_info()
        if account:
            print(f"餘額: {account.get('balance')}")
            print(f"淨值: {account.get('equity')}")

        # 2. 獲取報價
        print("\n獲取 EURUSD 報價...")
        quote = bridge.get_quote("EURUSD")
        if quote:
            print(f"Bid: {quote.get('bid')}, Ask: {quote.get('ask')}")

        # 3. 下單示例
        print("\n下買單...")
        order_result = bridge.place_order(
            symbol="EURUSD",
            order_type="BUY",
            volume=0.01,  # 0.01 手
            sl=0,  # 停損價 (0 表示不設)
            tp=0,  # 獲利價 (0 表示不設)
        )
        print(f"下單結果: {order_result}")

        # 4. 獲取持倉
        print("\n獲取持倉...")
        positions = bridge.get_positions()
        if positions:
            for pos in positions:
                print(
                    f"訂單 {pos['ticket']}: {pos['symbol']} {pos['type']} {pos['volume']}手"
                )

        # 5. 獲取歷史數據
        print("\n獲取歷史數據...")
        history = bridge.get_history("EURUSD", "H1", 100)
        if history:
            print(f"獲得 {len(history)} 根 K 線")

    finally:
        bridge.close()


if __name__ == "__main__":
    example_usage()
