# -*- coding: utf-8 -*-
"""
檔案通訊 Python-MT4 橋接 - Python 端
簡單但有效的通訊方式，適合初學者
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional
import threading
import queue


class FileBridge:
    def __init__(
        self,
        commands_dir: str = "C:/MT4_Bridge/Commands/",
        responses_dir: str = "C:/MT4_Bridge/Responses/",
    ):
        """
        初始化檔案橋接

        Args:
            commands_dir: 命令檔案目錄（Python → MT4）
            responses_dir: 回應檔案目錄（MT4 → Python）
        """
        self.commands_dir = commands_dir
        self.responses_dir = responses_dir

        # 創建目錄
        os.makedirs(commands_dir, exist_ok=True)
        os.makedirs(responses_dir, exist_ok=True)

        # 命令計數器
        self.command_id = 0

        # 回應佇列
        self.response_queue = queue.Queue()

        # 啟動監聽線程
        self.listening = True
        self.listener_thread = threading.Thread(target=self._listen_responses)
        self.listener_thread.daemon = True
        self.listener_thread.start()

    def _listen_responses(self):
        """監聽回應檔案"""
        while self.listening:
            try:
                for filename in os.listdir(self.responses_dir):
                    if filename.endswith(".json"):
                        filepath = os.path.join(self.responses_dir, filename)
                        try:
                            with open(filepath, "r") as f:
                                response = json.load(f)
                            self.response_queue.put(response)
                            os.remove(filepath)  # 刪除已讀取的檔案
                        except Exception:
                            pass
            except Exception:
                pass
            time.sleep(0.1)

    def send_command(self, command: str, **kwargs) -> str:
        """發送命令到 MT4"""
        self.command_id += 1
        command_data = {
            "id": self.command_id,
            "command": command,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }

        # 寫入命令檔案
        filename = f"cmd_{self.command_id}.json"
        filepath = os.path.join(self.commands_dir, filename)

        with open(filepath, "w") as f:
            json.dump(command_data, f)

        return str(self.command_id)

    def wait_response(
        self, command_id: str = None, timeout: float = 5.0
    ) -> Optional[Dict[str, Any]]:
        """等待回應"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = self.response_queue.get(timeout=0.1)
                if command_id is None or str(response.get("command_id")) == command_id:
                    return response
                else:
                    # 不是我們要的回應，放回佇列
                    self.response_queue.put(response)
            except queue.Empty:
                pass

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
        """下單"""
        cmd_id = self.send_command(
            "PLACE_ORDER",
            symbol=symbol,
            order_type=order_type,
            volume=volume,
            price=price,
            sl=sl,
            tp=tp,
        )

        response = self.wait_response(cmd_id)
        return response or {"error": "Timeout waiting for response"}

    def close_order(self, ticket: int) -> Dict[str, Any]:
        """平倉"""
        cmd_id = self.send_command("CLOSE_ORDER", ticket=ticket)
        response = self.wait_response(cmd_id)
        return response or {"error": "Timeout"}

    # === 數據功能 ===

    def get_quote(self, symbol: str) -> Optional[Dict[str, float]]:
        """獲取報價"""
        cmd_id = self.send_command("GET_QUOTE", symbol=symbol)
        return self.wait_response(cmd_id, timeout=2.0)

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """獲取帳戶資訊"""
        cmd_id = self.send_command("GET_ACCOUNT_INFO")
        return self.wait_response(cmd_id)

    def get_positions(self) -> Optional[list]:
        """獲取持倉"""
        cmd_id = self.send_command("GET_POSITIONS")
        response = self.wait_response(cmd_id)
        return response.get("positions") if response else None

    def close(self):
        """關閉橋接"""
        self.listening = False
        self.listener_thread.join(timeout=1)


# === 使用範例 ===


def example_usage():
    """示範如何使用檔案橋接"""

    # 創建橋接
    bridge = FileBridge()

    try:
        # 1. 獲取帳戶資訊
        print("獲取帳戶資訊...")
        account = bridge.get_account_info()
        if account:
            print(f"餘額: {account.get('balance')}")

        # 2. 獲取報價
        print("\n獲取 EURUSD 報價...")
        quote = bridge.get_quote("EURUSD")
        if quote:
            print(f"Bid: {quote.get('bid')}, Ask: {quote.get('ask')}")

        # 3. 下單
        print("\n下買單...")
        order_result = bridge.place_order(
            symbol="EURUSD", order_type="BUY", volume=0.01
        )
        print(f"下單結果: {order_result}")

    finally:
        bridge.close()


if __name__ == "__main__":
    example_usage()
