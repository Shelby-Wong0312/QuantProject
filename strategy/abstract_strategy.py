# 檔案位置: strategies/abstract_strategy.py

from abc import ABC, abstractmethod
import pandas as pd

class Signal:
    """
    一個標準化的資料類別，代表一個交易訊號。
    """
    def __init__(self, timestamp, symbol, action, quantity=None, price=None, sl=None, tp=None, order_type='MARKET', comment=""):
        self.timestamp = timestamp
        self.symbol = symbol
        self.action = action  # e.g., 'BUY_ENTRY', 'SELL_ENTRY', 'CLOSE_LONG_CONDITION', etc.
        self.quantity = quantity
        self.price = price
        self.sl = sl
        self.tp = tp
        self.order_type = order_type
        self.comment = comment

    def __repr__(self):
        """
        提供一個清晰的字串表示，方便日誌記錄和偵錯。
        """
        return (
            f"Signal(timestamp={self.timestamp}, symbol='{self.symbol}', action='{self.action}', "
            f"quantity={self.quantity}, price={self.price}, sl={self.sl}, tp={self.tp}, "
            f"order_type='{self.order_type}', comment='{self.comment}')"
        )

class AbstractStrategyBase(ABC):
    """
    所有交易策略都必須繼承的抽象基類。
    它定義了一個策略所需具備的通用介面和結構。
    """
    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.symbol = parameters.get('symbol', 'UNKNOWN_SYMBOL')
        # 初始化策略的特定參數
        self._initialize_parameters()

    @abstractmethod
    def _initialize_parameters(self):
        """
        從 self.parameters 字典中初始化策略所需的特定參數。
        例如：self.rsi_period = self.parameters.get('rsi_period', 14)
        """
        pass

    @abstractmethod
    def on_data(self, data_slice: pd.DataFrame) -> list:
        """
        處理新的市場數據並產生交易訊號的核心方法。
        
        :param data_slice: 一個包含最新市場數據的 pandas DataFrame。
                           最後一列(row)是當前的 K 棒。
        :return: 一個包含零或多個 Signal 物件的列表。
        """
        pass

    def get_indicator_definitions(self) -> dict:
        """
        (可選) 返回一個描述指標定義的字典，供適配器使用 (例如用於繪圖)。
        範例: {'rsi': (pandas_ta.rsi, {'length': 14}), ...}
        """
        return {}
    