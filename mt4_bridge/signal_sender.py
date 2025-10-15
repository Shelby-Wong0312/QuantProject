# -*- coding: utf-8 -*-
"""
MT4交易信號發送器模組
負責發送開倉/平倉信號、倉位管理命令、風險控制參數設定
提供完整的交易執行和管理功能
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from threading import Lock
import uuid

from .connector import MT4Connector, get_default_connector

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """訂單類型枚舉"""

    BUY = "BUY"
    SELL = "SELL"
    BUY_LIMIT = "BUY_LIMIT"
    SELL_LIMIT = "SELL_LIMIT"
    BUY_STOP = "BUY_STOP"
    SELL_STOP = "SELL_STOP"


class OrderStatus(Enum):
    """訂單狀態枚舉"""

    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    PARTIAL_FILLED = "PARTIAL_FILLED"


class SignalType(Enum):
    """信號類型枚舉"""

    ENTRY = "ENTRY"  # 進場信號
    EXIT = "EXIT"  # 出場信號
    MODIFY = "MODIFY"  # 修改訂單
    CANCEL = "CANCEL"  # 取消訂單


@dataclass
class TradingSignal:
    """交易信號數據結構"""

    signal_id: str
    symbol: str
    signal_type: SignalType
    order_type: OrderType
    volume: float
    price: float = 0.0  # 0表示市價
    stop_loss: float = 0.0  # 0表示不設停損
    take_profit: float = 0.0  # 0表示不設獲利
    expiry: datetime = None  # 訂單到期時間
    comment: str = ""
    magic_number: int = 0

    # 風險管理參數
    max_slippage: float = 3.0  # 最大滑點(點)
    risk_percent: float = 2.0  # 風險百分比

    # 時間戳
    created_at: datetime = None
    sent_at: datetime = None
    executed_at: datetime = None

    def __post_init__(self):
        if self.signal_id is None:
            self.signal_id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        asdict(self)
        # 轉換枚舉值為字符串
        data["signal_type"] = self.signal_type.value
        data["order_type"] = self.order_type.value
        # 轉換datetime為ISO格式字符串
        for key, value in data.items():
            if isinstance(value, datetime) and value:
                data[key] = value.isoformat()
        return data


@dataclass
class OrderResult:
    """訂單執行結果"""

    signal_id: str
    success: bool
    ticket: int = 0
    error_code: int = 0
    error_message: str = ""
    executed_price: float = 0.0
    executed_volume: float = 0.0
    execution_time: datetime = None

    def __post_init__(self):
        if self.execution_time is None:
            self.execution_time = datetime.now()


@dataclass
class PositionInfo:
    """持倉信息"""

    ticket: int
    symbol: str
    order_type: OrderType
    volume: float
    open_price: float
    current_price: float
    profit: float
    swap: float
    commission: float
    stop_loss: float = 0.0
    take_profit: float = 0.0
    open_time: datetime = None
    comment: str = ""
    magic_number: int = 0

    @property
    def unrealized_pnl(self) -> float:
        """未實現盈虧"""
        return self.profit + self.swap + self.commission


class RiskManager:
    """風險管理器"""

    def __init__(self):
        self.max_daily_loss = 0.0  # 每日最大虧損
        self.max_drawdown = 0.0  # 最大回撤
        self.max_positions_per_symbol = 1  # 每個品種最大持倉數
        self.max_total_positions = 10  # 總最大持倉數
        self.max_risk_per_trade = 0.02  # 每筆交易最大風險比例

        # 統計數據
        self.daily_pnl = 0.0
        self.current_drawdown = 0.0
        self.position_count = {}  # {symbol: count}
        self.total_positions = 0

    def validate_signal(
        self, signal: TradingSignal, account_balance: float, current_positions: List[PositionInfo]
    ) -> tuple[bool, str]:
        """
        驗證交易信號是否符合風險管理規則

        Args:
            signal: 交易信號
            account_balance: 賬戶餘額
            current_positions: 當前持倉列表

        Returns:
            tuple[bool, str]: (是否通過驗證, 錯誤信息)
        """
        try:
            # 檢查每日虧損限制
            if self.max_daily_loss > 0 and self.daily_pnl <= -self.max_daily_loss:
                return False, f"已達到每日最大虧損限制: {self.max_daily_loss}"

            # 檢查最大回撤限制
            if self.max_drawdown > 0 and self.current_drawdown >= self.max_drawdown:
                return False, f"已達到最大回撤限制: {self.max_drawdown}"

            # 檢查單一品種持倉限制
            symbol_positions = len([p for p in current_positions if p.symbol == signal.symbol])
            if symbol_positions >= self.max_positions_per_symbol:
                return (
                    False,
                    f"品種 {signal.symbol} 已達到最大持倉限制: {self.max_positions_per_symbol}",
                )

            # 檢查總持倉限制
            if len(current_positions) >= self.max_total_positions:
                return False, f"已達到總持倉限制: {self.max_total_positions}"

            # 檢查單筆交易風險
            if signal.risk_percent > self.max_risk_per_trade * 100:
                return (
                    False,
                    f"單筆交易風險過高: {signal.risk_percent}% > {self.max_risk_per_trade * 100}%",
                )

            # 計算交易金額是否超過可用資金
            position_value = signal.volume * signal.price * 100000  # 假設標準手
            risk_amount = account_balance * (signal.risk_percent / 100)

            if risk_amount > account_balance * self.max_risk_per_trade:
                return (
                    False,
                    f"風險金額過高: {risk_amount} > {account_balance * self.max_risk_per_trade}",
                )

            return True, "驗證通過"

        except Exception as e:
            return False, f"風險驗證時發生錯誤: {e}"

    def calculate_position_size(
        self, signal: TradingSignal, account_balance: float, pip_value: float
    ) -> float:
        """
        根據風險管理規則計算持倉大小

        Args:
            signal: 交易信號
            account_balance: 賬戶餘額
            pip_value: 點值

        Returns:
            float: 建議的持倉大小(手)
        """
        try:
            # 計算風險金額
            risk_amount = account_balance * (signal.risk_percent / 100)

            # 計算停損點數
            if signal.stop_loss > 0:
                if signal.order_type in [OrderType.BUY, OrderType.BUY_LIMIT, OrderType.BUY_STOP]:
                    stop_loss_pips = abs(signal.price - signal.stop_loss) * 10000  # 假設4位小數
                else:
                    stop_loss_pips = abs(signal.stop_loss - signal.price) * 10000

                # 計算持倉大小
                if stop_loss_pips > 0:
                    position_size = risk_amount / (stop_loss_pips * pip_value)
                    return min(position_size, signal.volume)  # 不超過原始信號的手數

            return signal.volume  # 如果無法計算，使用原始手數

        except Exception as e:
            logger.error(f"計算持倉大小時發生錯誤: {e}")
            return signal.volume


class MT4SignalSender:
    """MT4交易信號發送器"""

    def __init__(self, connector: MT4Connector = None):
        """
        初始化信號發送器

        Args:
            connector: MT4連接器，默認使用全局默認連接器
        """
        self.connector = connector or get_default_connector()
        self.risk_manager = RiskManager()

        # 信號追蹤
        self.pending_signals = {}  # {signal_id: TradingSignal}
        self.executed_orders = {}  # {signal_id: OrderResult}
        self.lock = Lock()

        # 回調函數
        self.order_callbacks = []  # [(callback_func, filter_criteria)]

        # 統計數據
        self.signals_sent = 0
        self.orders_filled = 0
        self.orders_rejected = 0

    def set_risk_parameters(self, **kwargs):
        """設置風險管理參數"""
        for key, value in kwargs.items():
            if hasattr(self.risk_manager, key):
                setattr(self.risk_manager, key, value)
                logger.info(f"風險參數已更新: {key} = {value}")

    def send_signal(self, signal: TradingSignal, validate_risk: bool = True) -> OrderResult:
        """
        發送交易信號

        Args:
            signal: 交易信號
            validate_risk: 是否進行風險驗證

        Returns:
            OrderResult: 訂單執行結果
        """
        try:
            with self.lock:
                signal.sent_at = datetime.now()
                self.pending_signals[signal.signal_id] = signal

            # 檢查連接
            if not self.connector or not self.connector.is_connected():
                result = OrderResult(
                    signal_id=signal.signal_id, success=False, error_message="MT4連接器未連接"
                )
                self._process_order_result(result)
                return result

            # 風險驗證
            if validate_risk:
                account_info = self._get_account_info()
                current_positions = self._get_current_positions()

                is_valid, error_msg = self.risk_manager.validate_signal(
                    signal, account_info.get("balance", 0), current_positions
                )

                if not is_valid:
                    result = OrderResult(
                        signal_id=signal.signal_id,
                        success=False,
                        error_message=f"風險驗證失敗: {error_msg}",
                    )
                    self._process_order_result(result)
                    return result

            # 根據信號類型處理
            if signal.signal_type == SignalType.ENTRY:
                return self._send_entry_order(signal)
            elif signal.signal_type == SignalType.EXIT:
                return self._send_exit_order(signal)
            elif signal.signal_type == SignalType.MODIFY:
                return self._send_modify_order(signal)
            elif signal.signal_type == SignalType.CANCEL:
                return self._send_cancel_order(signal)
            else:
                result = OrderResult(
                    signal_id=signal.signal_id,
                    success=False,
                    error_message=f"未知的信號類型: {signal.signal_type}",
                )
                self._process_order_result(result)
                return result

        except Exception as e:
            logger.error(f"發送交易信號時發生錯誤: {e}")
            result = OrderResult(signal_id=signal.signal_id, success=False, error_message=str(e))
            self._process_order_result(result)
            return result

    def _send_entry_order(self, signal: TradingSignal) -> OrderResult:
        """發送進場訂單"""
        try:
            # 構建訂單參數
            order_data = {
                "command": "PLACE_ORDER",
                "symbol": signal.symbol,
                "order_type": signal.order_type.value,
                "volume": signal.volume,
                "price": signal.price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "comment": signal.comment or f"Signal_{signal.signal_id[:8]}",
                "magic_number": signal.magic_number,
                "max_slippage": signal.max_slippage,
                "signal_id": signal.signal_id,
            }

            # 發送訂單
            response = self.connector.send_command(**order_data)

            if response and response.get("success"):
                result = OrderResult(
                    signal_id=signal.signal_id,
                    success=True,
                    ticket=response.get("ticket", 0),
                    executed_price=response.get("price", signal.price),
                    executed_volume=response.get("volume", signal.volume),
                )
                self.orders_filled += 1
            else:
                result = OrderResult(
                    signal_id=signal.signal_id,
                    success=False,
                    error_code=response.get("error_code", -1) if response else -1,
                    error_message=(
                        response.get("error_message", "未知錯誤") if response else "連接超時"
                    ),
                )
                self.orders_rejected += 1

            self._process_order_result(result)
            return result

        except Exception as e:
            logger.error(f"發送進場訂單時發生錯誤: {e}")
            result = OrderResult(signal_id=signal.signal_id, success=False, error_message=str(e))
            self._process_order_result(result)
            return result

    def _send_exit_order(self, signal: TradingSignal) -> OrderResult:
        """發送出場訂單"""
        try:
            # 查找要關閉的持倉
            positions = self._get_current_positions()
            target_positions = [p for p in positions if p.symbol == signal.symbol]

            if not target_positions:
                result = OrderResult(
                    signal_id=signal.signal_id,
                    success=False,
                    error_message=f"未找到品種 {signal.symbol} 的持倉",
                )
                self._process_order_result(result)
                return result

            # 選擇要關閉的持倉(這裡簡化為關閉第一個)
            position = target_positions[0]

            # 構建平倉參數
            close_data = {
                "command": "CLOSE_ORDER",
                "ticket": position.ticket,
                "volume": (
                    min(signal.volume, position.volume) if signal.volume > 0 else position.volume
                ),
                "price": signal.price,
                "max_slippage": signal.max_slippage,
                "signal_id": signal.signal_id,
            }

            # 發送平倉命令
            response = self.connector.send_command(**close_data)

            if response and response.get("success"):
                result = OrderResult(
                    signal_id=signal.signal_id,
                    success=True,
                    ticket=position.ticket,
                    executed_price=response.get("price", signal.price),
                    executed_volume=response.get("volume", position.volume),
                )
                self.orders_filled += 1
            else:
                result = OrderResult(
                    signal_id=signal.signal_id,
                    success=False,
                    error_code=response.get("error_code", -1) if response else -1,
                    error_message=(
                        response.get("error_message", "未知錯誤") if response else "連接超時"
                    ),
                )
                self.orders_rejected += 1

            self._process_order_result(result)
            return result

        except Exception as e:
            logger.error(f"發送出場訂單時發生錯誤: {e}")
            result = OrderResult(signal_id=signal.signal_id, success=False, error_message=str(e))
            self._process_order_result(result)
            return result

    def _send_modify_order(self, signal: TradingSignal) -> OrderResult:
        """發送修改訂單"""
        try:
            # 這裡需要額外的票號信息，可以從signal的comment或其他字段獲取
            # 簡化實現：假設magic_number就是要修改的票號
            ticket = signal.magic_number

            modify_data = {
                "command": "MODIFY_ORDER",
                "ticket": ticket,
                "price": signal.price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "signal_id": signal.signal_id,
            }

            response = self.connector.send_command(**modify_data)

            if response and response.get("success"):
                result = OrderResult(signal_id=signal.signal_id, success=True, ticket=ticket)
                self.orders_filled += 1
            else:
                result = OrderResult(
                    signal_id=signal.signal_id,
                    success=False,
                    error_code=response.get("error_code", -1) if response else -1,
                    error_message=(
                        response.get("error_message", "未知錯誤") if response else "連接超時"
                    ),
                )
                self.orders_rejected += 1

            self._process_order_result(result)
            return result

        except Exception as e:
            logger.error(f"發送修改訂單時發生錯誤: {e}")
            result = OrderResult(signal_id=signal.signal_id, success=False, error_message=str(e))
            self._process_order_result(result)
            return result

    def _send_cancel_order(self, signal: TradingSignal) -> OrderResult:
        """發送取消訂單"""
        try:
            # 同樣假設magic_number是要取消的票號
            ticket = signal.magic_number

            cancel_data = {
                "command": "DELETE_ORDER",
                "ticket": ticket,
                "signal_id": signal.signal_id,
            }

            response = self.connector.send_command(**cancel_data)

            if response and response.get("success"):
                result = OrderResult(signal_id=signal.signal_id, success=True, ticket=ticket)
            else:
                result = OrderResult(
                    signal_id=signal.signal_id,
                    success=False,
                    error_code=response.get("error_code", -1) if response else -1,
                    error_message=(
                        response.get("error_message", "未知錯誤") if response else "連接超時"
                    ),
                )

            self._process_order_result(result)
            return result

        except Exception as e:
            logger.error(f"發送取消訂單時發生錯誤: {e}")
            result = OrderResult(signal_id=signal.signal_id, success=False, error_message=str(e))
            self._process_order_result(result)
            return result

    def _process_order_result(self, result: OrderResult):
        """處理訂單結果"""
        with self.lock:
            self.executed_orders[result.signal_id] = result

            # 從待處理列表中移除
            if result.signal_id in self.pending_signals:
                signal = self.pending_signals[result.signal_id]
                signal.executed_at = result.execution_time
                del self.pending_signals[result.signal_id]

        # 更新統計
        self.signals_sent += 1

        # 觸發回調
        self._trigger_order_callbacks(result)

        # 記錄日誌
        if result.success:
            logger.info(f"訂單執行成功: {result.signal_id}, 票號: {result.ticket}")
        else:
            logger.error(f"訂單執行失敗: {result.signal_id}, 錯誤: {result.error_message}")

    def _get_account_info(self) -> Dict[str, Any]:
        """獲取賬戶信息"""
        try:
            response = self.connector.send_command("GET_ACCOUNT_INFO")
            return response if response else {}
        except Exception as e:
            logger.error(f"獲取賬戶信息失敗: {e}")
            return {}

    def _get_current_positions(self) -> List[PositionInfo]:
        """獲取當前持倉"""
        try:
            response = self.connector.send_command("GET_POSITIONS")
            if not response:
                return []

            positions = []
            position_data = response.get("positions", [])

            for pos_data in position_data:
                position = PositionInfo(
                    ticket=pos_data.get("ticket", 0),
                    symbol=pos_data.get("symbol", ""),
                    order_type=OrderType(pos_data.get("type", "BUY")),
                    volume=pos_data.get("volume", 0.0),
                    open_price=pos_data.get("open_price", 0.0),
                    current_price=pos_data.get("current_price", 0.0),
                    profit=pos_data.get("profit", 0.0),
                    swap=pos_data.get("swap", 0.0),
                    commission=pos_data.get("commission", 0.0),
                    stop_loss=pos_data.get("stop_loss", 0.0),
                    take_profit=pos_data.get("take_profit", 0.0),
                    comment=pos_data.get("comment", ""),
                    magic_number=pos_data.get("magic_number", 0),
                )

                # 解析開倉時間
                if pos_data.get("open_time"):
                    position.open_time = datetime.fromisoformat(pos_data["open_time"])

                positions.append(position)

            return positions

        except Exception as e:
            logger.error(f"獲取當前持倉失敗: {e}")
            return []

    def add_order_callback(self, callback: Callable[[OrderResult], None]):
        """添加訂單結果回調函數"""
        self.order_callbacks.append(callback)

    def _trigger_order_callbacks(self, result: OrderResult):
        """觸發訂單回調"""
        for callback in self.order_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"訂單回調函數執行錯誤: {e}")

    def get_pending_signals(self) -> List[TradingSignal]:
        """獲取待處理的信號"""
        with self.lock:
            return list(self.pending_signals.values())

    def get_executed_orders(self, limit: int = 100) -> List[OrderResult]:
        """獲取已執行的訂單"""
        with self.lock:
            orders = list(self.executed_orders.values())
            return sorted(orders, key=lambda x: x.execution_time, reverse=True)[:limit]

    def cancel_pending_signal(self, signal_id: str) -> bool:
        """取消待處理的信號"""
        with self.lock:
            if signal_id in self.pending_signals:
                del self.pending_signals[signal_id]
                logger.info(f"已取消待處理信號: {signal_id}")
                return True
            return False

    def get_stats(self) -> Dict[str, Any]:
        """獲取統計信息"""
        with self.lock:
            pending_count = len(self.pending_signals)
            executed_count = len(self.executed_orders)

        success_rate = (
            (self.orders_filled / self.signals_sent * 100) if self.signals_sent > 0 else 0
        )

        return {
            "signals_sent": self.signals_sent,
            "orders_filled": self.orders_filled,
            "orders_rejected": self.orders_rejected,
            "success_rate": f"{success_rate:.2f}%",
            "pending_signals": pending_count,
            "executed_orders": executed_count,
            "risk_manager": {
                "max_daily_loss": self.risk_manager.max_daily_loss,
                "max_drawdown": self.risk_manager.max_drawdown,
                "max_positions_per_symbol": self.risk_manager.max_positions_per_symbol,
                "max_total_positions": self.risk_manager.max_total_positions,
            },
        }


# 便利函數
def create_entry_signal(
    symbol: str,
    order_type: OrderType,
    volume: float,
    price: float = 0.0,
    stop_loss: float = 0.0,
    take_profit: float = 0.0,
    **kwargs,
) -> TradingSignal:
    """創建進場信號"""
    return TradingSignal(
        signal_id=str(uuid.uuid4()),
        symbol=symbol,
        signal_type=SignalType.ENTRY,
        order_type=order_type,
        volume=volume,
        price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        **kwargs,
    )


def create_exit_signal(
    symbol: str, volume: float = 0.0, price: float = 0.0, **kwargs
) -> TradingSignal:
    """創建出場信號"""
    return TradingSignal(
        signal_id=str(uuid.uuid4()),
        symbol=symbol,
        signal_type=SignalType.EXIT,
        order_type=OrderType.BUY,  # 平倉時會自動判斷方向
        volume=volume,
        price=price,
        **kwargs,
    )


def create_signal_sender(connector: MT4Connector = None) -> MT4SignalSender:
    """創建信號發送器實例"""
    return MT4SignalSender(connector)
