# -*- coding: utf-8 -*-
"""
MT4帳戶監控模組
負責監控賬戶餘額、持倉狀態、交易歷史、風險指標
提供實時監控和警報功能
"""

import logging
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from threading import Thread, Lock
from enum import Enum
import time
import json
import os

from .connector import MT4Connector, get_default_connector
from .signal_sender import PositionInfo, OrderType

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """警報級別枚舉"""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AlertType(Enum):
    """警報類型枚舉"""

    BALANCE_LOW = "BALANCE_LOW"
    MARGIN_CALL = "MARGIN_CALL"
    DRAWDOWN_HIGH = "DRAWDOWN_HIGH"
    POSITION_LOSS = "POSITION_LOSS"
    CONNECTION_LOST = "CONNECTION_LOST"
    UNUSUAL_ACTIVITY = "UNUSUAL_ACTIVITY"


@dataclass
class AccountSnapshot:
    """賬戶快照數據結構"""

    timestamp: datetime
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    profit: float

    # 附加計算字段
    drawdown: float = 0.0
    drawdown_percent: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_percent: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass
class Alert:
    """警報數據結構"""

    alert_id: str
    timestamp: datetime
    level: AlertLevel
    type: AlertType
    message: str
    data: Dict[str, Any] = None
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["level"] = self.level.value
        data["type"] = self.type.value
        return data


@dataclass
class TradeHistory:
    """交易歷史記錄"""

    ticket: int
    symbol: str
    order_type: OrderType
    volume: float
    open_price: float
    close_price: float
    profit: float
    commission: float
    swap: float
    open_time: datetime
    close_time: datetime
    comment: str = ""
    magic_number: int = 0

    @property
    def net_profit(self) -> float:
        """淨利潤"""
        return self.profit + self.commission + self.swap

    @property
    def pips(self) -> float:
        """點數(簡化計算)"""
        return abs(self.close_price - self.open_price) * 10000

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        data = asdict(self)
        data["order_type"] = self.order_type.value
        data["open_time"] = self.open_time.isoformat()
        data["close_time"] = self.close_time.isoformat()
        data["net_profit"] = self.net_profit
        data["pips"] = self.pips
        return data


class AccountMonitorStorage:
    """賬戶監控數據存儲"""

    def __init__(self, db_path: str = None):
        """
        初始化存儲

        Args:
            db_path: SQLite數據庫路徑
        """
        if db_path is None:
            # 使用項目根目錄下的test_storage/sqlite目錄
            project_root = os.path.dirname(os.path.dirname(__file__))
            storage_dir = os.path.join(project_root, "test_storage", "sqlite")
            os.makedirs(storage_dir, exist_ok=True)
            db_path = os.path.join(storage_dir, "account_monitor.db")

        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """初始化數據庫表"""
        with sqlite3.connect(self.db_path) as conn:
            # 賬戶快照表
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS account_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    balance REAL NOT NULL,
                    equity REAL NOT NULL,
                    margin REAL NOT NULL,
                    free_margin REAL NOT NULL,
                    margin_level REAL NOT NULL,
                    profit REAL NOT NULL,
                    drawdown REAL DEFAULT 0,
                    drawdown_percent REAL DEFAULT 0,
                    daily_pnl REAL DEFAULT 0,
                    daily_pnl_percent REAL DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # 警報表
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    timestamp DATETIME NOT NULL,
                    level TEXT NOT NULL,
                    type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    data TEXT,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # 交易歷史表
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket INTEGER UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    volume REAL NOT NULL,
                    open_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    profit REAL NOT NULL,
                    commission REAL NOT NULL,
                    swap REAL NOT NULL,
                    open_time DATETIME NOT NULL,
                    close_time DATETIME NOT NULL,
                    comment TEXT,
                    magic_number INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # 創建索引
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON account_snapshots(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp, level)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_close_time ON trade_history(close_time)"
            )

    def save_snapshot(self, snapshot: AccountSnapshot):
        """保存賬戶快照"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO account_snapshots 
                    (timestamp, balance, equity, margin, free_margin, margin_level, profit,
                     drawdown, drawdown_percent, daily_pnl, daily_pnl_percent)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        snapshot.timestamp,
                        snapshot.balance,
                        snapshot.equity,
                        snapshot.margin,
                        snapshot.free_margin,
                        snapshot.margin_level,
                        snapshot.profit,
                        snapshot.drawdown,
                        snapshot.drawdown_percent,
                        snapshot.daily_pnl,
                        snapshot.daily_pnl_percent,
                    ),
                )
        except Exception as e:
            logger.error(f"保存賬戶快照失敗: {e}")

    def save_alert(self, alert: Alert):
        """保存警報"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO alerts 
                    (alert_id, timestamp, level, type, message, data, acknowledged)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        alert.alert_id,
                        alert.timestamp,
                        alert.level.value,
                        alert.type.value,
                        alert.message,
                        json.dumps(alert.data) if alert.data else None,
                        alert.acknowledged,
                    ),
                )
        except Exception as e:
            logger.error(f"保存警報失敗: {e}")

    def save_trade(self, trade: TradeHistory):
        """保存交易歷史"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO trade_history 
                    (ticket, symbol, order_type, volume, open_price, close_price,
                     profit, commission, swap, open_time, close_time, comment, magic_number)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        trade.ticket,
                        trade.symbol,
                        trade.order_type.value,
                        trade.volume,
                        trade.open_price,
                        trade.close_price,
                        trade.profit,
                        trade.commission,
                        trade.swap,
                        trade.open_time,
                        trade.close_time,
                        trade.comment,
                        trade.magic_number,
                    ),
                )
        except Exception as e:
            logger.error(f"保存交易歷史失敗: {e}")

    def get_snapshots(self, hours: int = 24, limit: int = 1000) -> List[AccountSnapshot]:
        """獲取賬戶快照"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                start_time = datetime.now() - timedelta(hours=hours)
                cursor = conn.execute(
                    """
                    SELECT timestamp, balance, equity, margin, free_margin, margin_level,
                           profit, drawdown, drawdown_percent, daily_pnl, daily_pnl_percent
                    FROM account_snapshots 
                    WHERE timestamp >= ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """,
                    (start_time, limit),
                )

                snapshots = []
                for row in cursor.fetchall():
                    snapshot = AccountSnapshot(
                        timestamp=datetime.fromisoformat(row[0]),
                        balance=row[1],
                        equity=row[2],
                        margin=row[3],
                        free_margin=row[4],
                        margin_level=row[5],
                        profit=row[6],
                        drawdown=row[7] or 0.0,
                        drawdown_percent=row[8] or 0.0,
                        daily_pnl=row[9] or 0.0,
                        daily_pnl_percent=row[10] or 0.0,
                    )
                    snapshots.append(snapshot)

                return snapshots
        except Exception as e:
            logger.error(f"獲取賬戶快照失敗: {e}")
            return []

    def get_alerts(self, hours: int = 24, level: AlertLevel = None) -> List[Alert]:
        """獲取警報"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                start_time = datetime.now() - timedelta(hours=hours)

                query = """
                    SELECT alert_id, timestamp, level, type, message, data, acknowledged
                    FROM alerts WHERE timestamp >= ?
                """
                params = [start_time]

                if level:
                    query += " AND level = ?"
                    params.append(level.value)

                query += " ORDER BY timestamp DESC"

                cursor = conn.execute(query, params)

                alerts = []
                for row in cursor.fetchall():
                    alert_data = json.loads(row[5]) if row[5] else None

                    alert = Alert(
                        alert_id=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        level=AlertLevel(row[2]),
                        type=AlertType(row[3]),
                        message=row[4],
                        data=alert_data,
                        acknowledged=bool(row[6]),
                    )
                    alerts.append(alert)

                return alerts
        except Exception as e:
            logger.error(f"獲取警報失敗: {e}")
            return []


class PerformanceAnalyzer:
    """績效分析器"""

    @staticmethod
    def calculate_drawdown(snapshots: List[AccountSnapshot]) -> Dict[str, float]:
        """計算回撤指標"""
        if not snapshots:
            return {"max_drawdown": 0.0, "current_drawdown": 0.0}

        # 按時間排序
        sorted_snapshots = sorted(snapshots, key=lambda x: x.timestamp)

        peak_equity = 0.0
        max_drawdown = 0.0
        current_drawdown = 0.0

        for snapshot in sorted_snapshots:
            # 更新峰值
            if snapshot.equity > peak_equity:
                peak_equity = snapshot.equity

            # 計算當前回撤
            if peak_equity > 0:
                current_drawdown = (peak_equity - snapshot.equity) / peak_equity * 100
                max_drawdown = max(max_drawdown, current_drawdown)

        return {
            "max_drawdown": max_drawdown,
            "current_drawdown": current_drawdown,
            "peak_equity": peak_equity,
        }

    @staticmethod
    def calculate_daily_stats(snapshots: List[AccountSnapshot]) -> Dict[str, float]:
        """計算日統計數據"""
        if len(snapshots) < 2:
            return {"daily_return": 0.0, "volatility": 0.0}

        # 按時間排序
        sorted_snapshots = sorted(snapshots, key=lambda x: x.timestamp)

        daily_returns = []
        for i in range(1, len(sorted_snapshots)):
            prev_equity = sorted_snapshots[i - 1].equity
            curr_equity = sorted_snapshots[i].equity

            if prev_equity > 0:
                daily_return = (curr_equity - prev_equity) / prev_equity * 100
                daily_returns.append(daily_return)

        if not daily_returns:
            return {"daily_return": 0.0, "volatility": 0.0}

        avg_return = sum(daily_returns) / len(daily_returns)
        variance = sum((r - avg_return) ** 2 for r in daily_returns) / len(daily_returns)
        volatility = variance**0.5

        return {"daily_return": avg_return, "volatility": volatility}

    @staticmethod
    def analyze_trades(trades: List[TradeHistory]) -> Dict[str, Any]:
        """分析交易績效"""
        if not trades:
            return {}

        total_trades = len(trades)
        winning_trades = [t for t in trades if t.net_profit > 0]
        losing_trades = [t for t in trades if t.net_profit < 0]

        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0

        avg_win = (
            sum(t.net_profit for t in winning_trades) / len(winning_trades) if winning_trades else 0
        )
        avg_loss = (
            sum(t.net_profit for t in losing_trades) / len(losing_trades) if losing_trades else 0
        )

        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        total_profit = sum(t.net_profit for t in trades)

        return {
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "total_profit": total_profit,
        }


class MT4AccountMonitor:
    """MT4賬戶監控器"""

    def __init__(self, connector: MT4Connector = None, storage: AccountMonitorStorage = None):
        """
        初始化賬戶監控器

        Args:
            connector: MT4連接器
            storage: 數據存儲器
        """
        self.connector = connector or get_default_connector()
        self.storage = storage or AccountMonitorStorage()
        self.analyzer = PerformanceAnalyzer()

        # 監控設置
        self.monitor_interval = 60  # 監控間隔(秒)
        self.snapshot_interval = 300  # 快照間隔(秒)

        # 警報設置
        self.alert_thresholds = {
            "low_balance_ratio": 0.1,  # 餘額低於10%時警報
            "margin_level_warning": 200,  # 保證金水平低於200%時警報
            "margin_level_critical": 100,  # 保證金水平低於100%時嚴重警報
            "max_drawdown": 20.0,  # 最大回撤超過20%時警報
            "daily_loss_percent": 5.0,  # 日虧損超過5%時警報
        }

        # 回調函數
        self.alert_callbacks = []
        self.snapshot_callbacks = []

        # 線程控制
        self._running = False
        self._monitor_thread = None
        self._snapshot_thread = None
        self.lock = Lock()

        # 統計數據
        self.start_time = None
        self.snapshot_count = 0
        self.alert_count = 0

        # 緩存數據
        self.last_snapshot = None
        self.initial_balance = None
        self.peak_balance = None

    def start(self):
        """開始監控"""
        if not self.connector or not self.connector.is_connected():
            logger.error("MT4連接器未連接，無法開始賬戶監控")
            return False

        self._running = True
        self.start_time = datetime.now()

        # 獲取初始賬戶狀態
        self._initialize_account_state()

        # 啟動監控線程
        self._monitor_thread = Thread(target=self._monitor_worker, daemon=True)
        self._monitor_thread.start()

        # 啟動快照線程
        self._snapshot_thread = Thread(target=self._snapshot_worker, daemon=True)
        self._snapshot_thread.start()

        logger.info("MT4賬戶監控器已啟動")
        return True

    def stop(self):
        """停止監控"""
        self._running = False

        # 等待線程結束
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        if self._snapshot_thread and self._snapshot_thread.is_alive():
            self._snapshot_thread.join(timeout=5.0)

        logger.info("MT4賬戶監控器已停止")

    def _initialize_account_state(self):
        """初始化賬戶狀態"""
        try:
            account_info = self._get_account_info()
            if account_info:
                self.initial_balance = account_info.get("balance", 0)
                self.peak_balance = self.initial_balance
                logger.info(f"初始賬戶餘額: {self.initial_balance}")
        except Exception as e:
            logger.error(f"初始化賬戶狀態失敗: {e}")

    def _monitor_worker(self):
        """監控工作線程"""
        while self._running:
            try:
                self._check_account_status()
                time.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"監控工作線程錯誤: {e}")
                time.sleep(self.monitor_interval)

    def _snapshot_worker(self):
        """快照工作線程"""
        while self._running:
            try:
                self._take_snapshot()
                time.sleep(self.snapshot_interval)
            except Exception as e:
                logger.error(f"快照工作線程錯誤: {e}")
                time.sleep(self.snapshot_interval)

    def _check_account_status(self):
        """檢查賬戶狀態並生成警報"""
        try:
            account_info = self._get_account_info()
            if not account_info:
                self._create_alert(
                    AlertLevel.ERROR, AlertType.CONNECTION_LOST, "無法獲取賬戶信息，可能連接中斷"
                )
                return

            balance = account_info.get("balance", 0)
            equity = account_info.get("equity", 0)
            margin = account_info.get("margin", 0)
            free_margin = account_info.get("free_margin", 0)
            margin_level = account_info.get("margin_level", 0)

            # 檢查餘額警報
            if (
                self.initial_balance
                and balance < self.initial_balance * self.alert_thresholds["low_balance_ratio"]
            ):
                self._create_alert(
                    AlertLevel.WARNING,
                    AlertType.BALANCE_LOW,
                    f"賬戶餘額過低: {balance:.2f}, 初始餘額: {self.initial_balance:.2f}",
                )

            # 檢查保證金水平
            if margin > 0:
                if margin_level < self.alert_thresholds["margin_level_critical"]:
                    self._create_alert(
                        AlertLevel.CRITICAL,
                        AlertType.MARGIN_CALL,
                        f"保證金水平嚴重不足: {margin_level:.2f}%",
                    )
                elif margin_level < self.alert_thresholds["margin_level_warning"]:
                    self._create_alert(
                        AlertLevel.WARNING,
                        AlertType.MARGIN_CALL,
                        f"保證金水平偏低: {margin_level:.2f}%",
                    )

            # 檢查回撤
            if self.peak_balance:
                self.peak_balance = max(self.peak_balance, equity)
                current_drawdown = (self.peak_balance - equity) / self.peak_balance * 100

                if current_drawdown > self.alert_thresholds["max_drawdown"]:
                    self._create_alert(
                        AlertLevel.WARNING,
                        AlertType.DRAWDOWN_HIGH,
                        f"回撤過高: {current_drawdown:.2f}%",
                    )

            # 檢查日虧損
            if self.last_snapshot:
                time_diff = datetime.now() - self.last_snapshot.timestamp
                if time_diff.total_seconds() >= 86400:  # 24小時
                    daily_pnl_percent = (
                        (equity - self.last_snapshot.equity) / self.last_snapshot.equity * 100
                    )
                    if daily_pnl_percent <= -self.alert_thresholds["daily_loss_percent"]:
                        self._create_alert(
                            AlertLevel.WARNING,
                            AlertType.POSITION_LOSS,
                            f"日虧損過高: {daily_pnl_percent:.2f}%",
                        )

        except Exception as e:
            logger.error(f"檢查賬戶狀態時發生錯誤: {e}")

    def _take_snapshot(self):
        """拍攝賬戶快照"""
        try:
            account_info = self._get_account_info()
            if not account_info:
                return

            now = datetime.now()

            # 創建快照
            snapshot = AccountSnapshot(
                timestamp=now,
                balance=account_info.get("balance", 0),
                equity=account_info.get("equity", 0),
                margin=account_info.get("margin", 0),
                free_margin=account_info.get("free_margin", 0),
                margin_level=account_info.get("margin_level", 0),
                profit=account_info.get("profit", 0),
            )

            # 計算附加字段
            if self.peak_balance and self.peak_balance > 0:
                snapshot.drawdown = self.peak_balance - snapshot.equity
                snapshot.drawdown_percent = snapshot.drawdown / self.peak_balance * 100

            if self.last_snapshot:
                time_diff = (now - self.last_snapshot.timestamp).total_seconds()
                if time_diff >= 86400:  # 24小時
                    if self.last_snapshot.equity > 0:
                        snapshot.daily_pnl = snapshot.equity - self.last_snapshot.equity
                        snapshot.daily_pnl_percent = (
                            snapshot.daily_pnl / self.last_snapshot.equity * 100
                        )

            # 保存快照
            self.storage.save_snapshot(snapshot)

            with self.lock:
                self.last_snapshot = snapshot
                self.snapshot_count += 1

            # 觸發回調
            self._trigger_snapshot_callbacks(snapshot)

        except Exception as e:
            logger.error(f"拍攝賬戶快照時發生錯誤: {e}")

    def _get_account_info(self) -> Optional[Dict[str, Any]]:
        """獲取賬戶信息"""
        try:
            return self.connector.send_command("GET_ACCOUNT_INFO")
        except Exception as e:
            logger.error(f"獲取賬戶信息失敗: {e}")
            return None

    def _create_alert(
        self, level: AlertLevel, alert_type: AlertType, message: str, data: Dict[str, Any] = None
    ):
        """創建警報"""
        try:
            alert = Alert(
                alert_id=f"{alert_type.value}_{int(time.time())}",
                timestamp=datetime.now(),
                level=level,
                type=alert_type,
                message=message,
                data=data,
            )

            # 保存警報
            self.storage.save_alert(alert)

            with self.lock:
                self.alert_count += 1

            # 觸發回調
            self._trigger_alert_callbacks(alert)

            # 記錄日誌
            log_func = {
                AlertLevel.INFO: logger.info,
                AlertLevel.WARNING: logger.warning,
                AlertLevel.ERROR: logger.error,
                AlertLevel.CRITICAL: logger.critical,
            }.get(level, logger.info)

            log_func(f"賬戶警報 - {alert_type.value}: {message}")

        except Exception as e:
            logger.error(f"創建警報時發生錯誤: {e}")

    def _trigger_alert_callbacks(self, alert: Alert):
        """觸發警報回調"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"警報回調函數執行錯誤: {e}")

    def _trigger_snapshot_callbacks(self, snapshot: AccountSnapshot):
        """觸發快照回調"""
        for callback in self.snapshot_callbacks:
            try:
                callback(snapshot)
            except Exception as e:
                logger.error(f"快照回調函數執行錯誤: {e}")

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """添加警報回調函數"""
        self.alert_callbacks.append(callback)

    def add_snapshot_callback(self, callback: Callable[[AccountSnapshot], None]):
        """添加快照回調函數"""
        self.snapshot_callbacks.append(callback)

    def set_alert_threshold(self, threshold_name: str, value: float):
        """設置警報閾值"""
        if threshold_name in self.alert_thresholds:
            self.alert_thresholds[threshold_name] = value
            logger.info(f"警報閾值已更新: {threshold_name} = {value}")
        else:
            logger.warning(f"未知的警報閾值: {threshold_name}")

    def get_current_status(self) -> Dict[str, Any]:
        """獲取當前狀態"""
        account_info = self._get_account_info()
        if not account_info:
            return {}

        # 獲取最近的快照進行分析
        recent_snapshots = self.storage.get_snapshots(hours=24)

        drawdown_info = self.analyzer.calculate_drawdown(recent_snapshots)
        daily_stats = self.analyzer.calculate_daily_stats(recent_snapshots)

        with self.lock:
            uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0

            return {
                "account": account_info,
                "performance": {**drawdown_info, **daily_stats},
                "monitoring": {
                    "running": self._running,
                    "uptime_seconds": uptime,
                    "snapshot_count": self.snapshot_count,
                    "alert_count": self.alert_count,
                },
                "thresholds": self.alert_thresholds,
            }

    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """獲取最近的警報"""
        return self.storage.get_alerts(hours=hours)

    def acknowledge_alert(self, alert_id: str) -> bool:
        """確認警報"""
        try:
            # 這裡簡化實現，實際需要更新數據庫
            logger.info(f"警報已確認: {alert_id}")
            return True
        except Exception as e:
            logger.error(f"確認警報失敗: {e}")
            return False

    def get_performance_report(self, days: int = 30) -> Dict[str, Any]:
        """獲取績效報告"""
        try:
            # 獲取歷史快照
            snapshots = self.storage.get_snapshots(hours=days * 24)

            if not snapshots:
                return {}

            # 基本統計
            latest = snapshots[0] if snapshots else None
            earliest = snapshots[-1] if snapshots else None

            total_return = 0.0
            if earliest and earliest.balance > 0:
                total_return = (latest.balance - earliest.balance) / earliest.balance * 100

            # 績效分析
            drawdown_info = self.analyzer.calculate_drawdown(snapshots)
            daily_stats = self.analyzer.calculate_daily_stats(snapshots)

            return {
                "period_days": days,
                "total_return_percent": total_return,
                "current_balance": latest.balance if latest else 0,
                "current_equity": latest.equity if latest else 0,
                "peak_balance": drawdown_info.get("peak_equity", 0),
                "max_drawdown_percent": drawdown_info.get("max_drawdown", 0),
                "current_drawdown_percent": drawdown_info.get("current_drawdown", 0),
                "daily_return_percent": daily_stats.get("daily_return", 0),
                "volatility": daily_stats.get("volatility", 0),
                "snapshots_count": len(snapshots),
            }

        except Exception as e:
            logger.error(f"生成績效報告時發生錯誤: {e}")
            return {}


# 便利函數
def create_account_monitor(connector: MT4Connector = None) -> MT4AccountMonitor:
    """創建賬戶監控器實例"""
    return MT4AccountMonitor(connector)


def get_account_snapshot() -> Optional[AccountSnapshot]:
    """獲取當前賬戶快照"""
    connector = get_default_connector()
    if not connector:
        return None

    try:
        account_info = connector.send_command("GET_ACCOUNT_INFO")
        if not account_info:
            return None

        return AccountSnapshot(
            timestamp=datetime.now(),
            balance=account_info.get("balance", 0),
            equity=account_info.get("equity", 0),
            margin=account_info.get("margin", 0),
            free_margin=account_info.get("free_margin", 0),
            margin_level=account_info.get("margin_level", 0),
            profit=account_info.get("profit", 0),
        )
    except Exception as e:
        logger.error(f"獲取賬戶快照失敗: {e}")
        return None
