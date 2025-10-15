"""
Paper Trading Simulator
模擬交易系統
Cloud Quant - Task SYS-001
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from pathlib import Path
import asyncio
from collections import defaultdict
import uuid

logger = logging.getLogger(__name__)


@dataclass
class PaperOrder:
    """模擬訂單"""

    order_id: str
    symbol: str
    side: str  # BUY/SELL
    quantity: float
    order_type: str  # MARKET/LIMIT/STOP
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: str = "PENDING"  # PENDING/FILLED/CANCELLED
    filled_price: float = 0
    filled_quantity: float = 0
    timestamp: datetime = field(default_factory=datetime.now)
    commission: float = 0
    slippage: float = 0


@dataclass
class PaperPosition:
    """模擬持倉"""

    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float = 0
    total_commission: float = 0


@dataclass
class PaperAccount:
    """模擬帳戶"""

    account_id: str
    initial_balance: float
    cash_balance: float
    portfolio_value: float
    buying_power: float
    total_pnl: float = 0
    realized_pnl: float = 0
    unrealized_pnl: float = 0
    total_commission: float = 0
    margin_used: float = 0


class PaperTradingSimulator:
    """
    模擬交易模擬器

    提供真實的交易環境模擬，包含滑點、手續費等
    """

    def __init__(
        self,
        initial_balance: float = 100000,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        use_real_time_data: bool = False,
    ):
        """
        初始化模擬器

        Args:
            initial_balance: 初始資金
            commission_rate: 手續費率
            slippage_rate: 滑點率
            use_real_time_data: 是否使用實時數據
        """
        self.account = PaperAccount(
            account_id=str(uuid.uuid4()),
            initial_balance=initial_balance,
            cash_balance=initial_balance,
            portfolio_value=initial_balance,
            buying_power=initial_balance,
        )

        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.use_real_time_data = use_real_time_data

        # 持倉管理
        self.positions: Dict[str, PaperPosition] = {}

        # 訂單管理
        self.orders: Dict[str, PaperOrder] = {}
        self.order_history: List[PaperOrder] = []

        # 交易歷史
        self.trade_history: List[Dict] = []

        # 性能追蹤
        self.performance_history: List[Dict] = []
        self.daily_returns: List[float] = []

        # 市場數據緩存
        self.market_data_cache: Dict[str, Dict] = {}

        # 風險限制
        self.risk_limits = {
            "max_position_size": 0.1,  # 單個持倉最大10%
            "max_leverage": 1.0,  # 最大槓桿
            "max_daily_loss": 0.05,  # 單日最大虧損5%
            "max_orders_per_day": 100,  # 每日最大訂單數
        }

        # 當日統計
        self.daily_stats = {
            "orders_placed": 0,
            "trades_executed": 0,
            "daily_pnl": 0,
            "commission_paid": 0,
        }

        logger.info(f"Paper Trading Simulator initialized with ${initial_balance:,.2f}")

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> str:
        """
        下單

        Args:
            symbol: 股票代碼
            side: 買賣方向
            quantity: 數量
            order_type: 訂單類型
            price: 限價
            stop_price: 止損價

        Returns:
            訂單ID
        """
        # 風險檢查
        if not self._check_risk_limits(symbol, side, quantity):
            logger.warning(f"Order rejected due to risk limits: {symbol}")
            return None

        # 創建訂單
        str(uuid.uuid4())
        order = PaperOrder(
            order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            stop_price=stop_price,
            status="PENDING",
        )

        self.orders[order_id] = order
        self.daily_stats["orders_placed"] += 1

        logger.info(f"Order placed: {order_id} - {side} {quantity} {symbol} @ {order_type}")

        # 立即執行市價單
        if order_type == "MARKET":
            await self._execute_order(order_id)

        return order_id

    async def _execute_order(self, order_id: str):
        """執行訂單"""
        order = self.orders.get(order_id)
        if not order or order.status != "PENDING":
            return

        # 獲取市場價格
        market_price = await self._get_market_price(order.symbol)

        if market_price == 0:
            logger.error(f"Cannot execute order - no market price for {order.symbol}")
            return

        # 計算執行價格（包含滑點）
        if order.side == "BUY":
            execution_price = market_price * (1 + self.slippage_rate)
        else:
            execution_price = market_price * (1 - self.slippage_rate)

        # 限價單檢查
        if order.order_type == "LIMIT" and order.price:
            if order.side == "BUY" and execution_price > order.price:
                return  # 價格太高，不執行
            elif order.side == "SELL" and execution_price < order.price:
                return  # 價格太低，不執行

        # 計算手續費
        commission = order.quantity * execution_price * self.commission_rate

        # 檢查資金
        if order.side == "BUY":
            required_cash = order.quantity * execution_price + commission
            if self.account.cash_balance < required_cash:
                logger.warning(f"Insufficient funds for order {order_id}")
                order.status = "CANCELLED"
                return

        # 執行交易
        order.filled_price = execution_price
        order.filled_quantity = order.quantity
        order.commission = commission
        order.slippage = abs(market_price - execution_price) * order.quantity
        order.status = "FILLED"

        # 更新帳戶
        if order.side == "BUY":
            self.account.cash_balance -= order.quantity * execution_price + commission
            self._add_position(order.symbol, order.quantity, execution_price, commission)
        else:
            self.account.cash_balance += order.quantity * execution_price - commission
            self._reduce_position(order.symbol, order.quantity, execution_price, commission)

        self.account.total_commission += commission
        self.daily_stats["commission_paid"] += commission
        self.daily_stats["trades_executed"] += 1

        # 記錄交易
        self._record_trade(order)

        logger.info(f"Order executed: {order_id} - Filled @ ${execution_price:.2f}")

    def _add_position(self, symbol: str, quantity: float, price: float, commission: float):
        """增加持倉"""
        if symbol in self.positions:
            position = self.positions[symbol]
            total_cost = position.quantity * position.avg_price + quantity * price
            position.quantity += quantity
            position.avg_price = total_cost / position.quantity
            position.total_commission += commission
        else:
            self.positions[symbol] = PaperPosition(
                symbol=symbol,
                quantity=quantity,
                avg_price=price,
                current_price=price,
                market_value=quantity * price,
                unrealized_pnl=0,
                total_commission=commission,
            )

    def _reduce_position(self, symbol: str, quantity: float, price: float, commission: float):
        """減少持倉"""
        if symbol not in self.positions:
            logger.warning(f"No position to reduce for {symbol}")
            return

        position = self.positions[symbol]

        if quantity >= position.quantity:
            # 平倉
            realized_pnl = (price - position.avg_price) * position.quantity - commission
            position.realized_pnl += realized_pnl
            self.account.realized_pnl += realized_pnl
            del self.positions[symbol]
        else:
            # 部分平倉
            realized_pnl = (price - position.avg_price) * quantity - commission
            position.quantity -= quantity
            position.realized_pnl += realized_pnl
            position.total_commission += commission
            self.account.realized_pnl += realized_pnl

    async def _get_market_price(self, symbol: str) -> float:
        """獲取市場價格"""
        # 這裡簡化處理，實際應從數據源獲取
        if symbol in self.market_data_cache:
            return self.market_data_cache[symbol].get("price", 100)

        # 模擬價格
        return 100 + np.random.normal(0, 5)

    def update_market_prices(self, prices: Dict[str, float]):
        """更新市場價格"""
        for symbol, price in prices.items():
            self.market_data_cache[symbol] = {"price": price, "timestamp": datetime.now()}

            # 更新持倉市值
            if symbol in self.positions:
                position = self.positions[symbol]
                position.current_price = price
                position.market_value = position.quantity * price
                position.unrealized_pnl = (price - position.avg_price) * position.quantity

    def _check_risk_limits(self, symbol: str, side: str, quantity: float) -> bool:
        """檢查風險限制"""
        # 檢查每日訂單數
        if self.daily_stats["orders_placed"] >= self.risk_limits["max_orders_per_day"]:
            logger.warning("Daily order limit reached")
            return False

        # 檢查單個持倉大小
        if side == "BUY":
            market_price = self.market_data_cache.get(symbol, {}).get("price", 100)
            position_value = quantity * market_price

            if (
                position_value
                > self.account.portfolio_value * self.risk_limits["max_position_size"]
            ):
                logger.warning(f"Position size too large for {symbol}")
                return False

        # 檢查每日虧損
        daily_loss_limit = self.account.initial_balance * self.risk_limits["max_daily_loss"]
        if self.daily_stats["daily_pnl"] < -daily_loss_limit:
            logger.warning("Daily loss limit reached")
            return False

        return True

    def _record_trade(self, order: PaperOrder):
        """記錄交易"""
        trade = {
            "timestamp": datetime.now(),
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.filled_quantity,
            "price": order.filled_price,
            "commission": order.commission,
            "slippage": order.slippage,
        }

        self.trade_history.append(trade)
        self.order_history.append(order)

    def calculate_portfolio_value(self) -> float:
        """計算投資組合價值"""
        total_value = self.account.cash_balance

        for position in self.positions.values():
            total_value += position.market_value

        self.account.portfolio_value = total_value
        self.account.unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        self.account.total_pnl = self.account.realized_pnl + self.account.unrealized_pnl

        return total_value

    def get_performance_metrics(self) -> Dict[str, float]:
        """獲取性能指標"""
        portfolio_value = self.calculate_portfolio_value()
        initial_value = self.account.initial_balance

        metrics = {
            "total_return": (portfolio_value - initial_value) / initial_value,
            "portfolio_value": portfolio_value,
            "cash_balance": self.account.cash_balance,
            "positions_count": len(self.positions),
            "total_pnl": self.account.total_pnl,
            "realized_pnl": self.account.realized_pnl,
            "unrealized_pnl": self.account.unrealized_pnl,
            "total_commission": self.account.total_commission,
            "total_trades": len(self.trade_history),
            "winning_trades": sum(1 for t in self.trade_history if self._is_winning_trade(t)),
            "win_rate": 0,
        }

        if metrics["total_trades"] > 0:
            metrics["win_rate"] = metrics["winning_trades"] / metrics["total_trades"]

        # 計算 Sharpe Ratio
        if len(self.daily_returns) > 1:
            returns = np.array(self.daily_returns)
            metrics["sharpe_ratio"] = (
                np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
            )
        else:
            metrics["sharpe_ratio"] = 0

        # 計算最大回撤
        if len(self.performance_history) > 1:
            values = [p["portfolio_value"] for p in self.performance_history]
            peak = np.maximum.accumulate(values)
            drawdown = (values - peak) / peak
            metrics["max_drawdown"] = drawdown.min()
        else:
            metrics["max_drawdown"] = 0

        return metrics

    def _is_winning_trade(self, trade: Dict) -> bool:
        """判斷是否為獲利交易"""
        # 簡化邏輯：賣出時檢查是否獲利
        if trade["side"] == "SELL":
            symbol = trade["symbol"]
            # 查找對應的買入記錄
            buy_trades = [
                t
                for t in self.trade_history
                if t["symbol"] == symbol
                and t["side"] == "BUY"
                and t["timestamp"] < trade["timestamp"]
            ]
            if buy_trades:
                avg_buy_price = np.mean([t["price"] for t in buy_trades])
                return trade["price"] > avg_buy_price
        return False

    def reset_daily_stats(self):
        """重置每日統計"""
        # 記錄每日收益
        if len(self.performance_history) > 0:
            yesterday_value = self.performance_history[-1]["portfolio_value"]
            today_value = self.calculate_portfolio_value()
            daily_return = (today_value - yesterday_value) / yesterday_value
            self.daily_returns.append(daily_return)

        # 重置統計
        self.daily_stats = {
            "orders_placed": 0,
            "trades_executed": 0,
            "daily_pnl": 0,
            "commission_paid": 0,
        }

    def save_state(self, filepath: str):
        """保存狀態"""
        state = {
            "account": {
                "account_id": self.account.account_id,
                "initial_balance": self.account.initial_balance,
                "cash_balance": self.account.cash_balance,
                "portfolio_value": self.account.portfolio_value,
                "total_pnl": self.account.total_pnl,
                "total_commission": self.account.total_commission,
            },
            "positions": {
                symbol: {
                    "quantity": pos.quantity,
                    "avg_price": pos.avg_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "realized_pnl": pos.realized_pnl,
                }
                for symbol, pos in self.positions.items()
            },
            "performance": self.get_performance_metrics(),
            "trade_count": len(self.trade_history),
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2, default=str)

        logger.info(f"State saved to {filepath}")

    def generate_report(self) -> str:
        """生成報告"""
        metrics = self.get_performance_metrics()

        """
        ╔════════════════════════════════════════╗
        ║     PAPER TRADING PERFORMANCE REPORT   ║
        ╚════════════════════════════════════════╝
        
        Account ID: {self.account.account_id[:8]}...
        Date: {datetime.now():%Y-%m-%d %H:%M:%S}
        
        ═══ Portfolio Summary ═══
        Portfolio Value: ${metrics['portfolio_value']:,.2f}
        Cash Balance: ${metrics['cash_balance']:,.2f}
        Positions: {metrics['positions_count']}
        
        ═══ Performance Metrics ═══
        Total Return: {metrics['total_return']:.2%}
        Total P&L: ${metrics['total_pnl']:,.2f}
        Realized P&L: ${metrics['realized_pnl']:,.2f}
        Unrealized P&L: ${metrics['unrealized_pnl']:,.2f}
        
        Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
        Max Drawdown: {metrics['max_drawdown']:.2%}
        Win Rate: {metrics['win_rate']:.2%}
        
        ═══ Trading Statistics ═══
        Total Trades: {metrics['total_trades']}
        Winning Trades: {metrics['winning_trades']}
        Total Commission: ${metrics['total_commission']:,.2f}
        
        ═══ Top Positions ═══
        """

        # 添加前5大持倉
        sorted_positions = sorted(
            self.positions.items(), key=lambda x: x[1].market_value, reverse=True
        )[:5]

        for symbol, pos in sorted_positions:
            report += """
        {symbol}: {pos.quantity:.0f} @ ${pos.avg_price:.2f}
          Market Value: ${pos.market_value:,.2f}
          Unrealized P&L: ${pos.unrealized_pnl:,.2f}
            """

        return report


if __name__ == "__main__":
    print("Paper Trading Simulator - Cloud Quant Task SYS-001")
    print("=" * 50)

    # 測試模擬器
    async def test_simulator():
        simulator = PaperTradingSimulator(initial_balance=100000)

        # 模擬一些交易
        await simulator.place_order("AAPL", "BUY", 100, "MARKET")
        await simulator.place_order("GOOGL", "BUY", 50, "MARKET")

        # 更新價格
        simulator.update_market_prices({"AAPL": 105, "GOOGL": 2800})

        # 賣出
        await simulator.place_order("AAPL", "SELL", 50, "MARKET")

        # 生成報告
        print(simulator.generate_report())

    # 運行測試
    import asyncio

    asyncio.run(test_simulator())

    print("\n✓ Paper Trading Simulator ready!")
