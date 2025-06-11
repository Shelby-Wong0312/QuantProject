import logging
from decimal import Decimal
from core.event import FillEvent, SignalAction

logger = logging.getLogger(__name__)

class SimplePortfolioManager:
    """简化版的投资组合管理器"""
    
    def __init__(self, event_queue, initial_cash: float = 100000.0):
        self.event_queue = event_queue
        self.cash = Decimal(str(initial_cash))
        self.positions = {}  # {symbol: quantity}
        self.trades = []  # 交易记录
        
    async def handle_fill_event(self, fill_event: FillEvent):
        """处理成交事件，更新持仓和现金"""
        logger.info(f"处理成交事件: {fill_event.symbol} {fill_event.action} {fill_event.quantity} @ {fill_event.fill_price}")
        
        symbol = fill_event.symbol
        quantity = Decimal(str(fill_event.quantity))
        fill_price = Decimal(str(fill_event.fill_price))
        commission = Decimal(str(fill_event.commission))
        
        # 更新现金
        if fill_event.action == SignalAction.BUY:
            cost = quantity * fill_price + commission
            self.cash -= cost
            # 更新持仓
            if symbol in self.positions:
                self.positions[symbol] += quantity
            else:
                self.positions[symbol] = quantity
                
        elif fill_event.action == SignalAction.SELL:
            revenue = quantity * fill_price - commission
            self.cash += revenue
            # 更新持仓
            if symbol in self.positions:
                self.positions[symbol] -= quantity
                if self.positions[symbol] == 0:
                    del self.positions[symbol]
        
        # 记录交易
        self.trades.append({
            'timestamp': fill_event.timestamp,
            'symbol': symbol,
            'action': fill_event.action.value,
            'quantity': float(quantity),
            'price': float(fill_price),
            'commission': float(commission)
        })
        
        # 打印当前状态
        logger.info(f"当前现金: ${self.cash:.2f}")
        logger.info(f"当前持仓: {self.positions}")
        
    def get_portfolio_value(self, current_prices=None):
        """计算投资组合总价值"""
        total_value = self.cash
        
        if current_prices:
            for symbol, quantity in self.positions.items():
                if symbol in current_prices:
                    total_value += quantity * Decimal(str(current_prices[symbol]))
                    
        return float(total_value) 