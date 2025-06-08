# 檔案位置: portfolio/portfolio_manager.py

import logging
from collections import defaultdict
from core.event import FillEvent

logger = logging.getLogger(__name__)

class PortfolioManager:
    """
    一個簡單的投資組合管理器。
    它處理成交事件 (FillEvent)，並追蹤持倉和已實現損益。
    """
    def __init__(self, event_queue, initial_cash=100000.0):
        self.event_queue = event_queue
        self.initial_cash = initial_cash
        self.cash = initial_cash
        
        # defaultdict(float) 會在鍵不存在時，自動為其賦值一個預設的浮點數 0.0
        # 用於儲存每個 symbol 的持有數量
        self.current_positions = defaultdict(float) 
        # 用於儲存每個 symbol 的平均成本價
        self.average_cost = defaultdict(float)
        
        self.realized_pnl = 0.0

    def handle_fill_event(self, event: FillEvent):
        """
        處理成交事件，更新持倉、現金和已實現損益。
        """
        logger.info(f"投資組合管理器收到成交事件: {event.action} for {event.symbol}")

        symbol = event.symbol
        fill_qty = event.fill_quantity
        fill_price = event.fill_price
        
        # 處理開倉或加倉 (此處簡化為只處理 BUY_ENTRY)
        if event.action == 'BUY_ENTRY':
            current_qty = self.current_positions[symbol]
            current_total_cost = self.average_cost[symbol] * current_qty
            
            new_qty = current_qty + fill_qty
            new_total_cost = current_total_cost + (fill_qty * fill_price)
            
            self.average_cost[symbol] = new_total_cost / new_qty if new_qty != 0 else 0
            self.current_positions[symbol] = new_qty
            self.cash -= (fill_qty * fill_price)
            
        # 處理平倉或減倉 (此處簡化為只處理 CLOSE_LONG_CONDITION)
        elif event.action == 'CLOSE_LONG_CONDITION':
            # 假設此次平掉了 'fill_qty' 數量的倉位
            entry_cost = self.average_cost[symbol] * fill_qty
            exit_value = fill_qty * fill_price
            pnl = exit_value - entry_cost
            self.realized_pnl += pnl
            
            self.current_positions[symbol] -= fill_qty
            self.cash += exit_value
            
            # 如果該標的部位已完全清空，重置其平均成本
            if self.current_positions[symbol] == 0:
                self.average_cost.pop(symbol, None)
        
        self.log_portfolio_status()

    def log_portfolio_status(self):
        """印出目前的投資組合狀態日誌。"""
        # 這裡可以擴充計算未實現損益 (Mark-to-Market P&L) 等
        print("\n" + "-"*22 + " 投資組合狀態更新 " + "-"*22)
        print(f"  剩餘現金 (Cash): {self.cash:,.2f}")
        print(f"  已實現損益 (Realized PnL): {self.realized_pnl:,.2f}")
        if self.current_positions:
            print("  目前持倉 (Current Positions):")
            for symbol, qty in self.current_positions.items():
                if qty > 0:
                    print(f"    - {symbol}: {qty} (平均成本: {self.average_cost[symbol]:.2f})")
        else:
            print("  目前持倉: 無")
        print("-" * 62 + "\n")
        