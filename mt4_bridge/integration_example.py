# -*- coding: utf-8 -*-
"""
整合範例：將現有策略與 MT4 橋接
展示如何將您的 AI 策略信號傳送到 MT4 執行
"""

import asyncio
import time 
from typing import Dict, Any
from mt4_bridge.zeromq.python_side import MT4Bridge

class StrategyMT4Integration:
    """將現有策略整合到 MT4"""
    
    def __init__(self):
        self.bridge = MT4Bridge()
        self.positions = {}  # 追蹤持倉
        
    async def on_market_data(self, data: Dict[str, Any]):
        """處理市場數據（從您現有的事件系統）"""
        symbol = data['symbol']
        
        # 獲取 MT4 報價
        quote = self.bridge.get_quote(symbol)
        if not quote:
            return
            
        # 這裡呼叫您的策略邏輯
        signal = self.generate_signal(data, quote)
        
        if signal:
            await self.execute_signal(signal)
            
    def generate_signal(self, market_data: Dict, quote: Dict) -> Dict:
        """
        生成交易信號
        這裡應該整合您現有的策略邏輯
        """
        # 範例：簡單的 RSI 策略
        rsi = market_data.get('indicators', {}).get('rsi')
        
        if rsi and rsi < 30:
            return {
                'action': 'BUY',
                'symbol': market_data['symbol'],
                'volume': 0.01
            }
        elif rsi and rsi > 70:
            return {
                'action': 'SELL',
                'symbol': market_data['symbol'],
                'volume': 0.01
            }
            
        return None
        
    async def execute_signal(self, signal: Dict):
        """執行交易信號"""
        symbol = signal['symbol']
        action = signal['action']
        volume = signal['volume']
        
        # 檢查是否已有持倉
        if symbol in self.positions:
            # 如果信號相反，先平倉
            current_pos = self.positions[symbol]
            if (current_pos['type'] == 'BUY' and action == 'SELL') or \
               (current_pos['type'] == 'SELL' and action == 'BUY'):
                result = self.bridge.close_order(current_pos['ticket'])
                print(f"平倉 {symbol}: {result}")
                del self.positions[symbol]
        
        # 開新倉
        if symbol not in self.positions:
            result = self.bridge.place_order(
                symbol=symbol,
                order_type=action,
                volume=volume
            )
            
            if result.get('success'):
                self.positions[symbol] = {
                    'ticket': result['ticket'],
                    'type': action,
                    'volume': volume
                }
                print(f"開倉 {symbol} {action}: {result}")
            else:
                print(f"開倉失敗: {result}")
                
    def update_risk_management(self):
        """更新風險管理（停損/獲利）"""
        account = self.bridge.get_account_info()
        if not account:
            return
            
        # 範例：根據帳戶淨值調整停損
        for symbol, pos in self.positions.items():
            quote = self.bridge.get_quote(symbol)
            if quote:
                # 設置 2% 風險的停損
                if pos['type'] == 'BUY':
                    sl = quote['bid'] - (account['balance'] * 0.02 / pos['volume'])
                else:
                    sl = quote['ask'] + (account['balance'] * 0.02 / pos['volume'])
                    
                self.bridge.modify_order(pos['ticket'], sl=sl)
                
    def get_performance_metrics(self) -> Dict:
        """獲取績效指標"""
        account = self.bridge.get_account_info()
        positions = self.bridge.get_positions()
        
        if account and positions:
            total_profit = sum(pos.get('profit', 0) for pos in positions)
            return {
                'balance': account['balance'],
                'equity': account['equity'],
                'open_positions': len(positions),
                'unrealized_pnl': total_profit,
                'margin_level': account['equity'] / account['margin'] * 100 if account['margin'] > 0 else 0
            }
        return {}


# === 與現有系統整合範例 ===

async def integrate_with_existing_system():
    """展示如何與您現有的事件驅動系統整合"""
    
    # 創建整合器
    integrator = StrategyMT4Integration()
    
    # 模擬從您的 EventLoop 接收事件
    async def handle_market_event(event):
        if event.type == 'MARKET':
            # 將市場事件傳給 MT4 整合器
            await integrator.on_market_data({
                'symbol': event.symbol,
                'timestamp': event.timestamp,
                'indicators': {
                    'rsi': 35,  # 從您的策略計算
                    'ema_short': 1.1234,
                    'ema_long': 1.1230
                }
            })
            
    # 定期更新風險管理
    async def risk_management_loop():
        while True:
            integrator.update_risk_management()
            await asyncio.sleep(60)  # 每分鐘更新
            
    # 績效監控
    async def performance_monitor():
        while True:
            metrics = integrator.get_performance_metrics()
            print(f"績效指標: {metrics}")
            await asyncio.sleep(300)  # 每 5 分鐘
            
    # 啟動所有任務
    await asyncio.gather(
        risk_management_loop(),
        performance_monitor()
    )


# === 簡化的策略範例 ===

class SimpleMT4Strategy:
    """簡化的 MT4 策略實作"""
    
    def __init__(self):
        self.bridge = MT4Bridge()
        self.symbol = "EURUSD"
        self.volume = 0.01
        
    def run(self):
        """主執行循環"""
        while True:
            try:
                # 獲取最新報價
                quote = self.bridge.get_quote(self.symbol)
                if not quote:
                    continue
                    
                # 獲取歷史數據計算指標
                history = self.bridge.get_history(self.symbol, "H1", 100)
                if history and len(history) >= 20:
                    # 計算簡單移動平均
                    closes = [bar['close'] for bar in history[:20]]
                    sma = sum(closes) / len(closes)
                    
                    # 簡單策略：價格突破 SMA
                    current_price = quote['bid']
                    
                    positions = self.bridge.get_positions()
                    has_position = any(p['symbol'] == self.symbol for p in positions)
                    
                    if not has_position:
                        if current_price > sma * 1.001:  # 突破 SMA
                            result = self.bridge.place_order(
                                self.symbol, "BUY", self.volume
                            )
                            print(f"買入信號: {result}")
                        elif current_price < sma * 0.999:  # 跌破 SMA
                            result = self.bridge.place_order(
                                self.symbol, "SELL", self.volume
                            )
                            print(f"賣出信號: {result}")
                            
            except Exception as e:
                print(f"錯誤: {e}")
                
            time.sleep(60)  # 每分鐘檢查一次


if __name__ == "__main__":
    # 運行簡單策略
    strategy = SimpleMT4Strategy()
    strategy.run()