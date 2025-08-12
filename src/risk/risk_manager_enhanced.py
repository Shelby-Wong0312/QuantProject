"""
Enhanced Risk Management System
增強風險管理系統
Cloud Quant - Task Q-601
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
import json
from pathlib import Path

from src.risk.dynamic_stop_loss import DynamicStopLoss, ProfitProtection

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """風險指標"""
    portfolio_value: float
    cash_balance: float
    total_exposure: float
    leverage: float
    max_drawdown: float
    daily_pnl: float
    var_95: float
    position_count: int
    concentration_risk: float
    risk_score: int  # 0-100


@dataclass
class RiskAlert:
    """風險警報"""
    timestamp: datetime
    level: str  # INFO, WARNING, CRITICAL
    category: str  # STOP_LOSS, DRAWDOWN, LEVERAGE, etc
    message: str
    action_required: bool
    data: Dict


class EnhancedRiskManager:
    """
    增強風險管理器
    整合動態止損、風險監控、緊急控制
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 max_daily_loss: float = 0.02,
                 max_position_loss: float = 0.01,
                 max_drawdown: float = 0.10,
                 max_leverage: float = 2.0):
        """
        初始化風險管理器
        
        Args:
            initial_capital: 初始資金
            max_daily_loss: 每日最大虧損
            max_position_loss: 單一持倉最大虧損
            max_drawdown: 最大回撤
            max_leverage: 最大槓桿
        """
        # 風險參數
        self.initial_capital = initial_capital
        self.max_daily_loss = max_daily_loss
        self.max_position_loss = max_position_loss
        self.max_drawdown = max_drawdown
        self.max_leverage = max_leverage
        
        # 止損管理器
        self.stop_loss_manager = DynamicStopLoss()
        self.profit_protection = ProfitProtection()
        
        # 風險狀態
        self.daily_pnl = 0
        self.peak_value = initial_capital
        self.current_drawdown = 0
        self.is_emergency_mode = False
        
        # 警報系統
        self.alerts: List[RiskAlert] = []
        self.alert_callbacks = []
        
        # 風險歷史
        self.risk_history = []
        
        logger.info(f"Risk Manager initialized - Capital: ${initial_capital:,.0f}")
    
    async def monitor_positions(self, 
                               positions: Dict,
                               market_data: Dict) -> List[Dict]:
        """
        實時監控所有持倉
        
        Args:
            positions: 持倉字典
            market_data: 市場數據
            
        Returns:
            交易信號列表
        """
        signals = []
        total_value = 0
        total_exposure = 0
        
        for symbol, position in positions.items():
            if symbol not in market_data:
                logger.warning(f"No market data for {symbol}")
                continue
            
            current_price = market_data[symbol]['price']
            position_value = position['quantity'] * current_price
            total_value += position_value
            total_exposure += abs(position_value)
            
            # 更新追蹤止損
            self.stop_loss_manager.update_trailing_stop(symbol, current_price)
            
            # 更新利潤保護
            self.stop_loss_manager.update_profit_lock(symbol, current_price)
            
            # 檢查止損觸發
            triggered, reason = self.stop_loss_manager.check_stop_triggered(symbol, current_price)
            
            if triggered:
                signals.append({
                    'symbol': symbol,
                    'action': 'CLOSE',
                    'reason': reason,
                    'price': current_price,
                    'quantity': position['quantity'],
                    'urgency': 'HIGH'
                })
                
                # 記錄止損執行
                await self.log_stop_loss_execution(symbol, position, current_price, reason)
            
            # 檢查持倉風險
            position_pnl = (current_price - position['avg_price']) / position['avg_price']
            
            if position_pnl <= -self.max_position_loss:
                await self.create_alert(
                    level='WARNING',
                    category='POSITION_LOSS',
                    message=f"{symbol}: Position loss {position_pnl:.2%} exceeds limit",
                    action_required=True
                )
        
        # 計算風險指標
        risk_metrics = await self.calculate_risk_metrics(
            positions, market_data, total_value, total_exposure
        )
        
        # 檢查風險限制
        await self.check_risk_limits(risk_metrics, signals)
        
        return signals
    
    async def calculate_risk_metrics(self,
                                    positions: Dict,
                                    market_data: Dict,
                                    total_value: float,
                                    total_exposure: float) -> RiskMetrics:
        """
        計算風險指標
        
        Args:
            positions: 持倉
            market_data: 市場數據
            total_value: 總價值
            total_exposure: 總暴露
            
        Returns:
            風險指標
        """
        # 計算槓桿
        cash_balance = self.initial_capital - total_exposure
        leverage = total_exposure / max(1, self.initial_capital)
        
        # 計算回撤
        if total_value > self.peak_value:
            self.peak_value = total_value
        
        self.current_drawdown = (self.peak_value - total_value) / self.peak_value
        
        # 計算集中度風險
        position_values = []
        for symbol, position in positions.items():
            if symbol in market_data:
                value = position['quantity'] * market_data[symbol]['price']
                position_values.append(abs(value))
        
        concentration_risk = 0
        if position_values:
            max_position = max(position_values)
            concentration_risk = max_position / total_exposure if total_exposure > 0 else 0
        
        # 計算 VaR (簡化版)
        returns = []
        for symbol, position in positions.items():
            if symbol in market_data and 'returns' in market_data[symbol]:
                returns.extend(market_data[symbol]['returns'])
        
        var_95 = 0
        if returns:
            var_95 = np.percentile(returns, 5) * total_value
        
        # 計算風險評分
        risk_score = self.calculate_risk_score(
            leverage, self.current_drawdown, concentration_risk
        )
        
        return RiskMetrics(
            portfolio_value=total_value,
            cash_balance=cash_balance,
            total_exposure=total_exposure,
            leverage=leverage,
            max_drawdown=self.current_drawdown,
            daily_pnl=self.daily_pnl,
            var_95=var_95,
            position_count=len(positions),
            concentration_risk=concentration_risk,
            risk_score=risk_score
        )
    
    def calculate_risk_score(self, 
                            leverage: float,
                            drawdown: float,
                            concentration: float) -> int:
        """
        計算綜合風險評分
        
        Args:
            leverage: 槓桿率
            drawdown: 回撤
            concentration: 集中度
            
        Returns:
            風險評分 (0-100)
        """
        # 槓桿風險 (0-40分)
        leverage_score = min(40, leverage * 20)
        
        # 回撤風險 (0-30分)
        drawdown_score = min(30, drawdown * 300)
        
        # 集中度風險 (0-30分)
        concentration_score = min(30, concentration * 100)
        
        total_score = int(leverage_score + drawdown_score + concentration_score)
        
        return min(100, max(0, total_score))
    
    async def check_risk_limits(self, 
                               metrics: RiskMetrics,
                               signals: List[Dict]):
        """
        檢查風險限制
        
        Args:
            metrics: 風險指標
            signals: 信號列表
        """
        # 檢查每日虧損
        if self.daily_pnl <= -self.max_daily_loss * self.initial_capital:
            await self.create_alert(
                level='CRITICAL',
                category='DAILY_LOSS',
                message=f"Daily loss limit reached: ${self.daily_pnl:,.0f}",
                action_required=True
            )
            
            # 觸發緊急停止
            if not self.is_emergency_mode:
                await self.emergency_stop_all(signals)
        
        # 檢查最大回撤
        if metrics.max_drawdown >= self.max_drawdown:
            await self.create_alert(
                level='CRITICAL',
                category='MAX_DRAWDOWN',
                message=f"Maximum drawdown reached: {metrics.max_drawdown:.2%}",
                action_required=True
            )
        
        # 檢查槓桿
        if metrics.leverage > self.max_leverage:
            await self.create_alert(
                level='WARNING',
                category='LEVERAGE',
                message=f"Leverage exceeds limit: {metrics.leverage:.2f}x",
                action_required=True
            )
        
        # 檢查風險評分
        if metrics.risk_score >= 80:
            await self.create_alert(
                level='CRITICAL',
                category='RISK_SCORE',
                message=f"High risk score: {metrics.risk_score}/100",
                action_required=True
            )
        elif metrics.risk_score >= 60:
            await self.create_alert(
                level='WARNING',
                category='RISK_SCORE',
                message=f"Elevated risk score: {metrics.risk_score}/100",
                action_required=False
            )
    
    async def emergency_stop_all(self, signals: List[Dict]):
        """
        緊急止損 - 關閉所有持倉
        
        Args:
            signals: 信號列表
        """
        logger.critical("EMERGENCY STOP ACTIVATED!")
        self.is_emergency_mode = True
        
        await self.create_alert(
            level='CRITICAL',
            category='EMERGENCY_STOP',
            message="Emergency stop activated - closing all positions",
            action_required=True
        )
        
        # 添加緊急平倉信號
        for symbol in self.stop_loss_manager.position_stops.keys():
            signals.append({
                'symbol': symbol,
                'action': 'CLOSE_ALL',
                'reason': 'EMERGENCY_STOP',
                'urgency': 'CRITICAL'
            })
    
    async def log_stop_loss_execution(self, 
                                     symbol: str,
                                     position: Dict,
                                     exit_price: float,
                                     reason: str):
        """
        記錄止損執行
        
        Args:
            symbol: 股票代碼
            position: 持倉信息
            exit_price: 出場價格
            reason: 止損原因
        """
        entry_price = position.get('avg_price', 0)
        quantity = position.get('quantity', 0)
        
        pnl = (exit_price - entry_price) * quantity
        pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
        
        execution_log = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason
        }
        
        # 保存到文件
        log_file = Path('logs/stop_loss_executions.json')
        log_file.parent.mkdir(exist_ok=True)
        
        try:
            if log_file.exists():
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(execution_log)
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save stop loss log: {e}")
        
        logger.info(f"Stop loss executed: {symbol} - PnL: ${pnl:,.2f} ({pnl_pct:.2%})")
    
    async def create_alert(self, 
                          level: str,
                          category: str,
                          message: str,
                          action_required: bool = False,
                          data: Dict = None):
        """
        創建風險警報
        
        Args:
            level: 警報級別
            category: 警報類別
            message: 警報消息
            action_required: 是否需要行動
            data: 附加數據
        """
        alert = RiskAlert(
            timestamp=datetime.now(),
            level=level,
            category=category,
            message=message,
            action_required=action_required,
            data=data or {}
        )
        
        self.alerts.append(alert)
        
        # 觸發回調
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
        
        # 記錄警報
        if level == 'CRITICAL':
            logger.critical(f"[{category}] {message}")
        elif level == 'WARNING':
            logger.warning(f"[{category}] {message}")
        else:
            logger.info(f"[{category}] {message}")
    
    def register_alert_callback(self, callback):
        """
        註冊警報回調
        
        Args:
            callback: 回調函數
        """
        self.alert_callbacks.append(callback)
    
    def reset_daily_metrics(self):
        """重置每日指標"""
        self.daily_pnl = 0
        logger.info("Daily risk metrics reset")
    
    def get_risk_report(self) -> Dict:
        """
        獲取風險報告
        
        Returns:
            風險報告字典
        """
        recent_alerts = [
            {
                'timestamp': alert.timestamp.isoformat(),
                'level': alert.level,
                'category': alert.category,
                'message': alert.message
            }
            for alert in self.alerts[-10:]  # 最近10條警報
        ]
        
        stop_loss_stats = self.stop_loss_manager.get_statistics()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'risk_parameters': {
                'max_daily_loss': self.max_daily_loss,
                'max_position_loss': self.max_position_loss,
                'max_drawdown': self.max_drawdown,
                'max_leverage': self.max_leverage
            },
            'current_status': {
                'daily_pnl': self.daily_pnl,
                'current_drawdown': self.current_drawdown,
                'is_emergency_mode': self.is_emergency_mode,
                'active_alerts': len([a for a in self.alerts if a.action_required])
            },
            'stop_loss_statistics': stop_loss_stats,
            'recent_alerts': recent_alerts
        }


if __name__ == "__main__":
    # 測試風險管理器
    import asyncio
    
    async def test_risk_manager():
        print("Testing Enhanced Risk Manager...")
        print("=" * 50)
        
        # 初始化風險管理器
        risk_mgr = EnhancedRiskManager(initial_capital=100000)
        
        # 模擬持倉
        positions = {
            'AAPL': {'quantity': 100, 'avg_price': 180},
            'GOOGL': {'quantity': 50, 'avg_price': 140},
            'MSFT': {'quantity': 30, 'avg_price': 380}
        }
        
        # 模擬市場數據
        market_data = {
            'AAPL': {'price': 175, 'returns': [-0.02, -0.01, 0.01]},
            'GOOGL': {'price': 142, 'returns': [0.01, 0.02, -0.01]},
            'MSFT': {'price': 385, 'returns': [0.01, 0.01, 0.02]}
        }
        
        # 設置初始止損
        for symbol in positions:
            risk_mgr.stop_loss_manager.set_initial_stop(
                symbol, 
                positions[symbol]['avg_price'],
                2.0  # ATR值
            )
        
        # 監控持倉
        print("\nMonitoring positions...")
        signals = await risk_mgr.monitor_positions(positions, market_data)
        
        print(f"\nGenerated {len(signals)} signals:")
        for signal in signals:
            print(f"  {signal}")
        
        # 獲取風險報告
        report = risk_mgr.get_risk_report()
        
        print("\n" + "=" * 50)
        print("Risk Report:")
        print(f"Daily P&L: ${report['current_status']['daily_pnl']:,.2f}")
        print(f"Current Drawdown: {report['current_status']['current_drawdown']:.2%}")
        print(f"Active Alerts: {report['current_status']['active_alerts']}")
        
        print("\nStop Loss Statistics:")
        for key, value in report['stop_loss_statistics'].items():
            print(f"  {key}: {value}")
        
        print("\nTest Complete!")
    
    # 運行測試
    asyncio.run(test_risk_manager())