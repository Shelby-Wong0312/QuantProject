#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Capital.com 完整自動化交易系統
整合所有功能模組，實現7/24自動交易
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
import logging

# 導入Capital.com模組
from capital_data_collector import CapitalDataCollector
from capital_trading_system import CapitalTradingSystem
from capital_live_trading import CapitalLiveTrading

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("capital_automation.log"), logging.StreamHandler()],
)

logger = logging.getLogger("CapitalAutomation")


class CapitalAutomationSystem:
    """Capital.com 7/24 自動化交易系統"""

    def __init__(self):
        self.trading_system = CapitalTradingSystem()
        self.live_trading = CapitalLiveTrading()
        self.collector = self.trading_system.collector
        self.running = False
        self.threads = []

        # 系統配置
        self.config = {
            "max_positions": 5,
            "risk_per_trade": 0.02,
            "max_daily_trades": 20,
            "check_interval": 300,  # 5分鐘
            "enabled_strategies": ["momentum", "mean_reversion", "trend_following"],
        }

        # 績效追蹤
        self.performance = {
            "start_balance": 0,
            "current_balance": 0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "max_drawdown": 0,
            "best_trade": 0,
            "worst_trade": 0,
        }

        # 初始化帳戶
        self.initialize_account()

    def initialize_account(self):
        """初始化帳戶信息"""
        account = self.collector.get_account_info()
        if account:
            self.performance["start_balance"] = account.get("balance", 0)
            self.performance["current_balance"] = account.get("balance", 0)
            logger.info(f"Account initialized - Balance: ${account.get('balance'):,.2f}")

    def monitor_positions(self):
        """持續監控持倉"""
        while self.running:
            try:
                positions = self.collector.get_positions()

                if positions:
                    total_pnl = 0
                    for pos in positions:
                        market = pos.get("market", {})
                        position = pos.get("position", {})

                        epic = market.get("epic")
                        direction = position.get("direction")
                        size = position.get("size")
                        open_level = position.get("level")

                        # 計算盈虧
                        current_price = (
                            market.get("bid") if direction == "BUY" else market.get("offer")
                        )
                        if direction == "BUY":
                            pnl = (current_price - open_level) * size
                        else:
                            pnl = (open_level - current_price) * size

                        total_pnl += pnl

                        # 動態風險管理
                        pnl_pct = (pnl / (open_level * size)) * 100

                        if pnl_pct > 5:  # 盈利超過5%
                            logger.info(
                                f"Position {epic} profit {pnl_pct:.2f}% - Consider taking profit"
                            )
                            # 可以實現自動平倉邏輯
                        elif pnl_pct < -2:  # 虧損超過2%
                            logger.warning(f"Position {epic} loss {pnl_pct:.2f}% - Risk warning")

                    # 更新總盈虧
                    account = self.collector.get_account_info()
                    if account:
                        self.performance["current_balance"] = account.get("balance", 0)

                time.sleep(60)  # 每分鐘檢查一次

            except Exception as e:
                logger.error(f"Position monitoring error: {e}")
                time.sleep(60)

    def execute_trading_cycle(self):
        """執行交易週期"""
        while self.running:
            try:
                cycle_start = datetime.now()
                logger.info(f"Starting trading cycle at {cycle_start}")

                # 更新帳戶信息
                account = self.collector.get_account_info()
                if not account:
                    logger.error("Failed to get account info")
                    time.sleep(60)
                    continue

                # 檢查持倉數量
                positions = self.collector.get_positions()
                current_positions = len(positions)

                if current_positions >= self.config["max_positions"]:
                    logger.info(
                        f"Max positions reached ({current_positions}/{self.config['max_positions']})"
                    )
                else:
                    # 掃描市場機會
                    for market in self.live_trading.active_markets:
                        try:
                            # 執行策略
                            signal = self.analyze_market(market)

                            if signal:
                                self.execute_trade(market, signal)

                        except Exception as e:
                            logger.error(f"Error analyzing {market}: {e}")

                # 等待下個週期
                elapsed = (datetime.now() - cycle_start).seconds
                wait_time = max(0, self.config["check_interval"] - elapsed)

                if wait_time > 0:
                    logger.info(f"Waiting {wait_time} seconds for next cycle")
                    time.sleep(wait_time)

            except Exception as e:
                logger.error(f"Trading cycle error: {e}")
                time.sleep(60)

    def analyze_market(self, epic: str) -> Optional[Dict]:
        """分析市場並生成信號"""

        # 收集策略信號
        {}

        if "momentum" in self.config["enabled_strategies"]:
            signals["momentum"] = self.live_trading.momentum_strategy(epic)

        if "mean_reversion" in self.config["enabled_strategies"]:
            signals["mean_reversion"] = self.live_trading.mean_reversion_strategy(epic)

        if "trend_following" in self.config["enabled_strategies"]:
            signals["trend_following"] = self.live_trading.trend_following_strategy(epic)

        # 計算共識
        buy_count = sum(1 for s in signals.values() if s == "BUY")
        sell_count = sum(1 for s in signals.values() if s == "SELL")

        # 需要多數策略同意
        threshold = len(self.config["enabled_strategies"]) // 2 + 1

        if buy_count >= threshold:
            return {"signal": "BUY", "confidence": buy_count / len(signals)}
        elif sell_count >= threshold:
            return {"signal": "SELL", "confidence": sell_count / len(signals)}

        return None

    def execute_trade(self, epic: str, signal: Dict):
        """執行交易"""
        try:
            # 獲取市場信息
            market = self.collector.get_market_details(epic)
            if not market:
                logger.error(f"Failed to get market details for {epic}")
                return

            current_price = market.get("snapshot", {}).get("bid")

            # 計算倉位大小
            account = self.collector.get_account_info()
            balance = account.get("balance", 10000)
            risk_amount = balance * self.config["risk_per_trade"]

            # 設置止損止利
            if signal["signal"] == "BUY":
                stop_loss = current_price * 0.98
                take_profit = current_price * 1.04
            else:
                stop_loss = current_price * 1.02
                take_profit = current_price * 0.96

            stop_loss_pips = abs(current_price - stop_loss)
            size = self.trading_system.calculate_position_size(epic, stop_loss_pips)

            # 執行訂單
            logger.info(
                f"Executing {signal['signal']} on {epic} - Size: {size}, Confidence: {signal['confidence']:.1%}"
            )

            result = self.trading_system.place_order(
                epic=epic,
                direction=signal["signal"],
                size=size,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

            if result["success"]:
                self.performance["total_trades"] += 1
                logger.info(
                    f"Trade executed successfully - Deal: {result['data'].get('dealReference')}"
                )
            else:
                logger.error(f"Trade execution failed: {result.get('error')}")

        except Exception as e:
            logger.error(f"Trade execution error: {e}")

    def generate_report(self):
        """生成績效報告"""
        while self.running:
            try:
                # 每小時生成一次報告
                time.sleep(3600)

                # 計算績效指標
                pnl = self.performance["current_balance"] - self.performance["start_balance"]
                pnl_pct = (
                    (pnl / self.performance["start_balance"]) * 100
                    if self.performance["start_balance"] > 0
                    else 0
                )

                win_rate = 0
                if self.performance["total_trades"] > 0:
                    win_rate = (
                        self.performance["winning_trades"] / self.performance["total_trades"]
                    ) * 100

                """
                ========================================
                Capital.com Automation Report
                ========================================
                Time: {datetime.now()}
                
                Account Performance:
                - Start Balance: ${self.performance['start_balance']:,.2f}
                - Current Balance: ${self.performance['current_balance']:,.2f}
                - Total P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)
                
                Trading Statistics:
                - Total Trades: {self.performance['total_trades']}
                - Winning Trades: {self.performance['winning_trades']}
                - Losing Trades: {self.performance['losing_trades']}
                - Win Rate: {win_rate:.1f}%
                
                Risk Metrics:
                - Max Drawdown: ${self.performance['max_drawdown']:,.2f}
                - Best Trade: ${self.performance['best_trade']:,.2f}
                - Worst Trade: ${self.performance['worst_trade']:,.2f}
                ========================================
                """

                logger.info(report)

                # 保存報告到文件
                with open(f'reports/report_{datetime.now().strftime("%Y%m%d_%H")}.txt', "w") as f:
                    f.write(report)

            except Exception as e:
                logger.error(f"Report generation error: {e}")

    def start(self):
        """啟動自動化系統"""
        logger.info("Starting Capital.com Automation System")
        self.running = True

        # 創建報告目錄
        os.makedirs("reports", exist_ok=True)

        # 啟動監控線程
        monitor_thread = threading.Thread(target=self.monitor_positions, daemon=True)
        monitor_thread.start()
        self.threads.append(monitor_thread)

        # 啟動交易線程
        trading_thread = threading.Thread(target=self.execute_trading_cycle, daemon=True)
        trading_thread.start()
        self.threads.append(trading_thread)

        # 啟動報告線程
        report_thread = threading.Thread(target=self.generate_report, daemon=True)
        report_thread.start()
        self.threads.append(report_thread)

        logger.info("All systems started successfully")

        # 主循環
        try:
            while self.running:
                time.sleep(10)
                # 可以在這裡添加系統健康檢查

        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
            self.stop()

    def stop(self):
        """停止系統"""
        logger.info("Stopping automation system")
        self.running = False

        # 等待線程結束
        for thread in self.threads:
            thread.join(timeout=5)

        # 生成最終報告
        self.generate_final_report()

        logger.info("System stopped")

    def generate_final_report(self):
        """生成最終報告"""
        pnl = self.performance["current_balance"] - self.performance["start_balance"]
        pnl_pct = (
            (pnl / self.performance["start_balance"]) * 100
            if self.performance["start_balance"] > 0
            else 0
        )

        """
        ========================================
        FINAL TRADING REPORT
        ========================================
        Session End: {datetime.now()}
        
        Final Results:
        - Starting Balance: ${self.performance['start_balance']:,.2f}
        - Ending Balance: ${self.performance['current_balance']:,.2f}
        - Net P&L: ${pnl:,.2f}
        - Return: {pnl_pct:+.2f}%
        
        Total Trades: {self.performance['total_trades']}
        ========================================
        """

        print(report)

        # 保存最終報告
        with open(f'reports/final_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt', "w") as f:
            f.write(report)


def main():
    """主函數"""
    print("\n" + "=" * 60)
    print(" Capital.com 7/24 Automation System")
    print("=" * 60)
    print(" [1] Start Full Automation")
    print(" [2] Monitor Only Mode")
    print(" [3] Test Mode")
    print(" [0] Exit")
    print("=" * 60)

    choice = input("\nSelect mode: ").strip()

    if choice == "1":
        # 完全自動化模式
        system = CapitalAutomationSystem()
        system.start()

    elif choice == "2":
        # 監控模式
        print("\n[MONITOR MODE] Starting...")
        collector = CapitalDataCollector()
        collector.login()

        while True:
            positions = collector.get_positions()
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Positions: {len(positions)}")

            for pos in positions:
                market = pos.get("market", {})
                print(f"  - {market.get('instrumentName')}: {market.get('epic')}")

            time.sleep(30)

    elif choice == "3":
        # 測試模式
        print("\n[TEST MODE] Running system check...")
        system = CapitalAutomationSystem()

        # 測試各組件
        print("- Account: OK" if system.collector.cst else "- Account: FAIL")
        print(f"- Markets: {len(system.live_trading.active_markets)}")
        print(f"- Strategies: {len(system.config['enabled_strategies'])}")
        print("\nTest complete")

    else:
        print("Exiting...")


if __name__ == "__main__":
    main()
