"""
分層監控系統 - 智能三層股票監控
Tiered Monitoring System - Intelligent Three-Tier Stock Monitoring
"""

import asyncio
import threading
import time
import json
import sqlite3
import pandas as pd
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from enum import Enum

# 添加項目根目錄到路徑
import sys

sys.path.append(str(Path(__file__).parent.parent))

from data_pipeline.free_data_client import FreeDataClient
from monitoring.signal_scanner import SignalScanner, Signal

logger = logging.getLogger(__name__)


class TierLevel(Enum):
    """監控層級枚舉"""

    S_TIER = "S_tier"  # 實時監控
    A_TIER = "A_tier"  # 高頻監控
    B_TIER = "B_tier"  # 全市場掃描


@dataclass
class StockTierInfo:
    """股票層級資訊"""

    symbol: str
    tier: TierLevel
    last_update: datetime
    signal_strength: float
    signal_count: int
    promotion_score: float
    demotion_score: float
    metadata: Dict[str, Any] = None


class TieredMonitor:
    """
    分層監控系統

    管理三層監控架構：
    - S級：40支股票實時監控 (1秒更新)
    - A級：100支股票高頻監控 (1分鐘更新)
    - B級：4000+支股票全市場掃描 (5分鐘更新)
    """

    def __init__(self, config_path: str = "monitoring/config.yaml"):
        """初始化分層監控系統"""
        self.config = self._load_config(config_path)
        self.client = FreeDataClient()
        self.signal_scanner = SignalScanner(config_path)

        # 監控狀態
        self.is_running = False
        self.monitoring_threads = {}
        self.last_adjustment_time = datetime.now()

        # 股票分層管理
        self.stock_tiers: Dict[str, StockTierInfo] = {}
        self.tier_stocks: Dict[TierLevel, Set[str]] = {
            TierLevel.S_TIER: set(),
            TierLevel.A_TIER: set(),
            TierLevel.B_TIER: set(),
        }

        # 性能統計
        self.performance_stats = {
            "total_scans": 0,
            "total_signals": 0,
            "tier_adjustments": 0,
            "start_time": None,
        }

        # 初始化股票分配
        self._initialize_stock_allocation()

        logger.info("Tiered Monitor initialized with 3-tier architecture")

    def _load_config(self, config_path: str) -> Dict:
        """載入配置文件"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file {config_path} not found")
            raise

    def _initialize_stock_allocation(self):
        """初始化股票分層分配"""
        # 從配置載入預設股票
        default_stocks = self.config.get("default_stocks", {})

        # S級種子股票
        s_tier_seeds = default_stocks.get("S_tier_seeds", [])
        for symbol in s_tier_seeds[: self.config["tiers"]["S_tier"]["max_symbols"]]:
            self._add_stock_to_tier(symbol, TierLevel.S_TIER)

        # A級種子股票
        a_tier_seeds = default_stocks.get("A_tier_seeds", [])
        for symbol in a_tier_seeds[: self.config["tiers"]["A_tier"]["max_symbols"]]:
            if symbol not in self.tier_stocks[TierLevel.S_TIER]:
                self._add_stock_to_tier(symbol, TierLevel.A_TIER)

        # B級從股票清單載入
        try:
            df = pd.read_csv("data/csv/tradeable_stocks.csv")
            all_symbols = df["ticker"].tolist()

            # 添加到B級（排除已在S級和A級的）
            for symbol in all_symbols:
                if (
                    symbol not in self.tier_stocks[TierLevel.S_TIER]
                    and symbol not in self.tier_stocks[TierLevel.A_TIER]
                ):
                    self._add_stock_to_tier(symbol, TierLevel.B_TIER)

        except FileNotFoundError:
            logger.warning("tradeable_stocks.csv not found, using limited B-tier stocks")

        logger.info(
            f"Initialized stock allocation: S={len(self.tier_stocks[TierLevel.S_TIER])}, "
            f"A={len(self.tier_stocks[TierLevel.A_TIER])}, "
            f"B={len(self.tier_stocks[TierLevel.B_TIER])}"
        )

    def _add_stock_to_tier(self, symbol: str, tier: TierLevel):
        """將股票添加到指定層級"""
        # 從舊層級移除
        for old_tier in TierLevel:
            if symbol in self.tier_stocks[old_tier]:
                self.tier_stocks[old_tier].remove(symbol)

        # 添加到新層級
        self.tier_stocks[tier].add(symbol)

        # 更新股票層級信息
        self.stock_tiers[symbol] = StockTierInfo(
            symbol=symbol,
            tier=tier,
            last_update=datetime.now(),
            signal_strength=0.0,
            signal_count=0,
            promotion_score=0.0,
            demotion_score=0.0,
        )

    def start_monitoring(self):
        """啟動分層監控"""
        if self.is_running:
            logger.warning("Monitoring is already running")
            return

        self.is_running = True
        self.performance_stats["start_time"] = datetime.now()

        # 啟動各層級監控線程
        for tier in TierLevel:
            thread = threading.Thread(
                target=self._monitoring_loop,
                args=(tier,),
                daemon=True,
                name=f"Monitor-{tier.value}",
            )
            thread.start()
            self.monitoring_threads[tier] = thread

        # 啟動層級調整線程
        adjustment_thread = threading.Thread(
            target=self._tier_adjustment_loop, daemon=True, name="TierAdjustment"
        )
        adjustment_thread.start()
        self.monitoring_threads["adjustment"] = adjustment_thread

        logger.info("Tiered monitoring started successfully")

    def stop_monitoring(self):
        """停止分層監控"""
        self.is_running = False

        # 等待所有線程結束
        for thread in self.monitoring_threads.values():
            if thread.is_alive():
                thread.join(timeout=5)

        self.monitoring_threads.clear()
        logger.info("Tiered monitoring stopped")

    def _monitoring_loop(self, tier: TierLevel):
        """監控循環"""
        tier_config = self.config["tiers"][tier.value]
        update_interval = tier_config["update_interval"]

        logger.info(f"Started monitoring loop for {tier.value} (interval: {update_interval}s)")

        while self.is_running:
            try:
                start_time = time.time()

                # 獲取該層級的股票
                symbols = list(self.tier_stocks[tier])

                if symbols:
                    # 批量監控
                    self._monitor_tier_batch(tier, symbols)

                # 計算休眠時間
                elapsed = time.time() - start_time
                sleep_time = max(0, update_interval - elapsed)

                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in {tier.value} monitoring loop: {e}")
                time.sleep(update_interval)

    def _monitor_tier_batch(self, tier: TierLevel, symbols: List[str]):
        """批量監控指定層級的股票"""
        try:
            tier_config = self.config["tiers"][tier.value]
            batch_size = self.config.get("performance", {}).get("batch_size", 50)

            # 分批處理
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i : i + batch_size]

                # 獲取報價數據
                quotes = self.client.get_batch_quotes(batch_symbols, show_progress=False)

                # 為每個股票掃描信號
                for symbol in batch_symbols:
                    if symbol in quotes:
                        self._scan_and_update_stock(symbol, tier)

                self.performance_stats["total_scans"] += len(batch_symbols)

            logger.debug(f"Completed {tier.value} batch monitoring for {len(symbols)} symbols")

        except Exception as e:
            logger.error(f"Error in batch monitoring for {tier.value}: {e}")

    def _scan_and_update_stock(self, symbol: str, tier: TierLevel):
        """掃描並更新單個股票的信號"""
        try:
            # 根據層級選擇掃描頻率
            timeframes = ["1d"]
            if tier == TierLevel.S_TIER:
                timeframes = ["1m", "5m", "1d"]  # 多時間框架
            elif tier == TierLevel.A_TIER:
                timeframes = ["5m", "1d"]

            # 掃描信號
            signals = self.signal_scanner.scan_symbol_comprehensive(symbol, timeframes)

            if signals:
                # 計算組合信號強度
                signal_strength = self.signal_scanner.calculate_combined_signal_strength(signals)

                # 更新股票資訊
                if symbol in self.stock_tiers:
                    stock_info = self.stock_tiers[symbol]
                    stock_info.last_update = datetime.now()
                    stock_info.signal_strength = signal_strength
                    stock_info.signal_count = len(signals)

                    # 更新升級/降級分數
                    self._update_tier_scores(stock_info, signals)

                # 保存信號到數據庫
                self.signal_scanner.save_signals_to_db(signals)
                self.performance_stats["total_signals"] += len(signals)

                logger.debug(f"{symbol}: {len(signals)} signals, strength: {signal_strength:.2f}")

        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")

    def _update_tier_scores(self, stock_info: StockTierInfo, signals: List[Signal]):
        """更新股票的層級調整分數"""
        # 計算升級分數（基於信號強度和數量）
        promotion_factor = stock_info.signal_strength * (1 + len(signals) * 0.1)
        stock_info.promotion_score = min(
            1.0, stock_info.promotion_score * 0.9 + promotion_factor * 0.1
        )

        # 計算降級分數（基於無活動時間）
        inactive_duration = (datetime.now() - stock_info.last_update).total_seconds()
        if stock_info.signal_strength < 0.3:  # 弱信號
            demotion_factor = min(1.0, inactive_duration / 3600)  # 1小時正規化
            stock_info.demotion_score = min(
                1.0, stock_info.demotion_score * 0.9 + demotion_factor * 0.1
            )
        else:
            stock_info.demotion_score *= 0.95  # 有活動時衰減降級分數

    def _tier_adjustment_loop(self):
        """層級調整循環"""
        adjustment_interval = 300  # 5分鐘檢查一次

        while self.is_running:
            try:
                self._evaluate_tier_adjustments()
                time.sleep(adjustment_interval)
            except Exception as e:
                logger.error(f"Error in tier adjustment loop: {e}")
                time.sleep(adjustment_interval)

    def _evaluate_tier_adjustments(self):
        """評估並執行層級調整"""
        adjustment_rules = self.config.get("tier_adjustment", {})
        promotion_rules = adjustment_rules.get("promotion_rules", [])
        demotion_rules = adjustment_rules.get("demotion_rules", [])

        adjustments_made = 0

        # 檢查升級規則
        for rule in promotion_rules:
            adjustments_made += self._apply_promotion_rule(rule)

        # 檢查降級規則
        for rule in demotion_rules:
            adjustments_made += self._apply_demotion_rule(rule)

        if adjustments_made > 0:
            self.performance_stats["tier_adjustments"] += adjustments_made
            self.last_adjustment_time = datetime.now()
            logger.info(f"Made {adjustments_made} tier adjustments")

    def _apply_promotion_rule(self, rule: Dict) -> int:
        """應用升級規則"""
        from_tier = TierLevel(rule["from_tier"])
        to_tier = TierLevel(rule["to_tier"])
        conditions = rule["conditions"]

        # 檢查目標層級容量
        to_tier_config = self.config["tiers"][to_tier.value]
        max_symbols = to_tier_config["max_symbols"]
        current_count = len(self.tier_stocks[to_tier])

        if current_count >= max_symbols:
            return 0

        adjustments = 0
        candidates = []

        # 尋找升級候選
        for symbol in list(self.tier_stocks[from_tier]):
            if symbol in self.stock_tiers:
                stock_info = self.stock_tiers[symbol]

                # 檢查條件
                if stock_info.signal_strength >= conditions.get(
                    "signal_strength", 0.8
                ) and stock_info.signal_count >= conditions.get("min_signals", 2):

                    # 檢查時間窗口
                    time_window = conditions.get("time_window", 300)
                    if (datetime.now() - stock_info.last_update).total_seconds() <= time_window:
                        candidates.append((symbol, stock_info.promotion_score))

        # 按分數排序，升級最佳候選
        candidates.sort(key=lambda x: x[1], reverse=True)

        for symbol, score in candidates[: max_symbols - current_count]:
            self._move_stock_to_tier(symbol, to_tier)
            adjustments += 1
            logger.info(
                f"Promoted {symbol} from {from_tier.value} to {to_tier.value} (score: {score:.2f})"
            )

        return adjustments

    def _apply_demotion_rule(self, rule: Dict) -> int:
        """應用降級規則"""
        from_tier = TierLevel(rule["from_tier"])
        to_tier = TierLevel(rule["to_tier"])
        conditions = rule["conditions"]

        adjustments = 0

        # 尋找降級候選
        for symbol in list(self.tier_stocks[from_tier]):
            if symbol in self.stock_tiers:
                stock_info = self.stock_tiers[symbol]

                # 檢查無活動時間
                inactive_duration = (datetime.now() - stock_info.last_update).total_seconds()

                if inactive_duration >= conditions.get(
                    "inactive_duration", 3600
                ) and stock_info.signal_strength <= conditions.get("max_signal_strength", 0.3):

                    self._move_stock_to_tier(symbol, to_tier)
                    adjustments += 1
                    logger.info(
                        f"Demoted {symbol} from {from_tier.value} to {to_tier.value} "
                        f"(inactive: {inactive_duration/3600:.1f}h)"
                    )

        return adjustments

    def _move_stock_to_tier(self, symbol: str, new_tier: TierLevel):
        """將股票移動到新層級"""
        # 從當前層級移除
        for tier in TierLevel:
            if symbol in self.tier_stocks[tier]:
                self.tier_stocks[tier].remove(symbol)
                break

        # 添加到新層級
        self.tier_stocks[new_tier].add(symbol)

        # 更新股票資訊
        if symbol in self.stock_tiers:
            self.stock_tiers[symbol].tier = new_tier
            self.stock_tiers[symbol].promotion_score = 0.0
            self.stock_tiers[symbol].demotion_score = 0.0

    def get_monitoring_status(self) -> Dict:
        """獲取監控狀態"""
        current_time = datetime.now()

        status = {
            "is_running": self.is_running,
            "start_time": self.performance_stats.get("start_time"),
            "current_time": current_time,
            "uptime_seconds": (
                (current_time - self.performance_stats["start_time"]).total_seconds()
                if self.performance_stats["start_time"]
                else 0
            ),
            "tier_counts": {tier.value: len(self.tier_stocks[tier]) for tier in TierLevel},
            "performance_stats": self.performance_stats.copy(),
            "last_adjustment": self.last_adjustment_time,
            "active_threads": len([t for t in self.monitoring_threads.values() if t.is_alive()]),
        }

        return status

    def get_tier_details(self, tier: TierLevel = None) -> Dict:
        """獲取層級詳細資訊"""
        if tier:
            tiers_to_show = [tier]
        else:
            tiers_to_show = list(TierLevel)

        details = {}

        for t in tiers_to_show:
            symbols = list(self.tier_stocks[t])
            tier_info = []

            for symbol in symbols[:10]:  # 只顯示前10個
                if symbol in self.stock_tiers:
                    stock_info = self.stock_tiers[symbol]
                    tier_info.append(
                        {
                            "symbol": symbol,
                            "signal_strength": stock_info.signal_strength,
                            "signal_count": stock_info.signal_count,
                            "last_update": stock_info.last_update,
                            "promotion_score": stock_info.promotion_score,
                            "demotion_score": stock_info.demotion_score,
                        }
                    )

            details[t.value] = {
                "total_symbols": len(symbols),
                "config": self.config["tiers"][t.value],
                "top_stocks": tier_info,
            }

        return details

    def save_monitoring_report(self, filename: str = None) -> str:
        """保存監控報告"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tiered_monitoring_report_{timestamp}.json"

        report = {
            "monitoring_status": self.get_monitoring_status(),
            "tier_details": self.get_tier_details(),
            "config_summary": {
                "tiers": self.config["tiers"],
                "tier_adjustment": self.config.get("tier_adjustment", {}),
                "signals": self.config.get("signals", {}),
            },
            "generated_at": datetime.now().isoformat(),
        }

        filepath = Path("reports") / filename
        filepath.parent.mkdir(exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Monitoring report saved to {filepath}")
        return str(filepath)


# 示例使用
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("=" * 60)
    print("TIERED MONITORING SYSTEM TEST")
    print("=" * 60)

    # 創建監控系統
    monitor = TieredMonitor()

    # 顯示初始狀態
    status = monitor.get_monitoring_status()
    print(f"\n初始狀態:")
    print(f"  S級股票: {status['tier_counts']['S_tier']}")
    print(f"  A級股票: {status['tier_counts']['A_tier']}")
    print(f"  B級股票: {status['tier_counts']['B_tier']}")

    # 顯示層級詳情
    details = monitor.get_tier_details()
    for tier_name, tier_data in details.items():
        print(f"\n{tier_name} 詳情:")
        print(f"  總數: {tier_data['total_symbols']}")
        print(f"  更新間隔: {tier_data['config']['update_interval']}秒")
        if tier_data["top_stocks"]:
            print(f"  頂級股票: {[s['symbol'] for s in tier_data['top_stocks'][:5]]}")

    # 啟動監控（測試模式）
    print(f"\n啟動分層監控系統...")
    monitor.start_monitoring()

    # 運行一段時間進行測試
    try:
        time.sleep(30)  # 運行30秒
    except KeyboardInterrupt:
        print("\n收到中斷信號...")

    # 停止監控
    monitor.stop_monitoring()

    # 生成報告
    report_file = monitor.save_monitoring_report()
    print(f"\n監控報告已保存: {report_file}")

    # 顯示最終統計
    final_status = monitor.get_monitoring_status()
    print(f"\n最終統計:")
    print(f"  總掃描次數: {final_status['performance_stats']['total_scans']}")
    print(f"  總信號數: {final_status['performance_stats']['total_signals']}")
    print(f"  層級調整次數: {final_status['performance_stats']['tier_adjustments']}")
    print(f"  運行時間: {final_status['uptime_seconds']:.1f}秒")

    print("\n[SUCCESS] 分層監控系統測試完成！")
