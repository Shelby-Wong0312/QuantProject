#!/usr/bin/env python3
"""
分層監控系統測試 - 4000+股票處理能力驗證
Tiered Monitoring System Test - 4000+ Stock Processing Capability Verification
"""

import time
import json
import threading
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import pandas as pd

# 添加項目根目錄到路徑
import sys
sys.path.append(str(Path(__file__).parent))

from monitoring.tiered_monitor import TieredMonitor, TierLevel

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TieredMonitoringTest:
    """分層監控系統測試類"""
    
    def __init__(self):
        self.monitor = TieredMonitor()
        self.test_results = {}
        self.test_start_time = None
        
    def setup_large_scale_test(self, total_symbols: int = 4000):
        """
        設置大規模測試環境
        
        Args:
            total_symbols: 總測試股票數量
        """
        logger.info(f"Setting up large-scale test with {total_symbols} symbols")
        
        # 生成測試股票清單
        test_symbols = []
        
        # 1. 從真實股票清單載入
        try:
            df = pd.read_csv('data/csv/tradeable_stocks.csv')
            real_symbols = df['ticker'].tolist()
            test_symbols.extend(real_symbols[:min(len(real_symbols), total_symbols)])
            logger.info(f"Loaded {len(test_symbols)} real symbols")
        except FileNotFoundError:
            logger.warning("tradeable_stocks.csv not found")
            
        # 2. 補充模擬股票代碼
        if len(test_symbols) < total_symbols:
            needed = total_symbols - len(test_symbols)
            for i in range(needed):
                test_symbols.append(f"TEST{i:04d}")
            logger.info(f"Added {needed} simulated symbols")
            
        # 3. 重新分配股票到各層級
        self._reallocate_stocks_for_test(test_symbols)
        
        return test_symbols
        
    def _reallocate_stocks_for_test(self, symbols: List[str]):
        """重新分配股票到各層級進行測試"""
        # 清空現有分配
        for tier in TierLevel:
            self.monitor.tier_stocks[tier].clear()
            
        # 重新分配
        s_tier_size = self.monitor.config['tiers']['S_tier']['max_symbols']
        a_tier_size = self.monitor.config['tiers']['A_tier']['max_symbols']
        
        # S級：前40個
        for symbol in symbols[:s_tier_size]:
            self.monitor._add_stock_to_tier(symbol, TierLevel.S_TIER)
            
        # A級：接下來100個
        for symbol in symbols[s_tier_size:s_tier_size + a_tier_size]:
            self.monitor._add_stock_to_tier(symbol, TierLevel.A_TIER)
            
        # B級：剩餘的
        for symbol in symbols[s_tier_size + a_tier_size:]:
            self.monitor._add_stock_to_tier(symbol, TierLevel.B_TIER)
            
        logger.info(f"Reallocated stocks: S={len(self.monitor.tier_stocks[TierLevel.S_TIER])}, "
                   f"A={len(self.monitor.tier_stocks[TierLevel.A_TIER])}, "
                   f"B={len(self.monitor.tier_stocks[TierLevel.B_TIER])}")
                   
    def test_system_startup(self) -> Dict:
        """測試系統啟動性能"""
        logger.info("Testing system startup performance")
        
        start_time = time.time()
        
        # 啟動監控系統
        self.monitor.start_monitoring()
        
        # 等待所有線程啟動
        time.sleep(5)
        
        startup_time = time.time() - start_time
        
        # 檢查線程狀態
        status = self.monitor.get_monitoring_status()
        
        result = {
            'test_type': 'system_startup',
            'startup_time_seconds': startup_time,
            'threads_started': status['active_threads'],
            'is_running': status['is_running'],
            'tier_counts': status['tier_counts'],
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"System startup completed in {startup_time:.2f}s with {status['active_threads']} threads")
        return result
        
    def test_monitoring_performance(self, duration_minutes: int = 5) -> Dict:
        """
        測試監控性能
        
        Args:
            duration_minutes: 測試持續時間（分鐘）
        """
        logger.info(f"Testing monitoring performance for {duration_minutes} minutes")
        
        # 記錄初始統計
        initial_status = self.monitor.get_monitoring_status()
        initial_scans = initial_status['performance_stats']['total_scans']
        initial_signals = initial_status['performance_stats']['total_signals']
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        # 監控期間統計
        monitoring_stats = {
            'scans_per_minute': [],
            'signals_per_minute': [],
            'tier_adjustments': []
        }
        
        # 每分鐘收集一次統計
        while time.time() < end_time:
            time.sleep(60)  # 等待1分鐘
            
            current_status = self.monitor.get_monitoring_status()
            current_scans = current_status['performance_stats']['total_scans']
            current_signals = current_status['performance_stats']['total_signals']
            
            # 計算每分鐘速率
            scans_per_minute = current_scans - initial_scans
            signals_per_minute = current_signals - initial_signals
            
            monitoring_stats['scans_per_minute'].append(scans_per_minute)
            monitoring_stats['signals_per_minute'].append(signals_per_minute)
            monitoring_stats['tier_adjustments'].append(
                current_status['performance_stats']['tier_adjustments']
            )
            
            # 更新基準值
            initial_scans = current_scans
            initial_signals = current_signals
            
            logger.info(f"Minute stats - Scans: {scans_per_minute}, Signals: {signals_per_minute}")
            
        # 計算總體性能
        total_duration = time.time() - start_time
        final_status = self.monitor.get_monitoring_status()
        
        result = {
            'test_type': 'monitoring_performance',
            'duration_minutes': duration_minutes,
            'total_scans': final_status['performance_stats']['total_scans'],
            'total_signals': final_status['performance_stats']['total_signals'],
            'total_adjustments': final_status['performance_stats']['tier_adjustments'],
            'average_scans_per_minute': sum(monitoring_stats['scans_per_minute']) / len(monitoring_stats['scans_per_minute']) if monitoring_stats['scans_per_minute'] else 0,
            'average_signals_per_minute': sum(monitoring_stats['signals_per_minute']) / len(monitoring_stats['signals_per_minute']) if monitoring_stats['signals_per_minute'] else 0,
            'monitoring_stats': monitoring_stats,
            'final_tier_counts': final_status['tier_counts'],
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Performance test completed - "
                   f"Avg scans/min: {result['average_scans_per_minute']:.1f}, "
                   f"Avg signals/min: {result['average_signals_per_minute']:.1f}")
        
        return result
        
    def test_tier_adjustment_mechanism(self) -> Dict:
        """測試層級調整機制"""
        logger.info("Testing tier adjustment mechanism")
        
        # 記錄初始層級分布
        initial_counts = {}
        for tier in TierLevel:
            initial_counts[tier.value] = len(self.monitor.tier_stocks[tier])
            
        # 人工觸發一些調整（模擬強信號）
        test_symbols = list(self.monitor.tier_stocks[TierLevel.B_TIER])[:5]
        
        for symbol in test_symbols:
            if symbol in self.monitor.stock_tiers:
                stock_info = self.monitor.stock_tiers[symbol]
                # 模擬強信號
                stock_info.signal_strength = 0.9
                stock_info.signal_count = 5
                stock_info.promotion_score = 0.95
                
        # 等待調整發生
        time.sleep(120)  # 等待2分鐘
        
        # 檢查調整結果
        final_counts = {}
        for tier in TierLevel:
            final_counts[tier.value] = len(self.monitor.tier_stocks[tier])
            
        adjustments_made = self.monitor.performance_stats['tier_adjustments']
        
        result = {
            'test_type': 'tier_adjustment',
            'initial_tier_counts': initial_counts,
            'final_tier_counts': final_counts,
            'total_adjustments': adjustments_made,
            'test_symbols_used': test_symbols,
            'adjustment_working': adjustments_made > 0,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Tier adjustment test - {adjustments_made} adjustments made")
        return result
        
    def test_memory_usage(self) -> Dict:
        """測試內存使用情況"""
        logger.info("Testing memory usage")
        
        import psutil
        import os
        
        # 獲取當前進程
        process = psutil.Process(os.getpid())
        
        # 內存使用統計
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        # 估算數據結構大小
        total_stocks = sum(len(self.monitor.tier_stocks[tier]) for tier in TierLevel)
        stock_tiers_size = len(self.monitor.stock_tiers)
        
        result = {
            'test_type': 'memory_usage',
            'memory_rss_mb': memory_info.rss / 1024 / 1024,  # MB
            'memory_vms_mb': memory_info.vms / 1024 / 1024,  # MB
            'memory_percent': memory_percent,
            'total_stocks_tracked': total_stocks,
            'stock_info_objects': stock_tiers_size,
            'memory_per_stock_bytes': (memory_info.rss / total_stocks) if total_stocks > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Memory usage - RSS: {result['memory_rss_mb']:.1f}MB, "
                   f"Per stock: {result['memory_per_stock_bytes']:.1f} bytes")
        
        return result
        
    def test_scalability_limits(self) -> Dict:
        """測試可擴展性限制"""
        logger.info("Testing scalability limits")
        
        # 逐步增加負載測試
        test_sizes = [1000, 2000, 3000, 4000, 5000]
        scalability_results = []
        
        for size in test_sizes:
            logger.info(f"Testing scalability with {size} symbols")
            
            try:
                # 設置測試規模
                test_symbols = self.setup_large_scale_test(size)
                
                # 測試短期性能
                start_time = time.time()
                initial_status = self.monitor.get_monitoring_status()
                initial_scans = initial_status['performance_stats']['total_scans']
                
                # 運行1分鐘
                time.sleep(60)
                
                final_status = self.monitor.get_monitoring_status()
                final_scans = final_status['performance_stats']['total_scans']
                
                duration = time.time() - start_time
                scans_completed = final_scans - initial_scans
                throughput = scans_completed / duration
                
                # 內存使用
                import psutil
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                
                scalability_results.append({
                    'symbol_count': size,
                    'scans_per_second': throughput,
                    'memory_mb': memory_mb,
                    'tier_distribution': final_status['tier_counts'],
                    'success': True
                })
                
                logger.info(f"Size {size}: {throughput:.1f} scans/sec, {memory_mb:.1f}MB")
                
            except Exception as e:
                logger.error(f"Scalability test failed at {size} symbols: {e}")
                scalability_results.append({
                    'symbol_count': size,
                    'scans_per_second': 0,
                    'memory_mb': 0,
                    'error': str(e),
                    'success': False
                })
                break
                
        result = {
            'test_type': 'scalability_limits',
            'scalability_results': scalability_results,
            'max_successful_size': max([r['symbol_count'] for r in scalability_results if r['success']], default=0),
            'timestamp': datetime.now().isoformat()
        }
        
        return result
        
    def run_comprehensive_test(self, test_duration_minutes: int = 10) -> Dict:
        """
        運行全面測試
        
        Args:
            test_duration_minutes: 總測試時間（分鐘）
        """
        logger.info("="*60)
        logger.info("STARTING COMPREHENSIVE TIERED MONITORING TEST")
        logger.info("="*60)
        
        self.test_start_time = time.time()
        
        # 設置大規模測試環境
        test_symbols = self.setup_large_scale_test(4000)
        
        test_results = {}
        
        try:
            # 1. 系統啟動測試
            logger.info("\n" + "="*40)
            logger.info("1. SYSTEM STARTUP TEST")
            logger.info("="*40)
            test_results['startup'] = self.test_system_startup()
            
            # 2. 監控性能測試
            logger.info("\n" + "="*40)
            logger.info("2. MONITORING PERFORMANCE TEST")
            logger.info("="*40)
            test_results['performance'] = self.test_monitoring_performance(test_duration_minutes)
            
            # 3. 層級調整機制測試
            logger.info("\n" + "="*40)
            logger.info("3. TIER ADJUSTMENT MECHANISM TEST")
            logger.info("="*40)
            test_results['tier_adjustment'] = self.test_tier_adjustment_mechanism()
            
            # 4. 內存使用測試
            logger.info("\n" + "="*40)
            logger.info("4. MEMORY USAGE TEST")
            logger.info("="*40)
            test_results['memory_usage'] = self.test_memory_usage()
            
            # 5. 可擴展性限制測試
            logger.info("\n" + "="*40)
            logger.info("5. SCALABILITY LIMITS TEST")
            logger.info("="*40)
            test_results['scalability'] = self.test_scalability_limits()
            
        except Exception as e:
            logger.error(f"Test execution error: {e}")
            test_results['error'] = str(e)
            
        finally:
            # 停止監控
            self.monitor.stop_monitoring()
            
        # 生成綜合報告
        total_duration = time.time() - self.test_start_time
        
        comprehensive_report = {
            'test_suite': 'tiered_monitoring_comprehensive',
            'version': '1.0',
            'test_duration_seconds': total_duration,
            'symbols_tested': len(test_symbols),
            'test_timestamp': datetime.now().isoformat(),
            'test_results': test_results,
            'final_monitoring_status': self.monitor.get_monitoring_status(),
            'system_passed_4k_test': self._evaluate_4k_capability(test_results)
        }
        
        return comprehensive_report
        
    def _evaluate_4k_capability(self, test_results: Dict) -> Dict:
        """評估4000+股票處理能力"""
        evaluation = {
            'can_handle_4k_stocks': False,
            'performance_score': 0,
            'bottlenecks': [],
            'recommendations': []
        }
        
        try:
            # 檢查啟動性能
            startup_time = test_results.get('startup', {}).get('startup_time_seconds', 999)
            if startup_time > 30:
                evaluation['bottlenecks'].append("Slow startup time")
                
            # 檢查監控性能
            avg_scans_per_min = test_results.get('performance', {}).get('average_scans_per_minute', 0)
            if avg_scans_per_min < 1000:  # 期望每分鐘至少掃描1000個符號
                evaluation['bottlenecks'].append("Low scanning throughput")
                
            # 檢查內存使用
            memory_mb = test_results.get('memory_usage', {}).get('memory_rss_mb', 999)
            if memory_mb > 1000:  # 期望內存使用小於1GB
                evaluation['bottlenecks'].append("High memory usage")
                
            # 檢查可擴展性
            max_size = test_results.get('scalability', {}).get('max_successful_size', 0)
            if max_size >= 4000:
                evaluation['can_handle_4k_stocks'] = True
                
            # 計算性能分數
            score = 0
            if startup_time <= 30:
                score += 25
            if avg_scans_per_min >= 1000:
                score += 25
            if memory_mb <= 1000:
                score += 25
            if max_size >= 4000:
                score += 25
                
            evaluation['performance_score'] = score
            
            # 建議
            if not evaluation['can_handle_4k_stocks']:
                evaluation['recommendations'].extend([
                    "Optimize batch processing size",
                    "Implement more aggressive caching",
                    "Consider distributed architecture"
                ])
                
        except Exception as e:
            logger.error(f"Error evaluating 4K capability: {e}")
            
        return evaluation
        
    def save_test_report(self, results: Dict, filename: str = None) -> str:
        """保存測試報告"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tiered_monitoring_test_{timestamp}.json"
            
        filepath = Path("reports") / filename
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
        logger.info(f"Test report saved to {filepath}")
        return str(filepath)

def main():
    """主測試函數"""
    # 創建測試實例
    test_runner = TieredMonitoringTest()
    
    # 運行綜合測試
    results = test_runner.run_comprehensive_test(test_duration_minutes=5)
    
    # 保存報告
    report_file = test_runner.save_test_report(results)
    
    # 打印測試摘要
    print("\n" + "="*60)
    print("TIERED MONITORING TEST SUMMARY")
    print("="*60)
    
    print(f"Test Duration: {results['test_duration_seconds']:.1f} seconds")
    print(f"Symbols Tested: {results['symbols_tested']}")
    print(f"Report File: {report_file}")
    
    # 4K能力評估
    capability = results['system_passed_4k_test']
    print(f"\n4000+ STOCK PROCESSING CAPABILITY:")
    print(f"  Can Handle 4K Stocks: {capability['can_handle_4k_stocks']}")
    print(f"  Performance Score: {capability['performance_score']}/100")
    
    if capability['bottlenecks']:
        print(f"  Bottlenecks: {', '.join(capability['bottlenecks'])}")
        
    if capability['recommendations']:
        print(f"  Recommendations:")
        for rec in capability['recommendations']:
            print(f"    - {rec}")
            
    # 關鍵性能指標
    if 'performance' in results['test_results']:
        perf = results['test_results']['performance']
        print(f"\nKEY PERFORMANCE METRICS:")
        print(f"  Average Scans/Minute: {perf.get('average_scans_per_minute', 0):.1f}")
        print(f"  Average Signals/Minute: {perf.get('average_signals_per_minute', 0):.1f}")
        print(f"  Total Adjustments: {perf.get('total_adjustments', 0)}")
        
    if 'memory_usage' in results['test_results']:
        mem = results['test_results']['memory_usage']
        print(f"  Memory Usage: {mem.get('memory_rss_mb', 0):.1f} MB")
        print(f"  Memory per Stock: {mem.get('memory_per_stock_bytes', 0):.1f} bytes")
        
    if 'scalability' in results['test_results']:
        scale = results['test_results']['scalability']
        print(f"  Max Successful Scale: {scale.get('max_successful_size', 0)} stocks")
        
    print(f"\n[{'SUCCESS' if capability['can_handle_4k_stocks'] else 'NEEDS_IMPROVEMENT'}] "
          f"Tiered monitoring system test completed!")
          
    if capability['can_handle_4k_stocks']:
        print("System is ready for 4000+ stock monitoring in production!")
    else:
        print("System needs optimization before handling 4000+ stocks.")

if __name__ == "__main__":
    main()