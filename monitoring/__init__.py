"""
分層監控系統模組
Tiered Monitoring System Module

這個模組實現了智能的三層股票監控系統，支援大規模股票實時監控。

主要組件:
- TieredMonitor: 核心分層監控管理器
- SignalScanner: 技術信號掃描器
- config.yaml: 系統配置文件

使用示例:
    from monitoring.tiered_monitor import TieredMonitor
    
    monitor = TieredMonitor()
    monitor.start_monitoring()
    # ... 監控運行中 ...
    monitor.stop_monitoring()
"""

from .tiered_monitor import TieredMonitor, TierLevel, StockTierInfo
from .signal_scanner import SignalScanner, Signal

__version__ = "1.0.0"
__author__ = "Quant Developer Agent"

__all__ = [
    'TieredMonitor',
    'TierLevel', 
    'StockTierInfo',
    'SignalScanner',
    'Signal'
]