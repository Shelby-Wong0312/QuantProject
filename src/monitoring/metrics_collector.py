"""
Metrics Collector for Trading System
Cloud DevOps Agent - Monitoring Module
"""

import time
import psutil
import sqlite3
from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry, generate_latest
from prometheus_client import start_http_server
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Initialize Prometheus metrics
registry = CollectorRegistry()

# System metrics
cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percentage', registry=registry)
memory_usage = Gauge('system_memory_usage_percent', 'Memory usage percentage', registry=registry)
disk_usage = Gauge('system_disk_usage_percent', 'Disk usage percentage', registry=registry)

# Trading metrics
total_trades = Counter('trading_trades_total', 'Total number of trades executed', registry=registry)
successful_trades = Counter('trading_trades_successful', 'Number of successful trades', registry=registry)
failed_trades = Counter('trading_trades_failed', 'Number of failed trades', registry=registry)

# Performance metrics
order_latency = Histogram('trading_order_latency_seconds', 'Order execution latency', 
                         buckets=[0.1, 0.5, 1, 2, 5, 10], registry=registry)
strategy_performance = Gauge('trading_strategy_performance', 'Strategy performance score',
                            ['strategy_name'], registry=registry)

# Portfolio metrics
portfolio_value = Gauge('trading_portfolio_value', 'Current portfolio value', registry=registry)
portfolio_pnl = Gauge('trading_portfolio_pnl', 'Portfolio P&L', registry=registry)
open_positions = Gauge('trading_open_positions', 'Number of open positions', registry=registry)

# API metrics
api_requests = Counter('capital_api_requests_total', 'Total API requests', ['endpoint'], registry=registry)
api_errors = Counter('capital_api_errors_total', 'API request errors', ['endpoint', 'error_type'], registry=registry)
api_latency = Summary('capital_api_latency_seconds', 'API request latency', ['endpoint'], registry=registry)
api_connected = Gauge('capital_api_connected', 'Capital.com API connection status', registry=registry)

# Data metrics
data_quality = Gauge('data_quality_score', 'Data quality score', ['symbol'], registry=registry)
data_latency = Histogram('data_processing_latency_seconds', 'Data processing latency',
                         buckets=[0.01, 0.05, 0.1, 0.5, 1], registry=registry)

class MetricsCollector:
    """Collects and exposes metrics for monitoring"""
    
    def __init__(self, port: int = 8000):
        self.port = port
        self.running = False
        self.db_path = 'data/quant_trading.db'
        
    def start(self):
        """Start metrics server"""
        try:
            start_http_server(self.port, registry=registry)
            self.running = True
            logger.info(f"Metrics server started on port {self.port}")
            
            # Start collection loop
            while self.running:
                self.collect_metrics()
                time.sleep(15)  # Collect every 15 seconds
                
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
    
    def stop(self):
        """Stop metrics collection"""
        self.running = False
        logger.info("Metrics collector stopped")
    
    def collect_metrics(self):
        """Collect all metrics"""
        try:
            # System metrics
            self._collect_system_metrics()
            
            # Trading metrics
            self._collect_trading_metrics()
            
            # Portfolio metrics
            self._collect_portfolio_metrics()
            
            # API metrics
            self._collect_api_metrics()
            
            # Data metrics
            self._collect_data_metrics()
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
    def _collect_system_metrics(self):
        """Collect system resource metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_usage.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage.set(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage.set(disk.percent)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _collect_trading_metrics(self):
        """Collect trading related metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get trade counts
            cursor.execute("SELECT COUNT(*) FROM trades WHERE date >= date('now', '-1 day')")
            daily_trades = cursor.fetchone()[0]
            total_trades.inc(daily_trades)
            
            # Get open positions
            cursor.execute("SELECT COUNT(*) FROM positions WHERE status = 'open'")
            open_count = cursor.fetchone()[0]
            open_positions.set(open_count)
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error collecting trading metrics: {e}")
    
    def _collect_portfolio_metrics(self):
        """Collect portfolio metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get portfolio value
            cursor.execute("SELECT SUM(value) FROM positions")
            total_value = cursor.fetchone()[0] or 0
            portfolio_value.set(total_value)
            
            # Calculate P&L
            cursor.execute("""
                SELECT SUM(realized_pnl) + SUM(unrealized_pnl) 
                FROM positions 
                WHERE date >= date('now', '-1 day')
            """)
            daily_pnl = cursor.fetchone()[0] or 0
            portfolio_pnl.set(daily_pnl)
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error collecting portfolio metrics: {e}")
    
    def _collect_api_metrics(self):
        """Collect API metrics"""
        try:
            # Check API connection status
            from src.connectors.capital_com_api import CapitalComAPI
            api = CapitalComAPI()
            is_connected = 1 if api.is_connected() else 0
            api_connected.set(is_connected)
            
        except Exception as e:
            logger.error(f"Error collecting API metrics: {e}")
            api_connected.set(0)
    
    def _collect_data_metrics(self):
        """Collect data quality metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get data quality scores
            cursor.execute("""
                SELECT symbol, 
                       (1.0 - CAST(missing_count AS FLOAT) / total_count) * 100 as quality
                FROM data_quality_metrics
                LIMIT 10
            """)
            
            for symbol, quality in cursor.fetchall():
                data_quality.labels(symbol=symbol).set(quality)
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error collecting data metrics: {e}")
    
    def record_trade(self, success: bool, latency: float):
        """Record a trade execution"""
        if success:
            successful_trades.inc()
        else:
            failed_trades.inc()
        order_latency.observe(latency)
    
    def record_api_call(self, endpoint: str, latency: float, error: str = None):
        """Record an API call"""
        api_requests.labels(endpoint=endpoint).inc()
        api_latency.labels(endpoint=endpoint).observe(latency)
        
        if error:
            api_errors.labels(endpoint=endpoint, error_type=error).inc()
    
    def update_strategy_performance(self, strategy_name: str, performance: float):
        """Update strategy performance metric"""
        strategy_performance.labels(strategy_name=strategy_name).set(performance)
    
    def get_metrics(self) -> bytes:
        """Get current metrics in Prometheus format"""
        return generate_latest(registry)


# Health check endpoint
class HealthCheck:
    """Health check for the trading system"""
    
    @staticmethod
    def check() -> Dict[str, Any]:
        """Perform health check"""
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }
        
        try:
            # Check database
            conn = sqlite3.connect('data/quant_trading.db')
            conn.execute("SELECT 1")
            conn.close()
            health['checks']['database'] = 'ok'
        except:
            health['checks']['database'] = 'failed'
            health['status'] = 'unhealthy'
        
        try:
            # Check disk space
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                health['checks']['disk'] = 'warning'
                health['status'] = 'degraded'
            else:
                health['checks']['disk'] = 'ok'
        except:
            health['checks']['disk'] = 'failed'
        
        try:
            # Check memory
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                health['checks']['memory'] = 'warning'
                health['status'] = 'degraded'
            else:
                health['checks']['memory'] = 'ok'
        except:
            health['checks']['memory'] = 'failed'
        
        return health


if __name__ == "__main__":
    # Test metrics collector
    collector = MetricsCollector()
    
    print("Starting metrics collector...")
    print(f"Metrics available at http://localhost:{collector.port}/metrics")
    print("Health check at http://localhost:{collector.port}/health")
    
    try:
        collector.start()
    except KeyboardInterrupt:
        print("\nStopping metrics collector...")
        collector.stop()