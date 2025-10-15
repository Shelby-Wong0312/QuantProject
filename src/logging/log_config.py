"""
Centralized Logging Configuration
Cloud DevOps Agent - Logging System
"""

import os
import logging
import logging.handlers
import json
from datetime import datetime
from typing import Dict, Any
import traceback


class StructuredFormatter(logging.Formatter):
    """Structured JSON log formatter"""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
            "thread": record.thread,
            "thread_name": record.threadName,
            "process": record.process,
        }

        # Add extra fields
        if hasattr(record, "user_id"):
            log_obj["user_id"] = record.user_id
        if hasattr(record, "trade_id"):
            log_obj["trade_id"] = record.trade_id
        if hasattr(record, "strategy"):
            log_obj["strategy"] = record.strategy
        if hasattr(record, "symbol"):
            log_obj["symbol"] = record.symbol

        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(log_obj)


class TradingLogger:
    """Custom logger for trading system"""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.setup_handlers()

    def setup_handlers(self):
        """Setup log handlers"""

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)

        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            "logs/trading_system.log", maxBytes=10 * 1024 * 1024, backupCount=10  # 10MB
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(StructuredFormatter())

        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            "logs/errors.log", maxBytes=10 * 1024 * 1024, backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())

        # Syslog handler for centralized logging
        if os.environ.get("SYSLOG_HOST"):
            syslog_handler = logging.handlers.SysLogHandler(
                address=(os.environ.get("SYSLOG_HOST"), 514)
            )
            syslog_handler.setFormatter(StructuredFormatter())
            self.logger.addHandler(syslog_handler)

        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        self.logger.setLevel(logging.DEBUG)

    def trade_executed(
        self,
        trade_id: str,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        strategy: str = None,
    ):
        """Log trade execution"""
        self.logger.info(
            f"Trade executed: {action} {quantity} {symbol} @ {price}",
            extra={
                "trade_id": trade_id,
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "price": price,
                "strategy": strategy,
            },
        )

    def order_placed(
        self, order_id: str, symbol: str, order_type: str, quantity: float, price: float = None
    ):
        """Log order placement"""
        self.logger.info(
            f"Order placed: {order_type} {quantity} {symbol}",
            extra={
                "order_id": order_id,
                "symbol": symbol,
                "order_type": order_type,
                "quantity": quantity,
                "price": price,
            },
        )

    def strategy_signal(
        self, strategy: str, symbol: str, signal: str, strength: float, metadata: Dict = None
    ):
        """Log strategy signal"""
        self.logger.debug(
            f"Strategy signal: {strategy} - {signal} for {symbol} (strength: {strength})",
            extra={
                "strategy": strategy,
                "symbol": symbol,
                "signal": signal,
                "strength": strength,
                "metadata": metadata,
            },
        )

    def error_occurred(self, error_type: str, message: str, context: Dict = None, exc_info=None):
        """Log error with context"""
        self.logger.error(
            f"{error_type}: {message}",
            extra={"error_type": error_type, "context": context},
            exc_info=exc_info,
        )

    def performance_metric(
        self, metric_name: str, value: float, unit: str = None, tags: Dict = None
    ):
        """Log performance metric"""
        self.logger.info(
            f"Performance metric: {metric_name} = {value} {unit or ''}",
            extra={
                "metric_name": metric_name,
                "metric_value": value,
                "metric_unit": unit,
                "tags": tags,
            },
        )


class LogAggregator:
    """Aggregates logs from multiple sources"""

    def __init__(self):
        self.log_sources = []
        self.log_buffer = []
        self.max_buffer_size = 1000

    def add_source(self, source_path: str):
        """Add a log source to monitor"""
        self.log_sources.append(source_path)

    def collect_logs(self) -> list:
        """Collect logs from all sources"""
        all_logs = []

        for source in self.log_sources:
            try:
                with open(source, "r") as f:
                    # Read last 100 lines
                    lines = f.readlines()[-100:]
                    for line in lines:
                        try:
                            log_entry = json.loads(line)
                            all_logs.append(log_entry)
                        except json.JSONDecodeError:
                            # Handle non-JSON logs
                            all_logs.append({"message": line.strip()})
            except Exception as e:
                print(f"Error reading log source {source}: {e}")

        return all_logs

    def analyze_logs(self, logs: list) -> Dict[str, Any]:
        """Analyze log patterns"""
        analysis = {
            "total_logs": len(logs),
            "error_count": 0,
            "warning_count": 0,
            "info_count": 0,
            "top_errors": {},
            "performance_issues": [],
            "anomalies": [],
        }

        for log in logs:
            # Count log levels
            level = log.get("level", "INFO")
            if level == "ERROR":
                analysis["error_count"] += 1
                error_type = log.get("error_type", "Unknown")
                analysis["top_errors"][error_type] = analysis["top_errors"].get(error_type, 0) + 1
            elif level == "WARNING":
                analysis["warning_count"] += 1
            else:
                analysis["info_count"] += 1

            # Detect performance issues
            if "metric_name" in log and log.get("metric_name") == "latency":
                if log.get("metric_value", 0) > 1000:  # > 1 second
                    analysis["performance_issues"].append(log)

        return analysis


def setup_logging():
    """Setup logging for the entire application"""

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create specialized loggers
    trading_logger = TradingLogger("trading")
    api_logger = TradingLogger("api")
    strategy_logger = TradingLogger("strategy")
    data_logger = TradingLogger("data")

    return {
        "trading": trading_logger.logger,
        "api": api_logger.logger,
        "strategy": strategy_logger.logger,
        "data": data_logger.logger,
    }


# Log monitoring and alerting
class LogMonitor:
    """Monitor logs for critical events"""

    def __init__(self):
        self.alert_rules = []
        self.alert_callbacks = []

    def add_alert_rule(self, rule_name: str, condition: callable):
        """Add an alert rule"""
        self.alert_rules.append({"name": rule_name, "condition": condition})

    def add_alert_callback(self, callback: callable):
        """Add a callback for alerts"""
        self.alert_callbacks.append(callback)

    def check_log(self, log_entry: Dict):
        """Check if log triggers any alerts"""
        for rule in self.alert_rules:
            if rule["condition"](log_entry):
                self.trigger_alert(rule["name"], log_entry)

    def trigger_alert(self, rule_name: str, log_entry: Dict):
        """Trigger alert callbacks"""
        alert = {"rule": rule_name, "timestamp": datetime.utcnow().isoformat(), "log": log_entry}

        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Error in alert callback: {e}")


# Example alert rules
def setup_alert_rules(monitor: LogMonitor):
    """Setup common alert rules"""

    # High error rate
    monitor.add_alert_rule("high_error_rate", lambda log: log.get("level") == "ERROR")

    # API connection failure
    monitor.add_alert_rule(
        "api_connection_failure",
        lambda log: "Capital.com API" in log.get("message", "") and log.get("level") == "ERROR",
    )

    # High latency
    monitor.add_alert_rule(
        "high_latency",
        lambda log: log.get("metric_name") == "latency" and log.get("metric_value", 0) > 2000,
    )

    # Large drawdown
    monitor.add_alert_rule(
        "large_drawdown",
        lambda log: log.get("metric_name") == "drawdown" and log.get("metric_value", 0) < -0.1,
    )


if __name__ == "__main__":
    # Test logging setup
    loggers = setup_logging()

    # Test different log types
    trading_log = loggers["trading"]
    trading_log.info("System started")
    trading_log.error("Test error", exc_info=True)

    # Test log monitor
    monitor = LogMonitor()
    setup_alert_rules(monitor)

    # Test alert
    test_log = {"level": "ERROR", "message": "Capital.com API connection failed"}
    monitor.check_log(test_log)

    print("Logging system configured successfully")
