"""
Health Monitor - System health monitoring and alerting
"""

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import psutil
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DOWN = "down"


@dataclass
class ComponentHealth:
    """Health information for a component"""

    name: str
    status: HealthStatus
    last_check: datetime
    message: str
    metrics: Dict[str, Any]


class HealthMonitor:
    """
    Monitors system health and component status
    """

    def __init__(
        self,
        components: Dict[str, Any],
        check_interval: int = 60,
        alert_threshold: int = 3,
    ):
        """
        Initialize health monitor

        Args:
            components: Dictionary of component name to instance
            check_interval: Health check interval in seconds
            alert_threshold: Number of failures before alerting
        """
        self.components = components
        self.check_interval = check_interval
        self.alert_threshold = alert_threshold

        self.is_running = False
        self.health_history = []
        self.failure_counts = {name: 0 for name in components}

        # System resource thresholds
        self.resource_thresholds = {
            "cpu_percent": 80,
            "memory_percent": 85,
            "disk_percent": 90,
        }

        # Component-specific health checks
        self.health_checks = {
            "data_client": self._check_data_client,
            "lstm_predictor": self._check_lstm_predictor,
            "sentiment_analyzer": self._check_sentiment_analyzer,
            "rl_agent": self._check_rl_agent,
            "data_pipeline": self._check_data_pipeline,
        }

        logger.info(f"Health monitor initialized for {len(components)} components")

    async def start(self):
        """Start health monitoring"""
        self.is_running = True
        logger.info("Health monitor started")

        while self.is_running:
            try:
                # Perform health checks
                health_report = await self.check_all_components()

                # Check system resources
                resource_report = self.check_system_resources()

                # Combine reports
                full_report = {
                    "timestamp": datetime.now(),
                    "components": health_report,
                    "system_resources": resource_report,
                    "overall_status": self._determine_overall_status(
                        health_report, resource_report
                    ),
                }

                # Store history
                self.health_history.append(full_report)
                if len(self.health_history) > 1000:
                    self.health_history.pop(0)

                # Handle alerts
                await self._handle_alerts(full_report)

                # Save report
                self._save_health_report(full_report)

                # Wait for next check
                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in health monitor: {str(e)}")
                await asyncio.sleep(self.check_interval)

    async def stop(self):
        """Stop health monitoring"""
        self.is_running = False
        logger.info("Health monitor stopped")

    async def check_all_components(self) -> Dict[str, ComponentHealth]:
        """Check health of all components"""
        health_report = {}

        for name, component in self.components.items():
            try:
                if name in self.health_checks:
                    health = await self.health_checks[name](component)
                else:
                    health = await self._generic_health_check(name, component)

                health_report[name] = health

                # Update failure count
                if health.status in [HealthStatus.CRITICAL, HealthStatus.DOWN]:
                    self.failure_counts[name] += 1
                else:
                    self.failure_counts[name] = 0

            except Exception as e:
                logger.error(f"Error checking {name}: {str(e)}")
                health_report[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.DOWN,
                    last_check=datetime.now(),
                    message=f"Health check failed: {str(e)}",
                    metrics={},
                )
                self.failure_counts[name] += 1

        return health_report

    async def _check_data_client(self, client) -> ComponentHealth:
        """Check data client health"""
        try:
            # Check connection status
            is_connected = (
                await client.is_connected() if hasattr(client, "is_connected") else True
            )

            # Check data freshness
            last_update = getattr(client, "last_data_update", None)
            data_age = (
                (datetime.now() - last_update).seconds if last_update else float("inf")
            )

            # Determine status
            if not is_connected:
                status = HealthStatus.DOWN
                message = "Data client disconnected"
            elif data_age > 300:  # 5 minutes
                status = HealthStatus.WARNING
                message = f"No data updates for {data_age} seconds"
            else:
                status = HealthStatus.HEALTHY
                message = "Data client functioning normally"

            return ComponentHealth(
                name="data_client",
                status=status,
                last_check=datetime.now(),
                message=message,
                metrics={
                    "connected": is_connected,
                    "data_age_seconds": data_age,
                    "subscriptions": getattr(client, "active_subscriptions", 0),
                },
            )

        except Exception as e:
            return ComponentHealth(
                name="data_client",
                status=HealthStatus.DOWN,
                last_check=datetime.now(),
                message=str(e),
                metrics={},
            )

    async def _check_lstm_predictor(self, predictor) -> ComponentHealth:
        """Check LSTM predictor health"""
        try:
            # Check if model is loaded
            model_loaded = hasattr(predictor, "model") and predictor.model is not None

            # Check prediction latency
            if model_loaded and hasattr(predictor, "get_average_latency"):
                avg_latency = predictor.get_average_latency()
            else:
                avg_latency = 0

            # Determine status
            if not model_loaded:
                status = HealthStatus.WARNING
                message = "LSTM model not loaded"
            elif avg_latency > 1000:  # 1 second
                status = HealthStatus.WARNING
                message = f"High prediction latency: {avg_latency:.0f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = "LSTM predictor functioning normally"

            return ComponentHealth(
                name="lstm_predictor",
                status=status,
                last_check=datetime.now(),
                message=message,
                metrics={
                    "model_loaded": model_loaded,
                    "avg_latency_ms": avg_latency,
                    "predictions_count": getattr(predictor, "predictions_count", 0),
                },
            )

        except Exception as e:
            return ComponentHealth(
                name="lstm_predictor",
                status=HealthStatus.DOWN,
                last_check=datetime.now(),
                message=str(e),
                metrics={},
            )

    async def _check_sentiment_analyzer(self, analyzer) -> ComponentHealth:
        """Check sentiment analyzer health"""
        try:
            # Check if model is loaded
            model_loaded = hasattr(analyzer, "model") and analyzer.model is not None

            # Check cache hit rate
            if hasattr(analyzer, "get_cache_stats"):
                cache_stats = analyzer.get_cache_stats()
                cache_hit_rate = cache_stats.get("hit_rate", 0)
            else:
                cache_hit_rate = 0

            # Determine status
            if not model_loaded:
                status = HealthStatus.WARNING
                message = "FinBERT model not loaded"
            elif cache_hit_rate < 0.5 and getattr(analyzer, "analysis_count", 0) > 100:
                status = HealthStatus.WARNING
                message = f"Low cache hit rate: {cache_hit_rate:.1%}"
            else:
                status = HealthStatus.HEALTHY
                message = "Sentiment analyzer functioning normally"

            return ComponentHealth(
                name="sentiment_analyzer",
                status=status,
                last_check=datetime.now(),
                message=message,
                metrics={
                    "model_loaded": model_loaded,
                    "cache_hit_rate": cache_hit_rate,
                    "analysis_count": getattr(analyzer, "analysis_count", 0),
                },
            )

        except Exception as e:
            return ComponentHealth(
                name="sentiment_analyzer",
                status=HealthStatus.DOWN,
                last_check=datetime.now(),
                message=str(e),
                metrics={},
            )

    async def _check_rl_agent(self, agent) -> ComponentHealth:
        """Check RL agent health"""
        try:
            # Check if model is loaded
            model_loaded = hasattr(agent, "model") and agent.model is not None

            # Check recent performance
            if hasattr(agent, "get_recent_performance"):
                performance = agent.get_recent_performance()
                win_rate = performance.get("win_rate", 0)
            else:
                win_rate = 0

            # Determine status
            if not model_loaded:
                status = HealthStatus.WARNING
                message = "RL model not loaded"
            elif (
                win_rate < 0.3
                and hasattr(agent, "trades_count")
                and agent.trades_count > 50
            ):
                status = HealthStatus.WARNING
                message = f"Low win rate: {win_rate:.1%}"
            else:
                status = HealthStatus.HEALTHY
                message = "RL agent functioning normally"

            return ComponentHealth(
                name="rl_agent",
                status=status,
                last_check=datetime.now(),
                message=message,
                metrics={
                    "model_loaded": model_loaded,
                    "win_rate": win_rate,
                    "trades_count": getattr(agent, "trades_count", 0),
                },
            )

        except Exception as e:
            return ComponentHealth(
                name="rl_agent",
                status=HealthStatus.DOWN,
                last_check=datetime.now(),
                message=str(e),
                metrics={},
            )

    async def _check_data_pipeline(self, pipeline) -> ComponentHealth:
        """Check data pipeline health"""
        try:
            # Get data quality metrics
            if hasattr(pipeline, "get_data_quality_metrics"):
                metrics = pipeline.get_data_quality_metrics()
                success_rate = metrics.get("success_rate", 0)
                avg_latency = metrics.get("avg_latency_ms", 0)
            else:
                success_rate = 1
                avg_latency = 0

            # Determine status
            if success_rate < 0.8:
                status = HealthStatus.CRITICAL
                message = f"Low data processing success rate: {success_rate:.1%}"
            elif avg_latency > 100:
                status = HealthStatus.WARNING
                message = f"High data processing latency: {avg_latency:.0f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = "Data pipeline functioning normally"

            return ComponentHealth(
                name="data_pipeline",
                status=status,
                last_check=datetime.now(),
                message=message,
                metrics={
                    "success_rate": success_rate,
                    "avg_latency_ms": avg_latency,
                    "buffer_sizes": metrics.get("buffer_sizes", {}),
                },
            )

        except Exception as e:
            return ComponentHealth(
                name="data_pipeline",
                status=HealthStatus.DOWN,
                last_check=datetime.now(),
                message=str(e),
                metrics={},
            )

    async def _generic_health_check(self, name: str, component: Any) -> ComponentHealth:
        """Generic health check for components"""
        try:
            # Check if component has is_healthy method
            if hasattr(component, "is_healthy"):
                is_healthy = await component.is_healthy()
                status = HealthStatus.HEALTHY if is_healthy else HealthStatus.WARNING
            else:
                status = HealthStatus.HEALTHY

            return ComponentHealth(
                name=name,
                status=status,
                last_check=datetime.now(),
                message=f"{name} status: {status.value}",
                metrics={},
            )

        except Exception as e:
            return ComponentHealth(
                name=name,
                status=HealthStatus.DOWN,
                last_check=datetime.now(),
                message=str(e),
                metrics={},
            )

    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage("/")
            disk_percent = disk.percent

            # Determine status for each resource
            cpu_status = (
                HealthStatus.CRITICAL
                if cpu_percent > self.resource_thresholds["cpu_percent"]
                else HealthStatus.WARNING if cpu_percent > 70 else HealthStatus.HEALTHY
            )

            memory_status = (
                HealthStatus.CRITICAL
                if memory_percent > self.resource_thresholds["memory_percent"]
                else (
                    HealthStatus.WARNING
                    if memory_percent > 75
                    else HealthStatus.HEALTHY
                )
            )

            disk_status = (
                HealthStatus.CRITICAL
                if disk_percent > self.resource_thresholds["disk_percent"]
                else HealthStatus.WARNING if disk_percent > 80 else HealthStatus.HEALTHY
            )

            return {
                "cpu": {
                    "percent": cpu_percent,
                    "status": cpu_status.value,
                    "cores": psutil.cpu_count(),
                },
                "memory": {
                    "percent": memory_percent,
                    "status": memory_status.value,
                    "available_gb": memory.available / (1024**3),
                    "total_gb": memory.total / (1024**3),
                },
                "disk": {
                    "percent": disk_percent,
                    "status": disk_status.value,
                    "free_gb": disk.free / (1024**3),
                    "total_gb": disk.total / (1024**3),
                },
            }

        except Exception as e:
            logger.error(f"Error checking system resources: {str(e)}")
            return {}

    def _determine_overall_status(
        self,
        component_health: Dict[str, ComponentHealth],
        resource_report: Dict[str, Any],
    ) -> HealthStatus:
        """Determine overall system health status"""
        # Check component health
        component_statuses = [health.status for health in component_health.values()]

        # Check resource health
        resource_statuses = []
        for resource in ["cpu", "memory", "disk"]:
            if resource in resource_report:
                status_str = resource_report[resource].get("status", "healthy")
                resource_statuses.append(HealthStatus(status_str))

        all_statuses = component_statuses + resource_statuses

        # Determine overall status
        if any(status == HealthStatus.DOWN for status in all_statuses):
            return HealthStatus.DOWN
        elif any(status == HealthStatus.CRITICAL for status in all_statuses):
            return HealthStatus.CRITICAL
        elif any(status == HealthStatus.WARNING for status in all_statuses):
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY

    async def _handle_alerts(self, health_report: Dict[str, Any]):
        """Handle health alerts"""
        overall_status = health_report["overall_status"]

        # Check if we need to send alert
        if overall_status in [HealthStatus.CRITICAL, HealthStatus.DOWN]:
            # Check component failures
            for name, health in health_report["components"].items():
                if self.failure_counts[name] >= self.alert_threshold:
                    await self._send_alert(
                        component=name, status=health.status, message=health.message
                    )

            # Check resource alerts
            for resource, data in health_report["system_resources"].items():
                if data.get("status") in ["critical", "down"]:
                    await self._send_alert(
                        component=f"system_{resource}",
                        status=HealthStatus(data["status"]),
                        message=f"{resource.upper()} usage: {data.get('percent', 0):.1f}%",
                    )

    async def _send_alert(self, component: str, status: HealthStatus, message: str):
        """Send health alert"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "status": status.value,
            "message": message,
        }

        # Log alert
        logger.error(f"HEALTH ALERT: {component} - {status.value} - {message}")

        # Save alert to file
        alert_path = Path("alerts/health_alerts.jsonl")
        alert_path.parent.mkdir(exist_ok=True)

        with open(alert_path, "a") as f:
            f.write(json.dumps(alert) + "\n")

        # In production, this would send email/SMS/webhook

    def _save_health_report(self, report: Dict[str, Any]):
        """Save health report to file"""
        # Convert to serializable format
        serializable_report = {
            "timestamp": report["timestamp"].isoformat(),
            "overall_status": report["overall_status"].value,
            "components": {
                name: {
                    "status": health.status.value,
                    "message": health.message,
                    "metrics": health.metrics,
                }
                for name, health in report["components"].items()
            },
            "system_resources": report["system_resources"],
        }

        # Save to file
        report_path = Path(
            f'logs/health/health_report_{datetime.now().strftime("%Y%m%d")}.jsonl'
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "a") as f:
            f.write(json.dumps(serializable_report) + "\n")

    def get_health_summary(self) -> Dict[str, Any]:
        """Get current health summary"""
        if not self.health_history:
            return {"status": "No health data available"}

        latest_report = self.health_history[-1]

        return {
            "timestamp": latest_report["timestamp"],
            "overall_status": latest_report["overall_status"].value,
            "component_summary": {
                name: health.status.value
                for name, health in latest_report["components"].items()
            },
            "resource_summary": {
                resource: data.get("status", "unknown")
                for resource, data in latest_report["system_resources"].items()
            },
        }
