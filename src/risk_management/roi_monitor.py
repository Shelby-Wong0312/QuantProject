"""
ROI Monitoring and Verification Module
ROI Calculation, Performance Tracking, Alert System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging


@dataclass
class ROITarget:
    """ROI target configuration"""

    annual_target: float = 10.0  # 1000% annual target
    monthly_target: float = 0.15  # 15% monthly target
    degradation_threshold: float = 5.0  # 500% degradation threshold
    consecutive_months_limit: int = 2  # 2 consecutive months below threshold


class ROIMonitor:
    """ROI monitoring and verification system"""

    def __init__(self, targets: Optional[ROITarget] = None):
        self.targets = targets or ROITarget()
        self.performance_history: List[Dict] = []
        self.alerts: List[Dict] = []
        self.logger = logging.getLogger(__name__)

    def calculate_roi(
        self,
        returns: Union[List[float], np.ndarray, pd.Series, float],
        costs: Union[List[float], np.ndarray, pd.Series, float] = 0.0,
    ) -> float:
        """
        Calculate Return on Investment (ROI)

        Args:
            returns: Return series or cumulative return
            costs: Trading costs (can be series or single value)

        Returns:
            ROI as percentage
        """
        # Handle single float value
        if isinstance(returns, (int, float)):
            cumulative_return = returns
            total_costs = costs if isinstance(costs, (int, float)) else 0.0
        else:
            if isinstance(returns, (list, np.ndarray)):
                returns = pd.Series(returns)

            if isinstance(costs, (list, np.ndarray)):
                costs = pd.Series(costs)
            elif isinstance(costs, (int, float)):
                costs = pd.Series([costs] * len(returns)) if len(returns) > 1 else costs

            # Calculate net returns after costs
            if len(returns) == 0:
                return 0.0

            # Cumulative return calculation
            if isinstance(returns, pd.Series) and len(returns) > 1:
                cumulative_return = (1 + returns).prod() - 1
            else:
                cumulative_return = (
                    returns.iloc[0] if isinstance(returns, pd.Series) else returns
                )

            # Subtract costs
            total_costs = costs.sum() if hasattr(costs, "sum") else costs

        net_return = cumulative_return - total_costs

        # ROI as percentage
        return net_return * 100

    def calculate_monthly_roi(
        self, returns: pd.Series, dates: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Calculate monthly ROI

        Args:
            returns: Daily returns
            dates: Date index (optional)

        Returns:
            Monthly ROI series
        """
        if dates is not None:
            returns.index = pd.to_datetime(dates)

        # Group by month and calculate monthly returns
        monthly_returns = returns.groupby(pd.Grouper(freq="M")).apply(
            lambda x: (1 + x).prod() - 1
        )

        return monthly_returns * 100  # Convert to percentage

    def check_targets(
        self, current_roi: float, period: str = "annual"
    ) -> Dict[str, bool]:
        """
        Check if ROI meets targets

        Args:
            current_roi: Current ROI percentage
            period: "annual" or "monthly"

        Returns:
            Dictionary of target checks
        """
        results = {}

        if period == "annual":
            results["meets_target"] = current_roi >= self.targets.annual_target
            results["above_degradation"] = (
                current_roi >= self.targets.degradation_threshold
            )
        else:  # monthly
            results["meets_target"] = current_roi >= self.targets.monthly_target
            results["above_degradation"] = current_roi >= (
                self.targets.degradation_threshold / 12
            )

        return results

    def track_performance(
        self,
        returns: Union[float, List[float]],
        date: Optional[datetime] = None,
        costs: float = 0.0,
    ) -> None:
        """
        Track performance metrics

        Args:
            returns: Period returns
            date: Performance date
            costs: Trading costs
        """
        if date is None:
            date = datetime.now()

        # Calculate ROI
        roi = self.calculate_roi(returns, costs)

        # Store performance record
        performance_record = {
            "date": date,
            "returns": returns,
            "costs": costs,
            "roi": roi,
            "meets_annual_target": roi >= self.targets.annual_target,
            "meets_monthly_target": roi >= self.targets.monthly_target,
            "above_degradation": roi >= self.targets.degradation_threshold,
        }

        self.performance_history.append(performance_record)

        # Check for alerts
        self._check_alerts(performance_record)

    def _check_alerts(self, performance_record: Dict) -> None:
        """Check for performance alerts"""

        # Check degradation threshold
        if not performance_record["above_degradation"]:
            self._add_alert(
                "degradation",
                performance_record["date"],
                f"ROI below degradation threshold: {performance_record['roi']:.2f}%",
            )

        # Check consecutive months below threshold
        if len(self.performance_history) >= self.targets.consecutive_months_limit:
            recent_records = self.performance_history[
                -self.targets.consecutive_months_limit :
            ]
            if all(not record["above_degradation"] for record in recent_records):
                self._add_alert(
                    "consecutive_degradation",
                    performance_record["date"],
                    f"ROI below threshold for {self.targets.consecutive_months_limit} consecutive periods",
                )

    def _add_alert(self, alert_type: str, date: datetime, message: str) -> None:
        """Add alert to alert list"""
        alert = {
            "type": alert_type,
            "date": date,
            "message": message,
            "acknowledged": False,
        }
        self.alerts.append(alert)
        self.logger.warning(f"ROI Alert: {message}")

    def get_performance_summary(self, period_days: int = 30) -> Dict:
        """
        Get performance summary for specified period

        Args:
            period_days: Period in days for summary

        Returns:
            Performance summary dictionary
        """
        if not self.performance_history:
            return {"error": "No performance data available"}

        cutoff_date = datetime.now() - timedelta(days=period_days)
        recent_performance = [
            record
            for record in self.performance_history
            if record["date"] >= cutoff_date
        ]

        if not recent_performance:
            return {"error": f"No performance data in last {period_days} days"}

        # Calculate metrics
        roi_values = [record["roi"] for record in recent_performance]

        summary = {
            "period_days": period_days,
            "total_records": len(recent_performance),
            "average_roi": np.mean(roi_values),
            "max_roi": np.max(roi_values),
            "min_roi": np.min(roi_values),
            "current_roi": roi_values[-1] if roi_values else 0,
            "meets_annual_target": np.mean(
                [r["meets_annual_target"] for r in recent_performance]
            ),
            "meets_monthly_target": np.mean(
                [r["meets_monthly_target"] for r in recent_performance]
            ),
            "above_degradation": np.mean(
                [r["above_degradation"] for r in recent_performance]
            ),
            "active_alerts": len([a for a in self.alerts if not a["acknowledged"]]),
        }

        return summary

    def get_alerts(self, unacknowledged_only: bool = True) -> List[Dict]:
        """Get alerts"""
        if unacknowledged_only:
            return [alert for alert in self.alerts if not alert["acknowledged"]]
        return self.alerts

    def acknowledge_alert(self, alert_index: int) -> bool:
        """Acknowledge an alert"""
        if 0 <= alert_index < len(self.alerts):
            self.alerts[alert_index]["acknowledged"] = True
            return True
        return False

    def calculate_sharpe_roi(
        self, returns: pd.Series, risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate risk-adjusted ROI (Sharpe-based)

        Args:
            returns: Return series
            risk_free_rate: Risk-free rate

        Returns:
            Risk-adjusted ROI
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252
        sharpe = excess_returns.mean() / returns.std() * np.sqrt(252)

        # Convert Sharpe to ROI-like metric
        return sharpe * returns.std() * 100

    def zero_cost_validation(
        self, total_returns: float, initial_capital: float = 0.0
    ) -> Dict[str, bool]:
        """
        Validate zero-cost strategy performance

        Args:
            total_returns: Total returns generated
            initial_capital: Initial capital (should be 0 for zero-cost)

        Returns:
            Validation results
        """
        return {
            "is_zero_cost": initial_capital == 0.0,
            "positive_returns": total_returns > 0,
            "meets_1000_percent": total_returns >= 10.0,  # 1000% = 10x
            "sustainable": total_returns >= self.targets.degradation_threshold,
        }
