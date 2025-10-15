"""
Market Anomaly Detection System
市場異常檢測系統
Cloud Quant - Task Q-603
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Anomaly types"""

    PRICE_SPIKE = "price_spike"
    VOLUME_SURGE = "volume_surge"
    VOLATILITY_JUMP = "volatility_jump"
    CORRELATION_BREAK = "correlation_break"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    FLASH_CRASH = "flash_crash"


class SeverityLevel(Enum):
    """Anomaly severity levels"""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EXTREME = 5


@dataclass
class Anomaly:
    """Anomaly data structure"""

    timestamp: datetime
    symbol: str
    anomaly_type: AnomalyType
    severity: SeverityLevel
    score: float
    description: str
    metrics: Dict
    action_required: bool


class MarketAnomalyDetector:
    """
    Advanced market anomaly detection system
    Uses multiple ML algorithms for robust detection
    """

    def __init__(
        self, contamination: float = 0.01, sensitivity: float = 0.5, lookback_period: int = 100
    ):
        """
        Initialize anomaly detector

        Args:
            contamination: Expected proportion of outliers
            sensitivity: Detection sensitivity (0-1)
            lookback_period: Historical data lookback period
        """
        self.contamination = contamination
        self.sensitivity = sensitivity
        self.lookback_period = lookback_period

        # Initialize detectors
        self.isolation_forest = IsolationForest(
            contamination=contamination, n_estimators=100, max_samples="auto", random_state=42
        )

        self.dbscan = DBSCAN(eps=0.3, min_samples=5)

        self.scaler = StandardScaler()

        # Detection history
        self.anomaly_history: List[Anomaly] = []
        self.detection_cache = {}

        # Thresholds
        self.thresholds = {
            "price_change": 0.05,  # 5% price change
            "volume_ratio": 3.0,  # 3x average volume
            "volatility_ratio": 2.0,  # 2x normal volatility
            "correlation_break": 0.3,  # 30% correlation deviation
            "liquidity_ratio": 0.5,  # 50% liquidity drop
        }

        logger.info("Market Anomaly Detector initialized")

    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract features for anomaly detection

        Args:
            data: Market data DataFrame

        Returns:
            Feature array
        """
        features = []

        # Price features
        returns = data["close"].pct_change()
        features.append(returns.iloc[-1])  # Current return
        features.append(returns.std())  # Volatility
        features.append(returns.skew())  # Skewness
        features.append(returns.kurt())  # Kurtosis

        # Volume features
        volume_ma = data["volume"].rolling(20).mean()
        volume_ratio = data["volume"].iloc[-1] / volume_ma.iloc[-1]
        features.append(volume_ratio)

        # Technical indicators
        sma_20 = data["close"].rolling(20).mean()
        sma_50 = data["close"].rolling(50).mean()
        price_to_sma20 = data["close"].iloc[-1] / sma_20.iloc[-1]
        features.append(price_to_sma20)

        # Volatility features
        high_low_ratio = (data["high"] / data["low"]).mean()
        features.append(high_low_ratio)

        # Market microstructure
        spread = (data["high"] - data["low"]) / data["close"]
        features.append(spread.iloc[-1])

        return np.array(features).reshape(1, -1)

    def detect_anomalies(self, market_data: Dict[str, pd.DataFrame]) -> List[Anomaly]:
        """
        Detect anomalies in market data

        Args:
            market_data: Dictionary of symbol -> DataFrame

        Returns:
            List of detected anomalies
        """
        anomalies = []

        for symbol, data in market_data.items():
            if len(data) < self.lookback_period:
                continue

            # Extract features
            features = self.extract_features(data)

            # Scale features
            features_scaled = self.scaler.fit_transform(features)

            # Fit the model if not fitted yet (for testing)
            try:
                iso_score = self.isolation_forest.decision_function(features_scaled)[0]
                iso_anomaly = self.isolation_forest.predict(features_scaled)[0] == -1
            except Exception:
                # Fit the model with the current data if not fitted
                self.isolation_forest.fit(features_scaled)
                iso_score = self.isolation_forest.decision_function(features_scaled)[0]
                iso_anomaly = self.isolation_forest.predict(features_scaled)[0] == -1

            # Rule-based detection
            rule_anomalies = self._detect_rule_based(data)

            # Combine detections
            if iso_anomaly or rule_anomalies:
                severity = self._calculate_severity(iso_score, rule_anomalies)
                anomaly_type = self._determine_anomaly_type(data, rule_anomalies)

                anomaly = Anomaly(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    score=abs(iso_score),
                    description=self._generate_description(anomaly_type, data),
                    metrics=self._collect_metrics(data),
                    action_required=severity.value >= SeverityLevel.HIGH.value,
                )

                anomalies.append(anomaly)
                self.anomaly_history.append(anomaly)

                logger.warning(
                    f"Anomaly detected: {symbol} - {anomaly_type.value} - Severity: {severity.value}"
                )

        return anomalies

    def _detect_rule_based(self, data: pd.DataFrame) -> Dict[str, bool]:
        """
        Rule-based anomaly detection

        Args:
            data: Market data

        Returns:
            Dictionary of anomaly flags
        """
        anomalies = {}

        # Price spike detection
        returns = data["close"].pct_change()
        price_spike = abs(returns.iloc[-1]) > self.thresholds["price_change"]
        anomalies["price_spike"] = price_spike

        # Volume surge detection
        volume_ma = data["volume"].rolling(20).mean()
        volume_ratio = data["volume"].iloc[-1] / volume_ma.iloc[-1]
        anomalies["volume_surge"] = volume_ratio > self.thresholds["volume_ratio"]

        # Volatility jump detection
        volatility = returns.rolling(20).std()
        volatility_ratio = volatility.iloc[-1] / volatility.mean()
        anomalies["volatility_jump"] = volatility_ratio > self.thresholds["volatility_ratio"]

        # Flash crash detection
        intraday_drop = (data["low"].iloc[-1] - data["open"].iloc[-1]) / data["open"].iloc[-1]
        anomalies["flash_crash"] = intraday_drop < -0.03  # 3% intraday drop

        return anomalies

    def _calculate_severity(
        self, iso_score: float, rule_anomalies: Dict[str, bool]
    ) -> SeverityLevel:
        """
        Calculate anomaly severity

        Args:
            iso_score: Isolation Forest anomaly score
            rule_anomalies: Rule-based detection results

        Returns:
            Severity level
        """
        severity_score = 0

        # Isolation Forest contribution
        if abs(iso_score) > 0.5:
            severity_score += 3
        elif abs(iso_score) > 0.3:
            severity_score += 2
        elif abs(iso_score) > 0.1:
            severity_score += 1

        # Rule-based contribution
        rule_count = sum(rule_anomalies.values())
        severity_score += rule_count

        # Map to severity level
        if severity_score >= 5:
            return SeverityLevel.EXTREME
        elif severity_score >= 4:
            return SeverityLevel.CRITICAL
        elif severity_score >= 3:
            return SeverityLevel.HIGH
        elif severity_score >= 2:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW

    def _determine_anomaly_type(
        self, data: pd.DataFrame, rule_anomalies: Dict[str, bool]
    ) -> AnomalyType:
        """
        Determine the primary anomaly type

        Args:
            data: Market data
            rule_anomalies: Rule-based detection results

        Returns:
            Anomaly type
        """
        if rule_anomalies.get("flash_crash", False):
            return AnomalyType.FLASH_CRASH
        elif rule_anomalies.get("volume_surge", False):
            return AnomalyType.VOLUME_SURGE
        elif rule_anomalies.get("volatility_jump", False):
            return AnomalyType.VOLATILITY_JUMP
        elif rule_anomalies.get("price_spike", False):
            return AnomalyType.PRICE_SPIKE
        else:
            return AnomalyType.CORRELATION_BREAK

    def _generate_description(self, anomaly_type: AnomalyType, data: pd.DataFrame) -> str:
        """
        Generate anomaly description

        Args:
            anomaly_type: Type of anomaly
            data: Market data

        Returns:
            Description string
        """
        returns = data["close"].pct_change()
        current_return = returns.iloc[-1] * 100

        descriptions = {
            AnomalyType.PRICE_SPIKE: f"Abnormal price movement of {current_return:.2f}%",
            AnomalyType.VOLUME_SURGE: f"Trading volume {data['volume'].iloc[-1]/1e6:.1f}M, significantly above average",
            AnomalyType.VOLATILITY_JUMP: f"Volatility increased to {returns.std()*100:.2f}%",
            AnomalyType.FLASH_CRASH: f"Rapid price decline detected, down {current_return:.2f}%",
            AnomalyType.CORRELATION_BREAK: "Market correlation structure broken",
            AnomalyType.LIQUIDITY_CRISIS: "Severe liquidity shortage detected",
        }

        return descriptions.get(anomaly_type, "Unknown anomaly detected")

    def _collect_metrics(self, data: pd.DataFrame) -> Dict:
        """
        Collect anomaly metrics

        Args:
            data: Market data

        Returns:
            Metrics dictionary
        """
        returns = data["close"].pct_change()

        return {
            "price": data["close"].iloc[-1],
            "return": returns.iloc[-1],
            "volume": data["volume"].iloc[-1],
            "volatility": returns.std(),
            "high": data["high"].iloc[-1],
            "low": data["low"].iloc[-1],
            "spread": (data["high"].iloc[-1] - data["low"].iloc[-1]) / data["close"].iloc[-1],
        }

    def train_model(self, historical_data: pd.DataFrame):
        """
        Train the anomaly detection model

        Args:
            historical_data: Historical market data for training
        """
        # Extract features for all historical data
        features_list = []

        for i in range(self.lookback_period, len(historical_data)):
            window_data = historical_data.iloc[i - self.lookback_period : i]
            features = self.extract_features(window_data)
            features_list.append(features[0])

        if features_list:
            features_array = np.array(features_list)

            # Scale features
            features_scaled = self.scaler.fit_transform(features_array)

            # Train Isolation Forest
            self.isolation_forest.fit(features_scaled)

            logger.info(f"Model trained on {len(features_list)} samples")

    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """
        Update detection thresholds

        Args:
            new_thresholds: New threshold values
        """
        self.thresholds.update(new_thresholds)
        logger.info(f"Thresholds updated: {self.thresholds}")

    def get_anomaly_report(self) -> Dict:
        """
        Generate anomaly detection report

        Returns:
            Report dictionary
        """
        if not self.anomaly_history:
            return {"total_anomalies": 0, "by_type": {}, "by_severity": {}, "recent_anomalies": []}

        # Count by type
        by_type = {}
        for anomaly in self.anomaly_history:
            anomaly_type = anomaly.anomaly_type.value
            by_type[anomaly_type] = by_type.get(anomaly_type, 0) + 1

        # Count by severity
        by_severity = {}
        for anomaly in self.anomaly_history:
            severity = anomaly.severity.name
            by_severity[severity] = by_severity.get(severity, 0) + 1

        # Recent anomalies
        recent = (
            self.anomaly_history[-10:] if len(self.anomaly_history) > 10 else self.anomaly_history
        )

        return {
            "total_anomalies": len(self.anomaly_history),
            "by_type": by_type,
            "by_severity": by_severity,
            "recent_anomalies": [
                {
                    "timestamp": a.timestamp.isoformat(),
                    "symbol": a.symbol,
                    "type": a.anomaly_type.value,
                    "severity": a.severity.name,
                    "score": a.score,
                    "description": a.description,
                }
                for a in recent
            ],
            "detection_rate": len(self.anomaly_history) / max(1, len(self.detection_cache)),
        }


if __name__ == "__main__":
    # Test anomaly detector
    detector = MarketAnomalyDetector(contamination=0.01)

    # Generate test data
    dates = pd.date_range(end=datetime.now(), periods=200, freq="D")
    test_data = pd.DataFrame(
        {
            "date": dates,
            "open": np.random.uniform(100, 110, 200),
            "high": np.random.uniform(110, 120, 200),
            "low": np.random.uniform(90, 100, 200),
            "close": np.random.uniform(95, 115, 200),
            "volume": np.random.uniform(1000000, 5000000, 200),
        }
    )
    test_data.set_index("date", inplace=True)

    # Inject anomaly
    test_data.loc[test_data.index[-1], "close"] *= 0.92  # 8% drop
    test_data.loc[test_data.index[-1], "volume"] *= 5  # 5x volume

    # Test detection
    market_data = {"TEST": test_data}
    anomalies = detector.detect_anomalies(market_data)

    print("Anomaly Detection Test Results:")
    print(f"Detected {len(anomalies)} anomalies")

    for anomaly in anomalies:
        print(f"\n{anomaly.symbol}: {anomaly.anomaly_type.value}")
        print(f"Severity: {anomaly.severity.name}")
        print(f"Score: {anomaly.score:.3f}")
        print(f"Description: {anomaly.description}")

    # Generate report
    detector.get_anomaly_report()
    print("\nDetection Report:")
    print(f"Total anomalies: {report['total_anomalies']}")
    print(f"By type: {report['by_type']}")
    print(f"By severity: {report['by_severity']}")
