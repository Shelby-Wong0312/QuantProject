"""
Data Validation and Quality Checks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validate and check quality of financial data
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data validator

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.validation_results = {}

    def validate_ohlcv(
        self,
        df: pd.DataFrame,
        check_gaps: bool = True,
        check_outliers: bool = True,
        check_volumes: bool = True,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Comprehensive OHLCV data validation

        Args:
            df: DataFrame with OHLCV data
            check_gaps: Whether to check for time gaps
            check_outliers: Whether to check for price outliers
            check_volumes: Whether to check for volume anomalies

        Returns:
            Tuple of (is_valid, validation_report)
        """
        {
            "is_valid": True,
            "total_rows": len(df),
            "issues": [],
            "warnings": [],
            "statistics": {},
        }

        # Basic structure validation
        structure_valid, structure_report = self._validate_structure(df)
        if not structure_valid:
            report["is_valid"] = False
            report["issues"].extend(structure_report["issues"])

        # OHLC relationship validation
        ohlc_valid, ohlc_report = self._validate_ohlc_relationships(df)
        if not ohlc_valid:
            report["is_valid"] = False
            report["issues"].extend(ohlc_report["issues"])
        report["statistics"]["invalid_ohlc_relationships"] = ohlc_report.get("invalid_count", 0)

        # Missing data validation
        missing_valid, missing_report = self._validate_missing_data(df)
        if not missing_valid:
            report["is_valid"] = False
            report["issues"].extend(missing_report["issues"])
        report["statistics"]["missing_data"] = missing_report.get("missing_stats", {})

        # Time gap validation
        if check_gaps and isinstance(df.index, pd.DatetimeIndex):
            gaps_valid, gaps_report = self._validate_time_gaps(df)
            if not gaps_valid:
                report["warnings"].extend(gaps_report["warnings"])
            report["statistics"]["time_gaps"] = gaps_report.get("gaps", [])

        # Outlier validation
        if check_outliers:
            outliers_valid, outliers_report = self._validate_outliers(df)
            if not outliers_valid:
                report["warnings"].extend(outliers_report["warnings"])
            report["statistics"]["outliers"] = outliers_report.get("outlier_stats", {})

        # Volume validation
        if check_volumes and "volume" in df.columns:
            volume_valid, volume_report = self._validate_volume(df)
            if not volume_valid:
                report["warnings"].extend(volume_report["warnings"])
            report["statistics"]["volume_issues"] = volume_report.get("volume_stats", {})

        # Calculate data quality score
        report["quality_score"] = self._calculate_quality_score(report)

        self.validation_results = report
        return report["is_valid"], report

    def _validate_structure(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Validate basic DataFrame structure"""
        {"is_valid": True, "issues": []}

        # Check if DataFrame is empty
        if df.empty:
            report["is_valid"] = False
            report["issues"].append("DataFrame is empty")
            return report["is_valid"], report

        # Check for datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            report["issues"].append("Index is not DatetimeIndex")

        # Check for required columns
        required_columns = ["open", "high", "low", "close"]
        df_columns_lower = [col.lower() for col in df.columns]

        missing_columns = []
        for col in required_columns:
            if col not in df_columns_lower:
                missing_columns.append(col)

        if missing_columns:
            report["is_valid"] = False
            report["issues"].append(f"Missing required columns: {missing_columns}")

        # Check data types
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns and col not in numeric_columns:
                report["issues"].append(f"Column '{col}' is not numeric")

        return report["is_valid"], report

    def _validate_ohlc_relationships(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Validate OHLC price relationships"""
        {"is_valid": True, "issues": [], "invalid_count": 0}

        # Ensure columns exist
        required_cols = ["open", "high", "low", "close"]
        df_lower = df.copy()
        df_lower.columns = df_lower.columns.str.lower()

        if not all(col in df_lower.columns for col in required_cols):
            return True, report  # Skip if columns missing

        # Check OHLC relationships
        invalid_mask = (
            (df_lower["high"] < df_lower["low"])
            | (df_lower["high"] < df_lower["open"])
            | (df_lower["high"] < df_lower["close"])
            | (df_lower["low"] > df_lower["open"])
            | (df_lower["low"] > df_lower["close"])
        )

        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            report["is_valid"] = False
            report["invalid_count"] = invalid_count
            report["issues"].append(f"Found {invalid_count} rows with invalid OHLC relationships")

            # Get sample of invalid rows
            invalid_indices = df_lower[invalid_mask].index[:5].tolist()
            report["sample_invalid_indices"] = invalid_indices

        return report["is_valid"], report

    def _validate_missing_data(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Validate missing data"""
        {"is_valid": True, "issues": [], "missing_stats": {}}

        # Calculate missing data statistics
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df)) * 100

        # Store statistics
        report["missing_stats"] = {
            "counts": missing_counts.to_dict(),
            "percentages": missing_pct.to_dict(),
        }

        # Check for critical missing data
        critical_columns = ["open", "high", "low", "close"]
        for col in critical_columns:
            if col in df.columns:
                col_missing_pct = missing_pct.get(col, 0)
                if col_missing_pct > 5:  # More than 5% missing
                    report["is_valid"] = False
                    report["issues"].append(
                        f"Column '{col}' has {col_missing_pct:.2f}% missing data"
                    )

        return report["is_valid"], report

    def _validate_time_gaps(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Validate time gaps in data"""
        {"is_valid": True, "warnings": [], "gaps": []}

        if not isinstance(df.index, pd.DatetimeIndex):
            return True, report

        # Calculate time differences
        time_diffs = df.index.to_series().diff()

        # Find the mode (most common) time difference
        mode_diff = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else pd.Timedelta(hours=1)

        # Find gaps (more than 2x the mode difference)
        gap_threshold = mode_diff * 2
        gaps = time_diffs[time_diffs > gap_threshold]

        if len(gaps) > 0:
            report["warnings"].append(f"Found {len(gaps)} time gaps in data")

            # Store gap details
            gap_details = []
            for idx, gap in gaps.items():
                gap_details.append(
                    {"start": df.index[df.index.get_loc(idx) - 1], "end": idx, "duration": str(gap)}
                )

            report["gaps"] = gap_details[:10]  # Store first 10 gaps

        return report["is_valid"], report

    def _validate_outliers(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Validate outliers in price data"""
        {"is_valid": True, "warnings": [], "outlier_stats": {}}

        price_columns = ["open", "high", "low", "close"]
        outlier_counts = {}

        for col in price_columns:
            if col not in df.columns:
                continue

            # Calculate returns for outlier detection
            returns = df[col].pct_change()

            # Use IQR method for outlier detection
            Q1 = returns.quantile(0.25)
            Q3 = returns.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 3 * IQR  # Using 3x IQR for extreme outliers
            upper_bound = Q3 + 3 * IQR

            outliers = returns[(returns < lower_bound) | (returns > upper_bound)]
            outlier_counts[col] = len(outliers)

            if len(outliers) > 0:
                report["warnings"].append(f"Found {len(outliers)} outliers in {col} returns")

        report["outlier_stats"] = outlier_counts

        return report["is_valid"], report

    def _validate_volume(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Validate volume data"""
        {"is_valid": True, "warnings": [], "volume_stats": {}}

        if "volume" not in df.columns:
            return True, report

        volume = df["volume"]

        # Check for negative volumes
        negative_volumes = (volume < 0).sum()
        if negative_volumes > 0:
            report["warnings"].append(f"Found {negative_volumes} negative volume values")
            report["volume_stats"]["negative_count"] = negative_volumes

        # Check for zero volumes
        zero_volumes = (volume == 0).sum()
        zero_volume_pct = (zero_volumes / len(df)) * 100

        if zero_volume_pct > 20:  # More than 20% zero volume
            report["warnings"].append(f"{zero_volume_pct:.2f}% of data has zero volume")

        report["volume_stats"]["zero_count"] = zero_volumes
        report["volume_stats"]["zero_percentage"] = zero_volume_pct

        # Check for volume spikes
        volume_returns = volume.pct_change()
        volume_std = volume_returns.std()
        volume_spikes = volume_returns[volume_returns.abs() > 10 * volume_std]

        if len(volume_spikes) > 0:
            report["warnings"].append(f"Found {len(volume_spikes)} extreme volume spikes")
            report["volume_stats"]["spike_count"] = len(volume_spikes)

        return report["is_valid"], report

    def _calculate_quality_score(self, report: Dict[str, Any]) -> float:
        """
        Calculate overall data quality score

        Args:
            report: Validation report

        Returns:
            Quality score (0-100)
        """
        score = 100.0

        # Deduct for critical issues
        if not report["is_valid"]:
            score -= 50.0

        # Deduct for issues
        score -= len(report["issues"]) * 10

        # Deduct for warnings
        score -= len(report["warnings"]) * 5

        # Deduct for missing data
        if "missing_data" in report["statistics"]:
            missing_stats = report["statistics"]["missing_data"]
            if "percentages" in missing_stats:
                avg_missing = np.mean(list(missing_stats["percentages"].values()))
                score -= avg_missing * 2

        # Deduct for outliers
        if "outliers" in report["statistics"]:
            outlier_stats = report["statistics"]["outliers"]
            total_outliers = sum(outlier_stats.values())
            outlier_pct = (total_outliers / report["total_rows"]) * 100
            score -= outlier_pct * 5

        # Ensure score is between 0 and 100
        return max(0, min(100, score))

    def generate_validation_report(self) -> str:
        """
        Generate a human-readable validation report

        Returns:
            Formatted validation report string
        """
        if not self.validation_results:
            return "No validation results available. Run validate_ohlcv() first."

        self.validation_results

        output = []
        output.append("=" * 50)
        output.append("DATA VALIDATION REPORT")
        output.append("=" * 50)
        output.append(f"Total Rows: {report['total_rows']}")
        output.append(f"Valid: {'Yes' if report['is_valid'] else 'No'}")
        output.append(f"Quality Score: {report['quality_score']:.2f}/100")
        output.append("")

        if report["issues"]:
            output.append("CRITICAL ISSUES:")
            for issue in report["issues"]:
                output.append(f"  - {issue}")
            output.append("")

        if report["warnings"]:
            output.append("WARNINGS:")
            for warning in report["warnings"]:
                output.append(f"  - {warning}")
            output.append("")

        if report["statistics"]:
            output.append("STATISTICS:")
            stats = report["statistics"]

            if "missing_data" in stats and "percentages" in stats["missing_data"]:
                output.append("  Missing Data:")
                for col, pct in stats["missing_data"]["percentages"].items():
                    if pct > 0:
                        output.append(f"    - {col}: {pct:.2f}%")

            if "outliers" in stats:
                output.append("  Outliers:")
                for col, count in stats["outliers"].items():
                    if count > 0:
                        output.append(f"    - {col}: {count} outliers")

            if "time_gaps" in stats and stats["time_gaps"]:
                output.append(f"  Time Gaps: {len(stats['time_gaps'])} found")

        output.append("=" * 50)

        return "\n".join(output)
