from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rl3.features.registry import registry


ALL_FEATURES = [
    "logret",
    "rsi",
    "macd",
    "atr",
    "vol_ewma",
    "vol_zscore",
    "range_pct",
    "trend_persist",
    "breakout_flags",
]


def _sample_frame(rows: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    base = np.cumsum(rng.normal(0, 0.01, size=rows)) + 100.0
    high = base + np.abs(rng.normal(0, 0.02, size=rows))
    low = base - np.abs(rng.normal(0, 0.02, size=rows))
    return pd.DataFrame(
        {
            "open": base,
            "high": np.maximum(high, base),
            "low": np.minimum(low, base),
            "close": base + rng.normal(0, 0.01, size=rows),
            "volume": rng.integers(10_000, 50_000, size=rows),
        }
    )


def test_registry_build_produces_all_features_without_nan():
    df = _sample_frame()
    feats = registry.build(df, ALL_FEATURES)

    assert list(feats.columns) == ALL_FEATURES
    assert feats.shape[0] == df.shape[0]
    assert not np.isnan(feats.to_numpy()).any()


def test_registry_missing_feature_raises():
    df = _sample_frame()
    with pytest.raises(KeyError):
        registry.build(df, ALL_FEATURES + ["unknown_feature"])
