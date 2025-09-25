from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rl3.features.registry import registry


def _fake_bars(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2025-08-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.linspace(100, 102, n),
            "high": np.linspace(100.5, 102.5, n),
            "low": np.linspace(99.5, 101.5, n),
            "close": np.linspace(100, 102, n) + np.random.randn(n) * 0.05,
            "volume": np.random.randint(1_000, 2_000, n),
        },
        index=idx,
    )
    return df


def test_feature_shapes_and_finiteness() -> None:
    df = _fake_bars()
    feats = [
        "logret",
        "rsi",
        "macd",
        "atr",
        "vol_ewma",
        "range_pct",
        "vol_zscore",
        "trend_persist",
        "breakout_flags",
    ]
    out = registry.build(df, feats)
    assert list(out.columns) == feats
    assert out.shape[0] == df.shape[0]
    assert np.isfinite(out.to_numpy()).all()
