from __future__ import annotations

from unittest.mock import patch
import pandas as pd
import numpy as np

from src.quantproject.data_pipeline.loaders.bars import load_and_align


def _fake_df(n: int, start: str = "2025-08-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": np.linspace(100, 101, n),
            "high": np.linspace(100.2, 101.2, n),
            "low": np.linspace(99.8, 100.8, n),
            "close": np.linspace(100, 101, n),
            "volume": np.random.randint(100, 200, n),
        },
        index=idx,
    )


@patch("src.quantproject.data_pipeline.loaders.bars.DataRouter")
def test_align_intersection(MockRouter):
    router = MockRouter.return_value
    router.get_bars.side_effect = [_fake_df(110), _fake_df(120)]

    load_and_align(["A", "B"], "2025-08-01", "2025-09-01", "5min")

    assert router.get_bars.call_count == 2
    assert set(data.keys()) == {"A", "B"}
    lens = {symbol: df.shape[0] for symbol, df in data.items()}
    L = next(iter(lens.values()))
    assert all(length == L for length in lens.values())

    ref_index = next(iter(data.values())).index
    for df in data.values():
        assert df.index.equals(ref_index)


@patch("src.quantproject.data_pipeline.loaders.bars.DataRouter")
def test_skips_empty_symbols(MockRouter):
    router = MockRouter.return_value
    router.get_bars.side_effect = [
        _fake_df(50),
        pd.DataFrame(columns=["open", "high", "low", "close", "volume"]),
    ]

    load_and_align(["A", "B"], "2025-08-01", "2025-09-01", "5min")

    assert set(data.keys()) == {"A"}
    assert data["A"].shape[0] == 50


@patch("src.quantproject.data_pipeline.loaders.bars.DataRouter")
def test_loader_drops_na_after_alignment(MockRouter):
    router = MockRouter.return_value
    a = _fake_df(10)
    b = _fake_df(12)
    b.iloc[-1, 0] = np.nan
    router.get_bars.side_effect = [a, b]

    load_and_align(["A", "B"], "2025-08-01", "2025-08-02", "5min")

    assert set(data.keys()) == {"A", "B"}
    for df in data.values():
        assert df.notna().all().all()
