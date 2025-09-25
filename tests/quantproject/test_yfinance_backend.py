from __future__ import annotations

from unittest.mock import patch
import pandas as pd
import numpy as np

from src.quantproject.data_pipeline.backends.yfinance import YFinanceBackend


_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _mock_download(*args, **kwargs):
    n = 120
    idx = pd.date_range("2025-08-01", periods=n, freq="5min")
    data = {
        "Open": np.linspace(100, 101, n),
        "High": np.linspace(100.2, 101.2, n),
        "Low": np.linspace(99.8, 100.8, n),
        "Close": np.linspace(100, 101, n),
        "Volume": np.linspace(10_000, 20_000, n),
    }
    return pd.DataFrame(data, index=idx)


@patch("src.quantproject.data_pipeline.backends.yfinance.yf.download")
def test_get_bars_ok(mock_download, tmp_path):
    mock_download.return_value = _mock_download()

    backend = YFinanceBackend(cache_dir=tmp_path)
    df = backend.get_bars("SPY", "2025-08-01", "2025-09-01", "5min")

    assert not df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.tz is not None
    for col in df.columns:
        assert np.issubdtype(df[col].dtype, np.floating)

    cache_file = backend._cache_path("SPY", "5min")
    assert cache_file.exists()

    _, kwargs = mock_download.call_args
    assert kwargs["interval"] == "5m"
    assert kwargs["auto_adjust"] is False
    assert kwargs["progress"] is False
    assert kwargs["threads"] is False


@patch("src.quantproject.data_pipeline.backends.yfinance.yf.download")
def test_get_bars_exception_returns_empty(mock_download, tmp_path):
    mock_download.side_effect = RuntimeError("network error")

    backend = YFinanceBackend(cache_dir=tmp_path)
    df = backend.get_bars("SPY", "2025-08-01", "2025-09-01", "5min")

    assert df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    cache_file = backend._cache_path("SPY", "5min")
    assert not cache_file.exists()

    cache_file = backend._cache_path("SPY", "5min")
    assert not cache_file.exists()


@patch("src.quantproject.data_pipeline.backends.yfinance.yf.download")
def test_get_bars_truncates_large_5m_window(mock_download, tmp_path):
    mock_download.return_value = _mock_download()

    backend = YFinanceBackend(cache_dir=tmp_path)
    backend.get_bars("SPY", "2024-01-01", "2025-09-01", "5min")

    cache_file = backend._cache_path("SPY", "5min")
    assert cache_file.exists()

    _, kwargs = mock_download.call_args
    start_ts = pd.to_datetime(kwargs["start"], utc=True)
    end_ts = pd.to_datetime(kwargs["end"], utc=True)
    assert end_ts - start_ts == pd.Timedelta(days=55)
    assert kwargs["interval"] == "5m"


@patch("src.quantproject.data_pipeline.backends.yfinance.yf.download")
def test_get_bars_drops_rows_with_nan(mock_download, tmp_path):
    df_source = _mock_download()
    df_source.iloc[0, 0] = np.nan
    mock_download.return_value = df_source

    backend = YFinanceBackend(cache_dir=tmp_path)
    df = backend.get_bars("SPY", "2025-08-01", "2025-09-01", "5min")

    assert not df.empty
    assert df.shape[0] == df_source.shape[0] - 1

    cache_file = backend._cache_path("SPY", "5min")
    assert cache_file.exists()


@patch("src.quantproject.data_pipeline.backends.yfinance.yf.download")
def test_get_bars_missing_column_returns_empty(mock_download, tmp_path):
    df_source = _mock_download().drop(columns=["High"])
    mock_download.return_value = df_source

    backend = YFinanceBackend(cache_dir=tmp_path)
    df = backend.get_bars("SPY", "2025-08-01", "2025-09-01", "5min")

    assert df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
