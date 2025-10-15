from __future__ import annotations

from unittest.mock import patch
import pandas as pd
import numpy as np

from src.quantproject.data_pipeline.backends.yfinance import YFinanceBackend


_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _mock_download(*args, **kwargs):
    n = 120
    idx = pd.date_range("2025-08-01", periods=n, freq="5min")
    {
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
    assert "start" in kwargs and "end" in kwargs
    assert "period" not in kwargs


@patch(
    "src.quantproject.data_pipeline.backends.yfinance.YFinanceBackend._download_via_api",
    return_value=None,
)
@patch("src.quantproject.data_pipeline.backends.yfinance.time.sleep")
@patch("src.quantproject.data_pipeline.backends.yfinance.yf.download")
def test_get_bars_exception_returns_empty(mock_download, sleep_mock, mock_api, tmp_path):
    mock_download.side_effect = [RuntimeError("network error")] * 6

    backend = YFinanceBackend(cache_dir=tmp_path)
    df = backend.get_bars("SPY", "2025-08-01", "2025-09-01", "5min")

    assert df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert mock_api.call_count == 1

    cache_file = backend._cache_path("SPY", "5min")
    assert not cache_file.exists()

    assert mock_download.call_count == 6
    first_kwargs = mock_download.call_args_list[0][1]
    last_kwargs = mock_download.call_args_list[-1][1]
    assert "start" in first_kwargs and "end" in first_kwargs
    assert last_kwargs.get("period") == "60d"

    sleep_calls = [call.args[0] for call in sleep_mock.call_args_list]
    assert len(sleep_calls) == 4
    for got, base in zip(sleep_calls, (2.0, 5.0, 2.0, 5.0)):
        assert base <= got < base + 1.0


@patch(
    "src.quantproject.data_pipeline.backends.yfinance.YFinanceBackend._download_via_api",
    return_value=None,
)
@patch("src.quantproject.data_pipeline.backends.yfinance.time.sleep")
@patch("src.quantproject.data_pipeline.backends.yfinance.yf.download")
def test_get_bars_retry_then_success(mock_download, sleep_mock, mock_api, tmp_path):
    mock_download.side_effect = [RuntimeError("network error"), _mock_download()]

    backend = YFinanceBackend(cache_dir=tmp_path)
    df = backend.get_bars("SPY", "2025-08-01", "2025-09-01", "5min")

    assert not df.empty
    assert mock_download.call_count == 2
    assert mock_api.call_count == 0
    kwargs_first = mock_download.call_args_list[0][1]
    kwargs_second = mock_download.call_args_list[1][1]
    assert "start" in kwargs_first and "end" in kwargs_first
    assert "period" not in kwargs_first
    assert "start" in kwargs_second and "end" in kwargs_second

    sleep_calls = [call.args[0] for call in sleep_mock.call_args_list]
    assert len(sleep_calls) == 1
    assert 2.0 <= sleep_calls[0] < 3.0


@patch(
    "src.quantproject.data_pipeline.backends.yfinance.YFinanceBackend._download_via_api",
    return_value=None,
)
@patch("src.quantproject.data_pipeline.backends.yfinance.time.sleep")
@patch("src.quantproject.data_pipeline.backends.yfinance.yf.download")
def test_get_bars_period_fallback(mock_download, sleep_mock, mock_api, tmp_path):
    empty_df = pd.DataFrame(columns=_COLS)
    mock_download.side_effect = [empty_df, empty_df, empty_df, _mock_download()]

    backend = YFinanceBackend(cache_dir=tmp_path)
    df = backend.get_bars("SPY", "2025-08-01", "2025-09-01", "5min")

    assert not df.empty
    assert mock_download.call_count == 4
    assert mock_api.call_count == 1
    first_kwargs = mock_download.call_args_list[0][1]
    last_kwargs = mock_download.call_args_list[-1][1]
    assert "start" in first_kwargs and "end" in first_kwargs
    assert last_kwargs.get("period") == "60d"

    sleep_calls = [call.args[0] for call in sleep_mock.call_args_list]
    assert len(sleep_calls) == 2
    for got, base in zip(sleep_calls, (2.0, 5.0)):
        assert base <= got < base + 1.0


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


@patch("src.quantproject.data_pipeline.backends.yfinance.YFinanceBackend._download_via_api")
@patch("src.quantproject.data_pipeline.backends.yfinance.yf.download", return_value=pd.DataFrame())
def test_get_bars_api_fallback_success(mock_download, mock_api, tmp_path):
    api_df = _mock_download()
    mock_api.return_value = api_df

    backend = YFinanceBackend(cache_dir=tmp_path)
    df = backend.get_bars("SPY", "2025-08-01", "2025-09-01", "5min")

    assert not df.empty
    assert mock_download.call_count == 3
    mock_api.assert_called_once()
