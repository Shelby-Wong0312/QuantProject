from __future__ import annotations

from unittest.mock import patch
import pandas as pd

from src.quantproject.data_pipeline.backends.router import DataRouter


_COLS = ["open", "high", "low", "close", "volume"]


def _df(rows=5, start="2025-08-01"):
    idx = pd.date_range(start, periods=rows, freq="5min", tz="UTC")
    data = {
        "open": range(rows),
        "high": range(1, rows + 1),
        "low": range(rows),
        "close": range(1, rows + 1),
        "volume": [10.0] * rows,
    }
    return pd.DataFrame(data, index=idx)


@patch("src.quantproject.data_pipeline.backends.router.YFinanceBackend")
@patch("src.quantproject.data_pipeline.backends.router.BinanceBackend")
@patch("src.quantproject.data_pipeline.backends.router.AlphaVantageBackend")
def test_router_uses_yahoo_first(mock_alpha, mock_binance, mock_yf):
    yf = mock_yf.return_value
    yf.get_bars.return_value = _df()
    router = DataRouter()
    df = router.get_bars("AAPL", "2025-08-01", "2025-09-01", "5min")

    assert not df.empty
    yf.get_bars.assert_called_once()
    mock_alpha.return_value.get_bars.assert_not_called()
    mock_binance.return_value.get_bars.assert_not_called()


@patch("src.quantproject.data_pipeline.backends.router.YFinanceBackend")
@patch("src.quantproject.data_pipeline.backends.router.BinanceBackend")
@patch("src.quantproject.data_pipeline.backends.router.AlphaVantageBackend")
def test_router_crypto_fallbacks(mock_alpha, mock_binance, mock_yf):
    yf = mock_yf.return_value
    yf.get_bars.return_value = pd.DataFrame(columns=_COLS)
    mock_binance.return_value.get_bars.return_value = _df()

    router = DataRouter()
    df = router.get_bars("BTC-USD", "2025-08-01", "2025-09-01", "5min")

    assert not df.empty
    assert mock_binance.return_value.get_bars.called
    mock_alpha.return_value.get_bars.assert_not_called()


@patch("src.quantproject.data_pipeline.backends.router.YFinanceBackend")
@patch("src.quantproject.data_pipeline.backends.router.AlphaVantageBackend")
def test_router_returns_empty_if_all_fail(mock_alpha, mock_yf):
    mock_yf.return_value.get_bars.return_value = pd.DataFrame(columns=_COLS)
    mock_alpha.return_value.get_bars.return_value = pd.DataFrame(columns=_COLS)

    router = DataRouter()
    df = router.get_bars("MSFT", "2025-08-01", "2025-09-01", "5min")

    assert df.empty
    assert list(df.columns) == _COLS
