from __future__ import annotations

from unittest.mock import patch, MagicMock

from src.quantproject.data_pipeline.backends.binance import BinanceBackend


def _mock_response(payload, status=200):
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = payload

    def raise_for_status():
        if status != 200:
            raise ValueError("error")

    mock.raise_for_status.side_effect = raise_for_status
    return mock


@patch("src.quantproject.data_pipeline.backends.binance.requests.Session")
def test_binance_backend_success(mock_session):
    session_instance = mock_session.return_value
    klines = [
        [1693526400000, "26000", "26100", "25950", "26050", "12"],
        [1693526700000, "26050", "26200", "26000", "26150", "8"],
    ]
    session_instance.get.side_effect = [_mock_response(klines), _mock_response([])]

    backend = BinanceBackend(cache_dir="data_cache/binance_test")
    df = backend.get_bars("BTC-USD", "2023-08-31", "2023-09-01", "5min")

    assert not df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.tz is not None
    assert df.index.is_monotonic_increasing
    assert df.loc[df.index[0], "open"] == 26000.0


def test_binance_backend_unsupported_symbol():
    backend = BinanceBackend(cache_dir="data_cache/binance_test")
    df = backend.get_bars("SPY", "2023-08-31", "2023-09-01", "5min")
    assert df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
