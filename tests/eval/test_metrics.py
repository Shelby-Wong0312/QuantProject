import math

from eval.metrics import compute


def test_compute_basic_metrics():
    returns = [0.01, -0.005, 0.003, 0.0]
    equity = [1_000_000.0, 1_010_000.0, 1_004_500.0, 1_007_500.0]

    metrics = compute(returns, equity, timeframe="5min", symbols=["SPY", "QQQ"])

    assert set(metrics.keys()) == {"ann_return", "ann_vol", "sharpe", "max_drawdown"}
    assert metrics["ann_vol"] > 0.0
    assert metrics["ann_return"] != 0.0
    assert metrics["sharpe"] == metrics["sharpe"]  # not NaN
    assert metrics["max_drawdown"] < 0.0


def test_compute_uses_crypto_annualization():
    returns = [0.001] * 10
    equity = [1_000_000.0] * 10

    equity_hours = 252 * 6.5
    crypto_hours = 24 * 365
    expected_equity_ann = 0.001 * equity_hours * 12
    expected_crypto_ann = 0.001 * crypto_hours * 12

    equity_metrics = compute(returns, equity, timeframe="5min", symbols=["SPY"])
    crypto_metrics = compute(returns, equity, timeframe="5min", symbols=["BTC-USD"])

    assert math.isclose(equity_metrics["ann_return"], expected_equity_ann, rel_tol=1e-9)
    assert math.isclose(crypto_metrics["ann_return"], expected_crypto_ann, rel_tol=1e-9)
    assert crypto_metrics["ann_return"] > equity_metrics["ann_return"]
