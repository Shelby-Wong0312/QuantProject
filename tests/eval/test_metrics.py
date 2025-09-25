from eval.metrics import compute


def test_compute_basic_metrics():
    returns = [0.01, -0.005, 0.003, 0.0]
    equity = [1_000_000.0, 1_010_000.0, 1_004_500.0, 1_007_500.0]

    metrics = compute(returns, equity)

    assert set(metrics.keys()) == {"ann_return", "ann_vol", "sharpe", "max_drawdown"}
    assert metrics["ann_vol"] > 0.0
    assert metrics["ann_return"] != 0.0
    assert metrics["sharpe"] == metrics["sharpe"]  # not NaN
    assert metrics["max_drawdown"] < 0.0

