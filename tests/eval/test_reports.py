from pathlib import Path

from eval.reports import write_markdown


def test_write_markdown(tmp_path):
    metrics = {
        "ann_return": 0.12,
        "ann_vol": 0.20,
        "sharpe": 0.60,
        "max_drawdown": -0.05,
    }
    symbols = ["EURUSD", "BTCUSD"]
    out = tmp_path / "report.md"

    write_markdown(out, metrics, symbols)

    text = out.read_text()
    assert "RL3  Training Report" in text
    assert "Annualized Return" in text
    for symbol in symbols:
        assert symbol in text
