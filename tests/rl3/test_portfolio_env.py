import numpy as np
import pandas as pd
import pytest

from rl3.env.portfolio_env import EnvConfig, PortfolioEnv


def _fake(n: int = 100):
    idx = pd.date_range("2024-08-01", periods=n, freq="min", tz="UTC")

    def bars():
        return pd.DataFrame(
            {
                "open": np.linspace(100, 101, n),
                "high": np.linspace(100.2, 101.2, n),
                "low": np.linspace(99.8, 100.8, n),
                "close": np.linspace(100, 101, n) + np.random.randn(n) * 0.01,
                "volume": np.random.randint(100, 200, n),
            },
            index=idx,
        )

    prices = {"AAA": bars(), "BBB": bars()}
    feats = {
        s: pd.DataFrame(
            {
                "logret": np.log(prices[s]["close"]).diff().fillna(0.0),
                "range_pct": (prices[s]["high"] - prices[s]["low"])
                / prices[s]["close"],
            },
            index=prices[s].index,
        )
        for s in prices
    }
    return prices, feats


def _make_env(**overrides):
    prices, feats = _fake(100)
    cfg_defaults = dict(
        window=32,
        reward_cost_bps=10.0,
        lambda_turnover=0.05,
        reward_clip=0.05,
        dd_hard=0.05,
        dweight_threshold=0.0,
    )
    cfg_defaults.update(overrides)
    cfg = EnvConfig(features=["logret", "range_pct"], **cfg_defaults)
    return PortfolioEnv(prices, feats, symbols=["AAA", "BBB"], cfg=cfg)


def test_env_step_shapes_and_costs():
    env = _make_env()
    obs, _ = env.reset()
    assert obs.shape == (2 * 32 * 2,)

    action = np.array([0.3, -0.3], dtype=np.float32)
    _, reward, *_rest, info = env.step(action)

    assert np.isfinite(reward)
    assert env.action_space.shape == (2,)
    assert info["trade_costs"] >= 0.0
    assert "drawdown" in info
    assert "reward_clip" in info
    assert "pnl_ret" in info

    expected = (
        info["portfolio_return"] - info["total_cost_rate"] - info["turnover_penalty"]
    )
    recorded = info["per_step_return"]
    assert pytest.approx(recorded, rel=1e-9) == expected
    assert pytest.approx(info["pnl_ret"], rel=1e-9) == info["portfolio_return"]
    if abs(recorded) > info["reward_clip"] + 1e-12:
        assert abs(reward) == pytest.approx(info["reward_clip"], rel=1e-9)
    else:
        assert pytest.approx(reward, rel=1e-9) == recorded


def test_env_costs_non_negative():
    env = _make_env()
    env.reset()
    action = np.array([0.3, -0.3], dtype=np.float32)
    *_rest, info = env.step(action)
    assert info["trade_costs"] >= 0.0


def test_env_training_artifacts():
    env = _make_env(reward_cost_bps=0.0, lambda_turnover=0.0)
    env.reset()
    action = np.array([0.05, -0.05], dtype=np.float32)
    for _ in range(4):
        env.step(action)
    artifacts = env.get_training_artifacts()

    assert set(artifacts.keys()) == {"returns", "equity", "weights"}
    assert len(artifacts["equity"]) == len(env.nav_history)
    assert len(artifacts["returns"]) == len(artifacts["equity"]) - 1
    assert len(artifacts["weights"]) == len(artifacts["equity"])
    assert isinstance(artifacts["weights"][0], list)
    assert artifacts["equity"][0] == pytest.approx(1_000_000.0)
    assert all(v >= 1e-6 for v in artifacts["equity"])


def test_env_dd_hard_stop():
    env = _make_env(dd_hard=0.01, reward_cost_bps=0.0, lambda_turnover=0.0)
    env.reset()

    env.rets = {s: np.full_like(r, -0.5, dtype=np.float32) for s, r in env.rets.items()}

    action = np.array([0.3, 0.3], dtype=np.float32)
    env.step(action)
    _, reward, terminated, *_rest, info = env.step(action)

    assert terminated
    assert info["risk_stop"] is True
    assert info["drawdown"] <= -0.01
    assert pytest.approx(info["dd"], rel=1e-9) == info["drawdown"]

    expected = (
        info["portfolio_return"] - info["total_cost_rate"] - info["turnover_penalty"]
    )
    recorded = info["per_step_return"]
    assert pytest.approx(recorded, rel=1e-9) == expected
    if abs(recorded) > info["reward_clip"]:
        assert abs(reward) == pytest.approx(info["reward_clip"], rel=1e-9)
    else:
        assert pytest.approx(reward, rel=1e-9) == recorded


def test_reward_and_dd_ranges():
    env = _make_env()
    env.reset()
    action = np.array([0.3, -0.3], dtype=np.float32)
    for _ in range(5):
        _, reward, terminated, _, info = env.step(action)
        assert -0.05 <= reward <= 0.05
        assert info["dd"] <= 0.0 and info["dd"] >= -1.0
        if terminated:
            break


def test_dweight_threshold_blocks_small_changes():
    env = _make_env(
        dweight_threshold=0.05,
        max_dweight=0.2,
        reward_cost_bps=0.0,
        lambda_turnover=0.0,
    )
    env.reset()

    small_action = np.array([0.02, -0.02], dtype=np.float32)
    *_rest, info_small = env.step(small_action)
    assert np.allclose(info_small["weights"], 0.0)
    assert info_small["turnover"] == pytest.approx(0.0, abs=1e-12)

    large_action = np.array([0.15, -0.15], dtype=np.float32)
    *_rest, info_large = env.step(large_action)
    assert info_large["turnover"] > 0.0
    assert np.any(np.abs(info_large["weights"]) > 1e-6)
