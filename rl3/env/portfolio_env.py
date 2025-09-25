from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
import pandas as pd

from rl3.broker.simulator import BrokerParams, BrokerSim


@dataclass
class EnvConfig:
    features: List[str]
    window: int = 64
    max_weight: float = 0.30
    max_dweight: float = 0.10
    gross_leverage_cap: float = 1.20
    commission_bps: float = 1.0
    slippage_alpha: float = 0.10
    slippage_beta: float = 0.20
    participation_cap: float = 0.10
    reward_cost_bps: float = 0.0            # additional per-step cost rate (bps)
    lambda_turnover: float = 1e-3           # turnover penalty weight
    reward_clip: float = 0.05              # per-step reward clip
    dd_hard: float = 0.15                 # hard drawdown limit (fraction)


class PortfolioEnv(gym.Env):
    """Multi-asset portfolio environment using deterministic BrokerSim trade costs."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: Dict[str, pd.DataFrame],
        feats: Dict[str, pd.DataFrame],
        symbols: List[str],
        cfg: EnvConfig,
    ):
        super().__init__()
        self.symbols = list(symbols)
        self.N = len(self.symbols)
        self.cfg = cfg
        self.window = cfg.window
        self.features = list(cfg.features)
        self.F = len(self.features)

        aligned = self._align_prices_and_features(prices, feats)
        self.prices = aligned["prices"]
        self.feats = aligned["feats"]
        self.index = aligned["index"]

        self.rets = {
            s: self.prices[s]["close"].pct_change().fillna(0.0).to_numpy(dtype=np.float32)
            for s in self.symbols
        }

        obs_dim = self.N * self.window * self.F
        self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-self.cfg.max_weight,
            high=self.cfg.max_weight,
            shape=(self.N,),
            dtype=np.float32,
        )

        self.broker = BrokerSim(
            BrokerParams(
                self.cfg.commission_bps,
                self.cfg.slippage_alpha,
                self.cfg.slippage_beta,
                self.cfg.participation_cap,
            )
        )

        self.initial_nav = 1_000_000.0
        self.t0 = self.window
        self.weights = np.zeros(self.N, dtype=np.float32)
        self.nav = max(float(self.initial_nav), 1e-6)
        self.high_water = self.nav
        self.peak_nav = self.nav
        self.drawdown = 0.0

        self.nav_history: List[float] = [self.nav]
        self.weights_history: List[List[float]] = [self.weights.tolist()]
        self.returns_history: List[float] = []

        self.training_nav_history: List[float] = [self.nav]
        self.training_weights_history: List[List[float]] = [self.weights.tolist()]
        self.training_returns_history: List[float] = []

        self._override_bar: Optional[Dict[str, float]] = None
        self.t = self.t0

    # ------------------------------------------------------------------
    def _align_prices_and_features(
        self, prices: Dict[str, pd.DataFrame], feats: Dict[str, pd.DataFrame]
    ):
        idx: Optional[pd.DatetimeIndex] = None
        p2, f2 = {}, {}
        for s in self.symbols:
            p = prices[s][["open", "high", "low", "close", "volume"]].copy()
            p.index = pd.to_datetime(p.index, utc=True)
            assert p.index.is_monotonic_increasing, f"prices idx not monotonic for {s}"
            p2[s] = p.astype(float)
            idx = p.index if idx is None else idx.intersection(p.index)
        for s in self.symbols:
            f = feats[s].copy()
            f.index = pd.to_datetime(f.index, utc=True)
            for col in self.features:
                if col not in f.columns:
                    f[col] = 0.0
            f2[s] = f[self.features].astype(float)
            idx = idx.intersection(f2[s].index)
        for s in self.symbols:
            p2[s] = p2[s].reindex(idx)
            f2[s] = f2[s].reindex(idx).fillna(0.0)
        return {"prices": p2, "feats": f2, "index": idx}

    # ------------------------------------------------------------------
    def _make_obs(self, t: int) -> np.ndarray:
        t0 = t - self.window
        chunks = [self.feats[s].iloc[t0:t, :].to_numpy(dtype=np.float32).reshape(-1) for s in self.symbols]
        obs = np.concatenate(chunks, axis=0)
        np.nan_to_num(obs, copy=False)
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    def _apply_action_constraints(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(action, -self.cfg.max_weight, self.cfg.max_weight)
        d = np.clip(a - self.weights, -self.cfg.max_dweight, self.cfg.max_dweight)
        w = self.weights + d
        gross = float(np.sum(np.abs(w)))
        if gross > self.cfg.gross_leverage_cap + 1e-6:
            w *= self.cfg.gross_leverage_cap / gross
        return w.astype(np.float32)

    def update_bar(self, bar: Dict[str, float]) -> None:
        self._override_bar = bar

    def _current_bar(self) -> Dict[str, float]:
        if self._override_bar is not None:
            bar = self._override_bar
            self._override_bar = None
            return bar

        bar: Dict[str, float] = {"mid": 0.0, "volume": 0.0}
        for symbol in self.symbols:
            price = float(self.prices[symbol]["close"].iloc[self.t])
            volume = float(self.prices[symbol]["volume"].iloc[self.t])
            bar[f"price_{symbol}"] = price
            bar[f"volume_{symbol}"] = volume
            bar["mid"] += price
            bar["volume"] += volume
        if self.symbols:
            bar["mid"] /= float(len(self.symbols))
        return bar

    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.t = self.t0
        self.weights[:] = 0.0
        self.nav = max(float(self.initial_nav), 1e-6)
        self.high_water = self.nav
        self.peak_nav = self.nav
        self.drawdown = 0.0

        self.nav_history = [self.nav]
        self.weights_history = [self.weights.tolist()]
        self.returns_history = []
        self._override_bar = None
        return self._make_obs(self.t), {"t": int(self.t), "index": str(self.index[self.t])}

    def step(self, action: np.ndarray):
        assert action.shape == (self.N,)
        new_w = self._apply_action_constraints(action)
        prev_w = self.weights.copy()
        dweight = new_w - prev_w

        ar = np.array([self.rets[s][self.t] for s in self.symbols], dtype=np.float32)
        prices = np.array([self.prices[s]["close"].iloc[self.t] for s in self.symbols], dtype=np.float32)

        nav = float(self.nav)
        trade_costs_usd = 0.0
        for idx, _symbol in enumerate(self.symbols):
            delta = float(dweight[idx])
            if abs(delta) <= 1e-12:
                continue
            price = float(prices[idx])
            if price <= 0.0:
                continue
            trade_costs_usd += self.broker.execute(price, delta, nav)
        trade_costs = float(trade_costs_usd)

        portfolio_ret = float(np.dot(prev_w, ar))
        pnl = portfolio_ret * nav

        trade_cost_rate = trade_costs / nav if nav > 1e-12 else 0.0
        base_cost_rate = self.cfg.reward_cost_bps / 1e4
        total_cost_rate = trade_cost_rate + base_cost_rate

        turnover = float(np.sum(np.abs(dweight)))
        turnover_penalty = self.cfg.lambda_turnover * turnover

        per_step_return = portfolio_ret - total_cost_rate - turnover_penalty
        reward = float(np.clip(per_step_return, -abs(self.cfg.reward_clip), abs(self.cfg.reward_clip)))
        net_portfolio_ret = portfolio_ret - total_cost_rate

        self.training_returns_history.append(float(net_portfolio_ret))
        self.returns_history.append(float(net_portfolio_ret))

        self.weights = new_w
        self.nav = nav * (1.0 + net_portfolio_ret)
        if self.nav <= 0.0:
            self.nav = 1e-6

        self.nav_history.append(self.nav)
        self.weights_history.append(self.weights.tolist())
        self.training_nav_history.append(self.nav)
        self.training_weights_history.append(self.weights.tolist())

        if nav > 1e-12:
            net_return = (self.nav - nav) / nav
        else:
            net_return = 0.0

        next_t = self.t + 1
        self.t = next_t
        truncated = False

        self.high_water = max(self.high_water, self.nav)
        self.peak_nav = max(self.peak_nav, self.nav)
        if self.peak_nav > 0:
            dd = (self.nav - self.peak_nav) / self.peak_nav
        else:
            dd = 0.0
        self.drawdown = dd
        risk_stop = dd <= -abs(self.cfg.dd_hard)
        terminated = (next_t >= len(self.index) - 1) or risk_stop

        info = {
            "t": int(next_t),
            "index": str(self.index[min(next_t, len(self.index) - 1)]),
            "nav": float(self.nav),
            "weights": self.weights.copy(),
            "pnl_ret": float(portfolio_ret),
            "cost_rate": float(total_cost_rate),
            "turnover": float(turnover),
            "dd": float(dd),
            "trade_costs": float(trade_costs_usd),
            "portfolio_return": float(portfolio_ret),
            "net_return": float(net_return),
            "pnl": float(pnl),
            "trade_cost_rate": float(trade_cost_rate),
            "base_cost_rate": float(base_cost_rate),
            "total_cost_rate": float(total_cost_rate),
            "drawdown": float(self.drawdown),
            "risk_stop": risk_stop,
            "turnover_penalty": float(turnover_penalty),
            "reward_clip": float(self.cfg.reward_clip),
            "per_step_return": float(per_step_return),
        }

        obs = self._make_obs(self.t) if not terminated else self.observation_space.low
        return obs, reward, terminated, truncated, info
    def get_episode_trajectory(self) -> Dict[str, List]:
        equity = [float(v) for v in self.nav_history]
        weights = [list(w) for w in self.weights_history]
        returns = [float(r) for r in self.returns_history]
        return {"returns": returns, "equity": equity, "weights": weights}

    def get_training_artifacts(self) -> Dict[str, List]:
        returns = list(self.training_returns_history)
        if not returns:
            returns = [0.0]
        return {
            "returns": returns,
            "equity": list(self.training_nav_history),
            "weights": list(self.training_weights_history),
        }





















