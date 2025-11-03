from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import pandas as pd
import yaml

from src.quantproject.data_pipeline.backends.alpha_vantage import AlphaVantageBackend
from src.quantproject.data_pipeline.backends.binance import BinanceBackend
from src.quantproject.data_pipeline.backends.router import DataRouter
from src.quantproject.data_pipeline.backends.yfinance import YFinanceBackend


def load_config(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path).expanduser()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    text = cfg_path.read_text(encoding="utf-8")
    suffix = cfg_path.suffix.lower()
    if suffix in {".json", ".jsonc"}:
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise TypeError(f"Configuration at {cfg_path} must be a mapping.")
    return data


def parse_window_spec(spec: str) -> Tuple[str, str]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Invalid window spec '{spec}'. Expected 'start,end'.")
    return parts[0], parts[1]


def _slug(ts_value: Any) -> str:
    ts = pd.Timestamp(ts_value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y%m%d")


def _to_iso(ts_value: Any) -> str:
    ts = pd.Timestamp(ts_value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def _parse_timedelta(value: Any) -> pd.Timedelta:
    if isinstance(value, pd.Timedelta):
        return value
    if isinstance(value, (int, float)):
        return pd.Timedelta(days=float(value))
    if isinstance(value, str):
        try:
            return pd.Timedelta(value)
        except Exception:
            try:
                return pd.to_timedelta(float(value), unit="D")
            except Exception as exc:
                raise ValueError(f"Cannot parse timedelta from '{value}'.") from exc
    raise TypeError(f"Unsupported timedelta spec: {value!r}")


def build_eval_windows(
    cfg: Mapping[str, Any],
    window_override: str | None = None,
    *,
    prefix: str = "wf",
) -> List[Dict[str, str]]:
    eval_cfg = cfg.get("eval", {}) or {}
    windows: List[Dict[str, str]] = []

    if window_override:
        start_raw, end_raw = parse_window_spec(window_override)
        label = f"{prefix}_00_{_slug(start_raw)}_{_slug(end_raw)}"
        windows.append(
            {
                "label": label,
                "start": _to_iso(start_raw),
                "end": _to_iso(end_raw),
            }
        )
        return windows

    explicit = eval_cfg.get("windows")
    if explicit:
        for idx, window in enumerate(explicit):
            start = window.get("start")
            end = window.get("end")
            if not start or not end:
                raise ValueError("Each eval window must define start and end.")
            label = window.get("label") or f"{prefix}_{idx:02d}_{_slug(start)}_{_slug(end)}"
            windows.append(
                {
                    "label": str(label),
                    "start": _to_iso(start),
                    "end": _to_iso(end),
                }
            )
        return windows

    rolling = eval_cfg.get("rolling")
    if rolling:
        start_spec = rolling.get("start", cfg.get("start"))
        end_spec = rolling.get("end", cfg.get("end"))
        if start_spec is None or end_spec is None:
            raise ValueError("Rolling eval requires explicit start and end.")
        start_ts = pd.Timestamp(start_spec, tz="UTC")
        end_ts = pd.Timestamp(end_spec, tz="UTC")
        if start_ts >= end_ts:
            raise ValueError("Rolling eval requires start < end.")
        window_td = _parse_timedelta(rolling.get("window") or rolling.get("window_days"))
        step_td = _parse_timedelta(
            rolling.get("step") or rolling.get("step_days") or rolling.get("window")
        )
        label_prefix = rolling.get("label_prefix", prefix)
        idx = 0
        current_start = start_ts
        while current_start < end_ts:
            current_end = current_start + window_td
            if current_end > end_ts:
                current_end = end_ts
            windows.append(
                {
                    "label": f"{label_prefix}_{idx:02d}_{current_start.strftime('%Y%m%d')}_{current_end.strftime('%Y%m%d')}",
                    "start": current_start.isoformat(),
                    "end": current_end.isoformat(),
                }
            )
            current_start = current_start + step_td
            idx += 1
        if not windows:
            raise ValueError("Rolling eval produced no windows.")
        return windows

    base_start = eval_cfg.get("start") or cfg.get("start")
    base_end = eval_cfg.get("end") or cfg.get("end")
    if base_start and base_end:
        label = eval_cfg.get("label") or f"{prefix}_00_{_slug(base_start)}_{_slug(base_end)}"
        windows.append(
            {
                "label": str(label),
                "start": _to_iso(base_start),
                "end": _to_iso(base_end),
            }
        )
        return windows

    raise ValueError("No evaluation window defined; provide eval.windows, eval.rolling, or --window.")


def _select_kwargs(config: Mapping[str, Any], allowed: Iterable[str]) -> Dict[str, Any]:
    return {key: config[key] for key in allowed if key in config}


def build_data_router(config: Mapping[str, Any] | None) -> DataRouter:
    if config is None:
        return DataRouter()

    yahoo_cfg = config.get("yahoo", {}) or {}
    alpha_cfg = config.get("alphavantage", {}) or {}
    binance_cfg = config.get("binance", {}) or {}

    yahoo = YFinanceBackend(**_select_kwargs(yahoo_cfg, {"cache_dir"})) if yahoo_cfg else None
    alphavantage = (
        AlphaVantageBackend(**_select_kwargs(alpha_cfg, {"cache_dir", "api_key", "request_timeout"}))
        if alpha_cfg
        else None
    )
    binance = (
        BinanceBackend(**_select_kwargs(binance_cfg, {"base_url", "cache_dir", "request_timeout"}))
        if binance_cfg
        else None
    )

    return DataRouter(yahoo=yahoo, alphavantage=alphavantage, binance=binance)


__all__ = ["load_config", "parse_window_spec", "build_eval_windows", "build_data_router"]
