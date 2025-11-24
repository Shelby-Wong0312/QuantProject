import os, sys, inspect, traceback
import pandas as pd
import pyarrow.parquet as pq

PROJ = r"C:\Users\niuji\Documents\QuantProject"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "quant_project_RL"))

# vecenv 相容層
try:
    from rl3.eval.vecenv_compat import make_sb3_vecenv
except ImportError:
    from quant_project_RL.rl3.eval.vecenv_compat import make_sb3_vecenv

# env/設定
try:
    from quant_project_RL.envs.portfolio_env import PortfolioEnv, EnvConfig
except ImportError:
    from envs.portfolio_env import PortfolioEnv, EnvConfig

SYMS = ["BTCUSDT","ETHUSDT"]
TF = "5m"

CAND_TS = [
    "ts_utc","timestamp","time","ts","datetime",
    # 常見交易所欄位
    "open_time","openTime","kline_open_time","openTimeMs",
    # 你們自家欄位可能
    "ts_open","t_open"
]

def _filter_kwargs(func_or_cls, kwargs):
    try:
        sig = inspect.signature(func_or_cls)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return kwargs

def _find_binance_roots():
    hits = []
    for root, dirs, files in os.walk(PROJ):
        if os.path.basename(root).lower() != "binance":
            continue
        p_btc = os.path.join(root, "BTCUSDT", f"{TF}.parquet")
        p_eth = os.path.join(root, "ETHUSDT", f"{TF}.parquet")
        if os.path.exists(p_btc) and os.path.exists(p_eth):
            hits.append(root)
    return hits

def _to_utc_series(x):
    """將各種型別的時間欄轉成 UTC pandas.Series（支援 int 秒/毫秒、字串）。"""
    s = pd.Series(x)
    if pd.api.types.is_integer_dtype(s):
        # 判斷是秒還是毫秒：用量級估
        # 大於 10^12 多半是毫秒；介於 10^9~10^10 是秒
        med = s.median()
        if med > 1e12:   # ms
            return pd.to_datetime(s, unit="ms", utc=True, errors="coerce")
        elif med > 1e9:  # s
            return pd.to_datetime(s, unit="s", utc=True, errors="coerce")
        else:
            # 其他整數尺度，直接當秒
            return pd.to_datetime(s, unit="s", utc=True, errors="coerce")
    else:
        return pd.to_datetime(s, utc=True, errors="coerce")

def _ts_range_one(p):
    """回傳 (min, max, n, status)。若不可用會附上原因。"""
    try:
        pf = pq.ParquetFile(p)
        schema = pf.schema  # 舊新版皆可
        names = list(schema.names)
        # 先找候選欄位
        cand = [c for c in CAND_TS if c in names]
        if cand:
            col = cand[0]
            arr = pq.read_table(p, columns=[col])[col].to_pandas()
            s = _to_utc_series(arr).dropna()
            if len(s) == 0:
                return None, None, 0, f"EMPTY_{col}"
            return s.min(), s.max(), len(s), f"OK[{col}]"
        # 沒有候選欄位  取前幾列看內容
        tbl = pq.read_table(p)
        df = tbl.to_pandas()
        head = df.head(3)
        sample_cols = names[:5]
        print(f"[WARN] {os.path.basename(p)} 無常見時間欄。schema前5欄: {sample_cols}")
        print("[WARN] 樣本資料(前3列):")
        with pd.option_context("display.max_columns", 10, "display.width", 160):
            print(head)
        # 試著猜一個像時間的欄（包含 'time' 的欄位名）
        time_like = [c for c in names if "time" in c.lower()]
        if time_like:
            col = time_like[0]
            s = _to_utc_series(df[col]).dropna()
            if len(s) > 0:
                return s.min(), s.max(), len(s), f"GUESS[{col}]"
        return None, None, len(df), "NO_TS_COL"
    except Exception as e:
        return None, None, None, f"ERROR:{type(e).__name__}:{e}"

def _choose_window(btc, eth):
    want_s = pd.Timestamp("2025-09-03", tz="UTC")
    want_e = pd.Timestamp("2025-09-06", tz="UTC")
    if any(v[0] is None or v[1] is None for v in (btc, eth)):
        return None, None
    mn = max(btc[0], eth[0])
    mx = min(btc[1], eth[1])
    if mn is None or mx is None or mn >= mx:
        return None, None
    if mn <= want_s and mx >= want_e:
        return want_s, want_e
    adj_end = mx
    adj_start = max(mn, mx - pd.Timedelta(days=3))
    if adj_start >= adj_end:
        return None, None
    return adj_start, adj_end

def _make_oos_env():
    roots = _find_binance_roots()
    if not roots:
        raise RuntimeError("找不到同時含 BTC/ETH 的 binance 目錄，請先回補資料。")
    data_root = roots[0]
    p_btc = os.path.join(data_root, "BTCUSDT", f"{TF}.parquet")
    p_eth = os.path.join(data_root, "ETHUSDT", f"{TF}.parquet")

    b_min, b_max, b_n, b_status = _ts_range_one(p_btc)
    e_min, e_max, e_n, e_status = _ts_range_one(p_eth)
    print(f"[DATA] BTCUSDT {b_status} rows={b_n} min={b_min} max={b_max}")
    print(f"[DATA] ETHUSDT {e_status} rows={e_n} min={e_min} max={e_max}")

    if b_min is None or e_min is None:
        raise RuntimeError("Price parquet 存在但時間欄位不可用；請依上方 WARN 輸出修正欄名或回補正確口徑。")

    start, end = _choose_window((b_min, b_max), (e_min, e_max))
    if start is None or end is None:
        raise RuntimeError("兩檔資料交集為空或過短；請回補涵蓋 2025-09-03~09-06 的 5m。")

    base = dict(
        symbols=SYMS,
        timeframe=TF,
        start=str(start),
        end=str(end),
        commission_bps=2,
        slippage_alpha=0.5, slippage_beta=0.5,
        lambda_turnover=0.5,
        dweight_threshold=0.025,
        action_smooth_alpha=0.8,
        inaction_bonus=1e-4,
        data_root=data_root, data_dir=data_root, data_path=data_root,
        vecnorm_path=os.path.join(PROJ, r"runs\rl3\walkforward\wf_01\train\vecnormalize.pkl"),
        vecnorm_training=False,
    )

    try:
        cfg = EnvConfig(**_filter_kwargs(EnvConfig, base))
        return PortfolioEnv(cfg)
    except TypeError:
        return PortfolioEnv(**_filter_kwargs(PortfolioEnv, base))

def main():
    env_ctor = lambda: _make_oos_env()
    venv = make_sb3_vecenv([env_ctor], use_subproc=False)
    print("NUM_ENVS =", getattr(venv, "num_envs", None))
    print("single_action_space =", getattr(venv, "single_action_space", None))
    print("shape =", getattr(getattr(venv, "single_action_space", None), "shape", None))
    obs = venv.reset()
    action = [venv.single_action_space.sample()]
    obs, reward, done, info = venv.step(action)
    print("step ok")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FATAL]", type(e).__name__, e)
        traceback.print_exc()
        raise
