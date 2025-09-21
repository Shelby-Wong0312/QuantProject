# app_integration.py  -- Capital.com adapter (reads .env)
# 目的：
#   讀取 Capital.com 帳戶摘要與持倉，餵給 LiveTradingSystem 的背景快照
#   讓 LINE 指令 /status、/positions 顯示真實數據
#
# 支援的憑證環境變數（擇一組；會自動讀取 .env）：
#   A) 你現在的 .env 名稱：
#      CAPITAL_API_KEY, CAPITAL_IDENTIFIER, CAPITAL_API_PASSWORD, CAPITAL_DEMO_MODE=True|False
#   B) 替代名稱（若你之後想換）：
#      CAPITAL_API_KEY, CAPITAL_USERNAME, CAPITAL_PASSWORD, CAPITAL_ENV=demo|live
#
# 執行：
#   python app_integration.py --mode broker --period 60
#   （demo 模式）python app_integration.py --mode demo --period 60

from __future__ import annotations
import argparse, json, logging, os, sys, time
from typing import Iterable, List, Optional
from urllib import request, error

from live_trading_system import LiveTradingSystem, BrokerAdapter, Position

# ---------- 基礎設定 ----------
LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("app_integration")

# ---------- .env 讀取（無第三方套件） ----------
def _load_dotenv_into_environ(path: str = ".env") -> int:
    """把當前目錄的 .env 讀進 os.environ（已存在的 key 不覆寫）。回傳載入的鍵數。"""
    cnt = 0
    try:
        if not os.path.exists(path):
            return 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
                    cnt += 1
    except Exception as e:
        log.warning("讀取 .env 失敗：%s", e)
    return cnt

_loaded = _load_dotenv_into_environ()
if _loaded:
    log.info("loaded %d keys from .env", _loaded)

# ---------- 工具 ----------
def _as_bool(x) -> bool:
    return str(x).strip().lower() in ("1", "true", "yes", "y", "on")

# ---------- Capital.com REST Adapter ----------
class CapitalRESTAdapter(BrokerAdapter):
    """
    最小可用 Capital.com REST adapter：
      - POST /api/v1/session 取得 CST / X-SECURITY-TOKEN
      - GET  /api/v1/accounts   取 equity/cash/unrealisedPnL/realisedPnL
      - GET  /api/v1/positions  取未平倉部位（symbol/size/avg price/direction）
    """

    def __init__(self, api_key: str, identifier: str, password: str, demo_mode: bool = True, host: Optional[str] = None):
        if not (api_key and identifier and password):
            raise RuntimeError("缺少 Capital 憑證（CAPITAL_API_KEY / CAPITAL_IDENTIFIER / CAPITAL_API_PASSWORD）")
        self.api_key = api_key.strip()
        self.identifier = identifier.strip()
        self.password = password
        if not host:
            host = "api-demo-capital.backend-capital.com" if demo_mode else "api-capital.backend-capital.com"
        self.host = host
        self.base = f"https://{self.host}/api/v1"
        self._cst = None
        self._sec = None

        # 快取最新數據，避免過度打 API
        self._last_fetch_ts = 0.0
        self._equity = 0.0
        self._cash = 0.0
        self._upnl = 0.0
        self._rpnl = 0.0
        self._positions: List[Position] = []

        self._login()

    # ---- BrokerAdapter 介面 ----
    @property
    def equity(self) -> float: self._maybe_refresh(); return self._equity
    @property
    def cash(self) -> float:   self._maybe_refresh(); return self._cash
    @property
    def upnl(self) -> float:   self._maybe_refresh(); return self._upnl
    @property
    def rpnl(self) -> float:   self._maybe_refresh(); return self._rpnl
    def positions(self) -> Iterable[Position]:
        self._maybe_refresh()
        return list(self._positions)

    # ---- HTTP ----
    def _headers(self, with_auth: bool) -> dict:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-CAP-API-KEY": self.api_key,
        }
        if with_auth and self._cst and self._sec:
            h["CST"] = self._cst
            h["X-SECURITY-TOKEN"] = self._sec
        return h

    def _request(self, method: str, path: str, data: Optional[dict] = None, with_auth: bool = False) -> dict:
        url = self.base + path
        body = None if data is None else json.dumps(data).encode("utf-8")
        req = request.Request(url, data=body, headers=self._headers(with_auth), method=method.upper())
        try:
            with request.urlopen(req, timeout=15) as resp:
                if path == "/session":
                    self._cst = resp.headers.get("CST") or self._cst
                    self._sec = resp.headers.get("X-SECURITY-TOKEN") or self._sec
                raw = resp.read().decode("utf-8") or "{}"
                return json.loads(raw)
        except error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code} {e.reason} @ {path}: {e.read().decode('utf-8','ignore')}") from None
        except error.URLError as e:
            raise RuntimeError(f"Network error @ {path}: {e}") from None

    def _login(self) -> None:
        log.info("Capital.com login @ %s", self.host)
        _ = self._request("POST", "/session", {"identifier": self.identifier, "password": self.password}, with_auth=False)
        if not (self._cst and self._sec):
            raise RuntimeError("登入成功但未取得 CST / X-SECURITY-TOKEN（請檢查 API Key/帳密/環境 demo|live）")

    def _fetch_once(self) -> None:
        # 帳戶摘要
        acc = self._request("GET", "/accounts", None, with_auth=True)
        accounts = acc.get("accounts") or acc.get("accountList") or acc.get("data") or []
        a = accounts[0] if accounts else {}
        def pick_float(obj, *keys, default=0.0):
            for k in keys:
                if k in obj:
                    try: return float(obj[k])
                    except: pass
            return float(default)
        self._equity = pick_float(a, "equity", "equityValue", "equityCash")
        self._cash   = pick_float(a, "available", "cash", "availableCash", "balance")
        self._upnl   = pick_float(a, "unrealisedProfitLoss", "unrealisedPnL", "unrealizedPnL")
        self._rpnl   = pick_float(a, "realisedProfitLoss", "realisedPnL", "realizedPnL")

        # 持倉
        pos_raw = self._request("GET", "/positions", None, with_auth=True)
        items = pos_raw.get("positions") or pos_raw.get("data") or pos_raw.get("openPositions") or []
        parsed: List[Position] = []
        for p in items:
            pos = p.get("position") or p
            market = p.get("market") or {}
            inst = pos.get("instrument") or {}
            sym = inst.get("symbol") or pos.get("epic") or market.get("symbol") or market.get("instrumentName") or pos.get("symbol") or "UNKNOWN"
            qty = pos.get("size") or pos.get("dealSize") or pos.get("quantity") or 0
            level = pos.get("level") or pos.get("avgPrice") or pos.get("openLevel") or pos.get("price") or 0
            side = (pos.get("direction") or pos.get("side") or "").lower()
            try:
                qty = float(qty); level = float(level)
            except:
                qty, level = 0.0, 0.0
            if side == "sell" and qty > 0:  # sell 倉位視為負數
                qty = -qty
            parsed.append(Position(sym, qty, level))
        self._positions = parsed
        self._last_fetch_ts = time.time()

    def _maybe_refresh(self) -> None:
        if time.time() - self._last_fetch_ts >= 3.0:
            self._fetch_once()

# ---------- Demo Broker（保留驗證用） ----------
class DemoBroker(BrokerAdapter):
    def __init__(self):
        self._equity = 102345.67; self._cash = 53421.00; self._upnl = 123.45; self._rpnl = 987.65
        self._pos = [Position("XAUUSD", 1.0, 2405.0), Position("US100", -2.0, 18350.5)]
    @property
    def equity(self): return self._equity
    @property
    def cash(self):   return self._cash
    @property
    def upnl(self):   return self._upnl
    @property
    def rpnl(self):   return self._rpnl
    def positions(self): return list(self._pos)

# ---------- 建立 adapter（吃 .env / 環境變數） ----------
def build_capital_adapter_from_env() -> CapitalRESTAdapter:
    key = os.getenv("CAPITAL_API_KEY", "")
    # 兼容兩種命名：IDENTIFIER / USERNAME
    ident = os.getenv("CAPITAL_IDENTIFIER") or os.getenv("CAPITAL_USERNAME") or ""
    pwd = os.getenv("CAPITAL_API_PASSWORD") or os.getenv("CAPITAL_PASSWORD") or ""
    # 兼容 demo/live：CAPITAL_DEMO_MODE or CAPITAL_ENV
    demo_mode = _as_bool(os.getenv("CAPITAL_DEMO_MODE", "true"))
    env_str = os.getenv("CAPITAL_ENV")
    if env_str:
        demo_mode = (env_str.strip().lower() in ("demo", "paper", "test"))
    if not (key and ident and pwd):
        raise RuntimeError("缺少 Capital 憑證：CAPITAL_API_KEY / CAPITAL_IDENTIFIER / CAPITAL_API_PASSWORD（已讀取 .env）")
    return CapitalRESTAdapter(api_key=key, identifier=ident, password=pwd, demo_mode=demo_mode)

# ---------- Runner ----------
def _run(adapter: BrokerAdapter, period: int):
    lts = LiveTradingSystem(adapter, snapshot_enabled=True)
    lts.start_state_sync(period_sec=period)
    print(f"[state-sync] started; period={period}s.  Ctrl+C 停止。")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("\n[signal] stop...")
    finally:
        try: lts.stop()
        except Exception: pass
        print("[state-sync] stopped.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["broker","demo"], default="broker")
    ap.add_argument("--period", type=int, default=60, help="背景快照秒數（預設 60）")
    args = ap.parse_args()

    if args.mode == "demo":
        _run(DemoBroker(), args.period); return

    adapter = build_capital_adapter_from_env()
    _run(adapter, args.period)

if __name__ == "__main__":
    main()
