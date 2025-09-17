# path: line-trade-bot/src/ingest/app.py
# === LINE / SNS / DDB 整合層 ===
# 本檔在「收到一筆 JSON payload」時，會同時：
# 1) 依 payload 內容把交易事件發到 SNS（line-push 會推到 LINE）
# 2) 若是 filled/closed 事件，寫入 TradeEvents（/last N 用）
# 3) 若帶 summary/positions，也更新 SystemState（/status、/positions 用）

import base64
import json
import os
from typing import Dict, Any, Optional

# --- 動態把專案根目錄放到 sys.path，確保能 import 到 infra/* ---
import sys, pathlib
_THIS = pathlib.Path(__file__).resolve()
# 嘗試往上最多 5 層找 infra 目錄
for up in range(1, 6):
    cand = _THIS.parents[up-1] / "infra"
    if cand.is_dir():
        sys.path.append(str(_THIS.parents[up-1]))
        break

# 這裡會用到你前面建立的兩個模組
try:
    from infra.publish_to_sns import publish_trade_event
    from infra.state_writer import write_summary, write_positions_text, append_trade_event
except Exception as e:
    # 若還沒建立 infra/*，保留原 ingest 功能，但不做 LINE/DDB
    publish_trade_event = None  # type: ignore
    write_summary = None        # type: ignore
    write_positions_text = None # type: ignore
    append_trade_event = None   # type: ignore

# 原本就有的依賴（保留）
from common.ingest_core import process_payload


# ---------- 共用回應 ----------
def _resp(status: int, body: Any = None):
    if body is None:
        body = {"ok": status < 400}
    if not isinstance(body, str):
        body = json.dumps(body, ensure_ascii=False)
    return {"statusCode": status, "headers": {"Content-Type": "application/json"}, "body": body}


# ---------- 簡單的 header token 驗證（保留原樣） ----------
def _auth_ok(headers: Dict[str, str]) -> bool:
    token = None
    try:
        from common.secrets import get_param, get_secret
        token = get_param(os.environ.get("INGEST_TOKEN_PARAM", "")) or \
                get_secret(os.environ.get("INGEST_TOKEN_SECRET_ID", ""))
    except Exception:
        token = None
    if not token:
        token = os.environ.get("INGEST_TOKEN")
    if not token:
        return True  # No auth set
    hdr = None
    for k, v in (headers or {}).items():
        if k.lower() in ("x-auth-token", "x-api-key"):
            hdr = v
            break
    return hdr == token


# ---------- 交易事件格式化（for 記錄/除錯） ----------
def _format_event(ev: Dict[str, Any]) -> str:
    symbol = ev.get("symbol") or ev.get("ticker") or "?"
    side = (ev.get("side") or ev.get("action") or "").upper()
    qty = ev.get("qty") or ev.get("quantity")
    price = ev.get("price") or ev.get("fill_price")
    source = ev.get("source")
    note = ev.get("note") or ev.get("message")
    parts = ["TRADE", side, str(symbol)]
    if qty is not None:
        parts.append(str(qty))
    if price is not None:
        parts.append(f"@ {price}")
    if source:
        parts.append(f"#{source}")
    if note:
        parts.append(f"- {note}")
    return " ".join([p for p in parts if p])


# ---------- 將自由格式 payload 映射到我們的標準欄位 ----------
def _normalize_trade(ev: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    et = (ev.get("event") or ev.get("type") or ev.get("status") or ev.get("action") or "").strip().lower()
    # 映射可能的事件別名
    if et in ("submitted", "submit", "placed", "accepted", "ack"):
        status = "submitted"
    elif et in ("filled", "fill", "executed"):
        status = "filled"
    elif et in ("closed", "close", "flat"):
        status = "closed"
    elif et in ("rejected", "reject"):
        status = "rejected"
    else:
        # 若完全沒有事件類型，就不當作交易事件處理
        return None

    symbol = ev.get("symbol") or ev.get("ticker")
    side = (ev.get("side") or ev.get("action") or "").strip().lower()
    qty = ev.get("quantity") if "quantity" in ev else ev.get("qty")
    price = ev.get("price") or ev.get("fill_price") or ev.get("avg_price")
    deal_id = ev.get("dealId") or ev.get("deal_id") or ev.get("orderId") or ev.get("order_id")
    pnl = ev.get("pnl") or ev.get("profit")

    if not symbol or side not in ("buy", "sell") or qty is None or price is None:
        # 缺關鍵欄位就不處理（避免髒資料）
        return None

    out = {
        "symbol": symbol,
        "side": side,
        "quantity": float(qty),
        "price": float(price),
        "status": status,
        "dealId": deal_id,
        "pnl": (float(pnl) if isinstance(pnl, (int, float, str)) and str(pnl).replace('.', '', 1).lstrip('-').isdigit() else None),
        "extra": {k: v for k, v in ev.items() if k not in {"event","type","status","action","symbol","ticker","side","quantity","qty","price","fill_price","avg_price","dealId","deal_id","orderId","order_id","pnl","profit"}}
    }
    return out


def _maybe_update_state(ev: Dict[str, Any]) -> Dict[str, bool]:
    """
    若 payload 帶 summary 或 positions，就更新 SystemState。
    支援：
      - {"summary": {"equity":..., "cash":..., "upnl":..., "rpnl":...}}
      - {"positions_text": "XAUUSD 1@2405.0\nUS100 -2@18350.5"}
      - {"positions": [{"symbol":"XAUUSD","qty":1,"avg_price":2405.0}, ...]}
    """
    updated = {"summary": False, "positions": False}
    try:
        if write_summary and isinstance(ev.get("summary"), dict):
            s = ev["summary"]
            eq = float(s.get("equity", 0)); ca = float(s.get("cash", 0))
            up = float(s.get("upnl", 0));  rp = float(s.get("rpnl", 0))
            write_summary(equity=eq, cash=ca, upnl=up, rpnl=rp); updated["summary"] = True
    except Exception as e:
        print("write_summary failed:", e)

    try:
        if write_positions_text:
            if isinstance(ev.get("positions_text"), str) and ev["positions_text"].strip():
                write_positions_text(ev["positions_text"]); updated["positions"] = True
            elif isinstance(ev.get("positions"), list) and ev["positions"]:
                lines = []
                for p in ev["positions"]:
                    sym = p.get("symbol") or p.get("ticker") or "?"
                    qty = p.get("qty") or p.get("quantity") or 0
                    ap  = p.get("avg_price") or p.get("price") or 0
                    sign = "+" if float(qty) > 0 else ""
                    lines.append(f"{sym} {sign}{float(qty):.4f} @ {float(ap):.4f}")
                write_positions_text("\n".join(lines)); updated["positions"] = True
    except Exception as e:
        print("write_positions_text failed:", e)

    return updated


# ---------- Lambda 入口 ----------
def lambda_handler(event, context):
    if not _auth_ok(event.get("headers") or {}):
        return _resp(401, {"ok": False, "error": "Unauthorized"})

    body_str = event.get("body") or ""
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(body_str)
    else:
        raw = body_str.encode("utf-8")

    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return _resp(400, {"ok": False, "error": "Invalid JSON"})

    if not isinstance(payload, dict):
        return _resp(400, {"ok": False, "error": "Body must be a JSON object"})

    # === NEW: 先嘗試更新狀態（summary/positions） ===
    state_updated = _maybe_update_state(payload)

    # === NEW: 若是交易事件，發 SNS 並視情況寫入 TradeEvents ===
    trade_dispatched = False
    try:
        norm = _normalize_trade(payload)
        if norm and publish_trade_event:
            publish_trade_event(**norm)
            trade_dispatched = True
            # filled/closed 視為有實際成交，寫 /last N 用
            if norm["status"] in ("filled", "closed") and append_trade_event:
                try:
                    append_trade_event(
                        symbol=norm["symbol"],
                        side=norm["side"],
                        quantity=norm["quantity"],
                        price=norm["price"],
                    )
                except Exception as e:
                    print("append_trade_event failed:", e)
    except Exception as e:
        print("publish_trade_event failed:", e)

    # === 原有流程：讓 ingest_core 做它既有的事（比如轉發到其他訂閱者） ===
    try:
        result = process_payload(payload)
        delivered = result.get("delivered")
        subscribers = result.get("subscribers")
    except Exception as e:
        # 即便 ingest_core 失敗，也回傳我們自己處理的結果
        delivered = trade_dispatched
        subscribers = -1
        print("process_payload failed:", e)

    return _resp(200, {
        "ok": True,
        "delivered": bool(delivered or trade_dispatched),
        "subscribers": subscribers,
        "summary_updated": state_updated["summary"],
        "positions_updated": state_updated["positions"],
        "trade_dispatched": trade_dispatched,
    })
