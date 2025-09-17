import json

from common.ingest_core import process_payload
from common import db


def lambda_handler(event, context):
    delivered_total = 0
    subs_total = None
    records = event.get("Records", []) if isinstance(event, dict) else []
    for r in records:
        if r.get("EventSource") == "aws:sns" and "Sns" in r:
            message = r["Sns"].get("Message", "")
            try:
                payload = json.loads(message)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            ptype = str(payload.get("type") or "event").lower()
            # status payload example keys: equity, cash, unrealizedPnL, realizedPnL
            if ptype == "status" or any(k in payload for k in ("equity", "cash", "unrealizedPnL", "unrealized_pnl", "realizedPnL", "realized_pnl")):
                account_id = str(payload.get("accountId") or payload.get("account") or "default")
                status = {
                    "equity": payload.get("equity"),
                    "cash": payload.get("cash"),
                    "unrealizedPnL": payload.get("unrealizedPnL") or payload.get("unrealized_pnl") or payload.get("upnl"),
                    "realizedPnL": payload.get("realizedPnL") or payload.get("realized_pnl") or payload.get("rpnl"),
                    "accountId": account_id,
                }
                db.put_status(account_id, status)
                try:
                    db.put_system_summary(status)
                except Exception:
                    pass
                # Optional: if payload includes push=true, also broadcast latest status via process_payload
                if payload.get("push") is True:
                    # piggyback process_payload to push a line; formatter zh for consistency
                    # but do not count delivery towards totals to keep metrics focused on trade events
                    pass
            elif ptype == "positions" or (isinstance(payload, dict) and "positions" in payload):
                account_id = str(payload.get("accountId") or payload.get("account") or "default")
                db.put_positions(account_id, payload)
                try:
                    db.put_system_positions(payload)
                except Exception:
                    pass
            else:
                result = process_payload(payload, formatter="zh")
                delivered_total += result.get("delivered", 0)
                subs_total = result.get("subscribers", subs_total)

    return {
        "ok": True,
        "delivered": delivered_total,
        "subscribers": subs_total if subs_total is not None else 0,
    }
