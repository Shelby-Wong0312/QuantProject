#!/usr/bin/env python3
"""
Capital.com API connectivity check
- Loads credentials from environment/.env
- Attempts login against CAPITAL_BASE_URL (default demo)
- Prints a concise result without placing any orders
"""
import os
import sys
import requests
from dotenv import load_dotenv


def main() -> int:
    load_dotenv()

    base_url = os.getenv(
        "CAPITAL_BASE_URL",
        "https://demo-api-capital.backend-capital.com",
    ).rstrip("/")
    api_key = os.getenv("CAPITAL_API_KEY", "").strip()
    identifier = os.getenv("CAPITAL_IDENTIFIER", "").strip()
    password = os.getenv("CAPITAL_API_PASSWORD", "").strip()

    missing = [
        name
        for name, val in (
            ("CAPITAL_API_KEY", api_key),
            ("CAPITAL_IDENTIFIER", identifier),
            ("CAPITAL_API_PASSWORD", password),
        )
        if not val
    ]
    if missing:
        print(f"MISSING_CREDENTIALS: {', '.join(missing)}")
        return 2

    try:
        resp = requests.post(
            f"{base_url}/api/v1/session",
            headers={
                "X-CAP-API-KEY": api_key,
                "Content-Type": "application/json",
            },
            json={
                "identifier": identifier,
                "password": password,
            },
            timeout=12,
        )
    except Exception as e:
        print(f"NETWORK_ERROR: {e}")
        return 3

    if resp.status_code == 200:
        resp.headers.get("CST", "<hidden>")
        resp.headers.get("X-SECURITY-TOKEN", "<hidden>")
        print("OK: Capital.com connected (demo/live depends on CAPITAL_BASE_URL)")
        # Do not print tokens to stdout for safety.
        return 0
    else:
        txt = None
        try:
            txt = resp.text[:300]
        except Exception:
            txt = "<unavailable>"
        print(f"ERROR: HTTP {resp.status_code} - {txt}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
