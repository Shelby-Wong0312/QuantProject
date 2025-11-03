"""
Alpaca IP Allowlist Fix Guide
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()


def get_public_ip():
    """Get your public IP address"""
    try:
        response = requests.get("https://api.ipify.org?format=json", timeout=5)
        if response.status_code == 200:
            return response.json()["ip"]
    except Exception:
        pass

    try:
        response = requests.get("https://httpbin.org/ip", timeout=5)
        if response.status_code == 200:
            return response.json()["origin"]
    except Exception:
        pass

    return None


def test_alpaca():
    """Test Alpaca connection"""
    api_key = os.getenv("ALPACA_API_KEY_ID")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        return False, "Missing API credentials"

    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}

    try:
        response = requests.get(
            "https://paper-api.alpaca.markets/v2/account", headers=headers, timeout=10
        )

        if response.status_code == 200:
            account = response.json()
            return True, f"Connected! Cash: ${float(account.get('cash', 0)):,.2f}"
        else:
            return False, f"Error {response.status_code}: {response.text[:100]}"
    except Exception as e:
        return False, str(e)


def main():
    print("\n" + "=" * 80)
    print("ALPACA IP ALLOWLIST FIX")
    print("=" * 80)

    # Get IP
    ip = get_public_ip()
    if ip:
        print(f"\n[INFO] Your Public IP: {ip}")
    else:
        print("\n[WARNING] Could not get IP address")
        ip = "YOUR_IP"

    # Test Alpaca
    print("\n[TEST] Testing Alpaca connection...")
    success, message = test_alpaca()

    if success:
        print(f"[SUCCESS] {message}")
        print("\nYour Alpaca is working!")
    else:
        print(f"[FAILED] {message}")

        if "403" in message or "forbidden" in message:
            print("\n" + "=" * 60)
            print("HOW TO FIX IP ALLOWLIST")
            print("=" * 60)
            print("\n1. Go to: https://app.alpaca.markets/")
            print("2. Make sure you're in PAPER TRADING mode")
            print("3. Click on 'API Keys'")
            print("4. Find 'IP Allowlist (CIDR)' section")
            print("\n5. If it says 'Disabled':")
            print("   - Click to ENABLE it")
            print("   - Add one of these:")
            print("\n   OPTION A (Your IP only):")
            print(f"   {ip}/32")
            print("\n   OPTION B (Allow all - safe for paper trading):")
            print("   0.0.0.0/0")
            print("\n6. Click 'Save' or 'Update'")
            print("7. Wait 1-2 minutes")
            print("8. Run this script again")

            print("\n" + "=" * 60)
            print("WHY THIS HAPPENS")
            print("=" * 60)
            print("- IP Allowlist is a security feature")
            print("- When 'Disabled', it might block all requests")
            print("- You need to enable it and add allowed IPs")
            print("- For Paper Trading, allowing all IPs (0.0.0.0/0) is safe")

            print("\n" + "=" * 60)
            print("QUICK FIX")
            print("=" * 60)
            print("Just add: 0.0.0.0/0")
            print("This allows all IPs and is safe for Paper Trading!")


if __name__ == "__main__":
    main()
