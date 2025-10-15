"""
Capital.com API Connection Checker
Tests the Capital.com API integration status
"""

import requests
import json
import os
from datetime import datetime
import sys


def test_capital_api():
    """Test Capital.com API connectivity"""

    print("\n" + "=" * 60)
    print("CAPITAL.COM API CONNECTION TEST")
    print("=" * 60)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # API endpoints
    demo_url = "https://demo-api-capital.backend-capital.com"
    live_url = "https://api-capital.backend-capital.com"

    results = {"demo_api": False, "live_api": False, "websocket": False, "documentation": False}

    # Test 1: Demo API endpoint
    print("\n[Test 1] Checking Demo API endpoint...")
    try:
        response = requests.get(f"{demo_url}/api/v1/ping", timeout=5)
        if response.status_code == 200:
            print("[OK] Demo API is reachable")
            results["demo_api"] = True
        else:
            print(f"[FAIL] Demo API returned status: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("[FAIL] Cannot connect to Demo API")
    except requests.exceptions.Timeout:
        print("[FAIL] Demo API connection timeout")
    except Exception as e:
        print(f"[ERROR] {e}")

    # Test 2: Live API endpoint
    print("\n[Test 2] Checking Live API endpoint...")
    try:
        response = requests.get(f"{live_url}/api/v1/ping", timeout=5)
        if response.status_code == 200:
            print("[OK] Live API is reachable")
            results["live_api"] = True
        else:
            print(f"[FAIL] Live API returned status: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("[FAIL] Cannot connect to Live API")
    except requests.exceptions.Timeout:
        print("[FAIL] Live API connection timeout")
    except Exception as e:
        print(f"[ERROR] {e}")

    # Test 3: WebSocket endpoint
    print("\n[Test 3] Checking WebSocket endpoint...")
    ws_url = "wss://demo-api-streaming.backend-capital.com"
    try:
        # Just check if the URL is reachable via HTTPS
        https_url = ws_url.replace("wss://", "https://")
        response = requests.get(https_url, timeout=5)
        print("[INFO] WebSocket endpoint exists")
        results["websocket"] = True
    except:
        print("[INFO] WebSocket endpoint check skipped")

    # Test 4: Documentation availability
    print("\n[Test 4] Checking API documentation...")
    doc_url = "https://open-api.capital.com/"
    try:
        response = requests.get(doc_url, timeout=5)
        if response.status_code == 200:
            print("[OK] API documentation is accessible")
            results["documentation"] = True
        else:
            print(f"[FAIL] Documentation returned status: {response.status_code}")
    except:
        print("[FAIL] Cannot access documentation")

    # Check for API credentials
    print("\n[Test 5] Checking API credentials...")
    api_key = os.environ.get("CAPITAL_API_KEY")
    password = os.environ.get("CAPITAL_PASSWORD")
    identifier = os.environ.get("CAPITAL_IDENTIFIER")

    if api_key and password and identifier:
        print("[OK] API credentials found in environment")

        # Try authentication
        print("\n[Test 6] Testing authentication...")
        headers = {"Content-Type": "application/json", "X-CAP-API-KEY": api_key}

        auth_data = {"identifier": identifier, "password": password}

        try:
            response = requests.post(
                f"{demo_url}/api/v1/session", headers=headers, json=auth_data, timeout=10
            )

            if response.status_code == 200:
                print("[OK] Authentication successful")
                results["authentication"] = True

                # Get session tokens
                cst = response.headers.get("CST")
                token = response.headers.get("X-SECURITY-TOKEN")

                if cst and token:
                    print("[OK] Session tokens received")

                    # Test getting account info
                    print("\n[Test 7] Getting account information...")
                    headers.update({"CST": cst, "X-SECURITY-TOKEN": token})

                    acc_response = requests.get(
                        f"{demo_url}/api/v1/accounts", headers=headers, timeout=10
                    )

                    if acc_response.status_code == 200:
                        print("[OK] Account information retrieved")
                        accounts = acc_response.json().get("accounts", [])
                        for acc in accounts[:3]:  # Show first 3 accounts
                            print(f"  - Account: {acc.get('accountName', 'N/A')}")
                    else:
                        print(f"[FAIL] Could not get account info: {acc_response.status_code}")
            else:
                print(f"[FAIL] Authentication failed: {response.status_code}")
                if response.text:
                    error = json.loads(response.text)
                    print(f"  Error: {error.get('errorCode', 'Unknown')}")
                results["authentication"] = False

        except Exception as e:
            print(f"[ERROR] Authentication error: {e}")
            results["authentication"] = False
    else:
        print("[WARNING] No API credentials found")
        print("\nTo set up credentials, use:")
        print("  set CAPITAL_API_KEY=your_api_key")
        print("  set CAPITAL_PASSWORD=your_password")
        print("  set CAPITAL_IDENTIFIER=your_email")
        results["authentication"] = False

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    total = len(results)
    passed = sum(1 for v in results.values() if v)

    for test, status in results.items():
        status_text = "[PASS]" if status else "[FAIL]"
        print(f"{status_text} {test}")

    print(f"\nOverall: {passed}/{total} tests passed")

    # Save results
    with open("capital_api_status.json", "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "results": results,
                "summary": {"total": total, "passed": passed, "failed": total - passed},
            },
            f,
            indent=2,
        )

    print("\nResults saved to: capital_api_status.json")

    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    if not results.get("authentication", False):
        print("1. Register at https://capital.com")
        print("2. Go to Settings > API to generate API key")
        print("3. Set environment variables with your credentials")
        print("4. Use demo account for testing")
    else:
        print("[OK] API is ready for trading operations")

    print("\nUseful Resources:")
    print("- API Docs: https://open-api.capital.com/")
    print("- Support: https://help.capital.com/")
    print("- Status: https://status.capital.com/")

    return results


if __name__ == "__main__":
    try:
        results = test_capital_api()
        sys.exit(0 if results.get("demo_api", False) else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        sys.exit(1)
