"""
檢查你的公網IP地址
用於設置Alpaca IP Allowlist
"""

import requests


def get_my_ip():
    """獲取公網IP地址"""
    print("\n" + "=" * 60)
    print("檢查你的公網IP地址")
    print("=" * 60)

    # 方法1：使用 ipify
    try:
        response = requests.get("https://api.ipify.org?format=json", timeout=5)
        if response.status_code == 200:
            ip = response.json()["ip"]
            print(f"\n[OK] Your Public IP: {ip}")
            print(f"\n在Alpaca設置中添加: {ip}/32")

            # 檢查IP類型
            if ":" in ip:
                print("   (這是IPv6地址)")
            else:
                print("   (這是IPv4地址)")

            return ip
    except Exception as e:
        print(f"方法1失敗: {e}")

    # 方法2：使用 httpbin
    try:
        response = requests.get("https://httpbin.org/ip", timeout=5)
        if response.status_code == 200:
            ip = response.json()["origin"]
            print(f"\n[OK] Your Public IP: {ip}")
            return ip
    except Exception as e:
        print(f"方法2失敗: {e}")

    # 方法3：使用 ipinfo
    try:
        response = requests.get("https://ipinfo.io/json", timeout=5)
        if response.status_code == 200:
            response.json()
            ip = data["ip"]
            print(f"\n[OK] Your Public IP: {ip}")
            print(f"   位置: {data.get('city', '')}, {data.get('country', '')}")
            print(f"   ISP: {data.get('org', '')}")
            return ip
    except Exception as e:
        print(f"方法3失敗: {e}")

    return None


def test_alpaca_after_fix():
    """測試Alpaca連接（修復IP後）"""
    print("\n" + "=" * 60)
    print("測試Alpaca連接")
    print("=" * 60)

    import os
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("ALPACA_API_KEY_ID")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        print("❌ 缺少API憑證")
        return

    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}

    try:
        response = requests.get(
            "https://paper-api.alpaca.markets/v2/account", headers=headers, timeout=10
        )

        if response.status_code == 200:
            print("✅ Alpaca連接成功！")
            account = response.json()
            print(f"   帳戶狀態: {account.get('status')}")
            print(f"   現金: ${float(account.get('cash', 0)):,.2f}")
        else:
            print(f"❌ 連接失敗: {response.status_code}")
            print(f"   {response.text}")

            if response.status_code == 403:
                print("\n⚠️ 仍然是403錯誤，請確認：")
                print("   1. IP Allowlist已經正確設置")
                print("   2. 等待1-2分鐘讓設置生效")
                print("   3. 確保使用Paper Trading的API密鑰")

    except Exception as e:
        print(f"❌ 錯誤: {e}")


def main():
    print("\n" + "=" * 80)
    print("ALPACA IP ALLOWLIST FIX GUIDE")
    print("=" * 80)

    # 獲取IP
    ip = get_my_ip()

    if ip:
        print("\n" + "=" * 60)
        print("📝 修復步驟")
        print("=" * 60)
        print("\n1. 登入 https://app.alpaca.markets/")
        print("2. 確保在 Paper Trading 模式")
        print("3. 進入 API Keys 頁面")
        print("4. 找到 IP Allowlist (CIDR) 設置")
        print("5. 選擇以下其中一個選項：")
        print("\n   選項A（推薦用於測試）:")
        print("   - Enable IP Allowlist")
        print("   ✅ 添加: 0.0.0.0/0")
        print("      (這會允許所有IP，Paper Trading安全)")
        print("\n   選項B（更安全）:")
        print("   - Enable IP Allowlist")
        print(f"   - Add: {ip}/32")
        print("      (只允許你當前的IP)")
        print("\n6. 保存設置")
        print("7. 等待1-2分鐘")
        print("8. 再次運行這個腳本測試")

        print("\n" + "=" * 60)
        print("💡 重要提示")
        print("=" * 60)
        print("• 如果你的IP是動態的（會變化），使用 0.0.0.0/0")
        print("• Paper Trading使用 0.0.0.0/0 是安全的")
        print("• Live Trading建議使用特定IP")
        print("• 設置改變後可能需要等待幾分鐘生效")

        # 測試連接
        input("\n按Enter測試Alpaca連接...")
        test_alpaca_after_fix()
    else:
        print("\n❌ 無法獲取IP地址")


if __name__ == "__main__":
    main()
