import requests
import json

# --- 1. 請在此處填寫您的憑證 ---
# 將 YOUR_... 替換為您的真實資訊
API_KEY = "4prE28RZrTFIKzAx"  # 您最新的API金鑰
IDENTIFIER = "niujinheitaizi@gmail.com"     # 您的登入郵箱
PASSWORD = "@Nickatnyte3"        # 您的密碼

# --- 2. 交易設定 ---
EPIC = "BTCUSD"       # 交易商品 (比特幣/美元)
ORDER_SIZE = 0.1      # 下單數量
ORDER_DIRECTION = "BUY" # 交易方向 'BUY' 或 'SELL'

# --- 3. API 端點 ---
BASE_URL = "https://demo-api-capital.backend-capital.com/api/v1"

def place_btc_order():
    """
    連接到 Capital.com 並下單買入比特幣。
    """
    print("--- 開始執行下單程序 ---")

    # (A) 登入並獲取會話令牌 (Session Tokens)
    print("\n[步驟 1/3] 正在登入到 Capital.com...")
    session_url = f"{BASE_URL}/session"
    login_headers = {
        "X-CAP-API-KEY": API_KEY,
        "Content-Type": "application/json"
    }
    login_payload = {
        "identifier": IDENTIFIER,
        "password": PASSWORD
    }

    try:
        response = requests.post(session_url, headers=login_headers, json=login_payload, timeout=15)
        
        # 檢查登入是否成功
        if response.status_code != 200:
            print(f"❌ 登入失敗！ 狀態碼: {response.status_code}")
            print(f"伺服器回應: {response.text}")
            return

        # 從回應頭中提取會話令牌
        cst_token = response.headers.get("CST")
        x_security_token = response.headers.get("X-SECURITY-TOKEN")
        print("✅ 登入成功！已獲取會話令牌。")

    except requests.exceptions.RequestException as e:
        print(f"❌ 網路連線錯誤: {e}")
        return

    # (B) 準備並發送交易訂單
    print("\n[步驟 2/3] 準備發送交易訂單...")
    order_url = f"{BASE_URL}/positions"
    order_headers = {
        "X-CAP-API-KEY": API_KEY,
        "CST": cst_token,
        "X-SECURITY-TOKEN": x_security_token,
        "Content-Type": "application/json"
    }
    order_payload = {
        "epic": EPIC,
        "direction": ORDER_DIRECTION,
        "size": ORDER_SIZE
    }
    
    print(f"訂單詳情: {ORDER_DIRECTION} {ORDER_SIZE} {EPIC}")

    try:
        order_response = requests.post(order_url, headers=order_headers, json=order_payload, timeout=15)
        
        # (C) 打印最終結果
        print("\n[步驟 3/3] 收到券商最終回應:")
        print(f"狀態碼: {order_response.status_code}")
        
        # 美化輸出 JSON 格式
        try:
            response_json = order_response.json()
            print("伺服器回應:")
            print(json.dumps(response_json, indent=4))

            if order_response.status_code == 200 and response_json.get("dealReference"):
                print("\n🎉🎉🎉 恭喜！訂單成功送出！ 🎉🎉🎉")
            else:
                print("\n🔥🔥🔥 警告：訂單可能未成功，請檢查以上伺服器回應。 🔥🔥🔥")

        except json.JSONDecodeError:
            print(f"伺服器原始回應 (非JSON): {order_response.text}")

    except requests.exceptions.RequestException as e:
        print(f"❌ 下單時發生網路連線錯誤: {e}")


if __name__ == "__main__":
    place_btc_order()