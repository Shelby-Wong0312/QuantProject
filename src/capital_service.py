# src/capital_service.py

import requests
import logging
import time
import os
import json
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

CAPITAL_API_KEY = os.getenv("CAPITAL_API_KEY")
CAPITAL_IDENTIFIER = os.getenv("CAPITAL_IDENTIFIER")
CAPITAL_API_PASSWORD = os.getenv("CAPITAL_API_PASSWORD")
BASE_API_URL = os.getenv("CAPITAL_BASE_API_URL", "https://demo-api-capital.backend-capital.com/api/v1")

session_tokens = {
    "cst": None,
    "x_security_token": None,
    "expiry_time": 0  # Unix timestamp
}

def _get_headers(include_session_tokens=True):
    """輔助函數,產生請求標頭"""
    headers = {
        "X-CAP-API-KEY": CAPITAL_API_KEY,
        "Content-Type": "application/json"
    }
    if include_session_tokens:
        if not session_tokens["cst"] or not session_tokens["x_security_token"] or time.time() > session_tokens["expiry_time"]:
            logger.info("Session tokens are missing or expired. Attempting to log in.")
            if not login():
                logger.error("Failed to refresh session tokens during _get_headers.")
                raise Exception("Failed to refresh session tokens.")
        
        if session_tokens["cst"] and session_tokens["x_security_token"]:
            headers["CST"] = session_tokens["cst"]
            headers["X-SECURITY-TOKEN"] = session_tokens["x_security_token"]
        else:
            logger.error("Tokens are still unavailable after login attempt in _get_headers.")
            raise Exception("Session tokens are unavailable after login attempt.")
            
    return headers

def login():
    """執行登入操作以獲取CST和X-SECURITY-TOKEN"""
    global session_tokens
    login_url = f"{BASE_API_URL}/session"
    payload = {
        "identifier": CAPITAL_IDENTIFIER,
        "password": CAPITAL_API_PASSWORD,
        "encryptedPassword": False # 根據API文件,設為false
    }
    request_headers = {
        "X-CAP-API-KEY": CAPITAL_API_KEY,
        "Content-Type": "application/json"
    }

    try:
        logger.info(f"Attempting login to {login_url}")
        response = requests.post(login_url, headers=request_headers, json=payload)
        response.raise_for_status()

        cst_token = response.headers.get("CST")
        x_security_token_val = response.headers.get("X-SECURITY-TOKEN")

        if not cst_token or not x_security_token_val:
            logger.error(f"Login successful but CST or X-SECURITY-TOKEN missing in response headers. Headers: {response.headers}")
            return False

        session_tokens["cst"] = cst_token
        session_tokens["x_security_token"] = x_security_token_val
        session_tokens["expiry_time"] = time.time() + (9 * 60) # 假設 token 有效期為9分鐘
        logger.info(f"Login successful. CST and X-SECURITY-TOKEN obtained. Valid until: {time.ctime(session_tokens['expiry_time'])}")
        return True
    
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error during login: {http_err}. Status Code: {http_err.response.status_code if http_err.response else 'N/A'}. Response: {http_err.response.text if http_err.response else 'No response body'}")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Login failed due to RequestException: {e}. Response: {e.response.text if e.response else 'No response'}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during login: {e}")
        return False

def get_account_details():
    """獲取所有帳戶的詳細資訊"""
    try:
        logger.info("Fetching account details...")
        headers = _get_headers()
        url = f"{BASE_API_URL}/accounts"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        accounts_data = response.json()
        logger.info(f"Successfully fetched account details: {json.dumps(accounts_data, indent=2, ensure_ascii=False)}")
        
        if accounts_data and accounts_data.get("accounts"):
            for acc in accounts_data["accounts"]:
                logger.info(f"Account ID: {acc.get('accountId')}, Type: {acc.get('accountType')}, Balance: {acc.get('balance', {}).get('balance')}, P&L: {acc.get('balance', {}).get('profitAndLoss')}")
            return accounts_data["accounts"]
        else:
            logger.warning("No accounts found or unexpected response structure in account details.")
            return None
            
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error fetching account details: {http_err}. Status Code: {http_err.response.status_code if http_err.response else 'N/A'}. Response: {http_err.response.text if http_err.response else 'No response body'}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get account details due to RequestException: {e}. Response: {e.response.text if e.response else 'No response'}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching account details: {e}")
        return None

def get_market_details(epic):
    """獲取特定市場(epic)的詳細資訊"""
    try:
        logger.info(f"Fetching market details for epic: {epic}")
        headers = _get_headers()
        url = f"{BASE_API_URL}/markets/{epic}" # 請確認此端點是否存在且正確 (來自計畫書)
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        market_data = response.json()
        logger.info(f"Successfully fetched market details for {epic}: {json.dumps(market_data, indent=2, ensure_ascii=False)}")
        return market_data
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error fetching market details for {epic}: {http_err}. Status Code: {http_err.response.status_code if http_err.response else 'N/A'}. Response: {http_err.response.text if http_err.response else 'No response body'}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get market details for {epic}: {e}. Response: {e.response.text if e.response else 'No response'}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching market details for {epic}: {e}")
        return None

def place_market_order(epic, direction, size, stop_loss_price=None, take_profit_price=None, guaranteed_stop=False):
    """下市價單"""
    try:
        headers = _get_headers()
        url = f"{BASE_API_URL}/positions" # POST /positions 用於市價單
        payload = {
            "epic": epic,
            "direction": direction.upper(),
            "size": float(size),
            "guaranteedStop": guaranteed_stop
        }
        if stop_loss_price is not None: # 檢查是否為 None，因 0 可能是有效價格
            payload["stopLevel"] = float(stop_loss_price)
        if take_profit_price is not None: # 檢查是否為 None
            payload["profitLevel"] = float(take_profit_price)
        
        logger.info(f"Placing market order with payload: {payload}")
        response = requests.post(url, headers=headers, json=payload)
        response_data = response.json()
        logger.info(f"Place market order response status: {response.status_code}, data: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
        
        # Capital.com API 可能在 HTTP 200 OK 時仍在回應主體中包含業務邏輯錯誤
        # 需仔細檢查回應內容中的 dealReference 和可能的錯誤訊息
        if response.status_code == 200 and response_data.get("dealReference"):
            logger.info(f"Market order placed successfully for {epic}. Deal Reference: {response_data.get('dealReference')}")
            return {"success": True, "dealReference": response_data.get("dealReference"), "data": response_data}
        else:
            # 嘗試從回應中獲取更詳細的錯誤原因
            error_message = response_data.get("reason", response_data.get("message", "Unknown reason or error structure"))
            if "errorCode" in response_data: # 有些 API 會用 errorCode
                 error_message = f"{response_data.get('errorCode')}: {error_message}"
            logger.error(f"Market order for {epic} failed or was rejected. Status: {response.status_code}. Response: {response_data}")
            # 即使 HTTP 狀態碼不是 4xx/5xx，也可能因為業務邏輯拒絕
            if response.status_code >= 400:
                 response.raise_for_status() # 觸發 HTTPError 以便被下面的 except 捕獲
            return {"success": False, "message": error_message, "data": response_data}

    except requests.exceptions.HTTPError as http_err:
        error_response_text = http_err.response.text if http_err.response else "No response body"
        logger.error(f"HTTP error placing market order for {epic}: {http_err}. Response: {error_response_text}")
        try:
            error_data = http_err.response.json()
            return {"success": False, "message": error_data.get("errorCode", str(http_err)), "details": error_data, "data": error_data}
        except ValueError:
            return {"success": False, "message": str(http_err), "details": error_response_text, "data": None}
    except requests.exceptions.RequestException as e:
        logger.error(f"RequestException placing market order for {epic}: {e}. Response: {e.response.text if e.response else 'No response'}")
        return {"success": False, "message": str(e), "data": None}
    except Exception as e:
        logger.error(f"An unexpected error occurred placing market order for {epic}: {e}")
        return {"success": False, "message": f"Unexpected error: {str(e)}", "data": None}

# def place_limit_order(...): (計畫書中提到，但未提供完整範例)
#   logger.info("Placeholder for place_limit_order function")
#   pass

def get_open_positions():
    """獲取所有未平倉部位"""
    try:
        logger.info("Fetching open positions...")
        headers = _get_headers()
        url = f"{BASE_API_URL}/positions" # GET /positions
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        positions_data = response.json()
        logger.info(f"Successfully fetched open positions: {json.dumps(positions_data, indent=2, ensure_ascii=False)}")
        return positions_data.get("positions", []) # 返回空列表以防 "positions" 鍵不存在
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error fetching open positions: {http_err}. Status Code: {http_err.response.status_code if http_err.response else 'N/A'}. Response: {http_err.response.text if http_err.response else 'No response body'}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get open positions: {e}. Response: {e.response.text if e.response else 'No response'}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching open positions: {e}")
        return None

def close_position(deal_id): # 簡化參數，因為計畫書主要範例是透過 deal_id 刪除
    """平掉一個已存在的倉位 (透過 DELETE /positions/{dealId})"""
    try:
        logger.info(f"Attempting to close position with deal ID: {deal_id} via DELETE request.")
        headers = _get_headers()
        url = f"{BASE_API_URL}/positions/{deal_id}"
        
        # 計畫書範例暗示 DELETE /positions/{dealId} 可能不需要 body，或 body 內容需依最新文件確認
        # 這裡我們不傳遞 JSON payload 給 delete 方法，除非 API 明確要求
        response = requests.delete(url, headers=headers) 
        
        # 嘗試解析回應，即使是成功的 DELETE 也可能有回應主體
        try:
            response_data = response.json()
            logger.info(f"Close position (DELETE) response status: {response.status_code}, data: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
        except requests.exceptions.JSONDecodeError: # 如果沒有 JSON 回應 (例如 204 No Content)
            response_data = {"raw_text": response.text} # 記錄原始文本
            logger.info(f"Close position (DELETE) response status: {response.status_code}, no JSON body. Raw text: {response.text}")

        response.raise_for_status() # 檢查 HTTP 錯誤

        # 成功的 DELETE 可能返回 200 OK (帶有 dealReference) 或 204 No Content
        # 需要依據實際 API 行為調整成功判斷邏輯
        deal_reference = response_data.get("dealReference")
        if response.status_code == 200 and deal_reference:
            logger.info(f"Position {deal_id} closed successfully. Deal Reference: {deal_reference}")
            return {"success": True, "dealReference": deal_reference, "data": response_data}
        elif response.status_code == 204: # No Content 也算成功
             logger.info(f"Position {deal_id} closed successfully (Status 204 No Content).")
             return {"success": True, "message": "Closed with Status 204 No Content", "data": response_data}
        else: # 其他情況，或雖為 200 但無 dealReference
            logger.error(f"Position {deal_id} close (DELETE) might have failed or API behavior unexpected. Response: {response_data}")
            return {"success": False, "message": response_data.get("reason", "Unknown closure failure or unexpected API response"), "data": response_data}

    except requests.exceptions.HTTPError as http_err:
        error_response_text = http_err.response.text if http_err.response else "No response body"
        logger.error(f"HTTP error closing position {deal_id} (DELETE): {http_err}. Response: {error_response_text}")
        try:
            error_data = http_err.response.json()
            return {"success": False, "message": error_data.get("errorCode", str(http_err)), "details": error_data}
        except ValueError:
            return {"success": False, "message": str(http_err), "details": error_response_text}
    except requests.exceptions.RequestException as e:
        logger.error(f"RequestException closing position {deal_id} (DELETE): {e}. Response: {e.response.text if e.response else 'No response'}")
        return {"success": False, "message": str(e)}
    except Exception as e:
        logger.error(f"An unexpected error occurred closing position {deal_id} (DELETE): {e}")
        return {"success": False, "message": f"Unexpected error: {str(e)}"}


# if __name__ == '__main__':
#     # 設定基礎日誌以查看此模組的日誌輸出
#     logging.basicConfig(level=logging.INFO, 
#                         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                         handlers=[logging.StreamHandler()]) # 確保日誌輸出到控制台
#     logger.info("Attempting to test capital_service.py...")
#     
#     if login():
#         logger.info("Login test successful.")
#         accounts = get_account_details()
#         # epic_to_test = "EURUSD" # 請使用一個在您模擬帳戶中可用的EPIC
#         # market_info = get_market_details(epic_to_test)
#         # order_result = place_market_order(epic=epic_to_test, direction="BUY", size=0.01)
#         # open_positions = get_open_positions()
#         # if open_positions and len(open_positions) > 0:
#         #     deal_id_to_close = open_positions[0].get("position", {}).get("dealId")
#         #     if deal_id_to_close:
#         #         close_result = close_position(deal_id_to_close)
#     else:
#         logger.error("Login test failed.")
