# 檔案位置: execution/capital_client.py

import os
import requests
import logging
import time
from dotenv import load_dotenv

# 載入 .env 檔案中的環境變數到當前的執行環境中
load_dotenv()

logger = logging.getLogger(__name__)

class CapitalComClient:
    """
    用於與 Capital.com API 進行互動的客戶端類別。
    負責處理認證、會話管理以及發送交易請求。
    """
    def __init__(self):
        """
        初始化客戶端，從環境變數中讀取 API 憑證。
        """
        self.api_key = os.getenv("CAPITAL_API_KEY")
        self.identifier = os.getenv("CAPITAL_IDENTIFIER")
        self.api_password = os.getenv("CAPITAL_API_PASSWORD")
        self.base_url = os.getenv("CAPITAL_BASE_API_URL", "https://demo-api-capital.backend-capital.com/api/v1")

        if not all([self.api_key, self.identifier, self.api_password]):
            raise ValueError("Capital.com API 憑證未在 .env 檔案中完整設定。")

        self.cst = None
        self.x_security_token = None
        self.session_expiry_time = 0
        self.session_duration_seconds = 9 * 60

    def _is_session_active(self):
        """
        檢查當前的會話權杖是否仍然有效。
        """
        return self.cst and self.x_security_token and time.time() < self.session_expiry_time

    def _get_headers(self, include_session_tokens=True):
        """
        構建請求所需的 headers。如果需要，會自動包含會話權杖。
        如果會話過期，會嘗試重新登入。
        """
        headers = {
            "X-CAP-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        if include_session_tokens:
            if not self._is_session_active():
                logger.info("會話已過期或未啟用。正在嘗試重新登入...")
                if not self.login():
                    raise ConnectionError("無法刷新 Capital.com 會話權杖。")
            headers["CST"] = self.cst
            headers["X-SECURITY-TOKEN"] = self.x_security_token
        return headers

    def login(self):
        """
        登入到 Capital.com API 並獲取會話權杖。
        """
        login_url = f"{self.base_url}/session"
        payload = {
            "identifier": self.identifier,
            "password": self.api_password,
            "encryptedPassword": False
        }
        headers = {"X-CAP-API-KEY": self.api_key, "Content-Type": "application/json"}

        try:
            response = requests.post(login_url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()

            self.cst = response.headers.get("CST")
            self.x_security_token = response.headers.get("X-SECURITY-TOKEN")

            if not self.cst or not self.x_security_token:
                logger.error(f"登入成功但缺少 CST 或 X-SECURITY-TOKEN。 Headers: {response.headers}")
                return False

            self.session_expiry_time = time.time() + self.session_duration_seconds
            logger.info(f"Capital.com 登入成功。會話有效期至: {time.ctime(self.session_expiry_time)}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Capital.com 登入失敗: {e}")
            if e.response is not None:
                logger.error(f"登入失敗詳情: Status: {e.response.status_code}, Body: {e.response.text}")
            return False
        
    def get_account_details(self):
        """
        獲取帳戶詳細資訊。
        """
        if not self._is_session_active() and not self.login():
            return None
        
        url = f"{self.base_url}/accounts"
        try:
            headers = self._get_headers()
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            account_data = response.json()
            logger.info("成功獲取帳戶資訊。")
            return account_data
        except requests.exceptions.RequestException as e:
            logger.error(f"獲取帳戶資訊失敗: {e}. 回應: {e.response.text if e.response else 'No response'}")
            return None

    def place_market_order(self, epic: str, direction: str, size: float, stop_loss_price=None, take_profit_price=None, guaranteed_stop=False):
        """
        下達市價單。
        """
        if not self._is_session_active() and not self.login():
            return {"success": False, "message": "建立下單會話失敗。"}

        url = f"{self.base_url}/positions"
        payload = {
            "epic": epic,
            "direction": direction.upper(),
            "size": float(size),
            "guaranteedStop": guaranteed_stop
        }
        if stop_loss_price is not None:
            payload["stopLevel"] = float(stop_loss_price)
        if take_profit_price is not None:
            payload["profitLevel"] = float(take_profit_price)

        try:
            headers = self._get_headers()
            logger.info(f"正在下達市價單至 {url}，payload: {payload}")
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            response_data = response.json()
            logger.info(f"下單 API 回應狀態: {response.status_code}, data: {response_data}")
            response.raise_for_status()

            deal_reference = response_data.get("dealReference")
            if deal_reference:
                return {"success": True, "dealReference": deal_reference, "data": response_data}
            else:
                logger.error(f"對 {epic} 的市價單可能被 Capital.com 拒絕。回應: {response_data}")
                return {"success": False, "message": response_data.get("reason", "未知的拒絕原因"), "data": response_data}

        except requests.exceptions.RequestException as e:
            logger.error(f"下單時發生錯誤: {e}", exc_info=True)
            error_body = e.response.text if e.response else "No response body"
            return {"success": False, "message": str(e), "details": error_body}

    def get_open_positions(self):
        """
        獲取所有未平倉的部位。
        """
        if not self._is_session_active() and not self.login():
            return None
        
        url = f"{self.base_url}/positions"
        try:
            headers = self._get_headers()
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            positions_data = response.json()
            logger.info("成功獲取當前倉位。")
            return positions_data
        except requests.exceptions.RequestException as e:
            logger.error(f"獲取倉位失敗: {e}. 回應: {e.response.text if e.response else 'No response'}")
            return None

    def close_position(self, deal_id: str):
        """
        根據 dealId 平掉一個指定的倉位。
        """
        if not self._is_session_active() and not self.login():
            return {"success": False, "message": "建立平倉會話失敗。"}

        url = f"{self.base_url}/positions/{deal_id}"
        
        payload = {}

        try:
            headers = self._get_headers()
            logger.info(f"正在送出平倉請求至 {url}")
            response = requests.delete(url, headers=headers, timeout=15)
            response_data = response.json()
            logger.info(f"平倉 API 回應狀態: {response.status_code}, data: {response_data}")
            response.raise_for_status()
            
            deal_reference = response_data.get("dealReference")
            if deal_reference:
                return {"success": True, "dealReference": deal_reference, "data": response_data}
            else:
                logger.error(f"平倉 dealId {deal_id} 可能被拒絕。回應: {response_data}")
                return {"success": False, "message": response_data.get("reason", "未知的拒絕原因"), "data": response_data}

        except requests.exceptions.RequestException as e:
            logger.error(f"平倉時發生錯誤: {e}", exc_info=True)
            error_body = e.response.text if e.response else "No response body"
            return {"success": False, "message": str(e), "details": error_body}
        