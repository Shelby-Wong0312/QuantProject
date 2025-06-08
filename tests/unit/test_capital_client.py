# 檔案位置: tests/unit/test_capital_client.py

import unittest
from unittest.mock import patch, Mock
import os
import time
import requests

from execution.capital_client import CapitalComClient

class TestCapitalComClient(unittest.TestCase):

    def setUp(self):
        """
        在每個測試方法執行前設定環境。
        """
        self.mock_env = {
            "CAPITAL_API_KEY": "test_api_key",
            "CAPITAL_IDENTIFIER": "test_identifier",
            "CAPITAL_API_PASSWORD": "test_password"
        }
        self.env_patcher = patch.dict('os.environ', self.mock_env)
        self.env_patcher.start()
        self.client = CapitalComClient()

    def tearDown(self):
        """
        在每個測試方法執行後清理環境。
        """
        self.env_patcher.stop()

    @patch('execution.capital_client.requests.post')
    def test_login_success(self, mock_post):
        """
        測試成功登入的情境。
        """
        # 1. 準備：設定 mock 物件
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {
            'CST': 'mock_cst_token',
            'X-SECURITY-TOKEN': 'mock_security_token'
        }
        mock_response.raise_for_status.return_value = None
        
        mock_post.return_value = mock_response

        # 2. 執行：調用要測試的方法
        result = self.client.login()

        # 3. 斷言：驗證結果是否符合預期
        self.assertTrue(result)
        self.assertEqual(self.client.cst, 'mock_cst_token')
        self.assertEqual(self.client.x_security_token, 'mock_security_token')
        self.assertTrue(self.client._is_session_active())
        
        # 驗證 requests.post 是否以正確的參數被呼叫
        expected_url = f"{self.client.base_url}/session"
        
        # --- 以下是修正的部分 ---
        # 將 expected_url 作為位置參數傳入，以匹配實際的呼叫方式
        mock_post.assert_called_once_with(
            expected_url,  # <-- 修正點：從 url=expected_url 改為此處
            headers={"X-CAP-API-KEY": "test_api_key", "Content-Type": "application/json"},
            json={
                "identifier": "test_identifier",
                "password": "test_password",
                "encryptedPassword": False
            },
            timeout=10
        )

    @patch('execution.capital_client.requests.post')
    def test_login_failure_http_error(self, mock_post):
        """
        測試因 HTTP 錯誤導致登入失敗的情境。
        """
        # 1. 準備
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = '{"errorCode": "unauthorized"}'
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)

        mock_post.return_value = mock_response

        # 2. 執行
        result = self.client.login()

        # 3. 斷言
        self.assertFalse(result)
        self.assertIsNone(self.client.cst)
        self.assertIsNone(self.client.x_security_token)
        self.assertFalse(self.client._is_session_active())
        