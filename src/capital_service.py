# -*- coding: utf-8 -*-
"""
Capital.com Service Module
提供簡化的交易接口
"""

import os
import sys
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CapitalService:
    def __init__(self):
        self.api_key = os.getenv('CAPITAL_API_KEY', '').strip('"')
        self.identifier = os.getenv('CAPITAL_IDENTIFIER', '').strip('"')
        self.password = os.getenv('CAPITAL_API_PASSWORD', '').strip('"')
        self.demo_mode = os.getenv('CAPITAL_DEMO_MODE', 'True').lower() == 'true'
        
        if self.demo_mode:
            self.base_url = "https://demo-api-capital.backend-capital.com"
        else:
            self.base_url = "https://api-capital.backend-capital.com"
            
        self.cst = None
        self.x_security_token = None
        self.session = requests.Session()
        
    def login(self):
        """登入 Capital.com API"""
        login_url = f"{self.base_url}/api/v1/session"
        headers = {
            "X-CAP-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "identifier": self.identifier,
            "password": self.password,
            "encryptedPassword": False
        }
        
        try:
            response = self.session.post(login_url, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                self.cst = response.headers.get("CST")
                self.x_security_token = response.headers.get("X-SECURITY-TOKEN")
                return True, "登入成功"
            else:
                try:
                    error_data = response.json()
                    return False, f"登入失敗: {error_data.get('errorCode', 'Unknown error')}"
                except:
                    return False, f"登入失敗: Status {response.status_code}"
        except Exception as e:
            return False, f"登入時發生錯誤: {str(e)}"
    
    def place_order(self, epic, direction, size):
        """下市價單"""
        if not self.cst:
            success, msg = self.login()
            if not success:
                return {"success": False, "error": msg}
        
        order_url = f"{self.base_url}/api/v1/positions"
        headers = {
            "CST": self.cst,
            "X-SECURITY-TOKEN": self.x_security_token,
            "Content-Type": "application/json"
        }
        
        payload = {
            "epic": epic,
            "direction": direction,
            "size": size,
            "guaranteedStop": False,
            "trailingStop": False
        }
        
        try:
            response = self.session.post(order_url, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                error_data = response.json()
                return {"success": False, "error": error_data}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def search_stocks(self, search_term, limit=10):
        """搜尋股票/市場"""
        if not self.cst:
            success, msg = self.login()
            if not success:
                return []
        
        search_url = f"{self.base_url}/api/v1/markets"
        headers = {
            "CST": self.cst,
            "X-SECURITY-TOKEN": self.x_security_token,
            "Content-Type": "application/json"
        }
        params = {"searchTerm": search_term, "limit": limit}
        
        try:
            response = self.session.get(search_url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                return response.json().get('markets', [])
            else:
                print(f"Search failed: Status {response.status_code}")
                return []
        except Exception as e:
            print(f"Error searching stocks: {e}")
            return []
    
    def get_market_details(self, epic):
        """獲取市場詳細資訊"""
        if not self.cst:
            success, msg = self.login()
            if not success:
                return None
        
        detail_url = f"{self.base_url}/api/v1/markets/{epic}"
        headers = {
            "CST": self.cst,
            "X-SECURITY-TOKEN": self.x_security_token,
            "Content-Type": "application/json"
        }
        
        try:
            response = self.session.get(detail_url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Details failed: Status {response.status_code}")
                return None
        except Exception as e:
            print(f"Error getting market details: {e}")
            return None

# Global service instance
_service = None

def place_market_order(epic, direction, size):
    """簡化的下單接口"""
    global _service
    if _service is None:
        _service = CapitalService()
    
    return _service.place_order(epic, direction, size)