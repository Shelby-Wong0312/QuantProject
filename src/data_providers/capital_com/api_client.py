"""
Capital.com API Client
"""

import requests
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import logging
from enum import Enum

from .auth import CapitalComAuth
from .config import config


logger = logging.getLogger(__name__)


class Resolution(Enum):
    """Time resolution for market data"""

    MINUTE = "MINUTE"
    MINUTE_5 = "MINUTE_5"
    MINUTE_15 = "MINUTE_15"
    MINUTE_30 = "MINUTE_30"
    HOUR = "HOUR"
    HOUR_4 = "HOUR_4"
    DAY = "DAY"
    WEEK = "WEEK"


class CapitalComClient:
    """Main client for interacting with Capital.com API"""

    def __init__(self, custom_config: Optional[Any] = None):
        """
        Initialize Capital.com API client

        Args:
            custom_config: Optional custom configuration object
        """
        self.config = custom_config or config
        self.auth = CapitalComAuth(self.config)
        self._session = requests.Session()

    def connect(self) -> bool:
        """
        Establish connection to Capital.com API

        Returns:
            bool: True if connection successful, False otherwise
        """
        return self.auth.authenticate()

    def disconnect(self) -> bool:
        """
        Disconnect from Capital.com API

        Returns:
            bool: True if disconnection successful, False otherwise
        """
        return self.auth.logout()

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Make authenticated request to API

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (without base URL)
            **kwargs: Additional arguments for requests

        Returns:
            Optional[Dict[str, Any]]: Response data or None if error
        """
        if not self.auth.is_authenticated():
            logger.error("Not authenticated. Call connect() first.")
            return None

        url = f"{self.config.base_url}{endpoint}"
        headers = self.auth.get_authenticated_headers()

        # Merge any additional headers
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))

        # Set default timeout if not provided
        if "timeout" not in kwargs:
            kwargs["timeout"] = self.config.TIMEOUT

        try:
            response = self._session.request(method, url, headers=headers, **kwargs)

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Request failed with status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            return None

    def get_accounts(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get all accounts

        Returns:
            Optional[List[Dict[str, Any]]]: List of accounts or None if error
        """
        result = self._make_request("GET", "/api/v1/accounts")
        return result.get("accounts") if result else None

    def search_markets(self, search_term: str, limit: int = 50) -> Optional[List[Dict[str, Any]]]:
        """
        Search for markets

        Args:
            search_term: Search term (e.g., "AAPL", "EUR/USD")
            limit: Maximum number of results

        Returns:
            Optional[List[Dict[str, Any]]]: List of markets or None if error
        """
        params = {"searchTerm": search_term, "limit": limit}

        result = self._make_request("GET", "/api/v1/markets", params=params)
        return result.get("markets") if result else None

    def get_market_details(self, epic: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific market

        Args:
            epic: Market epic (identifier)

        Returns:
            Optional[Dict[str, Any]]: Market details or None if error
        """
        return self._make_request("GET", f"/api/v1/markets/{epic}")

    def get_historical_prices(
        self,
        epic: str,
        resolution: Union[str, Resolution],
        max_bars: Optional[int] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get historical price data

        Args:
            epic: Market epic (identifier)
            resolution: Time resolution (Resolution enum or string)
            max_bars: Maximum number of bars to retrieve
            from_date: Start date for historical data
            to_date: End date for historical data

        Returns:
            Optional[Dict[str, Any]]: Historical price data or None if error
        """
        # Convert resolution to string if enum
        if isinstance(resolution, Resolution):
            resolution = resolution.value

        # Build query parameters
        params = {"resolution": resolution}

        if max_bars:
            params["max"] = min(max_bars, self.config.MAX_BARS_PER_REQUEST)

        if from_date:
            params["from"] = from_date.strftime("%Y-%m-%dT%H:%M:%S")

        if to_date:
            params["to"] = to_date.strftime("%Y-%m-%dT%H:%M:%S")

        return self._make_request("GET", f"/api/v1/prices/{epic}", params=params)

    def get_positions(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get all open positions

        Returns:
            Optional[List[Dict[str, Any]]]: List of positions or None if error
        """
        result = self._make_request("GET", "/api/v1/positions")
        return result.get("positions") if result else None

    def create_position(
        self,
        epic: str,
        direction: str,  # "BUY" or "SELL"
        size: float,
        stop_level: Optional[float] = None,
        profit_level: Optional[float] = None,
        guaranteed_stop: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new position (place a trade)

        Args:
            epic: Market epic (identifier)
            direction: Trade direction ("BUY" or "SELL")
            size: Position size
            stop_level: Stop loss level
            profit_level: Take profit level
            guaranteed_stop: Whether to use guaranteed stop

        Returns:
            Optional[Dict[str, Any]]: Trade confirmation or None if error
        """
        payload = {
            "epic": epic,
            "direction": direction,
            "size": size,
            "guaranteedStop": guaranteed_stop,
        }

        if stop_level is not None:
            payload["stopLevel"] = stop_level

        if profit_level is not None:
            payload["profitLevel"] = profit_level

        return self._make_request("POST", "/api/v1/positions", json=payload)

    def close_position(self, deal_id: str) -> Optional[Dict[str, Any]]:
        """
        Close an existing position

        Args:
            deal_id: Deal identifier

        Returns:
            Optional[Dict[str, Any]]: Close confirmation or None if error
        """
        return self._make_request("DELETE", f"/api/v1/positions/{deal_id}")

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
