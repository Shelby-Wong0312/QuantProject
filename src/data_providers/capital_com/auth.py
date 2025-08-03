"""
Capital.com Authentication Module
"""
import requests
import json
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from .config import config


logger = logging.getLogger(__name__)


class CapitalComAuth:
    """Handle authentication with Capital.com API"""
    
    def __init__(self, custom_config: Optional[Any] = None):
        """
        Initialize authentication handler
        
        Args:
            custom_config: Optional custom configuration object
        """
        self.config = custom_config or config
        self.session_token: Optional[str] = None
        self.cst_token: Optional[str] = None
        self.token_expiry: Optional[datetime] = None
        self._session = requests.Session()
        
    def is_authenticated(self) -> bool:
        """Check if current authentication is valid"""
        if not self.session_token or not self.cst_token:
            return False
        
        if self.token_expiry and datetime.now() >= self.token_expiry:
            logger.warning("Authentication token has expired")
            return False
            
        return True
    
    def authenticate(self) -> bool:
        """
        Authenticate with Capital.com API
        
        Returns:
            bool: True if authentication successful, False otherwise
        """
        try:
            # Validate configuration
            self.config.validate()
            
            # Prepare authentication payload
            auth_payload = {
                "identifier": self.config.IDENTIFIER,
                "password": self.config.PASSWORD
            }
            
            # Add API key to headers
            headers = self.config.headers.copy()
            headers["X-CAP-API-KEY"] = self.config.API_KEY
            
            # Make authentication request
            url = f"{self.config.base_url}/api/v1/session"
            response = self._session.post(
                url,
                json=auth_payload,
                headers=headers,
                timeout=self.config.TIMEOUT
            )
            
            # Check response
            if response.status_code == 200:
                # Extract tokens from headers
                self.cst_token = response.headers.get("CST")
                self.session_token = response.headers.get("X-SECURITY-TOKEN")
                
                if self.cst_token and self.session_token:
                    # Set token expiry (Capital.com tokens typically last 10 hours)
                    self.token_expiry = datetime.now() + timedelta(hours=10)
                    
                    logger.info("Successfully authenticated with Capital.com API")
                    return True
                else:
                    logger.error("Authentication response missing required tokens")
                    return False
            else:
                logger.error(f"Authentication failed with status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error during authentication: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during authentication: {str(e)}")
            return False
    
    def get_authenticated_headers(self) -> Dict[str, str]:
        """
        Get headers with authentication tokens
        
        Returns:
            Dict[str, str]: Headers with authentication tokens
        """
        if not self.is_authenticated():
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        
        headers = self.config.headers.copy()
        headers["CST"] = self.cst_token
        headers["X-SECURITY-TOKEN"] = self.session_token
        
        return headers
    
    def logout(self) -> bool:
        """
        Logout from Capital.com API
        
        Returns:
            bool: True if logout successful, False otherwise
        """
        if not self.is_authenticated():
            logger.warning("Cannot logout - not authenticated")
            return False
        
        try:
            url = f"{self.config.base_url}/api/v1/session"
            headers = self.get_authenticated_headers()
            
            response = self._session.delete(
                url,
                headers=headers,
                timeout=self.config.TIMEOUT
            )
            
            if response.status_code == 200:
                # Clear authentication tokens
                self.session_token = None
                self.cst_token = None
                self.token_expiry = None
                
                logger.info("Successfully logged out from Capital.com API")
                return True
            else:
                logger.error(f"Logout failed with status code: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error during logout: {str(e)}")
            return False
    
    def __enter__(self):
        """Context manager entry"""
        self.authenticate()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.logout()