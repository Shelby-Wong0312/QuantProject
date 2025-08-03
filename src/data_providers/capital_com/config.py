"""
Capital.com API Configuration
"""
import os
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class CapitalComConfig:
    """Configuration for Capital.com API"""
    
    # API Endpoints
    DEMO_API_URL: str = "https://demo-api-capital.backend-capital.com"
    LIVE_API_URL: str = "https://api-capital.backend-capital.com"
    
    # Authentication
    API_KEY: str = os.getenv("CAPITAL_COM_API_KEY", "")
    IDENTIFIER: str = os.getenv("CAPITAL_COM_IDENTIFIER", "")
    PASSWORD: str = os.getenv("CAPITAL_COM_PASSWORD", "")
    
    # Trading Mode
    IS_DEMO: bool = os.getenv("CAPITAL_COM_DEMO_MODE", "True").lower() == "true"
    
    # Request Settings
    TIMEOUT: int = 30
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    
    # Data Settings
    MAX_BARS_PER_REQUEST: int = 1000
    DEFAULT_RESOLUTION: str = "HOUR"  # MINUTE, MINUTE_5, MINUTE_15, MINUTE_30, HOUR, HOUR_4, DAY, WEEK
    
    @property
    def base_url(self) -> str:
        """Get the appropriate base URL based on trading mode"""
        return self.DEMO_API_URL if self.IS_DEMO else self.LIVE_API_URL
    
    @property
    def headers(self) -> Dict[str, str]:
        """Get default headers for API requests"""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    def validate(self) -> None:
        """Validate configuration"""
        if not self.API_KEY:
            raise ValueError("CAPITAL_COM_API_KEY environment variable is not set")
        if not self.IDENTIFIER:
            raise ValueError("CAPITAL_COM_IDENTIFIER environment variable is not set")
        if not self.PASSWORD:
            raise ValueError("CAPITAL_COM_PASSWORD environment variable is not set")


# Create default configuration instance
config = CapitalComConfig()