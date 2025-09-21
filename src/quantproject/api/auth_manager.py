"""
Authentication Manager with Encryption
Secure credential management for Capital.com API
Cloud DE - Task DE-401
"""

import os
import json
import base64
from pathlib import Path
from typing import Dict, Optional
from cryptography.fernet import Fernet
import hashlib
import logging

logger = logging.getLogger(__name__)


class AuthManager:
    """
    Secure authentication manager for API credentials
    Implements encryption for sensitive data storage
    """
    
    def __init__(self, key_file: str = "config/.secret_key"):
        """
        Initialize authentication manager
        
        Args:
            key_file: Path to encryption key file
        """
        self.key_file = Path(key_file)
        self.key = self._get_or_create_key()
        self.cipher = Fernet(self.key)
        self.credentials_cache = {}
        
    def _get_or_create_key(self) -> bytes:
        """Get existing encryption key or create new one"""
        if self.key_file.exists():
            with open(self.key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            
            # Create directory if needed
            self.key_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save key securely
            with open(self.key_file, 'wb') as f:
                f.write(key)
            
            # Set restrictive permissions (Windows compatible)
            if os.name != 'nt':
                os.chmod(self.key_file, 0o600)
            
            logger.info(f"Generated new encryption key at {self.key_file}")
            return key
    
    def _derive_key_from_password(self, password: str, salt: bytes = None) -> bytes:
        """
        Derive encryption key from password using hashlib
        
        Args:
            password: Master password
            salt: Salt for key derivation
            
        Returns:
            Derived encryption key
        """
        if salt is None:
            salt = os.urandom(16)
        
        # Use hashlib for key derivation
        key_material = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt,
            100000,
            dklen=32
        )
        
        key = base64.urlsafe_b64encode(key_material)
        return key
    
    def encrypt_credentials(self, credentials: Dict) -> bytes:
        """
        Encrypt API credentials
        
        Args:
            credentials: Dictionary containing API credentials
            
        Returns:
            Encrypted credentials as bytes
        """
        json_str = json.dumps(credentials)
        encrypted = self.cipher.encrypt(json_str.encode())
        
        logger.debug("Credentials encrypted successfully")
        return encrypted
    
    def decrypt_credentials(self, encrypted: bytes) -> Dict:
        """
        Decrypt API credentials
        
        Args:
            encrypted: Encrypted credentials
            
        Returns:
            Decrypted credentials dictionary
        """
        try:
            decrypted = self.cipher.decrypt(encrypted)
            credentials = json.loads(decrypted.decode())
            
            logger.debug("Credentials decrypted successfully")
            return credentials
        except Exception as e:
            logger.error(f"Failed to decrypt credentials: {e}")
            raise ValueError("Invalid encryption key or corrupted data")
    
    def save_encrypted_credentials(self, filepath: str, credentials: Dict):
        """
        Save encrypted credentials to file
        
        Args:
            filepath: Path to save encrypted credentials
            credentials: Credentials dictionary to encrypt and save
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate credentials before saving
        self._validate_credentials(credentials)
        
        # Encrypt and save
        encrypted = self.encrypt_credentials(credentials)
        
        with open(filepath, 'wb') as f:
            f.write(encrypted)
        
        # Set restrictive permissions (Windows compatible)
        if os.name != 'nt':
            os.chmod(filepath, 0o600)
        
        logger.info(f"Encrypted credentials saved to {filepath}")
    
    def load_encrypted_credentials(self, filepath: str) -> Dict:
        """
        Load and decrypt credentials from file
        
        Args:
            filepath: Path to encrypted credentials file
            
        Returns:
            Decrypted credentials dictionary
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Credentials file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            encrypted = f.read()
        
        credentials = self.decrypt_credentials(encrypted)
        
        # Cache credentials in memory
        self.credentials_cache = credentials
        
        logger.info(f"Credentials loaded from {filepath}")
        return credentials
    
    def _validate_credentials(self, credentials: Dict):
        """
        Validate credential structure
        
        Args:
            credentials: Credentials to validate
            
        Raises:
            ValueError: If credentials are invalid
        """
        if 'capital_com' not in credentials:
            raise ValueError("Missing 'capital_com' section in credentials")
        
        required_fields = ['api_key', 'api_secret', 'environment']
        capital_creds = credentials['capital_com']
        
        for field in required_fields:
            if field not in capital_creds:
                raise ValueError(f"Missing required field: {field}")
            
            # Check for placeholder values
            if capital_creds[field].startswith("YOUR_"):
                raise ValueError(f"Please replace placeholder value for: {field}")
    
    def get_api_credentials(self) -> Dict:
        """
        Get API credentials from cache or environment
        
        Returns:
            API credentials dictionary
        """
        # Check cache first
        if self.credentials_cache:
            return self.credentials_cache
        
        # Check environment variables
        env_creds = self._load_from_environment()
        if env_creds:
            self.credentials_cache = env_creds
            return env_creds
        
        # Try loading from default encrypted file
        try:
            return self.load_encrypted_credentials("config/api_credentials.enc")
        except FileNotFoundError:
            logger.warning("No credentials found. Please configure API credentials.")
            return {}
    
    def _load_from_environment(self) -> Optional[Dict]:
        """
        Load credentials from environment variables
        
        Returns:
            Credentials dictionary or None
        """
        api_key = os.getenv('CAPITAL_API_KEY')
        api_secret = os.getenv('CAPITAL_API_SECRET')
        
        if api_key and api_secret:
            return {
                'capital_com': {
                    'api_key': api_key,
                    'api_secret': api_secret,
                    'environment': os.getenv('CAPITAL_ENV', 'demo'),
                    'account_id': os.getenv('CAPITAL_ACCOUNT_ID', ''),
                    'base_url': os.getenv(
                        'CAPITAL_BASE_URL',
                        'https://demo-api-capital.backend-capital.com/api/v1'
                    )
                }
            }
        
        return None
    
    def rotate_encryption_key(self, new_key: bytes = None):
        """
        Rotate encryption key and re-encrypt all credentials
        
        Args:
            new_key: New encryption key (generated if not provided)
        """
        # Decrypt with old key
        old_credentials = self.credentials_cache.copy()
        
        # Generate new key if not provided
        if new_key is None:
            new_key = Fernet.generate_key()
        
        # Update cipher with new key
        self.key = new_key
        self.cipher = Fernet(new_key)
        
        # Save new key
        with open(self.key_file, 'wb') as f:
            f.write(new_key)
        
        # Re-encrypt credentials with new key
        if old_credentials:
            self.save_encrypted_credentials(
                "config/api_credentials.enc",
                old_credentials
            )
        
        logger.info("Encryption key rotated successfully")
    
    def clear_cache(self):
        """Clear credentials from memory cache"""
        self.credentials_cache = {}
        logger.debug("Credentials cache cleared")


class TokenManager:
    """
    Manage OAuth 2.0 tokens for Capital.com API
    """
    
    def __init__(self, auth_manager: AuthManager):
        """
        Initialize token manager
        
        Args:
            auth_manager: Authentication manager instance
        """
        self.auth_manager = auth_manager
        self.access_token = None
        self.refresh_token = None
        self.token_expiry = None
        
    def is_token_valid(self) -> bool:
        """
        Check if current access token is valid
        
        Returns:
            True if token is valid, False otherwise
        """
        if not self.access_token:
            return False
        
        if self.token_expiry:
            from datetime import datetime
            return datetime.now() < self.token_expiry
        
        return True
    
    async def refresh_access_token(self) -> str:
        """
        Refresh access token using refresh token
        
        Returns:
            New access token
        """
        if not self.refresh_token:
            raise ValueError("No refresh token available")
        
        # TODO: Implement actual token refresh with Capital.com API
        logger.info("Refreshing access token")
        
        # Placeholder for actual implementation
        self.access_token = "new_access_token"
        
        return self.access_token
    
    def save_tokens(self, access_token: str, refresh_token: str = None, expiry: int = 3600):
        """
        Save tokens to secure storage
        
        Args:
            access_token: OAuth access token
            refresh_token: OAuth refresh token
            expiry: Token expiry time in seconds
        """
        from datetime import datetime, timedelta
        
        self.access_token = access_token
        
        if refresh_token:
            self.refresh_token = refresh_token
        
        self.token_expiry = datetime.now() + timedelta(seconds=expiry)
        
        # Save to encrypted storage
        token_data = {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'expiry': self.token_expiry.isoformat()
        }
        
        encrypted = self.auth_manager.encrypt_credentials(token_data)
        
        with open("config/.tokens.enc", 'wb') as f:
            f.write(encrypted)
        
        logger.info("Tokens saved securely")


if __name__ == "__main__":
    # Test authentication manager
    auth_mgr = AuthManager()
    
    # Test credentials
    test_creds = {
        "capital_com": {
            "api_key": "test_key_123",
            "api_secret": "test_secret_456",
            "account_id": "test_account",
            "environment": "demo"
        }
    }
    
    # Test encryption/decryption
    encrypted = auth_mgr.encrypt_credentials(test_creds)
    decrypted = auth_mgr.decrypt_credentials(encrypted)
    
    assert test_creds == decrypted
    print("Authentication manager test passed!")
    
    # Test token manager
    token_mgr = TokenManager(auth_mgr)
    token_mgr.save_tokens("test_access_token", "test_refresh_token")
    
    print("Token manager test passed!")