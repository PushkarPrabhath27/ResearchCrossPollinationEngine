"""
Security and Privacy Module

Handles API key management, data sanitization, rate limiting,
and privacy-preserving operations.
"""

from typing import Dict, Optional, Callable
from functools import wraps
from datetime import datetime, timedelta
import hashlib
import secrets
import re
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Rate limiting implementation"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # user_id -> list of timestamps
    
    def is_allowed(self, user_id: str) -> bool:
        """Check if request is allowed for user"""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.window_seconds)
        
        # Get user's requests in current window
        if user_id not in self.requests:
            self.requests[user_id] = []
        
        # Clean old requests
        self.requests[user_id] = [
            ts for ts in self.requests[user_id] 
            if ts > window_start
        ]
        
        # Check limit
        if len(self.requests[user_id]) >= self.max_requests:
            return False
        
        # Record this request
        self.requests[user_id].append(now)
        return True
    
    def remaining(self, user_id: str) -> int:
        """Get remaining requests for user"""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.window_seconds)
        
        if user_id not in self.requests:
            return self.max_requests
        
        current = len([ts for ts in self.requests[user_id] if ts > window_start])
        return max(0, self.max_requests - current)


class DataSanitizer:
    """Sanitizes user input and output data"""
    
    # Patterns to remove from text
    PII_PATTERNS = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone
        r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',  # SSN
    ]
    
    @classmethod
    def sanitize_input(cls, text: str) -> str:
        """Remove potentially harmful content from input"""
        # Remove HTML/script tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove potential injection attempts
        text = text.replace('{{', '').replace('}}', '')
        text = text.replace('${', '').replace('}$', '')
        
        return text.strip()
    
    @classmethod
    def remove_pii(cls, text: str) -> str:
        """Remove personally identifiable information"""
        for pattern in cls.PII_PATTERNS:
            text = re.sub(pattern, '[REDACTED]', text)
        return text
    
    @classmethod
    def hash_identifier(cls, identifier: str) -> str:
        """Create anonymous hash of identifier"""
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]


class APIKeyManager:
    """Manages API keys securely"""
    
    def __init__(self):
        self.keys = {}
    
    def generate_key(self, user_id: str) -> str:
        """Generate new API key for user"""
        key = f"hce_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        self.keys[key_hash] = {
            'user_id': user_id,
            'created_at': datetime.utcnow().isoformat(),
            'active': True
        }
        
        logger.info(f"Generated API key for user {user_id}")
        return key
    
    def validate_key(self, key: str) -> Optional[str]:
        """Validate API key, return user_id if valid"""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        if key_hash in self.keys and self.keys[key_hash]['active']:
            return self.keys[key_hash]['user_id']
        return None
    
    def revoke_key(self, key: str) -> bool:
        """Revoke an API key"""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        if key_hash in self.keys:
            self.keys[key_hash]['active'] = False
            logger.info(f"Revoked API key")
            return True
        return False


def require_auth(func: Callable) -> Callable:
    """Decorator requiring authentication"""
    @wraps(func)
    def wrapper(*args, api_key: str = None, **kwargs):
        if not api_key:
            raise PermissionError("API key required")
        
        manager = APIKeyManager()
        user_id = manager.validate_key(api_key)
        
        if not user_id:
            raise PermissionError("Invalid API key")
        
        return func(*args, user_id=user_id, **kwargs)
    return wrapper


# Global instances
_rate_limiter = RateLimiter()
_key_manager = APIKeyManager()


def get_rate_limiter() -> RateLimiter:
    return _rate_limiter


def get_key_manager() -> APIKeyManager:
    return _key_manager
