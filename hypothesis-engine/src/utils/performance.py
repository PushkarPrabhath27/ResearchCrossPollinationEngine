"""
Performance Optimization

Caching, connection pooling, and performance utilities
for production optimization.
"""

from typing import Any, Callable, Optional, Dict
from functools import wraps, lru_cache
import time
import threading
from collections import OrderedDict
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LRUCache:
    """Thread-safe LRU cache implementation"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            # Check TTL
            if time.time() - self.timestamps[key] > self.ttl_seconds:
                del self.cache[key]
                del self.timestamps[key]
                self.misses += 1
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
    
    def set(self, key: str, value: Any):
        """Set value in cache"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    # Remove oldest
                    oldest = next(iter(self.cache))
                    del self.cache[oldest]
                    del self.timestamps[oldest]
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hit_rate,
            'ttl_seconds': self.ttl_seconds
        }


def cached(cache_instance: LRUCache, key_func: Callable = None):
    """Decorator for caching function results"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            
            # Try cache
            result = cache_instance.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache
            result = func(*args, **kwargs)
            cache_instance.set(cache_key, result)
            return result
        
        return wrapper
    return decorator


class ConnectionPool:
    """Generic connection pool"""
    
    def __init__(self, factory: Callable, max_size: int = 10):
        self.factory = factory
        self.max_size = max_size
        self.pool = []
        self.in_use = set()
        self.lock = threading.Lock()
    
    def acquire(self):
        """Acquire connection from pool"""
        with self.lock:
            # Try to get existing connection
            if self.pool:
                conn = self.pool.pop()
                self.in_use.add(id(conn))
                return conn
            
            # Create new if under limit
            if len(self.in_use) < self.max_size:
                conn = self.factory()
                self.in_use.add(id(conn))
                return conn
            
            raise RuntimeError("Connection pool exhausted")
    
    def release(self, conn):
        """Release connection back to pool"""
        with self.lock:
            if id(conn) in self.in_use:
                self.in_use.remove(id(conn))
                self.pool.append(conn)


class PerformanceProfiler:
    """Simple performance profiler"""
    
    def __init__(self):
        self.timings = {}
    
    def start(self, name: str):
        """Start timing"""
        self.timings[name] = {'start': time.time(), 'end': None}
    
    def stop(self, name: str):
        """Stop timing"""
        if name in self.timings:
            self.timings[name]['end'] = time.time()
    
    def get_duration(self, name: str) -> Optional[float]:
        """Get duration for a timing"""
        if name in self.timings and self.timings[name]['end']:
            return self.timings[name]['end'] - self.timings[name]['start']
        return None
    
    def report(self) -> Dict:
        """Get all timings"""
        return {
            name: self.get_duration(name) 
            for name in self.timings 
            if self.get_duration(name) is not None
        }


# Global instances
_embedding_cache = LRUCache(max_size=10000, ttl_seconds=86400)  # 24h TTL
_search_cache = LRUCache(max_size=1000, ttl_seconds=3600)  # 1h TTL


def get_embedding_cache() -> LRUCache:
    return _embedding_cache


def get_search_cache() -> LRUCache:
    return _search_cache
