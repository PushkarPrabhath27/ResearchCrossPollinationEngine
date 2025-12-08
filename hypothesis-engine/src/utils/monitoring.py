"""
Monitoring and Metrics

Production monitoring, logging enhancement, and metrics collection
for system observability.
"""

from typing import Dict, Optional, Any
from datetime import datetime
import time
from functools import wraps
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """Collects and stores system metrics"""
    
    def __init__(self):
        self.metrics = {
            'requests': [],
            'latencies': [],
            'errors': [],
            'hypothesis_generated': 0,
            'papers_searched': 0
        }
        self.start_time = datetime.utcnow()
    
    def record_request(self, endpoint: str, duration: float, success: bool):
        """Record API request metric"""
        self.metrics['requests'].append({
            'endpoint': endpoint,
            'duration_ms': duration * 1000,
            'success': success,
            'timestamp': datetime.utcnow().isoformat()
        })
        self.metrics['latencies'].append(duration)
    
    def record_error(self, error: str, context: Dict):
        """Record error occurrence"""
        self.metrics['errors'].append({
            'error': error,
            'context': context,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def increment_hypothesis_count(self, count: int = 1):
        """Increment hypothesis generation counter"""
        self.metrics['hypothesis_generated'] += count
    
    def increment_search_count(self, count: int = 1):
        """Increment search counter"""
        self.metrics['papers_searched'] += count
    
    def get_summary(self) -> Dict:
        """Get metrics summary"""
        total_requests = len(self.metrics['requests'])
        successful = sum(1 for r in self.metrics['requests'] if r['success'])
        
        return {
            'uptime': str(datetime.utcnow() - self.start_time),
            'total_requests': total_requests,
            'success_rate': successful / total_requests if total_requests > 0 else 1.0,
            'avg_latency_ms': sum(self.metrics['latencies']) / len(self.metrics['latencies']) if self.metrics['latencies'] else 0,
            'total_errors': len(self.metrics['errors']),
            'hypotheses_generated': self.metrics['hypothesis_generated'],
            'papers_searched': self.metrics['papers_searched']
        }


def track_time(func):
    """Decorator to track function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start
            logger.debug(f"{func.__name__} executed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise
    return wrapper


class HealthChecker:
    """System health monitoring"""
    
    def __init__(self):
        self.components = {}
    
    def register_component(self, name: str, check_func):
        """Register health check for a component"""
        self.components[name] = check_func
    
    def check_all(self) -> Dict:
        """Run all health checks"""
        results = {}
        overall_healthy = True
        
        for name, check_func in self.components.items():
            try:
                healthy = check_func()
                results[name] = {'status': 'healthy' if healthy else 'unhealthy', 'healthy': healthy}
                if not healthy:
                    overall_healthy = False
            except Exception as e:
                results[name] = {'status': 'error', 'healthy': False, 'error': str(e)}
                overall_healthy = False
        
        return {
            'overall': 'healthy' if overall_healthy else 'unhealthy',
            'components': results,
            'timestamp': datetime.utcnow().isoformat()
        }


# Global metrics instance
_metrics = MetricsCollector()
_health = HealthChecker()


def get_metrics() -> MetricsCollector:
    return _metrics


def get_health_checker() -> HealthChecker:
    return _health
