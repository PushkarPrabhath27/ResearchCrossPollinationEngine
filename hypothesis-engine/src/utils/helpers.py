"""
Helper utility functions for the Hypothesis Engine.

This module provides common utilities used across the application including:
- Text processing
- File operations
- Data validation
- Retry logic
- Progress tracking
"""

import hashlib
import time
from typing import Any, Callable, List, Optional, Dict
from functools import wraps
import json
from pathlib import Path
import re


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator to retry function with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        raise last_exception
            
        return wrapper
    return decorator


def generate_id(text: str) -> str:
    """
    Generate a unique ID from text using MD5 hash
    
    Args:
        text: Input text
    
    Returns:
        Hexadecimal hash string
    """
    return hashlib.md5(text.encode()).hexdigest()


def clean_text(text: str) -> str:
    """
    Clean and normalize text
    
    Args:
        text: Raw text
    
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep scientific notation
    text = re.sub(r'[^\w\s\.\,\;\:\-\(\)\[\]\{\}]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks
    
    Args:
        lst: Input list
        chunk_size: Size of each chunk
    
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def save_json(data: Any, filepath: str, indent: int = 2):
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(filepath: str) -> Any:
    """
    Load data from JSON file
    
    Args:
        filepath: Input file path
    
    Returns:
        Loaded data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_dir(directory: str) -> Path:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        directory: Directory path
    
    Returns:
        Path object
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    Extract keywords from text (simple word frequency)
    
    Args:
        text: Input text
        top_n: Number of top keywords to return
    
    Returns:
        List of keywords
    """
    # Simple word frequency extraction
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    
    # Remove common stop words
    stop_words = {
        'this', 'that', 'with', 'from', 'have', 'been', 'were', 'will',
        'their', 'what', 'which', 'when', 'where', 'there', 'these',
        'those', 'could', 'would', 'should', 'about', 'other', 'such'
    }
    
    words = [w for w in words if w not in stop_words]
    
    # Count frequencies
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top N
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:top_n]]


def format_paper_citation(paper: Dict) -> str:
    """
    Format paper metadata as citation
    
    Args:
        paper: Paper metadata dictionary
    
    Returns:
        Formatted citation string
    """
    authors = paper.get('authors', [])
    if len(authors) > 3:
        author_str = f"{authors[0]} et al."
    else:
        author_str = ", ".join(authors)
    
    year = paper.get('year', 'n.d.')
    title = paper.get('title', 'Untitled')
    venue = paper.get('venue', '')
    
    citation = f"{author_str} ({year}). {title}."
    if venue:
        citation += f" {venue}."
    
    return citation


def calculate_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Cosine similarity score (0-1)
    """
    import math
    
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have same length")
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


class ProgressTracker:
    """Simple progress tracker for long-running operations"""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, n: int = 1):
        """Update progress by n steps"""
        self.current += n
        self._display()
    
    def _display(self):
        """Display progress"""
        if self.total > 0:
            percentage = (self.current / self.total) * 100
            elapsed = time.time() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / rate if rate > 0 else 0
            
            print(
                f"\r{self.description}: {self.current}/{self.total} "
                f"({percentage:.1f}%) | "
                f"Rate: {rate:.1f} it/s | "
                f"ETA: {eta:.0f}s",
                end=""
            )
    
    def close(self):
        """Finish progress tracking"""
        print()  # New line


def validate_email(email: str) -> bool:
    """
    Validate email format
    
    Args:
        email: Email address
    
    Returns:
        True if valid, False otherwise
    """
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
    
    Returns:
        Division result or default
    """
    return numerator / denominator if denominator != 0 else default
