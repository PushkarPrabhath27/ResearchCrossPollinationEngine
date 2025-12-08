"""
arXiv Paper Fetcher

Comprehensive fetcher for searching, downloading, and processing papers from arXiv.
Supports multiple search strategies, batch processing, and robust error handling.
"""

import arxiv
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Optional, Iterator
from datetime import datetime, timedelta
from pypdf import PdfReader
from tqdm import tqdm
import requests
from functools import wraps

from src.config import Settings
from src.utils.logger import get_logger
from src.utils.helpers import retry_with_backoff, ensure_dir, save_json

logger = get_logger(__name__)


def rate_limit(calls_per_second: float = 1.0):
    """
    Decorator to rate limit function calls
    
    Args:
        calls_per_second: Maximum calls per second
    """
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator


class ArxivFetcher:
    """
    Fetches and processes papers from arXiv
    
    Supports keyword, category, and author searches with date filtering.
    Downloads PDFs, extracts text, and stores structured metadata.
    """
    
    # arXiv rate limit: 1 request per 3 seconds to be respectful
    RATE_LIMIT = 1/3
    
    # arXiv category mappings
    CATEGORIES = {
        'cs': 'Computer Science',
        'math': 'Mathematics',
        'physics': 'Physics',
        'q-bio': 'Quantitative Biology',
        'q-fin': 'Quantitative Finance',
        'stat': 'Statistics',
        'eess': 'Electrical Engineering and Systems Science',
        'econ': 'Economics'
    }
    
    def __init__(self, config: Settings):
        """
        Initialize arXiv fetcher
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.raw_data_dir = Path(config.ingestion.raw_data_dir) / "arxiv"
        self.processed_data_dir = Path(config.ingestion.processed_data_dir) / "arxiv"
        
        # Create directories
        ensure_dir(self.raw_data_dir)
        ensure_dir(self.processed_data_dir)
        
        # Track downloaded papers to support resume
        self.download_cache_file = self.raw_data_dir / "download_cache.json"
        self.downloaded_papers = self._load_download_cache()
        
        logger.info(f"ArxivFetcher initialized. Data dir: {self.raw_data_dir}")
    
    def _load_download_cache(self) -> set:
        """Load set of already downloaded paper IDs"""
        if self.download_cache_file.exists():
            try:
                with open(self.download_cache_file, 'r') as f:
                    return set(json.load(f))
            except Exception as e:
                logger.warning(f"Could not load download cache: {e}")
                return set()
        return set()
    
    def _save_download_cache(self):
        """Save downloaded paper IDs to cache"""
        try:
            with open(self.download_cache_file, 'w') as f:
                json.dump(list(self.downloaded_papers), f)
        except Exception as e:
            logger.error(f"Could not save download cache: {e}")
    
    @rate_limit(calls_per_second=RATE_LIMIT)
    def search_papers(
        self,
        query: str,
        category: Optional[str] = None,
        max_results: int = 100,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        author: Optional[str] = None
    ) -> List[arxiv.Result]:
        """
        Search arXiv for papers
        
        Args:
            query: Search query string
            category: arXiv category (e.g., 'cs.AI', 'q-bio')
            max_results: Maximum number of results to return
            date_from: Start date in YYYY-MM-DD format
            date_to: End date in YYYY-MM-DD format
            author: Author name to filter by
        
        Returns:
            List of arXiv result objects
        """
        logger.info(f"Searching arXiv: query='{query}', category={category}, max={max_results}")
        
        # Build search query
        search_query = query
        
        if category:
            search_query = f"cat:{category} AND {search_query}"
        
        if author:
            search_query = f"au:{author} AND {search_query}"
        
        # Create search
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        # Execute search with retry
        @retry_with_backoff(max_retries=3, exceptions=(Exception,))
        def _execute_search():
            results = list(search.results())
            return results
        
        try:
            papers = _execute_search()
            
            # Filter by date if specified
            if date_from or date_to:
                papers = self._filter_by_date(papers, date_from, date_to)
            
            logger.info(f"Found {len(papers)} papers")
            return papers
            
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            raise
    
    def _filter_by_date(
        self,
        papers: List[arxiv.Result],
        date_from: Optional[str],
        date_to: Optional[str]
    ) -> List[arxiv.Result]:
        """
        Filter papers by date range
        
        Args:
            papers: List of papers
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
        
        Returns:
            Filtered list of papers
        """
        filtered = []
        
        from_date = datetime.fromisoformat(date_from) if date_from else None
        to_date = datetime.fromisoformat(date_to) if date_to else None
        
        for paper in papers:
            pub_date = paper.published.replace(tzinfo=None)
            
            if from_date and pub_date < from_date:
                continue
            if to_date and pub_date > to_date:
                continue
            
            filtered.append(paper)
        
        return filtered
    
    @retry_with_backoff(max_retries=3, exceptions=(requests.RequestException, IOError))
    @rate_limit(calls_per_second=RATE_LIMIT)
    def download_paper(
        self,
        paper_id: str,
        save_dir: Optional[Path] = None
    ) -> Path:
        """
        Download a single paper PDF
        
        Args:
            paper_id: arXiv paper ID (e.g., '2301.07041')
            save_dir: Directory to save PDF (defaults to raw_data_dir)
        
        Returns:
            Path to downloaded PDF
        
        Raises:
            ValueError: If paper ID is invalid
            IOError: If download fails
        """
        if not paper_id:
            raise ValueError("Paper ID cannot be empty")
        
        # Normalize paper ID (remove 'arxiv:' prefix if present)
        paper_id = paper_id.replace('arxiv:', '').replace('arXiv:', '')
        
        # Check if already downloaded
        if paper_id in self.downloaded_papers:
            logger.debug(f"Paper {paper_id} already downloaded, skipping")
            pdf_path = (save_dir or self.raw_data_dir) / f"{paper_id}.pdf"
            if pdf_path.exists():
                return pdf_path
        
        save_dir = Path(save_dir) if save_dir else self.raw_data_dir
        ensure_dir(save_dir)
        
        pdf_path = save_dir / f"{paper_id}.pdf"
        
        try:
            logger.info(f"Downloading paper {paper_id}...")
            
            # Search for the paper to get download URL
            search = arxiv.Search(id_list=[paper_id])
            paper = next(search.results())
            
            # Download PDF
            paper.download_pdf(dirpath=str(save_dir), filename=f"{paper_id}.pdf")
            
            # Add to cache
            self.downloaded_papers.add(paper_id)
            self._save_download_cache()
            
            logger.info(f"Downloaded: {pdf_path}")
            return pdf_path
            
        except StopIteration:
            raise ValueError(f"Paper {paper_id} not found on arXiv")
        except Exception as e:
            logger.error(f"Failed to download {paper_id}: {e}")
            raise IOError(f"Download failed: {e}")
    
    def download_batch(
        self,
        paper_ids: List[str],
        save_dir: Optional[Path] = None
    ) -> Dict[str, Path]:
        """
        Download multiple papers with progress tracking and resume capability
        
        Args:
            paper_ids: List of arXiv paper IDs
            save_dir: Directory to save PDFs
        
        Returns:
            Dictionary mapping paper IDs to downloaded file paths
        """
        save_dir = Path(save_dir) if save_dir else self.raw_data_dir
        results = {}
        
        logger.info(f"Downloading batch of {len(paper_ids)} papers...")
        
        # Filter out already downloaded papers
        to_download = [pid for pid in paper_ids if pid not in self.downloaded_papers]
        skipped = len(paper_ids) - len(to_download)
        
        if skipped > 0:
            logger.info(f"Skipping {skipped} already downloaded papers")
        
        # Download with progress bar
        with tqdm(total=len(to_download), desc="Downloading papers") as pbar:
            for paper_id in to_download:
                try:
                    pdf_path = self.download_paper(paper_id, save_dir)
                    results[paper_id] = pdf_path
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Failed to download {paper_id}: {e}")
                    results[paper_id] = None
                    pbar.update(1)
                
                # Small delay to respect rate limits
                time.sleep(0.5)
        
        success_count = sum(1 for v in results.values() if v is not None)
        logger.info(f"Batch download complete: {success_count}/{len(to_download)} successful")
        
        return results
    
    def extract_text(self, pdf_path: Path) -> str:
        """
        Extract text content from PDF
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Extracted text content
        
        Raises:
            IOError: If PDF cannot be read
        """
        if not Path(pdf_path).exists():
            raise IOError(f"PDF file not found: {pdf_path}")
        
        try:
            logger.debug(f"Extracting text from {pdf_path}")
            
            reader = PdfReader(str(pdf_path))
            text_parts = []
            
            for page_num, page in enumerate(reader.pages):
                try:
                    text_parts.append(page.extract_text())
                except Exception as e:
                    logger.warning(f"Failed to extract page {page_num}: {e}")
                    continue
            
            full_text = "\n\n".join(text_parts)
            
            logger.debug(f"Extracted {len(full_text)} characters from {len(reader.pages)} pages")
            return full_text
            
        except Exception as e:
            logger.error(f"PDF extraction failed for {pdf_path}: {e}")
            raise IOError(f"Could not extract text: {e}")
    
    def parse_metadata(self, arxiv_result: arxiv.Result) -> Dict:
        """
        Parse arXiv result into structured metadata
        
        Args:
            arxiv_result: arXiv Result object
        
        Returns:
            Dictionary with paper metadata
        """
        metadata = {
            'paper_id': arxiv_result.entry_id.split('/')[-1],
            'title': arxiv_result.title,
            'authors': [author.name for author in arxiv_result.authors],
            'abstract': arxiv_result.summary,
            'categories': arxiv_result.categories,
            'published_date': arxiv_result.published.isoformat(),
            'updated_date': arxiv_result.updated.isoformat(),
            'doi': arxiv_result.doi if hasattr(arxiv_result, 'doi') else None,
            'journal_reference': arxiv_result.journal_ref,
            'comments': arxiv_result.comment,
            'pdf_url': arxiv_result.pdf_url,
            'full_text': None  # Will be filled after extraction
        }
        
        return metadata
    
    def get_categories(self) -> Dict[str, str]:
        """
        Get list of arXiv categories
        
        Returns:
            Dictionary mapping category codes to names
        """
        return self.CATEGORIES.copy()
    
    def get_recent_papers(
        self,
        category: str,
        days: int = 7,
        max_results: int = 100
    ) -> List[arxiv.Result]:
        """
        Get recent papers from a specific category
        
        Args:
            category: arXiv category code (e.g., 'cs.AI')
            days: Number of days to look back
            max_results: Maximum number of papers to return
        
        Returns:
            List of recent papers
        """
        date_from = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        logger.info(f"Fetching papers from {category} since {date_from}")
        
        return self.search_papers(
            query="*",  # Match all
            category=category,
            max_results=max_results,
            date_from=date_from
        )
    
    def fetch_and_process(
        self,
        query: str,
        category: Optional[str] = None,
        max_results: int = 10,
        download_pdf: bool = True,
        extract_text: bool = True
    ) -> List[Dict]:
        """
        Search, download, and process papers in one go
        
        Args:
            query: Search query
            category: Optional category filter
            max_results: Maximum papers to process
            download_pdf: Whether to download PDFs
            extract_text: Whether to extract text from PDFs
        
        Returns:
            List of processed paper metadata dictionaries
        """
        logger.info(f"Fetching and processing papers for query: {query}")
        
        # Search papers
        papers = self.search_papers(query, category=category, max_results=max_results)
        
        processed_papers = []
        
        # Process each paper
        with tqdm(total=len(papers), desc="Processing papers") as pbar:
            for paper in papers:
                try:
                    # Parse metadata
                    metadata = self.parse_metadata(paper)
                    
                    # Download PDF if requested
                    if download_pdf:
                        try:
                            pdf_path = self.download_paper(metadata['paper_id'])
                            
                            # Extract text if requested
                            if extract_text and pdf_path:
                                text = self.extract_text(pdf_path)
                                metadata['full_text'] = text
                        except Exception as e:
                            logger.warning(f"Could not process PDF for {metadata['paper_id']}: {e}")
                    
                    processed_papers.append(metadata)
                    
                    # Save metadata
                    metadata_file = self.processed_data_dir / f"{metadata['paper_id']}.json"
                    save_json(metadata, str(metadata_file))
                    
                except Exception as e:
                    logger.error(f"Failed to process paper: {e}")
                finally:
                    pbar.update(1)
        
        logger.info(f"Processed {len(processed_papers)} papers successfully")
        return processed_papers


# Example usage
if __name__ == "__main__":
    from src.config import get_settings
    from src.utils.logger import setup_logging
    
    # Initialize
    setup_logging(level="INFO")
    config = get_settings()
    fetcher = ArxivFetcher(config)
    
    # Example 1: Search for papers
    print("\n=== Example 1: Search Papers ===")
    papers = fetcher.search_papers(
        query="cancer metastasis machine learning",
        category="q-bio",
        max_results=5,
        date_from="2023-01-01"
    )
    
    print(f"Found {len(papers)} papers:")
    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper.title[:80]}...")
        print(f"   Authors: {', '.join([a.name for a in paper.authors[:2]])} et al.")
        print(f"   Published: {paper.published.date()}")
        print()
    
    # Example 2: Download and process a single paper
    if papers:
        print("\n=== Example 2: Download and Process ===")
        paper = papers[0]
        paper_id = paper.entry_id.split('/')[-1]
        
        try:
            # Download
            pdf_path = fetcher.download_paper(paper_id)
            print(f"Downloaded to: {pdf_path}")
            
            # Extract text
            text = fetcher.extract_text(pdf_path)
            print(f"Extracted {len(text)} characters")
            print(f"First 200 characters: {text[:200]}...")
            
            # Parse metadata
            metadata = fetcher.parse_metadata(paper)
            print(f"\nMetadata keys: {list(metadata.keys())}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    # Example 3: Get recent papers
    print("\n=== Example 3: Recent Papers ===")
    recent = fetcher.get_recent_papers(category="cs.AI", days=7, max_results=3)
    print(f"Recent papers in cs.AI (last 7 days): {len(recent)}")
    
    # Example 4: Batch download
    print("\n=== Example 4: Batch Download ===")
    if len(papers) >= 2:
        paper_ids = [p.entry_id.split('/')[-1] for p in papers[:2]]
        results = fetcher.download_batch(paper_ids)
        print(f"Batch download complete: {len(results)} papers")
    
    print("\n✅ All examples completed successfully!")
