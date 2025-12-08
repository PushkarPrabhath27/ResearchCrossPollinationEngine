"""
Semantic Scholar API Fetcher

Fetches paper data, citations, and recommendations from Semantic Scholar.
Includes AI-generated summaries, influential citations, and author h-indices.
"""

import requests
import time
from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm

from src.config import Settings
from src.utils.logger import get_logger
from src.utils.helpers import retry_with_backoff, ensure_dir, save_json, load_json

logger = get_logger(__name__)


class SemanticScholarFetcher:
    """
    Fetches papers from Semantic Scholar API
    
    Free tier: 100 requests per 5 minutes (with API key)
    Provides citation counts, influential citations, and AI-generated TL;DR summaries.
    """
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    RATE_LIMIT = 5 / 100  # 100 requests per 5 minutes = 0.05 seconds between requests
    
    # Available fields
    PAPER_FIELDS = [
        'paperId', 'externalIds', 'url', 'title', 'abstract', 'venue', 'year',
        'referenceCount', 'citationCount', 'influentialCitationCount',
        'isOpenAccess', 'openAccessPdf', 'fieldsOfStudy', 'AUTHORS', 'citations',
        'references', 'embedding', 'tldr'
    ]
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Settings] = None):
        """
        Initialize Semantic Scholar fetcher
        
        Args:
            api_key: API key (optional but recommended for higher rate limits)
            config: Application configuration
        """
        self.api_key = api_key
        self.config = config
        self.session = requests.Session()
        
        # Set headers
        self.headers = {}
        if api_key:
            self.headers['x-api-key'] = api_key
        
        # Setup directories
        if config:
            self.cache_dir = Path(config.ingestion.raw_data_dir) / "semantic_scholar"
        else:
            self.cache_dir = Path("./data/raw/semantic_scholar")
        
        ensure_dir(self.cache_dir)
        
        # Load cache
        self.cache_file = self.cache_dir / "api_cache.json"
        self.cache = self._load_cache()
        
        self.last_request_time = 0
        
        logger.info(f"SemanticScholarFetcher initialized with API key: {'Yes' if api_key else 'No'}")
    
    def _load_cache(self) -> Dict:
        """Load API response cache"""
        if self.cache_file.exists():
            try:
                return load_json(str(self.cache_file))
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save API response cache"""
        try:
            save_json(self.cache, str(self.cache_file))
        except Exception as e:
            logger.error(f"Could not save cache: {e}")
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.RATE_LIMIT:
            time.sleep(self.RATE_LIMIT - elapsed)
        self.last_request_time = time.time()
    
    @retry_with_backoff(max_retries=3, exceptions=(requests.RequestException,))
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make API request with rate limiting and error handling
        
        Args:
            endpoint: API endpoint
            params: Query parameters
        
        Returns:
            JSON response
        """
        url = f"{self.BASE_URL}{endpoint}"
        
        # Check cache
        cache_key = f"{endpoint}_{params}"
        if cache_key in self.cache:
            logger.debug("Using cached response")
            return self.cache[cache_key]
        
        self._rate_limit()
        
        try:
            response = self.session.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Cache response
            self.cache[cache_key] = data
            self._save_cache()
            
            return data
            
        except requests.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning("Rate limit exceeded, waiting...")
                time.sleep(60)  # Wait 1 minute
                return self._make_request(endpoint, params)
            else:
                logger.error(f"HTTP error: {e}")
                raise
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
    
    def search_papers(
        self,
        query: str,
        limit: int = 100,
        fields: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Search for papers
        
        Args:
            query: Search query string
            limit: Maximum number of results
            fields: Fields to retrieve (default: all)
        
        Returns:
            List of paper dictionaries
        """
        if fields is None:
            fields = self.PAPER_FIELDS
        
        logger.info(f"Searching Semantic Scholar: '{query}', limit={limit}")
        
        params = {
            'query': query,
            'limit': min(limit, 100),  # API max is 100
            'fields': ','.join(fields)
        }
        
        try:
            response = self._make_request("/paper/search", params)
            papers = response.get('data', [])
            
            logger.info(f"Found {len(papers)} papers")
            return papers
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_paper(
        self,
        paper_id: str,
        fields: Optional[List[str]] = None
    ) -> Optional[Dict]:
        """
        Get details for a single paper
        
        Args:
            paper_id: Paper identifier (S2 ID, DOI, arXiv ID, etc.)
                     Examples: 'DOI:10.1038/nature...', 'ARXIV:2104.12345', 'S2:abc123'
            fields: Fields to retrieve
        
        Returns:
            Paper dictionary or None
        """
        if fields is None:
            fields = self.PAPER_FIELDS
        
        logger.debug(f"Fetching paper: {paper_id}")
        
        params = {'fields': ','.join(fields)}
        
        try:
            paper = self._make_request(f"/paper/{paper_id}", params)
            return paper
        except Exception as e:
            logger.error(f"Failed to get paper {paper_id}: {e}")
            return None
    
    def get_paper_citations(
        self,
        paper_id: str,
        limit: int = 100,
        fields: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Get papers that cite this paper
        
        Args:
            paper_id: Paper identifier
            limit: Maximum number of citations
            fields: Fields to retrieve for each citation
        
        Returns:
            List of citing paper dictionaries
        """
        if fields is None:
            fields = ['paperId', 'title', 'year', 'citationCount', 'influentialCitationCount']
        
        logger.info(f"Getting citations for {paper_id}")
        
        params = {
            'limit': min(limit, 1000),
            'fields': ','.join(fields)
        }
        
        try:
            response = self._make_request(f"/paper/{paper_id}/citations", params)
            citations = response.get('data', [])
            
            # Extract citing papers
            citing_papers = [c['citingPaper'] for c in citations if 'citingPaper' in c]
            
            logger.info(f"Found {len(citing_papers)} citations")
            return citing_papers
            
        except Exception as e:
            logger.error(f"Failed to get citations: {e}")
            return []
    
    def get_paper_references(
        self,
        paper_id: str,
        limit: int = 100,
        fields: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Get papers that this paper references
        
        Args:
            paper_id: Paper identifier
            limit: Maximum number of references
            fields: Fields to retrieve
        
        Returns:
            List of referenced paper dictionaries
        """
        if fields is None:
            fields = ['paperId', 'title', 'year', 'citationCount']
        
        logger.info(f"Getting references for {paper_id}")
        
        params = {
            'limit': min(limit, 1000),
            'fields': ','.join(fields)
        }
        
        try:
            response = self._make_request(f"/paper/{paper_id}/references", params)
            references = response.get('data', [])
            
            # Extract cited papers
            cited_papers = [r['citedPaper'] for r in references if 'citedPaper' in r]
            
            logger.info(f"Found {len(cited_papers)} references")
            return cited_papers
            
        except Exception as e:
            logger.error(f"Failed to get references: {e}")
            return []
    
    def get_author(self, author_id: str) -> Optional[Dict]:
        """
        Get author details including h-index
        
        Args:
            author_id: Semantic Scholar author ID
        
        Returns:
            Author dictionary with h-index, paper count, etc.
        """
        logger.debug(f"Fetching author: {author_id}")
        
        params = {
            'fields': 'authorId,name,affiliations,homepage,paperCount,citationCount,hIndex'
        }
        
        try:
            author = self._make_request(f"/author/{author_id}", params)
            return author
        except Exception as e:
            logger.error(f"Failed to get author {author_id}: {e}")
            return None
    
    def get_recommendations(
        self,
        paper_id: str,
        limit: int = 10,
        fields: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Get similar/recommended papers
        
        Args:
            paper_id: Paper identifier
            limit: Number of recommendations
            fields: Fields to retrieve
        
        Returns:
            List of recommended paper dictionaries
        """
        if fields is None:
            fields = ['paperId', 'title', 'year', 'abstract', 'citationCount']
        
        logger.info(f"Getting recommendations for {paper_id}")
        
        params = {
            'limit': min(limit, 100),
            'fields': ','.join(fields)
        }
        
        try:
            response = self._make_request(f"/paper/{paper_id}/recommendations", params)
            recommendations = response.get('recommendedPapers', [])
            
            logger.info(f"Found {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            return []
    
    def batch_get_papers(
        self,
        paper_ids: List[str],
        fields: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Get multiple papers efficiently
        
        Args:
            paper_ids: List of paper identifiers (max 500)
            fields: Fields to retrieve
        
        Returns:
            List of paper dictionaries
        """
        if fields is None:
            fields = self.PAPER_FIELDS
        
        logger.info(f"Batch fetching {len(paper_ids)} papers")
        
        # API supports max 500 IDs per request
        batch_size = 500
        all_papers = []
        
        for i in range(0, len(paper_ids), batch_size):
            batch = paper_ids[i:i+batch_size]
            
            payload = {
                'ids': batch
            }
            params = {'fields': ','.join(fields)}
            
            try:
                self._rate_limit()
                response = self.session.post(
                    f"{self.BASE_URL}/paper/batch",
                    headers=self.headers,
                    params=params,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                
                papers = response.json()
                all_papers.extend([p for p in papers if p is not None])
                
            except Exception as e:
                logger.error(f"Batch fetch failed: {e}")
        
        logger.info(f"Successfully fetched {len(all_papers)} papers")
        return all_papers


# Example usage
if __name__ == "__main__":
    from src.config import get_settings
    from src.utils.logger import setup_logging
    
    setup_logging(level="INFO")
    config = get_settings()
    
    # Initialize with API key if available
    api_key = config.api.semantic_scholar_api_key if hasattr(config.api, 'semantic_scholar_api_key') else None
    fetcher = SemanticScholarFetcher(api_key=api_key, config=config)
    
    # Example 1: Search papers
    print("\n=== Example 1: Search Papers ===")
    papers = fetcher.search_papers("machine learning cancer", limit=5)
    
    for paper in papers[:3]:
        print(f"\nTitle: {paper.get('title')}")
        print(f"Year: {paper.get('year')}")
        print(f"Citations: {paper.get('citationCount')}")
        print(f"Influential Citations: {paper.get('influentialCitationCount')}")
        if paper.get('tldr'):
            print(f"TL;DR: {paper['tldr'].get('text')}")
    
    # Example 2: Get paper details
    if papers:
        print("\n=== Example 2: Get Paper Details ===")
        paper_id = papers[0]['paperId']
        details = fetcher.get_paper(paper_id)
        
        print(f"Title: {details.get('title')}")
        print(f"Abstract: {details.get('abstract', 'N/A')[:200]}...")
        print(f"Fields of Study: {details.get('fieldsOfStudy')}")
        print(f"Open Access: {details.get('isOpenAccess')}")
    
    # Example 3: Get citations
    if papers:
        print("\n=== Example 3: Get Citations ===")
        citations = fetcher.get_paper_citations(papers[0]['paperId'], limit=5)
        print(f"Found {len(citations)} citing papers")
        
        for cit in citations[:2]:
            print(f"  - {cit.get('title')} ({cit.get('year')})")
    
    # Example 4: Get recommendations
    if papers:
        print("\n=== Example 4: Get Recommendations ===")
        recs = fetcher.get_recommendations(papers[0]['paperId'], limit=3)
        print(f"Found {len(recs)} recommendations:")
        
        for rec in recs:
            print(f"  - {rec.get('title')} ({rec.get('year')})")
    
    print("\n✅ All examples completed!")
