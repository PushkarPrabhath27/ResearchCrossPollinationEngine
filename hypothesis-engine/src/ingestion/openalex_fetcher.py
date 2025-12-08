"""
OpenAlex API Fetcher

Fetches comprehensive paper metadata from OpenAlex including institutions,
funding, concepts, and work lineage. No rate limits, completely free.
"""

import requests
import time
from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm

from src.config import Settings
from src.utils.logger import get_logger
from src.utils.helpers import retry_with_backoff, ensure_dir, save_json

logger = get_logger(__name__)


class OpenAlexFetcher:
    """
    Fetches papers from OpenAlex API
    
    Free, no rate limits. Provides comprehensive metadata including institutions,
    funding, concepts (research topics), and citation data.
    """
    
    BASE_URL = "https://api.openalex.org"
    
    def __init__(self, email: str, config: Optional[Settings] = None):
        """
        Initialize OpenAlex fetcher
        
        Args:
            email: Email for polite pool (gets faster response)
            config: Application configuration
        """
        self.email = email
        self.config = config
        
        # Polite pool - add email to get into faster lane
        self.params = {'mailto': email}
        
        # Setup directories
        if config:
            self.cache_dir = Path(config.ingestion.raw_data_dir) / "openalex"
        else:
            self.cache_dir = Path("./data/raw/openalex")
        
        ensure_dir(self.cache_dir)
        
        logger.info(f"OpenAlexFetcher initialized with email: {email}")
    
    @retry_with_backoff(max_retries=3, exceptions=(requests.RequestException,))
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make API request
        
        Args:
            endpoint: API endpoint
            params: Query parameters
        
        Returns:
            JSON response
        """
        url = f"{self.BASE_URL}{endpoint}"
        
        # Merge params
        all_params = {**self.params}
        if params:
            all_params.update(params)
        
        try:
            response = requests.get(url, params=all_params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
    
    def search_works(
        self,
        query: str,
        filters: Optional[Dict[str, str]] = None,
        per_page: int = 25,
        page: int = 1
    ) -> Dict:
        """
        Search for works (papers)
        
        Args:
            query: Search query
            filters: Filter dictionary (e.g., {"publication_year": ">2020"})
            per_page: Results per page (max 200)
            page: Page number
        
        Returns:
            Dictionary with results and metadata
        """
        logger.info(f"Searching OpenAlex: '{query}', per_page={per_page}")
        
        params = {
            'search': query,
            'per-page': min(per_page, 200),
            'page': page
        }
        
        # Add filters
        if filters:
            filter_strings = []
            for key, value in filters.items():
                filter_strings.append(f"{key}:{value}")
            params['filter'] = ','.join(filter_strings)
        
        try:
            response = self._make_request("/works", params)
            
            results = response.get('results', [])
            logger.info(f"Found {len(results)} works")
            
            return {
                'results': results,
                'meta': response.get('meta', {}),
                'count': response.get('meta', {}).get('count', 0)
            }
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {'results': [], 'meta': {}, 'count': 0}
    
    def get_work(self, openalex_id: str) -> Optional[Dict]:
        """
        Get single work by ID
        
        Args:
            openalex_id: OpenAlex ID or DOI
                        Examples: 'W2741809807', 'https://openalex.org/W2741809807',
                                 'https://doi.org/10.1234/example'
        
        Returns:
            Work dictionary or None
        """
        # Handle different ID formats
        if not openalex_id.startswith('http'):
            if not openalex_id.startswith('W'):
                openalex_id = f"W{openalex_id}"
            openalex_id = f"https://openalex.org/{openalex_id}"
        
        logger.debug(f"Fetching work: {openalex_id}")
        
        try:
            work = self._make_request(f"/works/{openalex_id.split('/')[-1]}")
            return work
        except Exception as e:
            logger.error(f"Failed to get work: {e}")
            return None
    
    def get_author(self, author_id: str) -> Optional[Dict]:
        """
        Get author information
        
        Args:
            author_id: OpenAlex author ID
        
        Returns:
            Author dictionary with works, h-index, etc.
        """
        if not author_id.startswith('A'):
            author_id = f"A{author_id}"
        
        logger.debug(f"Fetching author: {author_id}")
        
        try:
            author = self._make_request(f"/authors/{author_id}")
            return author
        except Exception as e:
            logger.error(f"Failed to get author: {e}")
            return None
    
    def get_institution(self, institution_id: str) -> Optional[Dict]:
        """
        Get institution information
        
        Args:
            institution_id: OpenAlex institution ID
        
        Returns:
            Institution dictionary
        """
        if not institution_id.startswith('I'):
            institution_id = f"I{institution_id}"
        
        logger.debug(f"Fetching institution: {institution_id}")
        
        try:
            institution = self._make_request(f"/institutions/{institution_id}")
            return institution
        except Exception as e:
            logger.error(f"Failed to get institution: {e}")
            return None
    
    def get_concept(self, concept_id: str) -> Optional[Dict]:
        """
        Get concept (research topic) details
        
        Args:
            concept_id: OpenAlex concept ID
        
        Returns:
            Concept dictionary with hierarchy
        """
        if not concept_id.startswith('C'):
            concept_id = f"C{concept_id}"
        
        logger.debug(f"Fetching concept: {concept_id}")
        
        try:
            concept = self._make_request(f"/concepts/{concept_id}")
            return concept
        except Exception as e:
            logger.error(f"Failed to get concept: {e}")
            return None
    
    def get_related_works(
        self,
        work_id: str,
        per_page: int = 10
    ) -> List[Dict]:
        """
        Find related works
        
        Args:
            work_id: OpenAlex work ID
            per_page: Number of related works to return
        
        Returns:
            List of related work dictionaries
        """
        logger.info(f"Finding related works for {work_id}")
        
        try:
            work = self.get_work(work_id)
            if not work:
                return []
            
            # Get works with similar concepts
            concepts = work.get('concepts', [])
            if not concepts:
                return []
            
            # Use top concepts
            top_concepts = sorted(concepts, key=lambda x: x.get('score', 0), reverse=True)[:3]
            concept_ids = [c['id'].split('/')[-1] for c in top_concepts]
            
            # Search for works with these concepts
            filters = {
                'concepts.id': '|'.join(concept_ids),
                'publication_year': f">={work.get('publication_year', 2020) - 5}"
            }
            
            results = self.search_works(
                query=work.get('title', ''),
                filters=filters,
                per_page=per_page
            )
            
            return results['results']
            
        except Exception as e:
            logger.error(f"Failed to get related works: {e}")
            return []
    
    def filter_by_concept(
        self,
        concept_name: str,
        filters: Optional[Dict[str, str]] = None,
        per_page: int = 25
    ) -> Dict:
        """
        Search works by research area/concept
        
        Args:
            concept_name: Concept name (e.g., "Machine Learning", "Cancer Research")
            filters: Additional filters
            per_page: Results per page
        
        Returns:
            Dictionary with results
        """
        logger.info(f"Searching by concept: {concept_name}")
        
        # Search for the concept first
        concept_params = {
            'search': concept_name,
            'per-page': 1
        }
        
        try:
            concept_response = self._make_request("/concepts", concept_params)
            concepts = concept_response.get('results', [])
            
            if not concepts:
                logger.warning(f"No concept found for: {concept_name}")
                return {'results': [], 'meta': {}, 'count': 0}
            
            concept_id = concepts[0]['id'].split('/')[-1]
            
            # Search works with this concept
            work_filters = {'concepts.id': concept_id}
            if filters:
                work_filters.update(filters)
            
            return self.search_works(
                query="",  # Empty query to get all works with this concept
                filters=work_filters,
                per_page=per_page
            )
            
        except Exception as e:
            logger.error(f"Concept search failed: {e}")
            return {'results': [], 'meta': {}, 'count': 0}
    
    def batch_fetch_works(
        self,
        work_ids: List[str],
        save_dir: Optional[Path] = None
    ) -> List[Dict]:
        """
        Fetch multiple works with progress tracking
        
        Args:
            work_ids: List of OpenAlex work IDs
            save_dir: Directory to save individual work JSONs
        
        Returns:
            List of work dictionaries
        """
        logger.info(f"Batch fetching {len(work_ids)} works")
        
        if save_dir:
            ensure_dir(save_dir)
        
        works = []
        
        with tqdm(total=len(work_ids), desc="Fetching OA works") as pbar:
            for work_id in work_ids:
                try:
                    work = self.get_work(work_id)
                    if work:
                        works.append(work)
                        
                        if save_dir:
                            filename = work_id.replace('https://openalex.org/', '').replace('/', '_')
                            save_json(work, str(save_dir / f"{filename}.json"))
                    
                except Exception as e:
                    logger.error(f"Failed to fetch {work_id}: {e}")
                finally:
                    pbar.update(1)
                    time.sleep(0.05)  # Small delay to be polite
        
        logger.info(f"Successfully fetched {len(works)} works")
        return works


# Example usage
if __name__ == "__main__":
    from src.config import get_settings
    from src.utils.logger import setup_logging
    
    setup_logging(level="INFO")
    config = get_settings()
    
    fetcher = OpenAlexFetcher(email=config.api.entrez_email, config=config)
    
    # Example 1: Search works
    print("\n=== Example 1: Search Works ===")
    results = fetcher.search_works(
        query="machine learning cancer",
        filters={"publication_year": ">2020", "cited_by_count": ">50"},
        per_page=5
    )
    
    print(f"Total count: {results['count']}")
    print(f"Results: {len(results['results'])}")
    
    for work in results['results'][:2]:
        print(f"\nTitle: {work.get('title')}")
        print(f"Year: {work.get('publication_year')}")
        print(f"Citations: {work.get('cited_by_count')}")
        print(f"Type: {work.get('type')}")
    
    # Example 2: Get work details
    if results['results']:
        print("\n=== Example 2: Get Work Details ===")
        work_id = results['results'][0]['id']
        work = fetcher.get_work(work_id)
        
        print(f"Title: {work.get('title')}")
        print(f"Concepts: {[c['display_name'] for c in work.get('concepts', [])[:5]]}")
        print(f"Authors: {', '.join([a['author']['display_name'] for a in work.get('authorships', [])[:3]])}")
        print(f"Open Access: {work.get('open_access', {}).get('is_oa')}")
    
    # Example 3: Search by concept
    print("\n=== Example 3: Search by Concept ===")
    concept_results = fetcher.filter_by_concept(
        "Machine Learning",
        filters={"publication_year": "2023"},
        per_page=3
    )
    
    print(f"Found {len(concept_results['results'])} ML papers from 2023")
    
    # Example 4: Get related works
    if results['results']:
        print("\n=== Example 4: Get Related Works ===")
        related = fetcher.get_related_works(results['results'][0]['id'], per_page=3)
        print(f"Found {len(related)} related works:")
        
        for r in related:
            print(f"  - {r.get('title')}")
    
    print("\n✅ All examples completed!")
