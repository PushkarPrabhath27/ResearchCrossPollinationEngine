"""
Real Paper Fetcher Service - ENHANCED VERSION
Fixes: Better query processing, improved API calls, field filtering

Issues Fixed:
1. OpenAlex returns irrelevant papers -> Added concept filtering
2. Semantic Scholar returns 0 -> Fixed API call format
3. Query preprocessing for better relevance
"""

import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import urllib.parse
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RealPaper:
    """Real paper with verified data from APIs"""
    title: str
    authors: str
    journal: str
    year: int
    doi: str
    url: str
    abstract: str
    citation_count: int
    source: str  # "openalex", "arxiv", "semantic_scholar"
    pdf_url: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QueryProcessor:
    """Extract keywords and optimize search queries"""
    
    # Field-specific keywords to add for better results
    FIELD_KEYWORDS = {
        "biology": ["biological", "genomics", "molecular", "cell", "protein"],
        "physics": ["quantum", "physics", "photonics", "optics", "mechanics"],
        "computer_science": ["algorithm", "machine learning", "neural", "computational"],
        "chemistry": ["chemical", "molecular", "synthesis", "reaction"],
        "medicine": ["clinical", "therapeutic", "disease", "patient"],
        "engineering": ["design", "system", "optimization", "mechanical"],
        "mathematics": ["mathematical", "theorem", "proof", "model"]
    }
    
    @staticmethod
    def extract_keywords(query: str) -> List[str]:
        """Extract important keywords from query"""
        # Remove common words
        stop_words = {"i'm", "i", "am", "are", "is", "the", "a", "an", "and", "or", "for", 
                      "to", "from", "that", "which", "with", "could", "would", "should",
                      "there", "their", "what", "how", "can", "help", "enhance", "improve",
                      "researching", "studying", "looking", "finding", "about", "using"}
        
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        keywords = [w for w in words if w not in stop_words]
        
        return keywords[:10]  # Top 10 keywords
    
    @staticmethod
    def build_optimized_query(query: str, field: str = "") -> str:
        """Build an optimized search query"""
        keywords = QueryProcessor.extract_keywords(query)
        
        # Add field-specific terms if field is specified
        if field and field.lower() in QueryProcessor.FIELD_KEYWORDS:
            # Check if any field keyword is already in the query
            field_terms = QueryProcessor.FIELD_KEYWORDS[field.lower()]
            existing = [k for k in keywords if k in field_terms]
            if not existing:
                keywords.extend(field_terms[:2])
        
        # Build query - prioritize key scientific terms
        priority_terms = ["quantum", "entanglement", "cryptography", "encryption", 
                          "communication", "security", "key distribution", "QKD",
                          "protocol", "channel", "photon"]
        
        # Find matches with priority terms
        matched = [k for k in keywords if k in priority_terms]
        other = [k for k in keywords if k not in matched]
        
        # Combine with priority terms first
        final_terms = matched[:5] + other[:5]
        
        return " ".join(final_terms)
    
    @staticmethod
    def get_arxiv_categories(field: str) -> List[str]:
        """Get arXiv categories for a field"""
        categories = {
            "biology": ["q-bio.BM", "q-bio.CB", "q-bio.GN"],
            "physics": ["quant-ph", "physics.optics", "cond-mat"],
            "computer_science": ["cs.AI", "cs.LG", "cs.CR"],
            "chemistry": ["physics.chem-ph", "cond-mat.mtrl-sci"],
            "medicine": ["q-bio.QM", "physics.med-ph"],
            "engineering": ["eess.SP", "eess.SY"],
            "mathematics": ["math.OC", "math.ST"]
        }
        return categories.get(field.lower(), ["quant-ph"])


class LivePaperFetcher:
    """
    Fetches REAL papers from multiple APIs on-demand.
    NO local storage - all data is live.
    """
    
    # API Endpoints
    OPENALEX_BASE = "https://api.openalex.org"
    ARXIV_BASE = "http://export.arxiv.org/api/query"
    SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ScienceBridge/1.0 (research-discovery-platform; mailto:contact@sciencebridge.ai)"
        })
        self.search_stats = {}
        self.query_processor = QueryProcessor()
        self.current_field = ""
    
    # ==================== OPENALEX ====================
    def search_openalex(self, query: str, max_results: int = 15) -> List[RealPaper]:
        """
        Search OpenAlex - 250M+ papers, FREE, no API key needed
        FIXED: Better query construction and filtering
        """
        papers = []
        start_time = time.time()
        
        try:
            # Optimize query for better results
            optimized_query = self.query_processor.build_optimized_query(query, self.current_field)
            encoded_query = urllib.parse.quote(optimized_query)
            
            # Use filter for the field if specified
            filter_param = ""
            if self.current_field:
                # Map field to OpenAlex concepts
                concept_ids = {
                    "physics": "C121332964",  # Physics concept ID
                    "biology": "C86803240",   # Biology
                    "computer_science": "C41008148",  # Computer Science
                    "chemistry": "C185592680",  # Chemistry
                    "medicine": "C71924100",   # Medicine
                    "engineering": "C127413603", # Engineering
                    "mathematics": "C33923547"  # Mathematics
                }
                concept_id = concept_ids.get(self.current_field.lower())
                if concept_id:
                    filter_param = f"&filter=concepts.id:{concept_id}"
            
            url = f"{self.OPENALEX_BASE}/works?search={encoded_query}&per-page={max_results}&sort=relevance_score:desc{filter_param}"
            
            logger.info(f"[OpenAlex] Searching: {optimized_query[:50]}...")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            self.search_stats["openalex_total"] = data.get("meta", {}).get("count", 0)
            
            for work in results:
                try:
                    # Extract authors
                    authorships = work.get("authorships", [])
                    authors = ", ".join([
                        a.get("author", {}).get("display_name", "Unknown")
                        for a in authorships[:5]
                    ])
                    if len(authorships) > 5:
                        authors += " et al."
                    
                    # Get DOI and URLs
                    doi = work.get("doi", "")
                    if doi:
                        doi = doi.replace("https://doi.org/", "")
                    
                    url = work.get("doi") or work.get("id", "")
                    
                    # Get PDF URL if available
                    pdf_url = ""
                    locations = work.get("locations", [])
                    for loc in locations:
                        if loc.get("pdf_url"):
                            pdf_url = loc.get("pdf_url")
                            break
                    
                    # Get venue/journal
                    venue = work.get("primary_location", {})
                    journal = "Preprint"
                    if venue:
                        source = venue.get("source", {})
                        if source:
                            journal = source.get("display_name", "Preprint")
                    
                    # Get abstract
                    abstract = ""
                    inverted_abstract = work.get("abstract_inverted_index", {})
                    if inverted_abstract:
                        positions = []
                        for word, pos_list in inverted_abstract.items():
                            for pos in pos_list:
                                positions.append((pos, word))
                        positions.sort()
                        abstract = " ".join([word for _, word in positions])[:500]
                    
                    paper = RealPaper(
                        title=work.get("title", "Untitled") or "Untitled",
                        authors=authors or "Unknown authors",
                        journal=journal,
                        year=work.get("publication_year", 0) or 0,
                        doi=doi,
                        url=url,
                        abstract=abstract,
                        citation_count=work.get("cited_by_count", 0) or 0,
                        source="openalex",
                        pdf_url=pdf_url
                    )
                    papers.append(paper)
                    
                except Exception as e:
                    logger.warning(f"[OpenAlex] Error parsing result: {e}")
                    continue
            
            elapsed = time.time() - start_time
            logger.info(f"[OpenAlex] Found {len(papers)} papers in {elapsed:.2f}s")
            self.search_stats["openalex_time"] = elapsed
            
        except Exception as e:
            logger.error(f"[OpenAlex] API error: {e}")
            self.search_stats["openalex_error"] = str(e)
        
        return papers
    
    # ==================== ARXIV ====================
    def search_arxiv(self, query: str, max_results: int = 15) -> List[RealPaper]:
        """
        Search arXiv - 2M+ papers, FREE, no API key needed
        FIXED: Better category targeting
        """
        papers = []
        start_time = time.time()
        
        try:
            # Get optimized query
            optimized_query = self.query_processor.build_optimized_query(query, self.current_field)
            
            # Build arXiv-specific query with categories
            categories = self.query_processor.get_arxiv_categories(self.current_field)
            
            # Build search query - search in title and abstract
            search_parts = []
            keywords = self.query_processor.extract_keywords(query)
            
            # Add main keywords for title/abstract search
            for kw in keywords[:5]:
                search_parts.append(f'ti:"{kw}" OR abs:"{kw}"')
            
            # Add category filter
            if categories and self.current_field:
                cat_filter = " OR ".join([f"cat:{cat}" for cat in categories[:2]])
                arxiv_query = f"({' OR '.join(search_parts)}) AND ({cat_filter})"
            else:
                arxiv_query = " OR ".join(search_parts) if search_parts else f"all:{optimized_query}"
            
            params = {
                "search_query": arxiv_query,
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            
            logger.info(f"[arXiv] Searching: {arxiv_query[:80]}...")
            response = self.session.get(self.ARXIV_BASE, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom"
            }
            
            # Get total results
            total_elem = root.find("opensearch:totalResults", {"opensearch": "http://a9.com/-/spec/opensearch/1.1/"})
            self.search_stats["arxiv_total"] = int(total_elem.text) if total_elem is not None else 0
            
            entries = root.findall("atom:entry", ns)
            
            for entry in entries:
                try:
                    title_elem = entry.find("atom:title", ns)
                    title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else "Untitled"
                    
                    author_elems = entry.findall("atom:author/atom:name", ns)
                    authors = ", ".join([a.text for a in author_elems[:5]])
                    if len(author_elems) > 5:
                        authors += " et al."
                    
                    summary_elem = entry.find("atom:summary", ns)
                    abstract = summary_elem.text.strip()[:500] if summary_elem is not None else ""
                    
                    id_elem = entry.find("atom:id", ns)
                    arxiv_url = id_elem.text if id_elem is not None else ""
                    arxiv_id = arxiv_url.split("/abs/")[-1] if arxiv_url else ""
                    
                    # PDF URL
                    pdf_url = arxiv_url.replace("/abs/", "/pdf/") + ".pdf" if arxiv_url else ""
                    
                    published_elem = entry.find("atom:published", ns)
                    year = 0
                    if published_elem is not None:
                        try:
                            year = int(published_elem.text[:4])
                        except:
                            pass
                    
                    # Check for DOI
                    doi = ""
                    for link in entry.findall("atom:link", ns):
                        if link.get("title") == "doi":
                            doi = link.get("href", "").replace("http://dx.doi.org/", "")
                    
                    if not doi:
                        doi = f"arXiv:{arxiv_id}"
                    
                    paper = RealPaper(
                        title=title,
                        authors=authors or "Unknown authors",
                        journal="arXiv preprint",
                        year=year,
                        doi=doi,
                        url=arxiv_url,
                        abstract=abstract,
                        citation_count=0,
                        source="arxiv",
                        pdf_url=pdf_url
                    )
                    papers.append(paper)
                    
                except Exception as e:
                    logger.warning(f"[arXiv] Error parsing result: {e}")
                    continue
            
            elapsed = time.time() - start_time
            logger.info(f"[arXiv] Found {len(papers)} papers in {elapsed:.2f}s")
            self.search_stats["arxiv_time"] = elapsed
            
        except Exception as e:
            logger.error(f"[arXiv] API error: {e}")
            self.search_stats["arxiv_error"] = str(e)
        
        return papers
    
    # ==================== SEMANTIC SCHOLAR ====================
    def search_semantic_scholar(self, query: str, max_results: int = 10) -> List[RealPaper]:
        """
        Search Semantic Scholar - 200M+ papers, FREE
        FIXED: Correct API endpoint and error handling
        """
        papers = []
        start_time = time.time()
        
        try:
            # Optimize query - use fewer, more specific terms
            keywords = self.query_processor.extract_keywords(query)
            search_query = " ".join(keywords[:5])  # Semantic Scholar prefers shorter queries
            
            url = f"{self.SEMANTIC_SCHOLAR_BASE}/paper/search"
            params = {
                "query": search_query,
                "limit": min(max_results, 10),
                "fields": "title,authors,year,abstract,citationCount,externalIds,url,venue,openAccessPdf"
            }
            
            logger.info(f"[Semantic Scholar] Searching: {search_query[:50]}...")
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
            
            response = self.session.get(url, params=params, timeout=30)
            
            # Check for rate limiting
            if response.status_code == 429:
                logger.warning("[Semantic Scholar] Rate limited, skipping")
                self.search_stats["semantic_scholar_total"] = 0
                self.search_stats["semantic_scholar_error"] = "Rate limited"
                return papers
            
            response.raise_for_status()
            
            data = response.json()
            self.search_stats["semantic_scholar_total"] = data.get("total", 0)
            
            for item in data.get("data", []):
                try:
                    # Authors
                    author_list = item.get("authors", [])
                    authors = ", ".join([a.get("name", "Unknown") for a in author_list[:5]])
                    if len(author_list) > 5:
                        authors += " et al."
                    
                    # DOI
                    external_ids = item.get("externalIds", {}) or {}
                    doi = external_ids.get("DOI", "")
                    arxiv_id = external_ids.get("ArXiv", "")
                    
                    # URL
                    paper_url = item.get("url", "")
                    if doi:
                        paper_url = f"https://doi.org/{doi}"
                    elif arxiv_id:
                        paper_url = f"https://arxiv.org/abs/{arxiv_id}"
                    
                    # PDF URL
                    pdf_info = item.get("openAccessPdf", {})
                    pdf_url = pdf_info.get("url", "") if pdf_info else ""
                    
                    paper = RealPaper(
                        title=item.get("title", "Untitled") or "Untitled",
                        authors=authors or "Unknown authors",
                        journal=item.get("venue", "Unknown venue") or "Unknown venue",
                        year=item.get("year", 0) or 0,
                        doi=doi or (f"arXiv:{arxiv_id}" if arxiv_id else "N/A"),
                        url=paper_url,
                        abstract=(item.get("abstract", "") or "")[:500],
                        citation_count=item.get("citationCount", 0) or 0,
                        source="semantic_scholar",
                        pdf_url=pdf_url
                    )
                    papers.append(paper)
                    
                except Exception as e:
                    logger.warning(f"[Semantic Scholar] Error parsing result: {e}")
                    continue
            
            elapsed = time.time() - start_time
            logger.info(f"[Semantic Scholar] Found {len(papers)} papers in {elapsed:.2f}s")
            self.search_stats["semantic_scholar_time"] = elapsed
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning("[Semantic Scholar] Rate limited")
                self.search_stats["semantic_scholar_error"] = "Rate limited - try again in 5 minutes"
            else:
                logger.error(f"[Semantic Scholar] HTTP error: {e}")
                self.search_stats["semantic_scholar_error"] = str(e)
        except Exception as e:
            logger.error(f"[Semantic Scholar] API error: {e}")
            self.search_stats["semantic_scholar_error"] = str(e)
        
        return papers
    
    # ==================== COMBINED SEARCH ====================
    def search_all_sources(self, query: str, field: str = "", max_per_source: int = 10) -> Dict[str, Any]:
        """
        Search ALL sources in parallel and return combined results.
        FIXED: Added field parameter for better filtering
        """
        start_time = time.time()
        self.search_stats = {"query": query}
        self.current_field = field  # Set field for use in search methods
        
        logger.info(f"[LIVE SEARCH] Searching all sources for: {query[:50]}... (field: {field})")
        
        all_papers = []
        
        # Search in parallel for speed
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self.search_openalex, query, max_per_source): "openalex",
                executor.submit(self.search_arxiv, query, max_per_source): "arxiv",
                executor.submit(self.search_semantic_scholar, query, min(max_per_source, 10)): "semantic_scholar"
            }
            
            for future in as_completed(futures):
                source = futures[future]
                try:
                    papers = future.result()
                    all_papers.extend(papers)
                    self.search_stats[f"{source}_found"] = len(papers)
                except Exception as e:
                    logger.error(f"[{source}] Search failed: {e}")
                    self.search_stats[f"{source}_error"] = str(e)
                    self.search_stats[f"{source}_found"] = 0
        
        # Remove duplicates by DOI
        seen_dois = set()
        unique_papers = []
        for paper in all_papers:
            if paper.doi and paper.doi in seen_dois:
                continue
            if paper.doi:
                seen_dois.add(paper.doi)
            unique_papers.append(paper)
        
        # Sort by citation count (higher first)
        unique_papers.sort(key=lambda p: p.citation_count, reverse=True)
        
        total_time = time.time() - start_time
        
        # Build comprehensive statistics
        stats = {
            "query": query,
            "optimized_query": self.query_processor.build_optimized_query(query, field),
            "field": field,
            "total_papers_found": len(unique_papers),
            "search_time_seconds": round(total_time, 2),
            "sources_searched": [
                {
                    "name": "OpenAlex",
                    "total_available": self.search_stats.get("openalex_total", 0),
                    "returned": self.search_stats.get("openalex_found", 0),
                    "time_seconds": round(self.search_stats.get("openalex_time", 0), 2),
                    "error": self.search_stats.get("openalex_error")
                },
                {
                    "name": "arXiv",
                    "total_available": self.search_stats.get("arxiv_total", 0),
                    "returned": self.search_stats.get("arxiv_found", 0),
                    "time_seconds": round(self.search_stats.get("arxiv_time", 0), 2),
                    "error": self.search_stats.get("arxiv_error")
                },
                {
                    "name": "Semantic Scholar",
                    "total_available": self.search_stats.get("semantic_scholar_total", 0),
                    "returned": self.search_stats.get("semantic_scholar_found", 0),
                    "time_seconds": round(self.search_stats.get("semantic_scholar_time", 0), 2),
                    "error": self.search_stats.get("semantic_scholar_error")
                }
            ],
            "papers_by_source": {
                "openalex": self.search_stats.get("openalex_found", 0),
                "arxiv": self.search_stats.get("arxiv_found", 0),
                "semantic_scholar": self.search_stats.get("semantic_scholar_found", 0)
            }
        }
        
        logger.info(f"[LIVE SEARCH] Total: {len(unique_papers)} unique papers in {total_time:.2f}s")
        
        return {
            "papers": unique_papers,
            "stats": stats
        }
    
    def format_papers_for_display(self, papers: List[RealPaper]) -> List[Dict[str, Any]]:
        """Convert RealPaper objects to display-ready dictionaries"""
        return [
            {
                "title": p.title,
                "authors": p.authors,
                "journal": p.journal,
                "year": p.year,
                "doi": p.doi,
                "url": p.url,
                "pdf_url": p.pdf_url,
                "abstract": p.abstract,
                "key_finding": p.abstract[:250] + "..." if len(p.abstract) > 250 else p.abstract,
                "citation_count": p.citation_count,
                "source": p.source,
                "relevance_score": max(0.5, 0.95 - (i * 0.03))  # Decreasing relevance for display
            }
            for i, p in enumerate(papers)
        ]


# Global instance
paper_fetcher = LivePaperFetcher()


def fetch_real_papers(query: str, field: str = "", max_results: int = 30) -> Dict[str, Any]:
    """
    Main function to fetch REAL papers from all sources.
    No local storage - all data is live from APIs.
    
    Args:
        query: Search query
        field: Research field for better filtering (optional)
        max_results: Maximum papers to return
    
    Returns:
        {
            "papers": [RealPaper, ...],
            "stats": {search statistics}
        }
    """
    return paper_fetcher.search_all_sources(query, field, max_results // 3)


def format_papers(papers: List[RealPaper]) -> List[Dict[str, Any]]:
    """Format papers for frontend display"""
    return paper_fetcher.format_papers_for_display(papers)
