"""
PubMed/PMC Paper Fetcher

Comprehensive fetcher for searching and downloading papers from PubMed and PubMed Central.
Uses Biopython's Entrez interface with proper rate limiting and caching.
"""

from Bio import Entrez
from Bio import Medline
import time
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from tqdm import tqdm
import requests

from src.config import Settings
from src.utils.logger import get_logger
from src.utils.helpers import retry_with_backoff, ensure_dir, save_json, load_json

logger = get_logger(__name__)


class PubMedFetcher:
    """
    Fetches papers from PubMed and PubMed Central
    
    Supports abstract and full-text retrieval with proper NCBI API compliance.
    Includes rate limiting, caching, and comprehensive metadata extraction.
    """
    
    # NCBI rate limits: 3 requests/second without API key, 10/second with key
    RATE_LIMIT_NO_KEY = 1/3
    RATE_LIMIT_WITH_KEY = 1/10
    
    def __init__(self, email: str, api_key: Optional[str] = None, config: Optional[Settings] = None):
        """
        Initialize PubMed fetcher
        
        Args:
            email: Email address (required by NCBI)
            api_key: NCBI API key (optional, increases rate limit)
            config: Application configuration
        """
        if not email or '@' not in email:
            raise ValueError("Valid email address required for NCBI Entrez")
        
        # Set Entrez credentials
        Entrez.email = email
        Entrez.api_key = api_key
        
        self.email = email
        self.api_key = api_key
        self.config = config
        
        # Set rate limit based on API key
        self.rate_limit = self.RATE_LIMIT_WITH_KEY if api_key else self.RATE_LIMIT_NO_KEY
        self.last_request_time = 0
        
        # Setup directories
        if config:
            self.raw_data_dir = Path(config.ingestion.raw_data_dir) / "pubmed"
            self.processed_data_dir = Path(config.ingestion.processed_data_dir) / "pubmed"
        else:
            self.raw_data_dir = Path("./data/raw/pubmed")
            self.processed_data_dir = Path("./data/processed/pubmed")
        
        ensure_dir(self.raw_data_dir)
        ensure_dir(self.processed_data_dir)
        
        # Cache for avoiding redundant API calls
        self.cache_file = self.raw_data_dir / "query_cache.json"
        self.cache = self._load_cache()
        
        logger.info(f"PubMedFetcher initialized with email: {email}, API key: {'Yes' if api_key else 'No'}")
    
    def _load_cache(self) -> Dict:
        """Load query cache"""
        if self.cache_file.exists():
            try:
                return load_json(str(self.cache_file))
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save query cache"""
        try:
            save_json(self.cache, str(self.cache_file))
        except Exception as e:
            logger.error(f"Could not save cache: {e}")
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    def search(
        self,
        query: str,
        max_results: int = 100,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        publication_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Search PubMed
        
        Args:
            query: PubMed query string (supports advanced search syntax)
            max_results: Maximum number of results
            date_from: Start date (YYYY/MM/DD)
            date_to: End date (YYYY/MM/DD)
            publication_types: Filter by publication types
        
        Returns:
            List of paper dictionaries with IDs
        """
        logger.info(f"Searching PubMed: '{query}', max_results={max_results}")
        
        # Build query with filters
        full_query = query
        
        if date_from and date_to:
            full_query += f" AND ({date_from}[PDAT]:{date_to}[PDAT])"
        elif date_from:
            full_query += f" AND ({date_from}[PDAT]:3000[PDAT])"
        elif date_to:
            full_query += f" AND (1800[PDAT]:{date_to}[PDAT])"
        
        if publication_types:
            pt_query = " OR ".join([f"{pt}[PT]" for pt in publication_types])
            full_query += f" AND ({pt_query})"
        
        # Check cache
        cache_key = f"{full_query}_{max_results}"
        if cache_key in self.cache:
            logger.info("Using cached search results")
            return self.cache[cache_key]
        
        # Perform search
        self._rate_limit()
        
        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=full_query,
                retmax=max_results,
                sort="relevance"
            )
            record = Entrez.read(handle)
            handle.close()
            
            id_list = record["IdList"]
            logger.info(f"Found {len(id_list)} PubMed IDs")
            
            results = [{"Id": pmid} for pmid in id_list]
            
            # Cache results
            self.cache[cache_key] = results
            self._save_cache()
            
            return results
            
        except Exception as e:
            logger.error(f"PubMed search failed: {e}", exc_info=True)
            raise
    
    def search_pmc(self, query: str, max_results: int = 100) -> List[Dict]:
        """
        Search PubMed Central for full-text articles
        
        Args:
            query: PMC query string
            max_results: Maximum number of results
        
        Returns:
            List of dictionaries with PMC IDs
        """
        logger.info(f"Searching PMC: '{query}', max_results={max_results}")
        
        self._rate_limit()
        
        try:
            handle = Entrez.esearch(
                db="pmc",
                term=query,
                retmax=max_results,
                sort="relevance"
            )
            record = Entrez.read(handle)
            handle.close()
            
            id_list = record["IdList"]
            logger.info(f"Found {len(id_list)} PMC IDs")
            
            return [{"PMC_Id": pmc_id} for pmc_id in id_list]
            
        except Exception as e:
            logger.error(f"PMC search failed: {e}", exc_info=True)
            raise
    
    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    def fetch_details(self, pubmed_ids: List[str]) -> List[Dict]:
        """
        Fetch detailed information for PubMed IDs
        
        Args:
            pubmed_ids: List of PubMed IDs
        
        Returns:
            List of detailed paper metadata dictionaries
        """
        if not pubmed_ids:
            return []
        
        logger.info(f"Fetching details for {len(pubmed_ids)} papers")
        
        details = []
        
        # Process in batches to respect rate limits  
        batch_size = 100
        for i in range(0, len(pubmed_ids), batch_size):
            batch = pubmed_ids[i:i+batch_size]
            
            self._rate_limit()
            
            try:
                # Fetch XML
                handle = Entrez.efetch(
                    db="pubmed",
                    id=",".join(batch),
                    retmode="xml"
                )
                xml_data = handle.read()
                handle.close()
                
                # Parse XML
                batch_details = self.parse_pubmed_xml(xml_data)
                details.extend(batch_details)
                
            except Exception as e:
                logger.error(f"Failed to fetch batch: {e}")
                continue
        
        logger.info(f"Successfully fetched {len(details)} paper details")
        return details
    
    def parse_pubmed_xml(self, xml_content: str) -> List[Dict]:
        """
        Parse PubMed XML response into structured metadata
        
        Args:
            xml_content: XML string from PubMed
        
        Returns:
            List of paper metadata dictionaries
        """
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
            
            for article in root.findall(".//PubmedArticle"):
                paper = {}
                
                # PMID
                pmid = article.find(".//PMID")
                paper['pmid'] = pmid.text if pmid is not None else None
                
                # PMC ID
                pmc_id = article.find(".//ArticleId[@IdType='pmc']")
                paper['pmc_id'] = pmc_id.text if pmc_id is not None else None
                
                # DOI
                doi = article.find(".//ArticleId[@IdType='doi']")
                paper['doi'] = doi.text if doi is not None else None
                
                # Title
                title = article.find(".//ArticleTitle")
                paper['title'] = title.text if title is not None else ""
                
                # Abstract
                abstract_parts = article.findall(".//AbstractText")
                if abstract_parts:
                    abstract = " ".join([a.text for a in abstract_parts if a.text])
                    paper['abstract'] = abstract
                else:
                    paper['abstract'] = ""
                
                # Authors
                authors = []
                author_list = article.findall(".//Author")
                for author in author_list:
                    last_name = author.find("LastName")
                    fore_name = author.find("ForeName")
                    affiliation = author.find(".//Affiliation")
                    
                    author_dict = {
                        'last_name': last_name.text if last_name is not None else "",
                        'fore_name': fore_name.text if fore_name is not None else "",
                        'affiliation': affiliation.text if affiliation is not None else ""
                    }
                    authors.append(author_dict)
                
                paper['authors'] = authors
                
                # Journal info
                journal = article.find(".//Journal")
                if journal is not None:
                    journal_title = journal.find(".//Title")
                    volume = journal.find(".//Volume")
                    issue = journal.find(".//Issue")
                    
                    paper['journal'] = {
                        'title': journal_title.text if journal_title is not None else "",
                        'volume': volume.text if volume is not None else "",
                        'issue': issue.text if issue is not None else ""
                    }
                
                # Publication date
                pub_date = article.find(".//PubDate")
                if pub_date is not None:
                    year = pub_date.find("Year")
                    month = pub_date.find("Month")
                    day = pub_date.find("Day")
                    
                    paper['publication_date'] = {
                        'year': year.text if year is not None else "",
                        'month': month.text if month is not None else "",
                        'day': day.text if day is not None else ""
                    }
                
                # Keywords/MeSH terms
                mesh_terms = article.findall(".//MeshHeading/DescriptorName")
                paper['mesh_terms'] = [term.text for term in mesh_terms if term.text]
                
                # Publication types
                pub_types = article.findall(".//PublicationType")
                paper['publication_types'] = [pt.text for pt in pub_types if pt.text]
                
                # Chemicals
                chemicals = article.findall(".//Chemical/NameOfSubstance")
                paper['chemicals'] = [chem.text for chem in chemicals if chem.text]
                
                # Grants
                grants = []
                grant_list = article.findall(".//Grant")
                for grant in grant_list:
                    grant_id = grant.find("GrantID")
                    agency = grant.find("Agency")
                    
                    grants.append({
                        'grant_id': grant_id.text if grant_id is not None else "",
                        'agency': agency.text if agency is not None else ""
                    })
                
                paper['grants'] = grants
                
                # Pages
                pages = article.find(".//MedlinePgn")
                paper['pages'] = pages.text if pages is not None else ""
                
                papers.append(paper)
                
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            raise
        
        return papers
    
    def fetch_full_text(self, pmc_id: str) -> Optional[Dict]:
        """
        Fetch full text from PubMed Central
        
        Args:
            pmc_id: PMC ID (can include 'PMC' prefix)
        
        Returns:
            Dictionary with full-text sections or None if not available
        """
        # Remove 'PMC' prefix if present
        pmc_id = pmc_id.replace('PMC', '')
        
        logger.info(f"Fetching full text for PMC{pmc_id}")
        
        self._rate_limit()
        
        try:
            handle = Entrez.efetch(
                db="pmc",
                id=pmc_id,
                retmode="xml"
            )
            xml_data = handle.read()
            handle.close()
            
            return self.parse_pmc_xml(xml_data)
            
        except Exception as e:
            logger.error(f"Failed to fetch PMC full text: {e}")
            return None
    
    def parse_pmc_xml(self, xml_content: str) -> Dict:
        """
        Parse PMC XML to extract full-text sections
        
        Args:
            xml_content: XML string from PMC
        
        Returns:
            Dictionary with sections, figures, tables, references
        """
        result = {
            'sections': {},
            'figures': [],
            'tables': [],
            'references': []
        }
        
        try:
            root = ET.fromstring(xml_content)
            
            # Extract sections
            body = root.find(".//body")
            if body is not None:
                for section in body.findall(".//sec"):
                    title_elem = section.find(".//title")
                    if title_elem is not None:
                        section_title = title_elem.text
                        
                        # Get all paragraphs in section
                        paragraphs = section.findall(".//p")
                        section_text = "\n\n".join([p.text for p in paragraphs if p.text])
                        
                        result['sections'][section_title] = section_text
            
            # Extract figures
            for fig in root.findall(".//fig"):
                fig_label = fig.find(".//label")
                fig_caption = fig.find(".//caption")
                
                figure_dict = {
                    'label': fig_label.text if fig_label is not None else "",
                    'caption': fig_caption.text if fig_caption is not None else ""
                }
                result['figures'].append(figure_dict)
            
            # Extract tables
            for table in root.findall(".//table-wrap"):
                table_label = table.find(".//label")
                table_caption = table.find(".//caption")
                
               table_dict = {
                    'label': table_label.text if table_label is not None else "",
                    'caption': table_caption.text if table_caption is not None else ""
                }
                result['tables'].append(table_dict)
            
            # Extract references
            for ref in root.findall(".//ref"):
                citation = ref.find(".//mixed-citation")
                if citation is not None:
                    result['references'].append(citation.text if citation.text else "")
            
        except ET.ParseError as e:
            logger.error(f"PMC XML parsing error: {e}")
        
        return result
    
    def get_citations(self, pubmed_id: str) -> List[str]:
        """
        Get papers that cite this paper
        
        Args:
            pubmed_id: PubMed ID
        
        Returns:
            List of citing paper PubMed IDs
        """
        logger.info(f"Getting citations for PMID: {pubmed_id}")
        
        self._rate_limit()
        
        try:
            handle = Entrez.elink(
                dbfrom="pubmed",
                id=pubmed_id,
                linkname="pubmed_pubmed_citedin"
            )
            record = Entenz.read(handle)
            handle.close()
            
            if record and record[0]["LinkSetDb"]:
                citing_ids = [link["Id"] for link in record[0]["LinkSetDb"][0]["Link"]]
                logger.info(f"Found {len(citing_ids)} citing papers")
                return citing_ids
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get citations: {e}")
            return []
    
    def get_references(self, pubmed_id: str) -> List[str]:
        """
        Get papers that this paper cites
        
        Args:
            pubmed_id: PubMed ID
        
        Returns:
            List of referenced paper PubMed IDs
        """
        logger.info(f"Getting references for PMID: {pubmed_id}")
        
        self._rate_limit()
        
        try:
            handle = Entrez.elink(
                dbfrom="pubmed",
                id=pubmed_id,
                linkname="pubmed_pubmed_refs"
            )
            record = Entrez.read(handle)
            handle.close()
            
            if record and record[0]["LinkSetDb"]:
                ref_ids = [link["Id"] for link in record[0]["LinkSetDb"][0]["Link"]]
                logger.info(f"Found {len(ref_ids)} references")
                return ref_ids
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get references: {e}")
            return []
    
    def batch_fetch(
        self,
        pubmed_ids: List[str],
        batch_size: int = 100,
        include_full_text: bool = False
    ) -> List[Dict]:
        """
        Fetch papers in batches with progress tracking
        
        Args:
            pubmed_ids: List of PubMed IDs
            batch_size: Number of papers per batch
            include_full_text: Whether to fetch PMC full text
        
        Returns:
            List of paper metadata dictionaries
        """
        logger.info(f"Batch fetching {len(pubmed_ids)} papers")
        
        all_papers = []
        
        with tqdm(total=len(pubmed_ids), desc="Fetching PubMed papers") as pbar:
            for i in range(0, len(pubmed_ids), batch_size):
                batch = pubmed_ids[i:i+batch_size]
                
                try:
                    # Fetch details
                    papers = self.fetch_details(batch)
                    
                    # Fetch full text if requested and PMC ID available
                    if include_full_text:
                        for paper in papers:
                            if paper.get('pmc_id'):
                                full_text = self.fetch_full_text(paper['pmc_id'])
                                if full_text:
                                    paper['full_text'] = full_text
                    
                    all_papers.extend(papers)
                    
                    # Save batch
                    for paper in papers:
                        if paper.get('pmid'):
                            save_path = self.processed_data_dir / f"{paper['pmid']}.json"
                            save_json(paper, str(save_path))
                    
                except Exception as e:
                    logger.error(f"Batch fetch failed: {e}")
                
                pbar.update(len(batch))
                time.sleep(0.5)  # Extra delay between batches
        
        logger.info(f"Successfully fetched {len(all_papers)} papers")
        return all_papers


# Example usage
if __name__ == "__main__":
    from src.config import get_settings
    from src.utils.logger import setup_logging
    
    # Initialize
    setup_logging(level="INFO")
    config = get_settings()
    
    # Create fetcher (use email from config)
    fetcher = PubMedFetcher(
        email=config.api.entrez_email,
        api_key=config.api.entrez_api_key
    )
    
    # Example 1: Simple search
    print("\n=== Example 1: Search PubMed ===")
    results = fetcher.search(
        query="(cancer[Title]) AND (metastasis[Title/Abstract])",
        max_results=5,
        date_from="2023/01/01",
        date_to="2024/12/31"
    )
    print(f"Found {len(results)} papers")
    
    # Example 2: Fetch details
    if results:
        print("\n=== Example 2: Fetch Details ===")
        pmids = [r['Id'] for r in results]
        details = fetcher.fetch_details(pmids)
        
        for paper in details[:2]:
            print(f"\nTitle: {paper.get('title', 'N/A')}")
            print(f"PMID: {paper.get('pmid', 'N/A')}")
            print(f"PMC ID: {paper.get('pmc_id', 'N/A')}")
            print(f"Authors: {len(paper.get('authors', []))} authors")
            print(f"Abstract length: {len(paper.get('abstract', ''))} chars")
            print(f"MeSH terms: {', '.join(paper.get('mesh_terms', [])[:5])}")
    
    # Example 3: Search PMC for full text
    print("\n=== Example 3: Search PMC ===")
    pmc_results = fetcher.search_pmc("machine learning cancer", max_results=3)
    print(f"Found {len(pmc_results)} PMC articles")
    
    # Example 4: Fetch full text
    if details and details[0].get('pmc_id'):
        print("\n=== Example 4: Fetch Full Text ===")
        pmc_id = details[0]['pmc_id']
        full_text = fetcher.fetch_full_text(pmc_id)
        
        if full_text:
            print(f"Sections: {list(full_text['sections'].keys())}")
            print(f"Figures: {len(full_text['figures'])}")
            print(f"References: {len(full_text['references'])}")
    
    print("\n✅ All examples completed!")
