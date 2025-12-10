"""
Dataset Fetcher - ENHANCED VERSION
Fetches REAL datasets dynamically based on query and field
Fixed: Dynamic dataset fetching, removed hardcoded field bias
"""

import requests
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import re

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RealDataset:
    """Real dataset with download information"""
    name: str
    source: str
    url: str
    description: str
    size: str
    format: str
    license: str
    downloads: int
    task_categories: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetFetcher:
    """
    Fetches REAL datasets from multiple sources.
    All live fetching - no local storage.
    """
    
    HF_API = "https://huggingface.co/api/datasets"
    PWC_API = "https://paperswithcode.com/api/v1/datasets"
    
    # Curated datasets by field (REAL datasets with working URLs)
    FIELD_DATASETS = {
        "physics": [
            RealDataset(
                name="CERN Open Data Portal",
                source="CERN",
                url="https://opendata.cern.ch/",
                description="Open access data from the Large Hadron Collider experiments including collision data and simulations",
                size="2+ petabytes",
                format="ROOT, CSV, HDF5",
                license="CC0",
                downloads=0,
                task_categories=["particle-physics", "high-energy-physics"]
            ),
            RealDataset(
                name="Quantum Machine Learning Datasets",
                source="IBM Quantum",
                url="https://qiskit.org/documentation/machine-learning/tutorials/index.html",
                description="Datasets for quantum machine learning experiments including classification and optimization problems",
                size="Various",
                format="NumPy, Qiskit",
                license="Apache-2.0",
                downloads=0,
                task_categories=["quantum-computing", "machine-learning"]
            ),
            RealDataset(
                name="Materials Project",
                source="Lawrence Berkeley National Lab",
                url="https://materialsproject.org/",
                description="Computed information on known and predicted materials for materials science research",
                size="150,000+ materials",
                format="JSON, API",
                license="CC BY 4.0",
                downloads=0,
                task_categories=["materials-science", "physics"]
            )
        ],
        "biology": [
            RealDataset(
                name="NCBI Gene Expression Omnibus (GEO)",
                source="NCBI",
                url="https://www.ncbi.nlm.nih.gov/geo/",
                description="Public repository for gene expression data supporting high-throughput functional genomics research",
                size="4+ million samples",
                format="CEL, TXT, CSV",
                license="Public Domain",
                downloads=0,
                task_categories=["genomics", "gene-expression"]
            ),
            RealDataset(
                name="UniProt Protein Database",
                source="UniProt Consortium",
                url="https://www.uniprot.org/",
                description="Comprehensive resource for protein sequence and annotation data",
                size="250+ million sequences",
                format="FASTA, XML, JSON",
                license="CC BY 4.0",
                downloads=0,
                task_categories=["proteomics", "bioinformatics"]
            ),
            RealDataset(
                name="Protein Data Bank (PDB)",
                source="RCSB",
                url="https://www.rcsb.org/",
                description="3D structural data of biological macromolecules",
                size="200,000+ structures",
                format="PDB, mmCIF",
                license="CC0",
                downloads=0,
                task_categories=["structural-biology", "protein-structure"]
            )
        ],
        "computer_science": [
            RealDataset(
                name="ImageNet",
                source="Stanford Vision Lab",
                url="https://www.image-net.org/",
                description="Large visual database for visual object recognition research",
                size="14+ million images",
                format="JPEG",
                license="Research only",
                downloads=0,
                task_categories=["image-classification", "computer-vision"]
            ),
            RealDataset(
                name="Common Crawl",
                source="Common Crawl Foundation",
                url="https://commoncrawl.org/",
                description="Open repository of web crawl data for NLP and web research",
                size="250+ billion pages",
                format="WARC, WET",
                license="CC0",
                downloads=0,
                task_categories=["nlp", "web-mining"]
            ),
            RealDataset(
                name="Papers With Code Datasets",
                source="Papers With Code",
                url="https://paperswithcode.com/datasets",
                description="Collection of ML/AI benchmark datasets with leaderboards",
                size="5000+ datasets",
                format="Various",
                license="Various",
                downloads=0,
                task_categories=["machine-learning", "benchmarks"]
            )
        ],
        "chemistry": [
            RealDataset(
                name="PubChem",
                source="NCBI",
                url="https://pubchem.ncbi.nlm.nih.gov/",
                description="Chemical molecules and their biological activities",
                size="110+ million compounds",
                format="SDF, SMILES, JSON",
                license="Public Domain",
                downloads=0,
                task_categories=["chemistry", "drug-discovery"]
            ),
            RealDataset(
                name="ChEMBL",
                source="EMBL-EBI",
                url="https://www.ebi.ac.uk/chembl/",
                description="Bioactive drug-like small molecules database",
                size="2+ million compounds",
                format="SDF, SQL",
                license="CC BY-SA 3.0",
                downloads=0,
                task_categories=["drug-discovery", "cheminformatics"]
            )
        ],
        "medicine": [
            RealDataset(
                name="MIMIC-III",
                source="PhysioNet",
                url="https://physionet.org/content/mimiciii/",
                description="Critical care data for over 40,000 patients",
                size="6+ GB",
                format="CSV, PostgreSQL",
                license="PhysioNet License",
                downloads=0,
                task_categories=["clinical-data", "healthcare"]
            ),
            RealDataset(
                name="NIH Chest X-ray Dataset",
                source="NIH Clinical Center",
                url="https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community",
                description="112,000+ chest X-ray images with disease labels",
                size="42 GB",
                format="PNG",
                license="CC0",
                downloads=0,
                task_categories=["medical-imaging", "radiology"]
            )
        ]
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ScienceBridge/1.0"
        })
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract relevant keywords from query"""
        stop_words = {"i'm", "i", "am", "are", "is", "the", "a", "an", "and", "or", "for", 
                      "to", "from", "that", "which", "with", "could", "would", "should"}
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        return [w for w in words if w not in stop_words][:5]
    
    def search_huggingface(self, query: str, max_results: int = 5) -> List[RealDataset]:
        """
        Search Hugging Face Datasets - FREE, no key required
        """
        datasets = []
        
        try:
            # Extract keywords for better search
            keywords = self._extract_keywords(query)
            search_query = " ".join(keywords[:3])
            
            url = f"{self.HF_API}"
            params = {
                "search": search_query,
                "limit": max_results,
                "sort": "downloads",
                "direction": -1
            }
            
            logger.info(f"[HuggingFace] Searching datasets: {search_query[:50]}...")
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            for item in response.json():
                try:
                    dataset = RealDataset(
                        name=item.get("id", "unknown"),
                        source="Hugging Face",
                        url=f"https://huggingface.co/datasets/{item.get('id', '')}",
                        description=item.get("description", "No description")[:300] if item.get("description") else "No description",
                        size=f"{item.get('downloads', 0):,} downloads",
                        format="Parquet, CSV, JSON",
                        license=item.get("license", "Unknown") or "Unknown",
                        downloads=item.get("downloads", 0) or 0,
                        task_categories=item.get("task_categories", [])[:3] if item.get("task_categories") else []
                    )
                    datasets.append(dataset)
                except Exception as e:
                    logger.warning(f"[HuggingFace] Parse error: {e}")
                    continue
            
            logger.info(f"[HuggingFace] Found {len(datasets)} datasets")
            
        except Exception as e:
            logger.error(f"[HuggingFace] API error: {e}")
        
        return datasets
    
    def search_papers_with_code(self, query: str, max_results: int = 5) -> List[RealDataset]:
        """
        Search Papers With Code Datasets - FREE, no key required
        """
        datasets = []
        
        try:
            keywords = self._extract_keywords(query)
            search_query = " ".join(keywords[:3])
            
            url = self.PWC_API
            params = {
                "q": search_query,
                "items_per_page": max_results
            }
            
            logger.info(f"[PapersWithCode] Searching datasets: {search_query[:50]}...")
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            for item in data.get("results", []):
                try:
                    # Get proper URL
                    dataset_url = item.get("url", "")
                    if not dataset_url:
                        dataset_url = f"https://paperswithcode.com/dataset/{item.get('id', '')}"
                    
                    dataset = RealDataset(
                        name=item.get("name", "unknown"),
                        source="Papers With Code",
                        url=dataset_url,
                        description=item.get("description", "No description")[:300] if item.get("description") else "No description",
                        size=f"{item.get('num_papers', 0)} related papers" if item.get("num_papers") else "Unknown size",
                        format="Various",
                        license="See website",
                        downloads=0,
                        task_categories=item.get("tasks", [])[:3] if isinstance(item.get("tasks"), list) else []
                    )
                    datasets.append(dataset)
                except Exception as e:
                    logger.warning(f"[PapersWithCode] Parse error: {e}")
                    continue
            
            logger.info(f"[PapersWithCode] Found {len(datasets)} datasets")
            
        except Exception as e:
            logger.error(f"[PapersWithCode] API error: {e}")
        
        return datasets
    
    def get_field_datasets(self, field: str) -> List[RealDataset]:
        """
        Return curated datasets for the specified field.
        These are real datasets with verified URLs.
        """
        field_lower = field.lower().replace(" ", "_")
        
        # Try exact match first
        if field_lower in self.FIELD_DATASETS:
            return self.FIELD_DATASETS[field_lower][:3]
        
        # Try partial match
        for key in self.FIELD_DATASETS:
            if key in field_lower or field_lower in key:
                return self.FIELD_DATASETS[key][:3]
        
        # Default to computer science datasets
        return self.FIELD_DATASETS.get("computer_science", [])[:2]
    
    def search_all(self, query: str, field: str = "", max_per_source: int = 3) -> List[RealDataset]:
        """
        Search all dataset sources and combine results.
        """
        all_datasets = []
        
        # Search HuggingFace
        hf_datasets = self.search_huggingface(query, max_per_source)
        all_datasets.extend(hf_datasets)
        
        # Search Papers With Code
        pwc_datasets = self.search_papers_with_code(query, max_per_source)
        all_datasets.extend(pwc_datasets)
        
        # Add field-specific curated datasets
        if field:
            curated = self.get_field_datasets(field)
            all_datasets.extend(curated)
        
        # Remove duplicates by name
        seen = set()
        unique = []
        for d in all_datasets:
            if d.name.lower() not in seen:
                seen.add(d.name.lower())
                unique.append(d)
        
        logger.info(f"[Datasets] Total found: {len(unique)}")
        return unique
    
    def format_datasets_for_display(self, datasets: List[RealDataset]) -> List[Dict[str, Any]]:
        """Format datasets for frontend display"""
        return [
            {
                "name": d.name,
                "source": d.source,
                "url": d.url,
                "description": d.description,
                "size": d.size,
                "format": d.format,
                "license": d.license,
                "downloads": d.downloads,
                "task_categories": d.task_categories,
                "download_instruction": f"Visit {d.url} to download"
            }
            for d in datasets
        ]


# Global instance
dataset_fetcher = DatasetFetcher()


def fetch_datasets(query: str, field: str = "", max_results: int = 6) -> List[Dict[str, Any]]:
    """
    Main function to fetch real datasets.
    Returns formatted datasets ready for display.
    """
    datasets = dataset_fetcher.search_all(query, field, max_results // 2)
    return dataset_fetcher.format_datasets_for_display(datasets)
