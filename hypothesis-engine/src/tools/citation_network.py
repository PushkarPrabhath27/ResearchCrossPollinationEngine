"""
Citation Network Tools

LangChain tools for exploring citation relationships, finding influential papers,
and tracing the evolution of ideas across scientific literature.
"""

from langchain.tools import tool
from typing import Optional, List, Dict
import json
from src.database.metadata_store import MetadataStore
from src.ingestion.citation_network import CitationNetworkBuilder
from src.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Singleton instances
_metadata_store = None
_citation_network = None


def get_metadata_store() -> MetadataStore:
    """Get or create metadata store instance"""
    global _metadata_store
    if _metadata_store is None:
        config = get_settings()
        _metadata_store = MetadataStore(config=config)
    return _metadata_store


def get_citation_network() -> CitationNetworkBuilder:
    """Get or create citation network instance"""
    global _citation_network
    if _citation_network is None:
        _citation_network = CitationNetworkBuilder()
    return _citation_network


@tool
def get_citations_tool(paper_id: str, direction: str = "citing", max_results: int = 20) -> str:
    """
    Get papers that cite or are cited by a given paper.
    
    Args:
        paper_id: ID of the paper
        direction: "citing" (papers that cite this) or "cited" (papers this cites)
        max_results: Maximum number of results
    
    Returns:
        JSON with citation information
    """
    logger.info(f"Getting {direction} papers for {paper_id}")
    
    try:
        store = get_metadata_store()
        citations = store.get_citations(paper_id, direction=direction)
        
        return json.dumps({
            "success": True,
            "paper_id": paper_id,
            "direction": direction,
            "num_citations": len(citations),
            "citations": citations[:max_results]
        }, indent=2)
    except Exception as e:
        logger.error(f"Citation lookup failed: {e}")
        return json.dumps({"success": False, "error": str(e)})


@tool
def find_influential_papers_tool(field: str, min_citations: int = 100, top_k: int = 10) -> str:
    """
    Find highly influential papers in a field.
    
    Args:
        field: Scientific field
        min_citations: Minimum citation count
        top_k: Number of papers to return
    
    Returns:
        JSON with influential papers
    """
    logger.info(f"Finding influential papers in {field}")
    
    try:
        store = get_metadata_store()
        # Query papers by field and citations
        papers = store.search_by_field(field=field)
        
        # Filter and sort by citations
        influential = [p for p in papers if p.get('citations_count', 0) >= min_citations]
        influential.sort(key=lambda x: x.get('citations_count', 0), reverse=True)
        
        return json.dumps({
            "success": True,
            "field": field,
            "num_results": len(influential[:top_k]),
            "influential_papers": influential[:top_k]
        }, indent=2)
    except Exception as e:
        logger.error(f"Influential papers search failed: {e}")
        return json.dumps({"success": False, "error": str(e)})


@tool
def trace_citation_path_tool(source_paper_id: str, target_paper_id: str, max_depth: int = 3) -> str:
    """
    Find citation path between two papers.
    
    Args:
        source_paper_id: Starting paper
        target_paper_id: Target paper
        max_depth: Maximum path depth
    
    Returns:
        JSON with citation path if exists
    """
    logger.info(f"Tracing path from {source_paper_id} to {target_paper_id}")
    
    try:
        network = get_citation_network()
        path = network.find_citation_path(source_paper_id, target_paper_id, max_depth)
        
        return json.dumps({
            "success": True,
            "source": source_paper_id,
            "target": target_paper_id,
            "path_exists": path is not None,
            "path_length": len(path) if path else 0,
            "path": path
        }, indent=2)
    except Exception as e:
        logger.error(f"Citation path tracing failed: {e}")
        return json.dumps({"success": False, "error": str(e)})


def get_all_citation_tools() -> List:
    """Get list of all citation network tools"""
    return [
        get_citations_tool,
        find_influential_papers_tool,
        trace_citation_path_tool
    ]
