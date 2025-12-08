"""
Vector Search Tools

LangChain tools for semantic search across scientific papers using vector embeddings.
Integrates with ChromaDB for efficient similarity search.
"""

from langchain.tools import tool
from typing import Optional, List, Dict
import json
import numpy as np

from src.database.chroma_manager import ChromaManager
from src.ingestion.embedder import PaperEmbedder
from src.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Initialize components (singleton pattern)
_embedder = None
_chroma_manager = None
_config = None


def get_embedder() -> PaperEmbedder:
    """Get or create embedder instance"""
    global _embedder, _config
    if _embedder is None:
        _config = get_settings()
        _embedder = PaperEmbedder(config=_config)
    return _embedder


def get_chroma_manager() -> ChromaManager:
    """Get or create Chroma manager instance"""
    global _chroma_manager, _config
    if _chroma_manager is None:
        _config = get_settings()
        _chroma_manager = ChromaManager(config=_config)
    return _chroma_manager


@tool
def vector_search_tool(
    query: str,
    field: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    top_k: int = 10,
    min_citations: Optional[int] = None
) -> str:
    """
    Search for research papers semantically similar to the query.
    
    This tool searches across millions of scientific papers using semantic similarity.
    It understands the meaning of your query, not just keywords.
    
    Args:
        query: Natural language description of what you're looking for
        field: Optional filter by scientific field (biology, physics, computer_science, etc.)
        year_from: Optional minimum publication year
        year_to: Optional maximum publication year
        top_k: Number of results to return (default 10, max 50)
        min_citations: Optional minimum number of citations
    
    Returns:
        JSON string with list of relevant papers including:
        - title, authors, year
        - abstract
        - field and subfield
        - citation count
        - DOI and URL
        - relevance score
    
    Example usage:
        Search for papers on "deep learning for protein folding"
        Search with filters: "cancer immunotherapy" in biology field from 2020-2024
    """
    logger.info(f"Vector search: {query[:100]}...")
    
    try:
        # Initialize components
        embedder = get_embedder()
        chroma_manager = get_chroma_manager()
        
        # Generate query embedding
        query_embedding = embedder.embed_text(query)
        
        # Build metadata filters
        filters = {}
        if field:
            filters["field"] = field
        if year_from:
            filters["year"] = {"$gte": year_from}
        if year_to:
            if "year" in filters:
                filters["year"]["$lte"] = year_to
            else:
                filters["year"] = {"$lte": year_to}
        if min_citations:
            filters["citations_count"] = {"$gte": min_citations}
        
        # Search
        results = chroma_manager.search(
            query_embedding=query_embedding,
            n_results=min(top_k, 50),
            filters=filters if filters else None
        )
        
        # Format results
        formatted_results = []
        for result in results:
            metadata = result.get("metadata", {})
            formatted_results.append({
                "title": metadata.get("title", "Unknown"),
                "authors": metadata.get("authors", []),
                "year": metadata.get("year", 0),
                "abstract": (metadata.get("abstract", "")[:500] + "...") if metadata.get("abstract") else "",
                "field": metadata.get("field", ""),
                "subfield": metadata.get("subfield", ""),
                "citations": metadata.get("citations_count", 0),
                "doi": metadata.get("doi", ""),
                "url": metadata.get("url", ""),
                "relevance_score": round(result.get("similarity", 0), 3)
            })
        
        logger.info(f"Found {len(formatted_results)} results")
        
        return json.dumps({
            "success": True,
            "query": query,
            "filters": filters,
            "num_results": len(formatted_results),
            "results": formatted_results
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "query": query
        })


@tool
def multi_field_search_tool(
    query: str,
    fields: str,  # Comma-separated string for LangChain compatibility
    results_per_field: int = 5
) -> str:
    """
    Search across multiple scientific fields simultaneously.
    
    Useful for finding cross-domain connections. Searches each field independently
    and aggregates results.
    
    Args:
        query: Research question or topic
        fields: Comma-separated fields (e.g., "biology,physics,computer_science")
        results_per_field: Number of papers to return from each field
    
    Returns:
        JSON string with results grouped by field
    
    Example:
        multi_field_search("neural networks", "biology,computer_science", 5)
    """
    logger.info(f"Multi-field search across: {fields}")
    
    try:
        # Parse fields
        field_list = [f.strip() for f in fields.split(',')]
        
        embedder = get_embedder()
        chroma_manager = get_chroma_manager()
        
        # Generate query embedding once
        query_embedding = embedder.embed_text(query)
        
        all_results = {}
        
        for field in field_list:
            try:
                field_results = chroma_manager.search(
                    query_embedding=query_embedding,
                    n_results=results_per_field,
                    filters={"field": field}
                )
                
                formatted = []
                for result in field_results:
                    metadata = result.get("metadata", {})
                    formatted.append({
                        "title": metadata.get("title", "Unknown"),
                        "year": metadata.get("year", 0),
                        "citations": metadata.get("citations_count", 0),
                        "relevance": round(result.get("similarity", 0), 3)
                    })
                
                if formatted:
                    all_results[field] = formatted
                    
            except Exception as e:
                logger.warning(f"Search in {field} failed: {e}")
                continue
        
        return json.dumps({
            "success": True,
            "query": query,
            "fields_searched": field_list,
            "results_by_field": all_results,
            "total_papers": sum(len(r) for r in all_results.values())
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Multi-field search failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "query": query
        })


@tool
def similar_papers_tool(
    paper_id: str,
    top_k: int = 10,
    same_field_only: bool = False
) -> str:
    """
    Find papers similar to a given paper.
    
    Useful for exploring related work or finding papers that cite similar concepts.
    
    Args:
        paper_id: ID of the reference paper
        top_k: Number of similar papers to return
        same_field_only: If True, only return papers from the same field
    
    Returns:
        JSON string with similar papers
    
    Example:
        Find papers similar to "arxiv_2024_12345"
    """
    logger.info(f"Finding papers similar to: {paper_id}")
    
    try:
        chroma_manager = get_chroma_manager()
        
        # Get reference paper
        ref_paper = chroma_manager.get_paper(paper_id)
        
        if not ref_paper:
            return json.dumps({
                "success": False,
                "error": f"Paper {paper_id} not found"
            })
        
        # Get embedding
        ref_embedding = ref_paper.get("embedding")
        if ref_embedding is None:
            return json.dumps({
                "success": False,
                "error": "Paper has no embedding"
            })
        
        # Build filters
        filters = None
        if same_field_only:
            ref_field = ref_paper.get("metadata", {}).get("field")
            if ref_field:
                filters = {"field": ref_field}
        
        # Search for similar
        results = chroma_manager.search(
            query_embedding=ref_embedding,
            n_results=top_k + 1,  # +1 to exclude self
            filters=filters
        )
        
        # Filter out the reference paper itself
        similar = []
        for result in results:
            if result.get("id") != paper_id:
                metadata = result.get("metadata", {})
                similar.append({
                    "id": result.get("id"),
                    "title": metadata.get("title", "Unknown"),
                    "year": metadata.get("year", 0),
                    "field": metadata.get("field", ""),
                    "similarity": round(result.get("similarity", 0), 3)
                })
        
        return json.dumps({
            "success": True,
            "reference_paper": ref_paper.get("metadata", {}).get("title", paper_id),
            "num_results": len(similar[:top_k]),
            "similar_papers": similar[:top_k]
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Similar papers search failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "paper_id": paper_id
        })


@tool
def trending_papers_tool(
    field: str,
    days: int = 30,
    min_citations: int = 5,
    top_k: int = 20
) -> str:
    """
    Find recently popular papers in a field.
    
    Identifies papers that have gained traction recently, useful for finding
    cutting-edge research and emerging trends.
    
    Args:
        field: Scientific field
        days: Look back this many days (default 30)
        min_citations: Minimum citations (default 5)
        top_k: Number of papers to return
    
    Returns:
        JSON string with trending papers
    
    Example:
        Find trending biology papers from last 60 days with 10+ citations
    """
    logger.info(f"Finding trending papers in {field}")
    
    try:
        chroma_manager = get_chroma_manager()
        
        # Use ChromaManager's get_statistics function
        # In a real implementation, would query by ingestion_date
        # For now, search with field filter and sort by citations
        
        # Calculate date threshold
        from datetime import datetime, timedelta
        threshold_date = datetime.utcnow() - timedelta(days=days)
        
        # Search with filters
        results = chroma_manager.search_by_metadata(
            metadata_filter={
                "field": field,
                "citations_count": {"$gte": min_citations}
            },
            n_results=top_k * 2  # Get more to filter by date
        )
        
        # Format and filter by date if available
        trending = []
        for result in results:
            metadata = result.get("metadata", {})
            
            # Check ingestion date if available
            ingestion_date = metadata.get("ingestion_date")
            if ingestion_date:
                try:
                    ing_date = datetime.fromisoformat(ingestion_date.replace('Z', '+00:00'))
                    if ing_date < threshold_date:
                        continue
                except:
                    pass
            
            trending.append({
                "title": metadata.get("title", "Unknown"),
                "year": metadata.get("year", 0),
                "citations": metadata.get("citations_count", 0),
                "field": metadata.get("field", ""),
                "url": metadata.get("url", "")
            })
        
        # Sort by citations
        trending.sort(key=lambda x: x["citations"], reverse=True)
        
        return json.dumps({
            "success": True,
            "field": field,
            "timeframe_days": days,
            "num_results": len(trending[:top_k]),
            "trending_papers": trending[:top_k]
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Trending papers search failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "field": field
        })


# Helper function to create all tools as a list
def get_all_vector_search_tools() -> List:
    """Get list of all vector search tools for agent initialization"""
    return [
        vector_search_tool,
        multi_field_search_tool,
        similar_papers_tool,
        trending_papers_tool
    ]


# Example usage
if __name__ == "__main__":
    from src.utils.logger import setup_logging
    
    setup_logging(level="INFO")
    
    print("\n=== Vector Search Tool Examples ===\n")
    
    # Test basic search
    print("1. Basic Search:")
    result1 = vector_search_tool(
        query="machine learning for cancer detection",
        field="biology",
        top_k=5
    )
    data1 = json.loads(result1)
    print(f"Success: {data1['success']}")
    print(f"Found: {data1.get('num_results', 0)} papers\n")
    
    # Test multi-field search
    print("2. Multi-Field Search:")
    result2 = multi_field_search_tool(
        query="neural networks",
        fields="biology,computer_science",
        results_per_field=3
    )
    data2 = json.loads(result2)
    print(f"Success: {data2['success']}")
    print(f"Total papers: {data2.get('total_papers', 0)}\n")
    
    # Test trending
    print("3. Trending Papers:")
    result3 = trending_papers_tool(
        field="computer_science",
        days=60,
        min_citations=10,
        top_k=5
    )
    data3 = json.loads(result3)
    print(f"Success: {data3['success']}")
    print(f"Trending: {data3.get('num_results', 0)} papers\n")
    
    print("✅ All tool examples completed!")
