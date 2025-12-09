"""
Tools Module

LangChain tools for research paper discovery, analysis, and comparison.
"""

from src.tools.vector_search import (
    vector_search_tool,
    multi_field_search_tool,
    similar_papers_tool,
    trending_papers_tool,
    get_all_vector_search_tools
)
from src.tools.citation_network import (
    get_citations_tool,
    find_influential_papers_tool,
    trace_citation_path_tool,
    get_all_citation_tools
)
from src.tools.dataset_finder import (
    find_datasets_tool,
    find_code_repositories_tool,
    get_all_dataset_tools
)
from src.tools.methodology_comparator import (
    compare_methodologies_tool,
    find_method_transfer_potential_tool,
    analyze_method_requirements_tool,
    get_all_methodology_tools
)


def get_all_tools():
    """Get all available tools for agent initialization"""
    return (
        get_all_vector_search_tools() +
        get_all_citation_tools() +
        get_all_dataset_tools() +
        get_all_methodology_tools()
    )


__all__ = [
    # Vector Search
    'vector_search_tool',
    'multi_field_search_tool',
    'similar_papers_tool',
    'trending_papers_tool',
    'get_all_vector_search_tools',
    # Citation Network
    'get_citations_tool',
    'find_influential_papers_tool',
    'trace_citation_path_tool',
    'get_all_citation_tools',
    # Dataset Finder
    'find_datasets_tool',
    'find_code_repositories_tool',
    'get_all_dataset_tools',
    # Methodology Comparator
    'compare_methodologies_tool',
    'find_method_transfer_potential_tool',
    'analyze_method_requirements_tool',
    'get_all_methodology_tools',
    # All tools
    'get_all_tools'
]
