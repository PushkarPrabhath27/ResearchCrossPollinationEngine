"""
Citation Network Builder

Builds unified citation graphs combining data from all sources (arXiv, PubMed, 
Semantic Scholar, OpenAlex). Enables citation depth analysis and path finding.
"""

import networkx as nx
import json
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict

from src.utils.logger import get_logger
from src.utils.helpers import ensure_dir, save_json

logger = get_logger(__name__)


class CitationNetworkBuilder:
    """
    Builds and analyzes citation networks from multiple data sources
    
    Maintains mappings between different paper IDs (DOI, arXiv, PubMed, etc.)
    and provides graph analysis capabilities.
    """
    
    def __init__(self):
        """Initialize citation network"""
        # Directed graph: edge from paper A to B means A cites B
        self.graph = nx.DiGraph()
        
        # Map different IDs to the same paper
        # Format: {id_type: {external_id: internal_node_id}}
        self.id_mappings = {
            'doi': {},
            'arxiv': {},
            'pubmed': {},
            'pmc': {},
            'semantic_scholar': {},
            'openalex': {}
        }
        
        # Paper metadata indexed by internal node ID
        self.paper_metadata = {}
        
        # Next internal node ID
        self.next_node_id = 1
        
        logger.info("CitationNetworkBuilder initialized")
    
    def _get_or_create_node(self, paper_data: Dict, source: str) -> str:
        """
        Get existing node or create new one for a paper
        
        Args:
            paper_data: Paper metadata
            source: Data source (arxiv, pubmed, semantic_scholar, openalex)
        
        Returns:
            Internal node ID
        """
        # Extract all identifiers
        ids = self._extract_identifiers(paper_data, source)
        
        # Check if we already have this paper
        for id_type, external_id in ids.items():
            if external_id in self.id_mappings[id_type]:
                node_id = self.id_mappings[id_type][external_id]
                logger.debug(f"Found existing node {node_id} for {id_type}:{external_id}")
                return node_id
        
        # Create new node
        node_id = f"N{self.next_node_id}"
        self.next_node_id += 1
        
        # Register all IDs
        for id_type, external_id in ids.items():
            self.id_mappings[id_type][external_id] = node_id
        
        # Add node to graph
        self.graph.add_node(node_id)
        
        logger.debug(f"Created new node {node_id}")
        return node_id
    
    def _extract_identifiers(self, paper_data: Dict, source: str) -> Dict[str, str]:
        """
        Extract all identifiers from paper data
        
        Args:
            paper_data: Paper metadata
            source: Data source
        
        Returns:
            Dictionary of {id_type: id_value}
        """
        ids = {}
        
        if source == 'arxiv':
            if 'paper_id' in paper_data:
                ids['arxiv'] = paper_data['paper_id']
            if 'doi' in paper_data and paper_data['doi']:
                ids['doi'] = paper_data['doi'].lower()
        
        elif source == 'pubmed':
            if 'pmid' in paper_data:
                ids['pubmed'] = paper_data['pmid']
            if 'pmc_id' in paper_data and paper_data['pmc_id']:
                ids['pmc'] = paper_data['pmc_id']
            if 'doi' in paper_data and paper_data['doi']:
                ids['doi'] = paper_data['doi'].lower()
        
        elif source == 'semantic_scholar':
            if 'paperId' in paper_data:
                ids['semantic_scholar'] = paper_data['paperId']
            
            if 'externalIds' in paper_data:
                ext_ids = paper_data['externalIds']
                if ext_ids.get('DOI'):
                    ids['doi'] = ext_ids['DOI'].lower()
                if ext_ids.get('ArXiv'):
                    ids['arxiv'] = ext_ids['ArXiv']
                if ext_ids.get('PubMed'):
                    ids['pubmed'] = ext_ids['PubMed']
        
        elif source == 'openalex':
            if 'id' in paper_data:
                ids['openalex'] = paper_data['id']
            
            if 'ids' in paper_data:
                id_dict = paper_data['ids']
                if id_dict.get('doi'):
                    ids['doi'] = id_dict['doi'].replace('https://doi.org/', '').lower()
                if id_dict.get('pmid'):
                    ids['pubmed'] = id_dict['pmid'].replace('https://pubmed.ncbi.nlm.nih.gov/', '')
        
        return ids
    
    def add_paper(self, paper_data: Dict, source: str) -> str:
        """
        Add a paper to the network
        
        Args:
            paper_data: Paper metadata
            source: Data source (arxiv, pubmed, semantic_scholar, openalex)
        
        Returns:
            Internal node ID
        """
        node_id = self._get_or_create_node(paper_data, source)
        
        # Merge metadata (later sources can overwrite)
        if node_id not in self.paper_metadata:
            self.paper_metadata[node_id] = {}
        
        # Extract core metadata
        self.paper_metadata[node_id].update({
            'title': paper_data.get('title'),
            'year': paper_data.get('year') or paper_data.get('publication_year'),
            'source': source
        })
        
        logger.debug(f"Added paper to node {node_id}")
        return node_id
   def add_citation(self, citing_id: str, cited_id: str):
        """
        Add a citation edge
        
        Args:
           citing_id: ID of the citing paper (any format)
            cited_id: ID of the cited paper (any format)
        """
        # Find nodes for these IDs
        citing_node = self._find_node(citing_id)
        cited_node = self._find_node(cited_id)
        
        if citing_node and cited_node:
            self.graph.add_edge(citing_node, cited_node)
            logger.debug(f"Added citation: {citing_node} -> {cited_node}")
        else:
            logger.warning(f"Could not add citation: nodes not found")
    
    def _find_node(self, paper_id: str) -> Optional[str]:
        """
        Find node by any identifier
        
        Args:
            paper_id: Paper ID in any format
        
        Returns:
            Internal node ID or None
        """
        # Check all ID types
        for id_type, mapping in self.id_mappings.items():
            if paper_id in mapping:
                return mapping[paper_id]
        
        # Maybe it's already an internal node ID
        if paper_id in self.graph:
            return paper_id
        
        return None
    
    def merge_paper_ids(self, id_list: List[str]):
        """
        Merge multiple IDs that represent the same paper
        
        Args:
            id_list: List of IDs that should map to same paper
        """
        # Find existing nodes
        nodes = [self._find_node(pid) for pid in id_list]
        nodes = [n for n in nodes if n is not None]
        
        if not nodes:
            logger.warning("No existing nodes found to merge")
            return
        
        # Use first node as canonical
        canonical_node = nodes[0]
        
        # Merge other nodes into canonical
        for node in nodes[1:]:
            if node != canonical_node:
                # Merge metadata
                self.paper_metadata[canonical_node].update(
                    self.paper_metadata.get(node, {})
                )
                
                # Redirect all edges
                for predecessor in self.graph.predecessors(node):
                    self.graph.add_edge(predecessor, canonical_node)
                for successor in self.graph.successors(node):
                    self.graph.add_edge(canonical_node, successor)
                
                # Remove old node
                self.graph.remove_node(node)
                del self.paper_metadata[node]
        
        # Update ID mappings to point to canonical
        for pid in id_list:
            for id_type, mapping in self.id_mappings.items():
                if pid in mapping:
                    mapping[pid] = canonical_node
        
        logger.info(f"Merged {len(nodes)} nodes into {canonical_node}")
    
    def get_citation_depth(self, paper_id: str, max_depth: int = 3) -> Dict[int, Set[str]]:
        """
        Get papers at different citation depths
        
        Args:
            paper_id: Starting paper ID
            max_depth: Maximum depth to traverse
        
        Returns:
            Dictionary {depth: set of node IDs}
        """
        node = self._find_node(paper_id)
        if not node:
            logger.warning(f"Paper not found: {paper_id}")
            return {}
        
        depths = defaultdict(set)
        depths[0].add(node)
        
        visited = {node}
        current_level = {node}
        
        for depth in range(1, max_depth + 1):
            next_level = set()
            
            for n in current_level:
                # Get papers cited by this paper
                for cited in self.graph.successors(n):
                    if cited not in visited:
                        next_level.add(cited)
                        visited.add(cited)
            
            if not next_level:
                break
            
            depths[depth] = next_level
            current_level = next_level
        
        return dict(depths)
    
    def find_citation_paths(self, paper1_id: str, paper2_id: str) -> List[List[str]]:
        """
        Find citation paths between two papers
        
        Args:
            paper1_id: First paper ID
            paper2_id: Second paper ID
        
        Returns:
            List of paths (each path is a list of node IDs)
        """
        node1 = self._find_node(paper1_id)
        node2 = self._find_node(paper2_id)
        
        if not node1 or not node2:
            logger.warning("One or both papers not found")
            return []
        
        try:
            paths = list(nx.all_simple_paths(self.graph, node1, node2, cutoff=5))
            logger.info(f"Found {len(paths)} paths between papers")
            return paths
        except nx.NetworkXNoPath:
            logger.info("No citation path found")
            return []
    
    def get_influential_papers(
        self,
        min_citations: int = 10,
        top_n: Optional[int] = None
    ) -> List[Tuple[str, int]]:
        """
        Get highly cited papers
        
        Args:
            min_citations: Minimum citation count
            top_n: Return only top N papers
        
        Returns:
            List of (node_id, citation_count) tuples
        """
        in_degrees = self.graph.in_degree()
        
        influential = [
            (node, degree) 
            for node, degree in in_degrees 
            if degree >= min_citations
        ]
        
        # Sort by citation count
        influential.sort(key=lambda x: x[1], reverse=True)
        
        if top_n:
            influential = influential[:top_n]
        
        logger.info(f"Found {len(influential)} influential papers")
        return influential
    
    def get_statistics(self) -> Dict:
        """
        Get network statistics
        
        Returns:
            Dictionary with network stats
        """
        stats = {
            'total_papers': self.graph.number_of_nodes(),
            'total_citations': self.graph.number_of_edges(),
            'average_citations_per_paper': (
                self.graph.number_of_edges() / self.graph.number_of_nodes()
                if self.graph.number_of_nodes() > 0 else 0
            ),
            'papers_by_source': defaultdict(int)
        }
        
        for node_id, metadata in self.paper_metadata.items():
            source = metadata.get('source', 'unknown')
            stats['papers_by_source'][source] += 1
        
        # Convert defaultdict to regular dict
        stats['papers_by_source'] = dict(stats['papers_by_source'])
        
        return stats
    
    def export_graph(self, filepath: Path, format: str = 'json'):
        """
        Export graph to file
        
        Args:
            filepath: Output file path
            format: Export format ('json', 'graphml', 'gexf')
        """
        logger.info(f"Exporting graph to {filepath} as {format}")
        
        ensure_dir(filepath.parent)
        
        if format == 'json':
            # Export as JSON with metadata
            data = {
                'nodes': [
                    {
                        'id': node,
                        **self.paper_metadata.get(node, {})
                    }
                    for node in self.graph.nodes()
                ],
                'edges': [
                    {'source': u, 'target': v}
                    for u, v in self.graph.edges()
                ]
            }
            save_json(data, str(filepath))
        
        elif format == 'graphml':
            # Add metadata as node attributes
            for node in self.graph.nodes():
                if node in self.paper_metadata:
                    for key, value in self.paper_metadata[node].items():
                        self.graph.nodes[node][key] = value
            
            nx.write_graphml(self.graph, str(filepath))
        
        elif format == 'gexf':
            # Add metadata as node attributes
            for node in self.graph.nodes():
                if node in self.paper_metadata:
                    for key, value in self.paper_metadata[node].items():
                        self.graph.nodes[node][key] = str(value)
            
            nx.write_gexf(self.graph, str(filepath))
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Graph exported successfully")


# Example usage
if __name__ == "__main__":
    from src.utils.logger import setup_logging
    
    setup_logging(level="INFO")
    
    # Create network
    network = CitationNetworkBuilder()
    
    # Add papers from different sources
    print("\n=== Adding Papers ===")
    
    # Paper from arXiv
    arxiv_paper = {
        'paper_id': '2104.12345',
        'doi': '10.1234/example',
        'title': 'Example Paper',
        'year': 2021
    }
    node1 = network.add_paper(arxiv_paper, source='arxiv')
    print(f"Added arXiv paper: {node1}")
    
    # Same paper from Semantic Scholar
    ss_paper = {
        'paperId': 'abc123',
        'externalIds': {
            'DOI': '10.1234/example',
            'ArXiv': '2104.12345'
        },
        'title': 'Example Paper',
        'year': 2021
    }
    node2 = network.add_paper(ss_paper, source='semantic_scholar')
    print(f"Added S2 paper: {node2}")
    
    # Add another paper
   cited_paper = {
        'paperId': 'def456',
        'title': 'Cited Paper',
        'year': 2020
    }
    node3 = network.add_paper(cited_paper, source='semantic_scholar')
    
    # Add citation
    print("\n=== Adding Citations ===")
    network.add_citation('abc123', 'def456')
    
    # Get statistics
    print("\n=== Network Statistics ===")
    stats = network.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Get influential papers
    print("\n=== Influential Papers ===")
    influential = network.get_influential_papers(min_citations=1)
    for node_id, citations in influential:
        metadata = network.paper_metadata.get(node_id, {})
        print(f"{node_id}: {citations} citations - {metadata.get('title', 'N/A')}")
    
    # Export
    print("\n=== Exporting ===")
    network.export_graph(Path("./test_network.json"), format='json')
    print("✅ Graph exported to test_network.json")
