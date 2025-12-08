"""
Chroma DB Manager

Comprehensive vector database manager for storing and querying paper embeddings.
Supports multiple search strategies, metadata filtering, and batch operations.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
import numpy as np
from typing import List, Dict, Optional, Any
from pathlib import Path
from tqdm import tqdm
import json
from collections import defaultdict

from src.config import Settings
from src.utils.logger import get_logger
from src.utils.helpers import ensure_dir

logger = get_logger(__name__)


class ChromaManager:
    """
    Manages Chroma vector database for scientific papers
    
    Provides semantic search, metadata filtering, hybrid search,
    and efficient batch operations.
    """
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "scientific_papers",
        config: Optional[Settings] = None
    ):
        """
        Initialize Chroma manager
        
        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the collection
            config: Application configuration
        """
        self.config = config
        
        # Set persist directory
        if persist_directory:
            self.persist_dir = Path(persist_directory)
        elif config:
            self.persist_dir = Path(config.database.chroma_persist_dir)
        else:
            self.persist_dir = Path("./data/embeddings")
        
        ensure_dir(self.persist_dir)
        
        # Initialize Chroma client
        logger.info(f"Initializing ChromaDB at {self.persist_dir}")
        
        chroma_settings = ChromaSettings(
            persist_directory=str(self.persist_dir),
            anonymized_telemetry=False
        )
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=chroma_settings
        )
        
        # Get or create collection
        self.collection_name = collection_name
        self.collection = self.get_collection(collection_name)
        
        logger.info(f"ChromaManager initialized with collection: {collection_name}")
    
    def create_collection(
        self,
        name: str,
        metadata: Optional[Dict] = None
    ) -> chromadb.Collection:
        """
        Create a new collection
        
        Args:
            name: Collection name
            metadata: Collection metadata
        
        Returns:
            Created collection
        """
        logger.info(f"Creating collection: {name}")
        
        try:
            collection = self.client.create_collection(
                name=name,
                metadata=metadata or {}
            )
            logger.info(f"Collection '{name}' created successfully")
            return collection
        except Exception as e:
            if "already exists" in str(e):
                logger.warning(f"Collection '{name}' already exists")
                return self.client.get_collection(name)
            else:
                logger.error(f"Failed to create collection: {e}")
                raise
    
    def get_collection(self, name: str) -> chromadb.Collection:
        """
        Get existing collection or create if it doesn't exist
        
        Args:
            name: Collection name
        
        Returns:
            Collection object
        """
        try:
            collection = self.client.get_collection(name)
            logger.info(f"Retrieved existing collection: {name}")
            return collection
        except Exception:
            logger.info(f"Collection '{name}' not found, creating new one")
            return self.create_collection(name)
    
    def delete_collection(self, name: str) -> bool:
        """
        Delete a collection
        
        Args:
            name: Collection name
        
        Returns:
            True if successful
        """
        logger.warning(f"Deleting collection: {name}")
        
        try:
            self.client.delete_collection(name)
            logger.info(f"Collection '{name}' deleted")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
    
    def add_paper(
        self,
        paper_id: str,
        embedding: np.ndarray,
        metadata: Dict,
        text: str
    ) -> bool:
        """
        Add a single paper to the collection
        
        Args:
            paper_id: Unique paper identifier
            embedding: Embedding vector
            metadata: Paper metadata
            text: Paper text/chunk
        
        Returns:
            True if successful
        """
        try:
            # Convert numpy array to list if needed
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            self.collection.add(
                ids=[paper_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[text]
            )
            
            logger.debug(f"Added paper: {paper_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add paper {paper_id}: {e}")
            return False
    
    def add_papers(
        self,
        papers_data: List[Dict],
        batch_size: int = 100
    ) -> Dict[str, int]:
        """
        Add multiple papers in batches
        
        Args:
            papers_data: List of dicts with 'id', 'embedding', 'metadata', 'text'
            batch_size: Batch size for processing
        
        Returns:
            Dictionary with success/failure counts
        """
        logger.info(f"Adding {len(papers_data)} papers in batches of {batch_size}")
        
        results = {'success': 0, 'failed': 0, 'duplicates': 0}
        
        with tqdm(total=len(papers_data), desc="Adding papers") as pbar:
            for i in range(0, len(papers_data), batch_size):
                batch = papers_data[i:i+batch_size]
                
                # Prepare batch data
                ids = []
                embeddings = []
                metadatas = []
                documents = []
                
                for paper in batch:
                    try:
                        # Validate required fields
                        if not all(k in paper for k in ['id', 'embedding', 'metadata', 'text']):
                            logger.warning(f"Paper missing required fields, skipping")
                            results['failed'] += 1
                            continue
                        
                        # Check for duplicates
                        existing = self.get_paper(paper['id'])
                        if existing:
                            logger.debug(f"Paper {paper['id']} already exists, skipping")
                            results['duplicates'] += 1
                            continue
                        
                        # Convert embedding if needed
                        emb = paper['embedding']
                        if isinstance(emb, np.ndarray):
                            emb = emb.tolist()
                        
                        ids.append(paper['id'])
                        embeddings.append(emb)
                        metadatas.append(paper['metadata'])
                        documents.append(paper['text'])
                        
                    except Exception as e:
                        logger.error(f"Error preparing paper: {e}")
                        results['failed'] += 1
                
                # Add batch
                if ids:
                    try:
                        self.collection.add(
                            ids=ids,
                            embeddings=embeddings,
                            metadatas=metadatas,
                            documents=documents
                        )
                        results['success'] += len(ids)
                    except Exception as e:
                        logger.error(f"Batch add failed: {e}")
                        results['failed'] += len(ids)
                
                pbar.update(len(batch))
        
        logger.info(f"Results: {results['success']} added, {results['failed']} failed, {results['duplicates']} duplicates")
        return results
    
    def update_paper(self, paper_id: str, updates: Dict) -> bool:
        """
        Update paper data
        
        Args:
            paper_id: Paper ID
            updates: Dictionary with fields to update
        
        Returns:
            True if successful
        """
        try:
            update_dict = {}
            
            if 'embedding' in updates:
                emb = updates['embedding']
                if isinstance(emb, np.ndarray):
                    emb = emb.tolist()
                update_dict['embeddings'] = [emb]
            
            if 'metadata' in updates:
                update_dict['metadatas'] = [updates['metadata']]
            
            if 'text' in updates:
                update_dict['documents'] = [updates['text']]
            
            if update_dict:
                self.collection.update(
                    ids=[paper_id],
                    **update_dict
                )
                logger.debug(f"Updated paper: {paper_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update paper {paper_id}: {e}")
            return False
    
    def delete_paper(self, paper_id: str) -> bool:
        """
        Delete a paper
        
        Args:
            paper_id: Paper ID
        
        Returns:
            True if successful
        """
        try:
            self.collection.delete(ids=[paper_id])
            logger.debug(f"Deleted paper: {paper_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete paper {paper_id}: {e}")
            return False
    
    def search(
        self,
        query_embedding: np.ndarray,
        n_results: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Semantic search using query embedding
        
        Args:
            query_embedding: Query vector
            n_results: Number of results
            filters: Metadata filters
        
        Returns:
            List of result dictionaries
        """
        logger.debug(f"Semantic search for {n_results} results")
        
        try:
            # Convert to list if needed
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filters
            )
            
            # Format results
            formatted = []
            for i in range(len(results['ids'][0])):
                formatted.append({
                    'id': results['ids'][0][i],
                    'distance': results['distances'][0][i],
                    'similarity': 1 - results['distances'][ 0][i],  # Convert distance to similarity
                    'metadata': results['metadatas'][0][i],
                    'text': results['documents'][0][i]
                })
            
            logger.info(f"Found {len(formatted)} results")
            return formatted
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_by_metadata(
        self,
        metadata_filter: Dict,
        n_results: int = 100
    ) -> List[Dict]:
        """
        Filter papers by metadata only
        
        Args:
            metadata_filter: Metadata filter dict
            n_results: Maximum results
        
        Returns:
            List of matching papers
        """
        logger.debug(f"Metadata search with filter: {metadata_filter}")
        
        try:
            results = self.collection.get(
                where=metadata_filter,
                limit=n_results
            )
            
            # Format results
            formatted = []
            for i in range(len(results['ids'])):
                formatted.append({
                    'id': results['ids'][i],
                    'metadata': results['metadatas'][i],
                    'text': results['documents'][i]
                })
            
            logger.info(f"Found {len(formatted)} results")
            return formatted
            
        except Exception as e:
            logger.error(f"Metadata search failed: {e}")
            return []
    
    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        metadata_filter: Dict,
        n_results: int = 10
    ) -> List[Dict]:
        """
        Combined semantic and metadata search
        
        Args:
            query_embedding: Query vector
            metadata_filter: Metadata filters
            n_results: Number of results
        
        Returns:
            List of results
        """
        logger.debug("Hybrid search (semantic + metadata)")
        
        # Use Chroma's built-in hybrid search
        return self.search(
            query_embedding=query_embedding,
            n_results=n_results,
            filters=metadata_filter
        )
    
    def search_across_fields(
        self,
        query_embedding: np.ndarray,
        fields: List[str],
        n_results_per_field: int = 5
    ) -> List[Dict]:
        """
        Search across multiple scientific fields
        
        Args:
            query_embedding: Query vector
            fields: List of field names
            n_results_per_field: Results from each field
        
        Returns:
            Aggregated and deduplicated results
        """
        logger.info(f"Multi-field search across: {fields}")
        
        all_results = []
        seen_ids = set()
        
        for field in fields:
            # Search in this field
            field_results = self.search(
                query_embedding=query_embedding,
                n_results=n_results_per_field,
                filters={"field": field}
            )
            
            # Deduplicate
            for result in field_results:
                if result['id'] not in seen_ids:
                    result['source_field'] = field
                    all_results.append(result)
                    seen_ids.add(result['id'])
        
        # Sort by similarity
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        logger.info(f"Found {len(all_results)} unique results across {len(fields)} fields")
        return all_results
    
    def get_paper(self, paper_id: str) -> Optional[Dict]:
        """
        Retrieve specific paper
        
        Args:
            paper_id: Paper ID
        
        Returns:
            Paper dictionary or None
        """
        try:
            results = self.collection.get(
                ids=[paper_id],
                include=['embeddings', 'metadatas', 'documents']
            )
            
            if results['ids']:
                return {
                    'id': results['ids'][0],
                    'embedding': results['embeddings'][0] if results.get('embeddings') else None,
                    'metadata': results['metadatas'][0],
                    'text': results['documents'][0]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get paper {paper_id}: {e}")
            return None
    
    def get_statistics(self) -> Dict:
        """
        Get collection statistics
        
        Returns:
            Dictionary with stats
        """
        logger.info("Calculating collection statistics")
        
        try:
            # Get all items (may be slow for large collections)
            all_data = self.collection.get(include=['metadatas'])
            
            total_papers = len(all_data['ids'])
            
            # Analyze metadata
            fields = defaultdict(int)
            years = defaultdict(int)
            sources = defaultdict(int)
            keywords = defaultdict(int)
            
            total_citations = 0
            citation_counts = []
            
            for metadata in all_data['metadatas']:
                # Field distribution
                if 'field' in metadata:
                    fields[metadata['field']] += 1
                
                # Year distribution
                if 'year' in metadata:
                    years[metadata['year']] += 1
                
                # Source distribution
                if 'source' in metadata:
                    sources[metadata['source']] += 1
                
                # Citations
                if 'citations_count' in metadata:
                    count = metadata['citations_count']
                    total_citations += count
                    citation_counts.append(count)
                
                # Keywords
                if 'keywords' in metadata:
                    for kw in metadata['keywords']:
                        keywords[kw] += 1
            
            # Calculate stats
            avg_citations = total_citations / total_papers if total_papers > 0 else 0
            
            # Top keywords
            top_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:10]
            
            stats = {
                'total_papers': total_papers,
                'papers_by_field': dict(fields),
                'papers_by_year': dict(years),
                'papers_by_source': dict(sources),
                'total_citations': total_citations,
                'average_citations': avg_citations,
                'top_keywords': dict(top_keywords),
                'collection_name': self.collection_name,
                'persist_directory': str(self.persist_dir)
            }
            
            logger.info(f"Statistics calculated for {total_papers} papers")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to calculate statistics: {e}")
            return {'error': str(e)}
    
    def export_collection(self, output_path: Path) -> bool:
        """
        Backup collection to file
        
        Args:
            output_path: Output file path
        
        Returns:
            True if successful
        """
        logger.info(f"Exporting collection to {output_path}")
        
        try:
            # Get all data
            data = self.collection.get(
                include=['embeddings', 'metadatas', 'documents']
            )
            
            # Save as JSON
            export_data = {
                'collection_name': self.collection_name,
                'count': len(data['ids']),
                'data': {
                    'ids': data['ids'],
                    'embeddings': data['embeddings'],
                    'metadatas': data['metadatas'],
                    'documents': data['documents']
                }
            }
            
            ensure_dir(output_path.parent)
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f)
            
            logger.info(f"Exported {len(data['ids'])} papers")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def import_collection(self, input_path: Path) -> bool:
        """
        Restore collection from backup
        
        Args:
            input_path: Input file path
        
        Returns:
            True if successful
        """
        logger.info(f"Importing collection from {input_path}")
        
        try:
            with open(input_path, 'r') as f:
                import_data = json.load(f)
            
            data = import_data['data']
            
            # Add in batches
            batch_size = 100
            total = len(data['ids'])
            
            for i in range(0, total, batch_size):
                batch_ids = data['ids'][i:i+batch_size]
                batch_embeddings = data['embeddings'][i:i+batch_size]
                batch_metadatas = data['metadatas'][i:i+batch_size]
                batch_documents = data['documents'][i:i+batch_size]
                
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_documents
                )
            
            logger.info(f"Imported {total} papers")
            return True
            
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return False


# Example usage
if __name__ == "__main__":
    from src.utils.logger import setup_logging
    import numpy as np
    
    setup_logging(level="INFO")
    
    # Initialize manager
    print("\n=== Initializing ChromaManager ===")
    manager = ChromaManager(
        persist_directory="./test_chroma_db",
        collection_name="test_papers"
    )
    
    # Add sample papers
    print("\n=== Adding Papers ===")
    papers_data = []
    for i in range(5):
        papers_data.append({
            'id': f"paper_{i}",
            'embedding': np.random.rand(768).tolist(),
            'metadata': {
                'title': f"Paper {i}",
                'year': 2020 + i,
                'field': 'biology' if i % 2 == 0 else 'physics',
                'citations_count': i * 10
            },
            'text': f"This is the text of paper {i}"
        })
    
    results = manager.add_papers(papers_data, batch_size=2)
    print(f"Added: {results}")
    
    # Search
    print("\n=== Semantic Search ===")
    query_emb = np.random.rand(768)
    search_results = manager.search(query_emb, n_results=3)
    print(f"Found {len(search_results)} results:")
    for r in search_results:
        print(f"  - {r['id']}: similarity={r['similarity']:.3f}")
    
    # Metadata search
    print("\n=== Metadata Search ===")
    meta_results = manager.search_by_metadata(
        {'field': 'biology'},
        n_results=10
    )
    print(f"Found {len(meta_results)} biology papers")
    
    # Statistics
    print("\n=== Statistics ===")
    stats = manager.get_statistics()
    print(f"Total papers: {stats['total_papers']}")
    print(f"By field: {stats['papers_by_field']}")
    print(f"By year: {stats['papers_by_year']}")
    
    print("\n✅ All examples completed!")
