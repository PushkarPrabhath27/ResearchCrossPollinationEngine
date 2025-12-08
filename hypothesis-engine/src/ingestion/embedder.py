"""
Paper Embedder

Generates embeddings for scientific papers using sentence-transformers.
Supports batch processing, GPU acceleration, and caching.
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Union
from pathlib import Path
from tqdm import tqdm
import pickle

from src.config import Settings
from src.utils.logger import get_logger
from src.utils.helpers import ensure_dir

logger = get_logger(__name__)


class PaperEmbedder:
    """
    Generates embeddings for scientific papers
    
    Uses sentence-transformers with support for GPU acceleration,
    batch processing, and caching. Optimized for scientific text.
    """
    
    # Supported models
    MODELS = {
        'specter': 'allenai-specter',  # Scientific papers, 768 dim
        'minilm': 'sentence-transformers/all-MiniLM-L6-v2',  # General, 384 dim
        'mpnet': 'sentence-transformers/all-mpnet-base-v2',  # General, 768 dim
        'qa': 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'  # Q&A, 384 dim
    }
    
    def __init__(
        self,
        model_name: str = 'allenai-specter',
        device: Optional[str] = None,
        config: Optional[Settings] = None
    ):
        """
        Initialize embedder
        
        Args:
            model_name: Name of sentence-transformer model
            device: Device to use ('cuda', 'cpu', or None for auto)
            config: Application configuration
        """
        self.model_name = model_name
        self.config = config
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Loading model: {model_name} on {self.device}")
        
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Setup cache
        if config:
            self.cache_dir = Path(config.ingestion.processed_data_dir) / "embeddings_cache"
        else:
            self.cache_dir = Path("./data/processed/embeddings_cache")
        
        ensure_dir(self.cache_dir)
        self.cache = {}
    
    def embed_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for single text
        
        Args:
            text: Input text
            normalize: Whether to normalize embedding
        
        Returns:
            Embedding vector as numpy array
        """
        if not text or not text.strip():
            logger.warning("Empty text provided")
            return np.zeros(self.embedding_dim)
        
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return np.zeros(self.embedding_dim)
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = True
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
            show_progress: Show progress bar
        
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
                device=self.device
            )
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            return list(embeddings)
            
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            # Return zero vectors as fallback
            return [np.zeros(self.embedding_dim) for _ in texts]
    
    def embed_paper(
        self,
        paper_dict: Dict,
        batch_size: int = 32
    ) -> Dict:
        """
        Embed all chunks of a paper
        
        Args:
            paper_dict: Dictionary with paper data including 'chunks'
            batch_size: Batch size for processing
        
        Returns:
            Dictionary with chunks and their embeddings
        """
        if 'chunks' not in paper_dict:
            logger.warning("No chunks found in paper dictionary")
            return paper_dict
        
        chunks = paper_dict['chunks']
        texts = [chunk['text'] for chunk in chunks]
        
        logger.info(f"Embedding paper with {len(chunks)} chunks")
        
        # Generate embeddings
        embeddings = self.embed_batch(texts, batch_size=batch_size)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
        
        paper_dict['chunks'] = chunks
        paper_dict['embedding_dim'] = self.embedding_dim
        paper_dict['model_name'] = self.model_name
        
        return paper_dict
    
    def compute_similarity(
        self,
        emb1: Union[np.ndarray, List[float]],
        emb2: Union[np.ndarray, List[float]]
    ) -> float:
        """
        Calculate cosine similarity between embeddings
        
        Args:
            emb1: First embedding
            emb2: Second embedding
        
        Returns:
            Cosine similarity score (-1 to 1)
        """
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def find_similar_chunks(
        self,
        query_emb: np.ndarray,
        chunk_embeddings: List[np.ndarray],
        top_k: int = 10
    ) -> List[tuple]:
        """
        Find most similar chunks to query
        
        Args:
            query_emb: Query embedding
            chunk_embeddings: List of chunk embeddings
            top_k: Number of top results to return
        
        Returns:
            List of (index, similarity_score) tuples
        """
        if not chunk_embeddings:
            return []
        
        # Calculate similarities
        similarities = []
        for idx, chunk_emb in enumerate(chunk_embeddings):
            sim = self.compute_similarity(query_emb, chunk_emb)
            similarities.append((idx, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        return similarities[:top_k]
    
    def embed_with_cache(
        self,
        text: str,
        cache_key: Optional[str] = None
    ) -> np.ndarray:
        """
        Embed text with caching
        
        Args:
            text: Input text
            cache_key: Key for caching (default: hash of text)
        
        Returns:
            Embedding vector
        """
        if cache_key is None:
            cache_key = str(hash(text))
        
        # Check cache
        if cache_key in self.cache:
            logger.debug(f"Cache hit for {cache_key}")
            return self.cache[cache_key]
        
        # Generate embedding
        embedding = self.embed_text(text)
        
        # Cache it
        self.cache[cache_key] = embedding
        
        return embedding
    
    def save_cache(self, filepath: Optional[Path] = None):
        """
        Save embedding cache to disk
        
        Args:
            filepath: Path to save cache (default: cache_dir/cache.pkl)
        """
        if filepath is None:
            filepath = self.cache_dir / "cache.pkl"
        
        logger.info(f"Saving cache to {filepath}")
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.cache, f)
            logger.info(f"Saved {len(self.cache)} cached embeddings")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def load_cache(self, filepath: Optional[Path] = None):
        """
        Load embedding cache from disk
        
        Args:
            filepath: Path to load cache from
        """
        if filepath is None:
            filepath = self.cache_dir / "cache.pkl"
        
        if not filepath.exists():
            logger.info("No cache file found")
            return
        
        logger.info(f"Loading cache from {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                self.cache = pickle.load(f)
            logger.info(f"Loaded {len(self.cache)} cached embeddings")
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            self.cache = {}
    
    def batch_embed_papers(
        self,
        papers: List[Dict],
        batch_size: int = 32,
        save_individually: bool = True,
        output_dir: Optional[Path] = None
    ) -> List[Dict]:
        """
        Embed multiple papers with progress tracking
        
        Args:
            papers: List of paper dictionaries with chunks
            batch_size: Batch size for embedding
            save_individually: Save each paper separately
            output_dir: Directory to save individual papers
        
        Returns:
            List of papers with embeddings
        """
        logger.info(f"Batch embedding {len(papers)} papers")
        
        if save_individually and output_dir:
            ensure_dir(output_dir)
        
        embedded_papers = []
        
        with tqdm(total=len(papers), desc="Embedding papers") as pbar:
            for paper in papers:
                try:
                    embedded = self.embed_paper(paper, batch_size=batch_size)
                    embedded_papers.append(embedded)
                    
                    # Save individual paper
                    if save_individually and output_dir:
                        paper_id = paper.get('paper_id', f"paper_{len(embedded_papers)}")
                        save_path = output_dir / f"{paper_id}_embedded.pkl"
                        
                        with open(save_path, 'wb') as f:
                            pickle.dump(embedded, f)
                    
                except Exception as e:
                    logger.error(f"Failed to embed paper: {e}")
                    embedded_papers.append(paper)  # Add without embeddings
                finally:
                    pbar.update(1)
        
        logger.info(f"Successfully embedded {len(embedded_papers)} papers")
        return embedded_papers
    
    def get_model_info(self) -> Dict:
        """
        Get information about the current model
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'device': self.device,
            'max_seq_length': self.model.max_seq_length,
            'cache_size': len(self.cache)
        }


# Example usage
if __name__ == "__main__":
    from src.utils.logger import setup_logging
    from src.ingestion.parser import PaperParser
    
    setup_logging(level="INFO")
    
    # Initialize
    print("\n=== Initializing Embedder ===")
    embedder = PaperEmbedder(model_name='allenai-specter')
    info = embedder.get_model_info()
    print(f"Model: {info['model_name']}")
    print(f"Dimension: {info['embedding_dim']}")
    print(f"Device: {info['device']}")
    
    # Example 1: Embed single text
    print("\n=== Example 1: Single Text ===")
    text1 = "Machine learning applications in cancer research"
    emb1 = embedder.embed_text(text1)
    print(f"Embedding shape: {emb1.shape}")
    print(f"First 5 values: {emb1[:5]}")
    
    # Example 2: Batch embedding
    print("\n=== Example 2: Batch Embedding ===")
    texts = [
        "Deep learning for medical imaging",
        "Natural language processing in healthcare",
        "Quantum computing applications"
    ]
    embeddings = embedder.embed_batch(texts, batch_size=2, show_progress=True)
    print(f"Generated {len(embeddings)} embeddings")
    
    # Example 3: Similarity calculation
    print("\n=== Example 3: Similarity ===")
    sim_1_2 = embedder.compute_similarity(embeddings[0], embeddings[1])
    sim_1_3 = embedder.compute_similarity(embeddings[0], embeddings[2])
    print(f"Similarity (1 vs 2): {sim_1_2:.3f}")
    print(f"Similarity (1 vs 3): {sim_1_3:.3f}")
    
    # Example 4: Find similar chunks
    print("\n=== Example 4: Find Similar ===")
    query_emb = embeddings[0]
    similar = embedder.find_similar_chunks(query_emb, embeddings, top_k=2)
    print(f"Top similar chunks:")
    for idx, score in similar:
        print(f"  {idx}: {score:.3f} - {texts[idx]}")
    
    # Example 5: Embed paper with chunks
    print("\n=== Example 5: Embed Paper ===")
    parser = PaperParser()
    sample_text = "Introduction: This paper discusses X. Methods: We used Y. Results: We found Z."
    chunks = parser.chunk_text(sample_text, chunk_size=10, overlap=2)
    
    paper_dict = {
        'paper_id': 'test_001',
        'chunks': chunks
    }
    
    embedded_paper = embedder.embed_paper(paper_dict, batch_size=2)
    print(f"Embedded paper with {len(embedded_paper['chunks'])} chunks")
    print(f"Each chunk now has 'embedding' field")
    
    print("\n✅ All examples completed!")
