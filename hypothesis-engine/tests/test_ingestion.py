"""
Test Suite for Ingestion Modules

Unit tests for data fetchers, parsers, and embedders.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestArxivFetcher:
    """Tests for arXiv fetcher"""
    
    def test_query_construction(self):
        """Test search query construction"""
        from src.ingestion.arxiv_fetcher import ArxivFetcher
        
        fetcher = ArxivFetcher()
        query = fetcher.build_query(
            keywords=["machine learning", "cancer"],
            categories=["cs.LG", "q-bio.QM"]
        )
        
        assert "machine learning" in query
        assert "cancer" in query
    
    def test_metadata_extraction(self):
        """Test paper metadata extraction"""
        from src.ingestion.arxiv_fetcher import ArxivFetcher
        
        fetcher = ArxivFetcher()
        
        # Mock paper result
        mock_paper = Mock()
        mock_paper.title = "Test Paper Title"
        mock_paper.authors = [Mock(name="Author One")]
        mock_paper.published = Mock()
        mock_paper.published.year = 2024
        mock_paper.abstract = "This is a test abstract"
        mock_paper.categories = ["cs.LG"]
        mock_paper.pdf_url = "https://arxiv.org/pdf/test.pdf"
        
        metadata = fetcher.extract_metadata(mock_paper)
        
        assert metadata['title'] == "Test Paper Title"
        assert metadata['year'] == 2024
        assert 'abstract' in metadata


class TestPubMedFetcher:
    """Tests for PubMed fetcher"""
    
    def test_search_construction(self):
        """Test search query building"""
        from src.ingestion.pubmed_fetcher import PubMedFetcher
        
        fetcher = PubMedFetcher()
        query = fetcher.build_search_query(
            terms=["CRISPR", "gene editing"],
            mesh_terms=["Gene Editing"]
        )
        
        assert "CRISPR" in query or "gene editing" in query
    
    def test_xml_parsing(self):
        """Test XML response parsing"""
        # Would test actual XML parsing
        pass


class TestPaperParser:
    """Tests for paper parser"""
    
    def test_text_cleaning(self):
        """Test text normalization"""
        from src.ingestion.parser import PaperParser
        
        parser = PaperParser()
        
        dirty = "This   has     weird   spacing\n\n\n\nand newlines"
        clean = parser.clean_text(dirty)
        
        # Should have normalized spacing
        assert "  " not in clean or clean.count("  ") < dirty.count("  ")
    
    def test_section_detection(self):
        """Test section identification"""
        from src.ingestion.parser import PaperParser
        
        parser = PaperParser()
        
        text = """
        Abstract
        This is the abstract section.
        
        Introduction  
        This is the introduction.
        
        Methods
        These are the methods.
        
        Results
        Here are results.
        
        Conclusion
        This is the conclusion.
        """
        
        sections = parser.extract_sections(text)
        
        assert len(sections) > 0
        # Should detect at least some sections
    
    def test_chunking(self):
        """Test text chunking"""
        from src.ingestion.parser import PaperParser
        
        parser = PaperParser()
        
        # Create long text
        words = ["word"] * 1000
        text = " ".join(words)
        
        chunks = parser.chunk_text(text, chunk_size=100, overlap=10)
        
        assert len(chunks) > 0
        assert all(chunk.get('text') for chunk in chunks)


class TestEmbedder:
    """Tests for paper embedder"""
    
    @pytest.fixture
    def embedder(self):
        """Create embedder with small model"""
        from src.ingestion.embedder import PaperEmbedder
        from src.config import get_settings
        
        config = get_settings()
        return PaperEmbedder(
            config=config,
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
    
    def test_embedding_dimension(self, embedder):
        """Test embedding dimension"""
        text = "This is a test sentence"
        embedding = embedder.embed_text(text)
        
        assert len(embedding) == embedder.embedding_dim
    
    def test_batch_processing(self, embedder):
        """Test batch embedding"""
        texts = [f"Test sentence number {i}" for i in range(10)]
        embeddings = embedder.embed_batch(texts)
        
        assert len(embeddings) == 10
    
    def test_similarity(self, embedder):
        """Test similarity calculation"""
        emb1 = embedder.embed_text("machine learning artificial intelligence")
        emb2 = embedder.embed_text("machine learning AI")
        emb3 = embedder.embed_text("quantum physics particles")
        
        sim12 = embedder.compute_similarity(emb1, emb2)
        sim13 = embedder.compute_similarity(emb1, emb3)
        
        # Similar texts should have higher similarity
        assert sim12 > sim13


class TestCitationNetwork:
    """Tests for citation network builder"""
    
    def test_graph_creation(self):
        """Test graph initialization"""
        from src.ingestion.citation_network import CitationNetworkBuilder
        
        builder = CitationNetworkBuilder()
        
        # Add papers
        builder.add_paper("paper1", {"title": "Paper 1"})
        builder.add_paper("paper2", {"title": "Paper 2"})
        
        # Add citation
        builder.add_citation("paper1", "paper2")
        
        assert builder.graph.number_of_nodes() == 2
        assert builder.graph.number_of_edges() == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
