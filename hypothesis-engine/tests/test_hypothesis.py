"""
Comprehensive Testing Suite

Unit and integration tests for the Scientific Hypothesis Cross-Pollination Engine.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import get_settings
from src.ingestion.parser import PaperParser
from src.ingestion.embedder import PaperEmbedder
from src.database.chroma_manager import ChromaManager
from src.agents.base_agent import BaseResearchAgent
from src.hypothesis.generator import HypothesisGenerator
from src.hypothesis.validator import HypothesisValidator


class TestConfiguration:
    """Test configuration management"""
    
    def test_settings_load(self):
        """Test settings can be loaded"""
        config = get_settings()
        assert config is not None
        assert hasattr(config, 'agent')
        assert hasattr(config, 'database')
    
    def test_environment_variables(self):
        """Test environment variable loading"""
        config = get_settings()
        assert config.agent.llm_provider in ['openai', 'ollama']


class TestPaperParser:
    """Test paper parsing functionality"""
    
    def test_text_cleaning(self):
        """Test text cleaning"""
        parser = PaperParser()
        
        dirty_text = "This  is   a    test\n\n\n\nwith extra   spaces"
        clean = parser.clean_text(dirty_text)
        
        assert "  " not in clean
        assert "\n\n\n" not in clean
    
    def test_section_extraction(self):
        """Test section detection"""
        parser = PaperParser()
        
        text = """
        ABSTRACT
        This is the abstract.
        
        INTRODUCTION
        This is the introduction.
        
        METHODS
        These are the methods.
        """
        
        sections = parser.extract_sections(text)
        assert 'abstract' in sections
        assert 'introduction' in sections
        assert 'methods' in sections
    
    def test_chunking(self):
        """Test text chunking with overlap"""
        parser = PaperParser()
        
        text = " ".join(["word"] * 1000)  # 1000 words
        chunks = parser.chunk_text(text, chunk_size=100, overlap=10)
        
        assert len(chunks) > 0
        assert all('text' in chunk for chunk in chunks)
        assert all('index' in chunk for chunk in chunks)


class TestEmbedder:
    """Test embedding generation"""
    
    @pytest.fixture
    def embedder(self):
        """Create embedder instance"""
        config = get_settings()
        return PaperEmbedder(config=config, model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    def test_single_embedding(self, embedder):
        """Test generating single embedding"""
        text = "This is a test sentence for embedding generation"
        embedding = embedder.embed_text(text)
        
        assert embedding is not None
        assert len(embedding) == embedder.embedding_dim
        assert embedding.dtype == 'float32' or embedding.dtype == 'float64'
    
    def test_batch_embedding(self, embedder):
        """Test batch embedding"""
        texts = [f"Test sentence {i}" for i in range(5)]
        embeddings = embedder.embed_batch(texts, batch_size=2)
        
        assert len(embeddings) == 5
        assert all(len(emb) == embedder.embedding_dim for emb in embeddings)
    
    def test_similarity_calculation(self, embedder):
        """Test similarity computation"""
        emb1 = embedder.embed_text("machine learning")
        emb2 = embedder.embed_text("machine learning")  # Same text
        emb3 = embedder.embed_text("quantum physics")  # Different text
        
        sim_same = embedder.compute_similarity(emb1, emb2)
        sim_diff = embedder.compute_similarity(emb1, emb3)
        
        assert sim_same > 0.9  # Should be very similar
        assert sim_diff < sim_same  # Should be less similar


class TestHypothesisGeneration:
    """Test hypothesis generation and validation"""
    
    def test_hypothesis_generator(self):
        """Test hypothesis generation"""
        config = get_settings()
        generator = HypothesisGenerator(config)
        
        # Mock findings
        primary_findings = {
            'field': {'primary': 'biology'},
            'knowledge_gaps': [
                {'description': 'Lack of ML methods for X', 'category': 'methodology'}
            ]
        }
        
        cross_domain_findings = {
            'promising_analogies': [
                {
                    'field': 'physics',
                    'analogy': {'similarity_score': 0.85, 'explanation': 'Both involve networks'}
                }
            ]
        }
        
        hypotheses = generator.generate_hypotheses(
            primary_findings,
            cross_domain_findings,
            []
        )
        
        assert len(hypotheses) > 0
        assert all('id' in h for h in hypotheses)
        assert all('title' in h for h in hypotheses)
        assert all('novelty_score' in h for h in hypotheses)
    
    def test_hypothesis_validator(self):
        """Test hypothesis validation"""
        config = get_settings()
        validator = HypothesisValidator(config)
        
        hypothesis = {
            'id': 'test_1',
            'title': 'Test Hypothesis',
            'novelty_score': 0.8,
            'feasibility_score': 0.7
        }
        
        validation = validator.validate_hypothesis(hypothesis)
        
        assert 'hypothesis_id' in validation
        assert 'overall_score' in validation
        assert 'recommendation' in validation
        assert 0 <= validation['overall_score'] <= 1


class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_workflow(self):
        """Test complete hypothesis generation workflow"""
        # This would test the full pipeline
        # Skipped for now as it requires all services running
        pytest.skip("Requires running services")
    
    def test_api_endpoints(self):
        """Test API endpoints"""
        # Would test FastAPI endpoints
        pytest.skip("Requires running API server")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
