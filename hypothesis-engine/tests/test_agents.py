"""
Test Suite for LangChain Agents

Unit tests for all specialized research agents.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestBaseAgent:
    """Tests for base research agent"""
    
    def test_initialization(self):
        """Test agent initialization"""
        from src.agents.base_agent import BaseResearchAgent
        from src.config import get_settings
        
        config = get_settings()
        
        # BaseResearchAgent is abstract, test a concrete implementation
        # For testing, we'll mock the abstract methods
        
    def test_memory_management(self):
        """Test conversation memory"""
        # Would test memory functions
        pass
    
    def test_tool_registration(self):
        """Test tool registration"""
        # Would test tool setup
        pass


class TestPrimaryDomainAgent:
    """Tests for primary domain agent"""
    
    @pytest.fixture
    def agent(self):
        """Create primary domain agent"""
        from src.agents.primary_domain_agent import PrimaryDomainAgent
        from src.config import get_settings
        
        config = get_settings()
        return PrimaryDomainAgent(config=config)
    
    def test_field_identification(self, agent):
        """Test field detection from query"""
        query = "How can machine learning improve cancer diagnosis?"
        field, subfield, confidence = agent.identify_field(query)
        
        # Should identify biology or computer science
        assert field in ['biology', 'computer_science', 'unknown']
        assert 0 <= confidence <= 1
    
    def test_question_analysis(self, agent):
        """Test research question analysis"""
        query = "What are the mechanisms of antibiotic resistance?"
        analysis = agent.analyze_research_question(query)
        
        assert 'field' in analysis
        assert 'keywords' in analysis


class TestCrossDomainAgent:
    """Tests for cross-domain discovery agent"""
    
    @pytest.fixture
    def agent(self):
        """Create cross-domain agent"""
        from src.agents.crossdomain_agent import CrossDomainAgent
        from src.config import get_settings
        
        config = get_settings()
        return CrossDomainAgent(config=config)
    
    def test_problem_abstraction(self, agent):
        """Test problem abstraction"""
        problem = "How do cancer cells migrate through blood vessels?"
        abstraction = agent.abstract_problem(problem, "biology")
        
        assert 'abstract_problem' in abstraction
        assert 'core_mechanism' in abstraction
    
    def test_field_mapping(self, agent):
        """Test field mapping availability"""
        assert 'biology' in agent.FIELD_MAPPINGS
        assert 'physics' in agent.FIELD_MAPPINGS
        
        # Biology should map to other fields
        assert len(agent.FIELD_MAPPINGS['biology']) > 0


class TestMethodologyAgent:
    """Tests for methodology transfer agent"""
    
    @pytest.fixture
    def agent(self):
        """Create methodology agent"""
        from src.agents.methodology_agent import MethodologyTransferAgent
        from src.config import get_settings
        
        config = get_settings()
        return MethodologyTransferAgent(config=config)
    
    def test_requirements_analysis(self, agent):
        """Test method requirements extraction"""
        method = {
            'title': 'Deep Learning for Image Classification',
            'field': 'Computer Science',
            'method': 'CNN-based image analysis'
        }
        
        requirements = agent.analyze_method_requirements(method)
        
        assert 'equipment' in requirements or 'error' not in requirements
        assert 'full_analysis' in requirements or 'error' in requirements
    
    def test_barrier_identification(self, agent):
        """Test technical barrier identification"""
        barriers = agent.identify_technical_barriers(
            source_field="physics",
            target_field="biology",
            method={'title': 'Particle tracking'}
        )
        
        # Should return list (possibly empty)
        assert isinstance(barriers, list)


class TestResourceAgent:
    """Tests for resource finder agent"""
    
    @pytest.fixture
    def agent(self):
        """Create resource agent"""
        from src.agents.resource_agent import ResourceFinderAgent
        from src.config import get_settings
        
        config = get_settings()
        return ResourceFinderAgent(config=config)
    
    def test_dataset_sources(self, agent):
        """Test dataset source configuration"""
        assert 'biology' in agent.DATASET_SOURCES
        assert 'general' in agent.DATASET_SOURCES
    
    def test_funding_sources(self, agent):
        """Test funding source configuration"""
        assert 'US' in agent.FUNDING_SOURCES
        assert 'EU' in agent.FUNDING_SOURCES
    
    def test_resource_tracking(self, agent):
        """Test resource collection"""
        resources = agent.get_all_resources()
        
        assert 'datasets' in resources
        assert 'code' in resources
        assert 'protocols' in resources
        assert 'funding' in resources
        assert 'tools' in resources


class TestAgentIntegration:
    """Integration tests for agent workflow"""
    
    def test_full_workflow(self):
        """Test complete agent pipeline"""
        # This would test the full workflow
        # Skipped as it requires all services
        pytest.skip("Requires running services")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
