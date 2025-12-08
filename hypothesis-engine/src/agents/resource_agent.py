"""
Resource Finder Agent

Helps researchers find datasets, code repositories, protocols, and other
resources needed to test hypotheses and implement research projects.
"""

from typing import List, Dict, Optional
import re
from langchain.tools import Tool

from src.agents.base_agent import BaseResearchAgent
from src.config import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ResourceFinderAgent(BaseResearchAgent):
    """
    Agent for discovering research resources
    
    Finds datasets, code, protocols, funding, tools, and software
    needed for research projects.
    """
    
    SYSTEM_PROMPT = """
You are a research resource specialist who helps researchers find datasets, code, and protocols.

Your responsibilities:
1. Identify what resources are needed for a research project
2. Search for publicly available datasets
3. Find relevant code repositories
4. Locate experimental protocols
5. Identify funding opportunities
6. Find relevant tools and software

Prioritize:
- Open access and free resources
- Well-documented and maintained resources
- Resources with appropriate licenses
- High-quality, validated data

Available tools:
{tools}
"""
    
    # Common dataset repositories
    DATASET_SOURCES = {
        'biology': ['NCBI', 'GEO', 'ArrayExpress', 'TCGA', 'UK Biobank'],
        'physics': ['CERN Open Data', 'NASA Data Portal', 'arXiv datasets'],
        'computer_science': ['Kaggle', 'UCI ML Repository', 'Papers with Code'],
        'general': ['Zenodo', 'Figshare', 'Dryad', 'OSF', 'Google Dataset Search']
    }
    
    # Code repository sources
    CODE_SOURCES = ['GitHub', 'GitLab', 'BitBucket', 'Papers with Code', 'SourceForge']
    
    # Funding sources
    FUNDING_SOURCES = {
        'US': ['NIH', 'NSF', 'DOE', 'DARPA', 'private foundations'],
        'EU': ['ERC', 'Horizon Europe', 'Marie Curie'],
        'UK': ['UKRI', 'Wellcome Trust', 'Royal Society'],
        'general': ['Bill & Melinda Gates Foundation', 'Open Philanthropy']
    }
    
    def __init__(
        self,
        config: Settings,
        tools: Optional[List[Tool]] = None,
        temperature: float = 0.5  # Lower temp for factual accuracy
    ):
        """
        Initialize resource finder agent
        
        Args:
            config: Application configuration
            tools: List of LangChain tools
            temperature: LLM temperature
        """
        super().__init__(
            config=config,
            tools=tools,
            name="ResourceFinderAgent",
            temperature=temperature
        )
        
        # Track found resources
        self.found_resources = {
            'datasets': [],
            'code': [],
            'protocols': [],
            'funding': [],
            'tools': []
        }
        
        logger.info("ResourceFinderAgent initialized")
    
    def get_system_prompt(self) -> str:
        """Get agent's system prompt"""
        return self.SYSTEM_PROMPT
    
    def find_datasets(
        self,
        research_area: str,
        data_type: str,
        field: Optional[str] = None
    ) -> List[Dict]:
        """
        Find publicly available datasets
        
        Args:
            research_area: Research area description
            data_type: Type of data needed
            field: Scientific field
        
        Returns:
            List of dataset resources
        """
        logger.info(f"Finding datasets for {research_area}")
        
        # Get relevant sources
        sources = self.DATASET_SOURCES.get(field, []) + self.DATASET_SOURCES['general']
        sources_str = ', '.join(sources)
        
        search_prompt = f"""
Find publicly available datasets for this research:

Area: {research_area}
Data Type: {data_type}
Field: {field or 'general'}

Search in: {sources_str}

For each dataset provide:
1. Name
2. Source/repository
3. Description
4. Size (if known)
5. Format
6. License
7. Access URL

Prioritize open access, well-documented datasets.
"""
        
        try:
            result = self.llm.invoke(search_prompt)
            response = result.content if hasattr(result, 'content') else str(result)
            
            datasets = self._parse_datasets(response)
            self.found_resources['datasets'].extend(datasets)
            
            logger.info(f"Found {len(datasets)} datasets")
            return datasets
            
        except Exception as e:
            logger.error(f"Failed to find datasets: {e}")
            return []
    
    def _parse_datasets(self, text: str) -> List[Dict]:
        """Parse dataset information from text"""
        datasets = []
        lines = text.split('\n')
        
        current_dataset = {}
        for line in lines:
            line = line.strip()
            
            if 'name:' in line.lower():
                if current_dataset:
                    datasets.append(current_dataset)
                current_dataset = {'name': line.split(':', 1)[-1].strip()}
            elif current_dataset:
                if 'source:' in line.lower() or 'repository:' in line.lower():
                    current_dataset['source'] = line.split(':', 1)[-1].strip()
                elif 'description:' in line.lower():
                    current_dataset['description'] = line.split(':', 1)[-1].strip()
                elif 'license:' in line.lower():
                    current_dataset['license'] = line.split(':', 1)[-1].strip()
                elif 'url:' in line.lower() or 'link:' in line.lower():
                    current_dataset['url'] = line.split(':', 1)[-1].strip()
        
        if current_dataset:
            datasets.append(current_dataset)
        
        return datasets[:10]
    
    def find_code_repositories(
        self,
        method_name: str,
        language: Optional[str] = None
    ) -> List[Dict]:
        """
        Find code implementations
        
        Args:
            method_name: Method or algorithm name
            language: Programming language preference
        
        Returns:
            List of code repositories
        """
        logger.info(f"Finding code for {method_name}")
        
        search_prompt = f"""
Find code implementations for: {method_name}

Search platforms: {', '.join(self.CODE_SOURCES)}
{f'Preferred language: {language}' if language else ''}

For each repository provide:
1. Repository name
2. Platform (GitHub, etc.)
3. Programming language
4. Description
5. Stars/popularity
6. Last updated
7. License
8. URL

Prioritize well-maintained, documented repositories with permissive licenses.
"""
        
        try:
            result = self.llm.invoke(search_prompt)
            response = result.content if hasattr(result, 'content') else str(result)
            
            repos = self._parse_repositories(response)
            self.found_resources['code'].extend(repos)
            
            logger.info(f"Found {len(repos)} code repositories")
            return repos
            
        except Exception as e:
            logger.error(f"Failed to find code: {e}")
            return []
    
    def _parse_repositories(self, text: str) -> List[Dict]:
        """Parse repository information from text"""
        repos = []
        lines = text.split('\n')
        
        current_repo = {}
        for line in lines:
            line = line.strip()
            
            if ('name:' in line.lower() or 'repository:' in line.lower()) and current_repo:
                repos.append(current_repo)
                current_repo = {}
            
            if 'name:' in line.lower() or 'repository:' in line.lower():
                current_repo['name'] = line.split(':', 1)[-1].strip()
            elif current_repo:
                if 'platform:' in line.lower():
                    current_repo['platform'] = line.split(':', 1)[-1].strip()
                elif 'language:' in line.lower():
                    current_repo['language'] = line.split(':', 1)[-1].strip()
                elif 'url:' in line.lower() or 'link:' in line.lower():
                    current_repo['url'] = line.split(':', 1)[-1].strip()
                elif 'stars:' in line.lower():
                    current_repo['stars'] = line.split(':', 1)[-1].strip()
        
        if current_repo:
            repos.append(current_repo)
        
        return repos[:10]
    
    def find_protocols(
        self,
        procedure_type: str,
        field: str
    ) -> List[Dict]:
        """
        Find experimental protocols
        
        Args:
            procedure_type: Type of procedure
            field: Scientific field
        
        Returns:
            List of protocol resources
        """
        logger.info(f"Finding protocols for {procedure_type} in {field}")
        
        protocol_prompt = f"""
Find experimental protocols for: {procedure_type}

Field: {field}

Search in: protocols.io, Nature Protocols, JoVE, Bio-protocol, PLoS protocols

For each protocol provide:
1. Title
2. Source
3. Brief description
4. Difficulty level
5. Time required
6. Key equipment needed
7. URL

Focus on peer-reviewed, validated protocols.
"""
        
        try:
            result = self.llm.invoke(protocol_prompt)
            response = result.content if hasattr(result, 'content') else str(result)
            
            protocols = self._parse_protocols(response)
            self.found_resources['protocols'].extend(protocols)
            
            logger.info(f"Found {len(protocols)} protocols")
            return protocols
            
        except Exception as e:
            logger.error(f"Failed to find protocols: {e}")
            return []
    
    def _parse_protocols(self, text: str) -> List[Dict]:
        """Parse protocol information from text"""
        protocols = []
        lines = text.split('\n')
        
        current_protocol = {}
        for line in lines:
            line = line.strip()
            
            if 'title:' in line.lower():
                if current_protocol:
                    protocols.append(current_protocol)
                current_protocol = {'title': line.split(':', 1)[-1].strip()}
            elif current_protocol:
                if 'source:' in line.lower():
                    current_protocol['source'] = line.split(':', 1)[-1].strip()
                elif 'time:' in line.lower() or 'duration:' in line.lower():
                    current_protocol['duration'] = line.split(':', 1)[-1].strip()
                elif 'url:' in line.lower():
                    current_protocol['url'] = line.split(':', 1)[-1].strip()
        
        if current_protocol:
            protocols.append(current_protocol)
        
        return protocols[:8]
    
    def find_funding(
        self,
        research_area: str,
        region: str = 'US',
        amount_needed: Optional[str] = None
    ) -> List[Dict]:
        """
        Identify funding opportunities
        
        Args:
            research_area: Research area
            region: Geographic region
            amount_needed: Funding amount needed
        
        Returns:
            List of funding opportunities
        """
        logger.info(f"Finding funding for {research_area} in {region}")
        
        sources = self.FUNDING_SOURCES.get(region, []) + self.FUNDING_SOURCES['general']
        
        funding_prompt = f"""
Find funding opportunities for research in: {research_area}

Region: {region}
{f'Amount needed: {amount_needed}' if amount_needed else ''}

Search sources: {', '.join(sources)}

For each opportunity provide:
1. Funding agency
2. Program name
3. Typical award amount
4. Eligibility requirements
5. Deadline (if known)
6. Website URL

Focus on opportunities suitable for this research area.
"""
        
        try:
            result = self.llm.invoke(funding_prompt)
            response = result.content if hasattr(result, 'content') else str(result)
            
            opportunities = self._parse_funding(response)
            self.found_resources['funding'].extend(opportunities)
            
            logger.info(f"Found {len(opportunities)} funding opportunities")
            return opportunities
            
        except Exception as e:
            logger.error(f"Failed to find funding: {e}")
            return []
    
    def _parse_funding(self, text: str) -> List[Dict]:
        """Parse funding information from text"""
        opportunities = []
        lines = text.split('\n')
        
        current_opp = {}
        for line in lines:
            line = line.strip()
            
            if 'agency:' in line.lower() or 'program:' in line.lower():
                if current_opp and 'agency' in current_opp:
                    opportunities.append(current_opp)
                    current_opp = {}
            
            if 'agency:' in line.lower():
                current_opp['agency'] = line.split(':', 1)[-1].strip()
            elif 'program:' in line.lower():
                current_opp['program'] = line.split(':', 1)[-1].strip()
            elif 'amount:' in line.lower():
                current_opp['amount'] = line.split(':', 1)[-1].strip()
            elif 'url:' in line.lower() or 'website:' in line.lower():
                current_opp['url'] = line.split(':', 1)[-1].strip()
        
        if current_opp:
            opportunities.append(current_opp)
        
        return opportunities[:8]
    
    def find_tools(
        self,
        task_description: str,
        field: Optional[str] = None
    ) -> List[Dict]:
        """
        Find relevant research tools and software
        
        Args:
            task_description: Description of task
            field: Scientific field
        
        Returns:
            List of tools
        """
        logger.info(f"Finding tools for {task_description}")
        
        tools_prompt = f"""
Find software tools and platforms for: {task_description}

{f'Field: {field}' if field else ''}

For each tool provide:
1. Tool name
2. Purpose
3. Platform (Web, Desktop, CLI)
4. License type
5. Learning curve
6. Documentation quality
7. URL

Prioritize open-source, well-supported tools.
"""
        
        try:
            result = self.llm.invoke(tools_prompt)
            response = result.content if hasattr(result, 'content') else str(result)
            
            tools = self._parse_tools(response)
            self.found_resources['tools'].extend(tools)
            
            logger.info(f"Found {len(tools)} tools")
            return tools
            
        except Exception as e:
            logger.error(f"Failed to find tools: {e}")
            return []
    
    def _parse_tools(self, text: str) -> List[Dict]:
        """Parse tool information from text"""
        tools = []
        lines = text.split('\n')
        
        current_tool = {}
        for line in lines:
            line = line.strip()
            
            if 'name:' in line.lower() or 'tool:' in line.lower():
                if current_tool:
                    tools.append(current_tool)
                current_tool = {'name': line.split(':', 1)[-1].strip()}
            elif current_tool:
                if 'purpose:' in line.lower():
                    current_tool['purpose'] = line.split(':', 1)[-1].strip()
                elif 'license:' in line.lower():
                    current_tool['license'] = line.split(':', 1)[-1].strip()
                elif 'url:' in line.lower():
                    current_tool['url'] = line.split(':', 1)[-1].strip()
        
        if current_tool:
            tools.append(current_tool)
        
        return tools[:10]
    
    def get_all_resources(self) -> Dict[str, List[Dict]]:
        """Get all resources found in session"""
        return self.found_resources.copy()
    
    def generate_resource_report(self, research_project: str) -> Dict:
        """
        Generate comprehensive resource report for a project
        
        Args:
            research_project: Project description
        
        Returns:
            Complete resource report
        """
        logger.info("Generating comprehensive resource report")
        
        report = {
            'project': research_project,
            'datasets': self.found_resources['datasets'],
            'code': self.found_resources['code'],
            'protocols': self.found_resources['protocols'],
            'funding': self.found_resources['funding'],
            'tools': self.found_resources['tools'],
            'total_resources': sum(len(v) for v in self.found_resources.values())
        }
        
        return report


# Example usage
if __name__ == "__main__":
    from src.config import get_settings
    from src.utils.logger import setup_logging
    
    setup_logging(level="INFO")
    config = get_settings()
    
    # Initialize agent
    print("\n=== Initializing Resource Finder Agent ===")
    agent = ResourceFinderAgent(config=config)
    
    # Find datasets
    print("\n=== Finding Datasets ===")
    datasets = agent.find_datasets(
        "cancer genomics",
        "RNA-seq data",
        "biology"
    )
    print(f"Found {len(datasets)} datasets")
    for ds in datasets[:2]:
        print(f"  - {ds.get('name', 'N/A')}")
    
    # Find code
    print("\n=== Finding Code ===")
    code = agent.find_code_repositories(
        "differential gene expression analysis",
        "Python"
    )
    print(f"Found {len(code)} repositories")
    
    # Find protocols
    print("\n=== Finding Protocols ===")
    protocols = agent.find_protocols(
        "RNA extraction",
        "biology"
    )
    print(f"Found {len(protocols)} protocols")
    
    # Find funding
    print("\n=== Finding Funding ===")
    funding = agent.find_funding(
        "cancer research",
        "US",
        "$500K"
    )
    print(f"Found {len(funding)} funding opportunities")
    
    # Generate report
    print("\n=== Resource Report ===")
    report = agent.generate_resource_report("Cancer genomics study")
    print(f"Total resources: {report['total_resources']}")
    
    print("\n✅ All examples completed!")
