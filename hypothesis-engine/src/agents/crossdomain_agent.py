"""
Cross-Domain Discovery Agent

Finds unexpected connections and analogous solutions across different scientific fields.
Specializes in creative problem abstraction and cross-field methodology transfer.
"""

from typing import List, Dict, Optional, Tuple
import re
from langchain.tools import Tool

from src.agents.base_agent import BaseResearchAgent
from src.config import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CrossDomainAgent(BaseResearchAgent):
    """
    Agent for discovering cross-field connections and solutions
    
    Abstracts problems to find analogous challenges in different domains
    and identifies transferable methodologies.
    """
    
    SYSTEM_PROMPT = """
You are a creative research assistant specializing in finding unexpected connections between different scientific fields.

Your mission:
1. Take a research problem from one field
2. Search for similar problems or patterns in completely different fields
3. Identify methodologies that could transfer
4. Think creatively about analogies and metaphors
5. Find "hidden" solutions that domain experts might miss

Think like:
- A biologist studying cancer might learn from studying traffic flow (both involve propagation through networks)
- A neuroscientist might use techniques from social network analysis
- A materials  scientist might apply quantum mechanics principles

Be bold and creative, but ground suggestions in actual published research.

Search strategy:
1. Abstract the core problem (remove domain-specific terminology)
2. Search for that abstract problem in other fields
3. Find papers that solved similar challenges
4. Identify transferable methodologies

Available tools:
{tools}
"""
    
    # Common problem patterns that appear across fields
    PROBLEM_PATTERNS = {
        'network_propagation': [
            'spread', 'diffusion', 'propagation', 'transmission', 'flow', 'cascade'
        ],
        'optimization': [
            'optimize', 'maximize', 'minimize', 'efficient', 'optimal', 'improve'
        ],
        'pattern_recognition': [
            'detect', 'identify', 'classify', 'recognize', 'distinguish'
        ],
        'prediction': [
            'predict', 'forecast', 'estimate', 'anticipate', 'model'
        ],
        'clustering': [
            'group', 'cluster', 'segment', 'categorize', 'partition'
        ],
        'stability': [
            'stable', 'equilibrium', 'balance', 'steady-state', 'homeostasis'
        ]
    }
    
    # Field mappings for cross-domain search
    FIELD_MAPPINGS = {
        'biology': ['physics', 'computer_science', 'mathematics', 'engineering'],
        'physics': ['biology', 'chemistry', 'mathematics', 'computer_science'],
        'computer_science': ['biology', 'physics', 'mathematics', 'engineering'],
        'chemistry': ['physics', 'biology', 'materials_science'],
        'mathematics': ['physics', 'biology', 'computer_science', 'economics']
    }
    
    def __init__(
        self,
        config: Settings,
        tools: Optional[List[Tool]] = None,
        temperature: float = 0.8  # Higher temperature for creativity
    ):
        """
        Initialize cross-domain agent
        
        Args:
            config: Application configuration
            tools: List of LangChain tools
            temperature: LLM temperature (higher for creativity)
        """
        super().__init__(
            config=config,
            tools=tools,
            name="CrossDomainAgent",
            temperature=temperature
        )
        
        # Track discovered connections
        self.discovered_analogies = []
        
        logger.info("CrossDomainAgent initialized with creative mode")
    
    def get_system_prompt(self) -> str:
        """Get agent's system prompt"""
        return self.SYSTEM_PROMPT
    
    def abstract_problem(self, domain_specific_query: str, source_field: str) -> Dict:
        """
        Convert domain-specific problem to abstract form
        
        Args:
            domain_specific_query: Original problem description
            source_field: Field the problem comes from
        
        Returns:
            Dictionary with abstract problem and metadata
        """
        logger.info(f"Abstracting problem from {source_field}")
        
        abstraction_prompt = f"""
Convert this domain-specific research problem into an abstract form that could apply to any field.

Original problem: "{domain_specific_query}"
Source field: {source_field}

Guidelines:
1. Remove field-specific terminology
2. Focus on the core challenge or pattern
3. Identify the fundamental mechanism or process
4. Express in general terms

Example:
"How do cancer cells migrate through blood vessels?" (Biology)
Becomes:
"How do particles navigate through constrained networks with varying permeability?"

Provide:
1. Abstract problem statement
2. Core mechanism (e.g., network propagation, optimization, prediction)
3. Key variables involved
4. Desired outcome
"""
        
        try:
            result = self.llm.invoke(abstraction_prompt)
            response = result.content if hasattr(result, 'content') else str(result)
            
            # Parse abstraction
            abstraction = self._parse_abstraction(response, domain_specific_query)
            
            logger.info(f"Abstracted to: {abstraction['abstract_problem']}")
            return abstraction
            
        except Exception as e:
            logger.error(f"Failed to abstract problem: {e}")
            return {
                'abstract_problem': domain_specific_query,
                'core_mechanism': 'unknown',
                'original_problem': domain_specific_query,
                'source_field': source_field
            }
    
    def _parse_abstraction(self, response: str, original: str) -> Dict:
        """Parse abstraction from LLM response"""
        # Detect pattern type
        pattern_type = 'other'
        response_lower = response.lower()
        
        for pattern, keywords in self.PROBLEM_PATTERNS.items():
            if any(kw in response_lower for kw in keywords):
                pattern_type = pattern
                break
        
        return {
            'abstract_problem': response[:300],  # First 300 chars
            'core_mechanism': pattern_type,
            'original_problem': original,
            'pattern_keywords': self.PROBLEM_PATTERNS.get(pattern_type, [])
        }
    
    def search_multiple_fields(
        self,
        abstract_problem: str,
        exclude_field: str,
        max_per_field: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        Search across multiple fields for similar problems
        
        Args:
            abstract_problem: Abstracted problem description
            exclude_field: Field to exclude (the source field)
            max_per_field: Maximum papers per field
        
        Returns:
            Dictionary mapping fields to relevant papers
        """
        logger.info(f"Searching multiple fields (excluding {exclude_field})")
        
        # Get fields to search
        target_fields = self.FIELD_MAPPINGS.get(exclude_field, [
            'biology', 'physics', 'chemistry', 'computer_science', 'mathematics'
        ])
        
        results = {}
        
        for field in target_fields:
            field_query = f"""
Find papers in {field} that address: {abstract_problem}

Focus on papers that solve similar fundamental challenges, even if the application domain is different.
"""
            
            try:
                # Use agent to search this field
                search_result = self.run(field_query)
                
                if search_result['success']:
                    # Parse papers
                    papers = self._extract_papers_from_response(
                        search_result['output'],
                        max_papers=max_per_field
                    )
                    
                    if papers:
                        results[field] = papers
                        logger.info(f"Found {len(papers)} papers in {field}")
                
            except Exception as e:
                logger.warning(f"Search in {field} failed: {e}")
                continue
        
        logger.info(f"Cross-domain search complete: {len(results)} fields")
        return results
    
    def _extract_papers_from_response(
        self,
        response: str,
        max_papers: int = 10
    ) -> List[Dict]:
        """Extract paper information from LLM response"""
        papers = []
        lines = response.split('\n')
        
        current_paper = {}
        for line in lines:
            line = line.strip()
            if not line:
                if current_paper and 'title' in current_paper:
                    papers.append(current_paper)
                    current_paper = {}
                continue
            
            # Simple pattern matching
            if any(indicator in line for indicator in ['Title:', '**', '##', '1.', '2.']):
                title = re.sub(r'^[\d\.\*\#\s]+', '', line)
                title = title.replace('Title:', '').strip()
                if title:
                    current_paper['title'] = title
            elif 'method' in line.lower() or 'approach' in line.lower():
                current_paper['method'] = line.split(':', 1)[-1].strip()
        
        if current_paper and 'title' in current_paper:
            papers.append(current_paper)
        
        return papers[:max_papers]
    
    def find_analogies(
        self,
        problem: str,
        other_field_papers: List[Dict],
        source_field: str
    ) -> List[Dict]:
        """
        Identify structural similarities between problems
        
        Args:
            problem: Original problem
            other_field_papers: Papers from other fields
            source_field: Source field
        
        Returns:
            List of analogies with transferability scores
        """
        logger.info("Finding analogies across fields")
        
        analogies = []
        
        for paper in other_field_papers[:10]:  # Limit to top 10
            analogy_prompt = f"""
Compare these two research approaches:

Original problem: {problem} (in {source_field})

Other field approach:
Title: {paper.get('title', 'N/A')}
Method: {paper.get('method', 'N/A')}

Analyze:
1. What aspects are analogous?
2. How might the method transfer?
3. What adaptations would be needed?
4. Rate transferability (0-10)

Be specific about the analogy.
"""
            
            try:
                result = self.llm.invoke(analogy_prompt)
                response = result.content if hasattr(result, 'content') else str(result)
                
                # Parse analogy
                analogy = {
                    'source_paper': paper,
                    'analogy_description': response[:500],
                    'transferability_score': self._extract_score(response),
                    'adaptations_needed': self._extract_adaptations(response)
                }
                
                analogies.append(analogy)
                self.discovered_analogies.append(analogy)
                
            except Exception as e:
                logger.warning(f"Failed to analyze analogy: {e}")
                continue
        
        # Sort by transferability
        analogies.sort(key=lambda x: x['transferability_score'], reverse=True)
        
        logger.info(f"Found {len(analogies)} analogies")
        return analogies
    
    def _extract_score(self, text: str) -> float:
        """Extract transferability score from text"""
        # Look for ratings like "8/10", "7 out of 10", "score: 6"
        patterns = [
            r'(\d+)\s*/\s*10',
            r'(\d+)\s+out of\s+10',
            r'score[:\s]+(\d+)',
            r'rating[:\s]+(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                return min(score / 10.0, 1.0)  # Normalize to 0-1
        
        return 0.5  # Default middle score
    
    def _extract_adaptations(self, text: str) -> List[str]:
        """Extract needed adaptations from text"""
        adaptations = []
        
        # Look for bullet points or numbered lists after "adaptations" or "needed"
        lines = text.split('\n')
        in_adaptations  = False
        
        for line in lines:
            if 'adaptation' in line.lower() or 'needed' in line.lower():
                in_adaptations = True
                continue
            
            if in_adaptations and re.match(r'^[\d\-\*]+[\.\)]\s+', line):
                adaptation = re.sub(r'^[\d\-\*]+[\.\)]\s+', '', line).strip()
                if adaptation:
                    adaptations.append(adaptation)
        
        return adaptations[:5]  # Top 5
    
    def assess_transferability(
        self,
        method_paper: Dict,
        target_problem: str,
        target_field: str
    ) -> Dict:
        """
        Evaluate if a method from another field could work
        
        Args:
            method_paper: Paper with the method
            target_problem: Problem to solve
            target_field: Field to transfer to
        
        Returns:
            Assessment with scores and recommendations
        """
        logger.info("Assessing method transferability")
        
        assessment_prompt = f"""
Evaluate the transferability of this method:

Method source:
- Title: {method_paper.get('title', 'N/A')}
- Field: {method_paper.get('field', 'unknown')}
- Approach: {method_paper.get('method', 'N/A')}

Target:
- Problem: {target_problem}
- Field: {target_field}

Rate on these dimensions (0-10 each):
1. Conceptual similarity: How similar are the underlying problems?
2. Technical feasibility: Can this be implemented in the target field?
3. Resource requirements: What resources would be needed?
4. Likely effectiveness: How well would this work?

Provide specific recommendations for adaptation.
"""
        
        try:
            result = self.llm.invoke(assessment_prompt)
            response = result.content if hasattr(result, 'content') else str(result)
            
            assessment = {
                'conceptual_similarity': self._extract_dimension_score(response, 'conceptual'),
                'technical_feasibility': self._extract_dimension_score(response, 'feasibility'),
                'resource_requirements': self._extract_dimension_score(response, 'resource'),
                'likely_effectiveness': self._extract_dimension_score(response, 'effectiveness'),
                'overall_score': 0.0,
                'recommendations': self._extract_adaptations(response),
                'assessment_text': response[:500]
            }
            
            # Calculate overall score
            assessment['overall_score'] = sum([
                assessment['conceptual_similarity'],
                assessment['technical_feasibility'],
                assessment['likely_effectiveness']
            ]) / 3.0
            
            return assessment
            
        except Exception as e:
            logger.error(f"Failed to assess transferability: {e}")
            return {
                'overall_score': 0.0,
                'error': str(e)
            }
    
    def _extract_dimension_score(self, text: str, dimension: str) -> float:
        """Extract score for a specific dimension"""
        # Look for the dimension followed by a score
        pattern = rf'{dimension}[:\s]+(\d+)'
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            score = int(match.group(1))
            return min(score / 10.0, 1.0)
        
        return 0.5
    
    def get_discovered_analogies(self) -> List[Dict]:
        """Get all discovered analogies from session"""
        return self.discovered_analogies.copy()


# Example usage
if __name__ == "__main__":
    from src.config import get_settings
    from src.utils.logger import setup_logging
    
    setup_logging(level="INFO")
    config = get_settings()
    
    # Initialize agent
    print("\n=== Initializing Cross-Domain Agent ===")
    agent = CrossDomainAgent(config=config)
    
    # Test problem abstraction
    print("\n=== Abstracting Problem ===")
    problem = "How can we predict cancer metastasis patterns in patients?"
    abstraction = agent.abstract_problem(problem, "biology")
    
    print(f"Original: {problem}")
    print(f"Abstract: {abstraction['abstract_problem'][:150]}...")
    print(f"Pattern: {abstraction['core_mechanism']}")
    
    # Test cross-field search
    print("\n=== Cross-Field Search ===")
    results = agent.search_multiple_fields(
        abstraction['abstract_problem'],
        exclude_field="biology",
        max_per_field=2
    )
    
    print(f"Found papers in {len(results)} fields:")
    for field, papers in results.items():
        print(f"  {field}: {len(papers)} papers")
    
    # Get discovered analogies
    print("\n=== Discovered Analogies ===")
    analogies = agent.get_discovered_analogies()
    print(f"Total analogies: {len(analogies)}")
    
    print("\n✅ All examples completed!")
