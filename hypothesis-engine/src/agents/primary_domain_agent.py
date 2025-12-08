"""
Primary Domain Agent

Specialized agent for searching within the user's specific research field.
Identifies current approaches, knowledge gaps, and limitations in the domain.
"""

from typing import List, Dict, Optional
import re
from langchain.tools import Tool

from src.agents.base_agent import BaseResearchAgent
from src.config import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PrimaryDomainAgent(BaseResearchAgent):
    """
    Agent specializing in the user's primary research field
    
    Focuses on understanding domain-specific knowledge, current approaches,
    and identifying gaps in existing research.
    """
    
    SYSTEM_PROMPT = """
You are an expert research assistant specializing in understanding a researcher's primary field of study.

Your responsibilities:
1. Analyze the user's research question to identify their specific domain
2. Search for the most relevant papers within that domain
3. Identify current state-of-the-art approaches
4. Find knowledge gaps and limitations in current research
5. Understand what has already been tried and what hasn't worked

When responding:
- Be precise and cite specific papers
- Acknowledge uncertainties
- Identify both successful approaches and dead ends
- Note methodological limitations in existing work
- Consider recent advances (prioritize papers from last 2-3 years)

Available tools:
{tools}

Use the tools to search the research database, then synthesize findings into a clear summary.
"""
    
    def __init__(
        self,
        config: Settings,
        tools: Optional[List[Tool]] = None,
        temperature: float = 0.7
    ):
        """
        Initialize primary domain agent
        
        Args:
            config: Application configuration
            tools: List of LangChain tools
            temperature: LLM temperature
        """
        super().__init__(
            config=config,
            tools=tools,
            name="PrimaryDomainAgent",
            temperature=temperature
        )
        
        # Domain-specific state
        self.identified_field = None
        self.identified_subfield = None
        self.field_confidence = 0.0
        
        logger.info("PrimaryDomainAgent initialized")
    
    def get_system_prompt(self) -> str:
        """Get agent's system prompt"""
        return self.SYSTEM_PROMPT
    
    def analyze_research_question(self, query: str) -> Dict:
        """
        Extract key information from user's question
        
        Args:
            query: Research question
        
        Returns:
            Dictionary with extracted information
        """
        logger.info("Analyzing research question")
        
        # Use LLM to analyze query
        analysis_prompt = f"""
Analyze this research question and extract:
1. Primary field (biology, physics, computer science, etc.)
2. Specific subfield
3. Core research problem
4. Key search keywords
5. Time constraints if mentioned

Research question: "{query}"

Provide a structured analysis.
"""
        
        try:
            result = self.llm.invoke(analysis_prompt)
            
            # Extract field information
            field_info = self._parse_field_from_response(result.content if hasattr(result, 'content') else str(result))
            
            # Store identified field
            self.identified_field = field_info.get('field')
            self.identified_subfield = field_info.get('subfield')
            self.field_confidence = field_info.get('confidence', 0.5)
            
            logger.info(f"Identified field: {self.identified_field}/{self.identified_subfield} (confidence: {self.field_confidence})")
            
            return field_info
            
        except Exception as e:
            logger.error(f"Failed to analyze question: {e}")
            return {
                'field': 'unknown',
                'subfield': 'unknown',
                'problem': query,
                'keywords': [],
                'confidence': 0.0
            }
    
    def _parse_field_from_response(self, response: str) -> Dict:
        """
        Parse field information from LLM response
        
        Args:
            response: LLM response text
        
        Returns:
            Parsed field information
        """
        # Common fields
        fields = {
            'biology': ['biology', 'biolog', 'life science', 'medical', 'biomedical'],
            'computer_science': ['computer science', 'cs', 'computing', 'ai', 'machine learning'],
            'physics': ['physics', 'physical', 'quantum'],
            'chemistry': ['chemistry', 'chemical'],
            'mathematics': ['math', 'mathematics'],
            'engineering': ['engineering', 'engineer']
        }
        
        response_lower = response.lower()
        
        # Find field
        identified_field = 'unknown'
        for field, keywords in fields.items():
            if any(kw in response_lower for kw in keywords):
                identified_field = field
                break
        
        # Extract keywords (look for lists or comma-separated items)
        keywords = []
        keyword_pattern = r'keywords?[:\s]+(.*?)(?:\n|$)'
        match = re.search(keyword_pattern, response_lower, re.IGNORECASE)
        if match:
            keyword_text = match.group(1)
            keywords = [k.strip() for k in re.split(r'[,;]', keyword_text) if k.strip()]
        
        return {
            'field': identified_field,
            'subfield': None,  # Could be extracted with more complex parsing
            'problem': response[:200],
            'keywords': keywords[:10],  # Limit to top 10
            'confidence': 0.7 if identified_field != 'unknown' else 0.3
        }
    
    def identify_field(self, query: str) -> tuple:
        """
        Determine the scientific field from query
        
        Args:
            query: Research question
        
        Returns:
            Tuple of (field, subfield, confidence_score)
        """
        logger.debug(f"Identifying field for query: {query[:100]}...")
        
        analysis = self.analyze_research_question(query)
        
        return (
            analysis.get('field', 'unknown'),
            analysis.get('subfield'),
            analysis.get('confidence', 0.0)
        )
    
    def find_current_approaches(self, problem_description: str, max_papers: int = 20) -> List[Dict]:
        """
        Search for existing solutions to this problem
        
        Args:
            problem_description: Description of the research problem
            max_papers: Maximum papers to find
        
        Returns:
            List of papers with their approaches and outcomes
        """
        logger.info(f"Finding current approaches for problem")
        
        # Build search query focused on methods and results
        search_query = f"""
Find papers that address: {problem_description}

Focus on:
- Methodological approaches used
- Results achieved
- Limitations noted by authors
- Recent work (last 2-3 years preferred)

Return the {max_papers} most relevant papers.
"""
        
        try:
            # Use agent to search
            result = self.run(search_query)
            
            if result['success']:
                # Parse papers from output
                papers = self._extract_papers_from_response(result['output'])
                
                logger.info(f"Found {len(papers)} current approaches")
                return papers
            else:
                logger.warning("Search for current approaches failed")
                return []
                
        except Exception as e:
            logger.error(f"Failed to find current approaches: {e}")
            return []
    
    def _extract_papers_from_response(self, response: str) -> List[Dict]:
        """
        Extract paper information from LLM response
        
        Args:
            response: LLM response with paper information
        
        Returns:
            List of paper dictionaries
        """
        # Simple extraction - in practice, this would parse structured output
        papers = []
        
        # Look for paper-like patterns (title, authors, year)
        # This is a simplified version
        lines = response.split('\n')
        current_paper = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_paper:
                    papers.append(current_paper)
                    current_paper = {}
                continue
            
            # Look for title indicators
            if line.startswith(('Title:', '**', '##')):
                current_paper['title'] = line.split(':', 1)[-1].strip()
            elif 'approach:' in line.lower():
                current_paper['approach'] = line.split(':', 1)[-1].strip()
            elif 'limitation' in line.lower():
                current_paper['limitations'] = line.split(':', 1)[-1].strip()
        
        if current_paper:
            papers.append(current_paper)
        
        return papers[:20]  # Limit results
    
    def identify_knowledge_gaps(
        self,
        problem: str,
        existing_work: List[Dict]
    ) -> List[Dict]:
        """
        Analyze existing work to find knowledge gaps
        
        Args:
            problem: Research problem
            existing_work: List of existing papers
        
        Returns:
            List of identified gaps with descriptions
        """
        logger.info("Identifying knowledge gaps")
        
        # Summarize existing work
        work_summary = "\n".join([
            f"- {p.get('title', 'Unknown')}: {p.get('approach', 'N/A')}"
            for p in existing_work[:10]
        ])
        
        gap_prompt = f"""
Given this research problem: {problem}

And existing work:
{work_summary}

Identify knowledge gaps:
1. What hasn't been tried?
2. What has failed and why?
3. What is theoretically possible but not yet done?
4. What assumptions are limiting progress?

Provide specific, actionable gaps.
"""
        
        try:
            result = self.llm.invoke(gap_prompt)
            response = result.content if hasattr(result, 'content') else str(result)
            
            # Parse gaps
            gaps = self._parse_gaps_from_response(response)
            
            logger.info(f"Identified {len(gaps)} knowledge gaps")
            return gaps
            
        except Exception as e:
            logger.error(f"Failed to identify gaps: {e}")
            return []
    
    def _parse_gaps_from_response(self, response: str) -> List[Dict]:
        """Parse knowledge gaps from LLM response"""
        gaps = []
        
        lines = response.split('\n')
        current_gap = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for numbered items or bullet points
            if re.match(r'^[\d\-\*]+[\.\)]\s+', line):
                if current_gap:
                    gaps.append(current_gap)
                
                gap_text = re.sub(r'^[\d\-\*]+[\.\)]\s+', '', line)
                current_gap = {
                    'description': gap_text,
                    'category': self._categorize_gap(gap_text)
                }
        
        if current_gap:
            gaps.append(current_gap)
        
        return gaps
    
    def _categorize_gap(self, gap_text: str) -> str:
        """Categorize a knowledge gap"""
        gap_lower = gap_text.lower()
        
        if any(word in gap_lower for word in ['not tried', 'unexplored', 'missing']):
            return 'unexplored'
        elif any(word in gap_lower for word in ['failed', 'limitation', 'challenge']):
            return 'limitation'
        elif any(word in gap_lower for word in ['theoretical', 'possible', 'potential']):
            return 'theoretical'
        elif any(word in gap_lower for word in ['assumption', 'constraint', 'barrier']):
            return 'assumption'
        else:
            return 'other'
    
    def get_domain_summary(self) -> Dict:
        """
        Get summary of current domain understanding
        
        Returns:
            Dictionary with domain information
        """
        return {
            'field': self.identified_field,
            'subfield': self.identified_subfield,
            'confidence': self.field_confidence,
            'session_duration': self.session_start,
            'reasoning_steps': len(self.reasoning_history)
        }


# Example usage
if __name__ == "__main__":
    from src.config import get_settings
    from src.utils.logger import setup_logging
    
    setup_logging(level="INFO")
    config = get_settings()
    
    # Initialize agent
    print("\n=== Initializing Primary Domain Agent ===")
    agent = PrimaryDomainAgent(config=config)
    
    # Test field identification
    print("\n=== Identifying Field ===")
    query = "How can machine learning improve early detection of metastatic cancer?"
    field, subfield, confidence = agent.identify_field(query)
    print(f"Field: {field}")
    print(f"Subfield: {subfield}")
    print(f"Confidence: {confidence}")
    
    # Analyze research question
    print("\n=== Analyzing Question ===")
    analysis = agent.analyze_research_question(query)
    print(f"Keywords: {analysis.get('keywords', [])}")
    
    # Get domain summary
    print("\n=== Domain Summary ===")
    summary = agent.get_domain_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\n✅ All examples completed!")
