"""
Methodology Transfer Agent

Helps researchers adapt techniques from other fields by providing detailed
implementation guidance, identifying barriers, and creating adaptation plans.
"""

from typing import List, Dict, Optional
from langchain.tools import Tool

from src.agents.base_agent import BaseResearchAgent
from src.config import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MethodologyTransferAgent(BaseResearchAgent):
    """
    Agent for transferring methodologies across fields
    
    Analyzes methods from source field and provides detailed plans
    for adapting them to target field.
    """
    
    SYSTEM_PROMPT = """
You are a research methodology expert who helps researchers adapt techniques from other fields.

Your responsibilities:
1. Take a methodology from one field
2. Analyze its core principles and requirements
3. Identify what needs to be adapted for a different field
4. Provide step-by-step implementation guidance
5. Anticipate technical challenges
6. Suggest validation approaches

Be specific and practical. Provide enough detail that a researcher could actually implement your suggestions.

Available tools:
{tools}
"""
    
    def __init__(
        self,
        config: Settings,
        tools: Optional[List[Tool]] = None,
        temperature: float = 0.7
    ):
        """
        Initialize methodology transfer agent
        
        Args:
            config: Application configuration
            tools: List of LangChain tools
            temperature: LLM temperature
        """
        super().__init__(
            config=config,
            tools=tools,
            name="MethodologyTransferAgent",
            temperature=temperature
        )
        
        logger.info("MethodologyTransferAgent initialized")
    
    def get_system_prompt(self) -> str:
        """Get agent's system prompt"""
        return self.SYSTEM_PROMPT
    
    def analyze_method_requirements(self, method_paper: Dict) -> Dict:
        """
        Extract what the method needs
        
        Args:
            method_paper: Paper describing the method
        
        Returns:
            Dictionary with requirements
        """
        logger.info(f"Analyzing requirements for method")
        
        analysis_prompt = f"""
Analyze the requirements for this methodology:

Paper: {method_paper.get('title', 'N/A')}
Field: {method_paper.get('field', 'N/A')}
Method: {method_paper.get('method', 'N/A')}

Extract:
1. Equipment needed
2. Required expertise/skills
3. Software/algorithms required
4. Data requirements
5. Time investment
6. Budget estimate (if mentioned)

Be specific and comprehensive.
"""
        
        try:
            result = self.llm.invoke(analysis_prompt)
            response = result.content if hasattr(result, 'content') else str(result)
            
            requirements = {
                'equipment': self._extract_section(response, 'equipment'),
                'expertise': self._extract_section(response, 'expertise'),
                'software': self._extract_section(response, 'software'),
                'data': self._extract_section(response, 'data'),
                'time': self._extract_section(response, 'time'),
                'budget': self._extract_section(response, 'budget'),
                'full_analysis': response
            }
            
            return requirements
            
        except Exception as e:
            logger.error(f"Failed to analyze requirements: {e}")
            return {'error': str(e)}
    
    def _extract_section(self, text: str, section_name: str) -> List[str]:
        """Extract items from a section of text"""
        items = []
        lines = text.split('\n')
        in_section = False
        
        for line in lines:
            line_lower = line.lower()
            
            # Check if we're entering the section
            if section_name in line_lower:
                in_section = True
                continue
            
            # Check if we're leaving the section
            if in_section and line_lower and line[0].isalpha() and ':' in line:
                in_section = False
            
            # Extract items
            if in_section and line.strip():
                # Remove bullet points and numbering
                item = line.strip().lstrip('-*•0123456789.)').strip()
                if item:
                    items.append(item)
        
        return items[:10]  # Limit to top 10
    
    def generate_adaptation_plan(
        self,
        source_method: Dict,
        target_problem: str,
        target_field: str
    ) -> Dict:
        """
        Create detailed adaptation plan
        
        Args:
            source_method: Method from source field
            target_problem: Problem in target field
            target_field: Target field
        
        Returns:
            Comprehensive adaptation plan
        """
        logger.info(f"Generating adaptation plan for {target_field}")
        
        plan_prompt = f"""
Create a detailed adaptation plan for transferring this methodology:

Source Method:
- Name: {source_method.get('title', 'N/A')}
- Field: {source_method.get('field', 'N/A')}
- Core Principle: {source_method.get('method', 'N/A')}

Target:
- Problem: {target_problem}
- Field: {target_field}

Provide:
1. What stays the same (core principles that transfer directly)
2. What needs modification (and how)
3. What's completely different (requires new approach)
4. Step-by-step implementation plan (numbered steps)
5. Validation strategy (how to test if it works)
6. Expected timeline

Be specific and actionable.
"""
        
        try:
            result = self.llm.invoke(plan_prompt)
            response = result.content if hasattr(result, 'content') else str(result)
            
            plan = {
                'unchanged_aspects': self._extract_section(response, 'same'),
                'modifications': self._extract_section(response, 'modification'),
                'new_aspects': self._extract_section(response, 'different'),
                'implementation_steps': self._parse_steps(response),
                'validation': self._extract_section(response, 'validation'),
                'timeline': self._extract_timeline(response),
                'full_plan': response
            }
            
            return plan
            
        except Exception as e:
            logger.error(f"Failed to generate adaptation plan: {e}")
            return {'error': str(e)}
    
    def _parse_steps(self, text: str) -> List[Dict]:
        """Parse implementation steps from text"""
        steps = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            # Look for numbered steps
            if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-')):
                step_text = line.strip().lstrip('0123456789.-)*').strip()
                if step_text:
                    steps.append({
                        'step': len(steps) + 1,
                        'action': step_text,
                        'estimated_time': self._estimate_step_time(step_text)
                    })
        
        return steps[:15]  # Limit to 15 steps
    
    def _estimate_step_time(self, step_text: str) -> str:
        """Estimate time for a step based on keywords"""
        step_lower = step_text.lower()
        
        if any(word in step_lower for word in ['design', 'develop', 'create', 'build']):
            return "2-4 weeks"
        elif any(word in step_lower for word in ['test', 'validate', 'verify']):
            return "1-2 weeks"
        elif any(word in step_lower for word in ['analyze', 'evaluate', 'assess']):
            return "1 week"
        elif any(word in step_lower for word in ['prepare', 'setup', 'configure']):
            return "2-3 days"
        else:
            return "1 week"
    
    def _extract_timeline(self, text: str) -> str:
        """Extract overall timeline from text"""
        # Look for timeline mentions
        import re
        patterns = [
            r'(\d+)\s+(weeks?|months?|days?)',
            r'timeline[:\s]+([^\n]+)',
            r'expected\s+duration[:\s]+([^\n]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return "6-12 months (estimated)"
    
    def identify_technical_barriers(
        self,
        source_field: str,
        target_field: str,
        method: Dict
    ) -> List[Dict]:
        """
        Find challenges in transferring method
        
        Args:
            source_field: Source field
            target_field: Target field
            method: Method details
        
        Returns:
            List of technical barriers
        """
        logger.info(f"Identifying barriers for {source_field} -> {target_field} transfer")
        
        barrier_prompt = f"""
Identify technical challenges in transferring this methodology:

Source: {source_field}
Target: {target_field}
Method: {method.get('title', 'N/A')}

Consider:
1. Different scales (nano vs macro, temporal scales)
2. Different constraints (biological vs physical)
3. Different measurement capabilities
4. Different theoretical frameworks
5. Equipment limitations
6. Ethical considerations

List specific, concrete barriers.
"""
        
        try:
            result = self.llm.invoke(barrier_prompt)
            response = result.content if hasattr(result, 'content') else str(result)
            
            barriers = self._parse_barriers(response)
            
            logger.info(f"Identified {len(barriers)} technical barriers")
            return barriers
            
        except Exception as e:
            logger.error(f"Failed to identify barriers: {e}")
            return []
    
    def _parse_barriers(self, text: str) -> List[Dict]:
        """Parse barriers from text"""
        barriers = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                barrier_text = line.lstrip('0123456789.-)*').strip()
                if barrier_text:
                    barriers.append({
                        'barrier': barrier_text,
                        'severity': self._assess_severity(barrier_text),
                        'category': self._categorize_barrier(barrier_text)
                    })
        
        return barriers[:10]
    
    def _assess_severity(self, barrier_text: str) -> str:
        """Assess severity of a barrier"""
        barrier_lower = barrier_text.lower()
        
        if any(word in barrier_lower for word in ['impossible', 'cannot', 'insurmountable']):
            return 'critical'
        elif any(word in barrier_lower for word in ['difficult', 'challenging', 'major']):
            return 'high'
        elif any(word in barrier_lower for word in ['moderate', 'some', 'minor']):
            return 'medium'
        else:
            return 'low'
    
    def _categorize_barrier(self, barrier_text: str) -> str:
        """Categorize a barrier"""
        barrier_lower = barrier_text.lower()
        
        if any(word in barrier_lower for word in ['scale', 'size', 'dimension']):
            return 'scale'
        elif any(word in barrier_lower for word in ['equipment', 'instrument', 'device']):
            return 'equipment'
        elif any(word in barrier_lower for word in ['theoretical', 'framework', 'model']):
            return 'theoretical'
        elif any(word in barrier_lower for word in ['measurement', 'detection', 'sensing']):
            return 'measurement'
        elif any(word in barrier_lower for word in ['ethical', 'regulatory', 'approval']):
            return 'ethical'
        else:
            return 'other'
    
    def find_implementation_examples(
        self,
        method_type: str,
        target_field: str
    ) -> List[Dict]:
        """
        Search for cases where similar transfers worked
        
        Args:
            method_type: Type of methodology
            target_field: Target field
        
        Returns:
            List of successful transfer examples
        """
        logger.info(f"Finding implementation examples for {method_type} in {target_field}")
        
        search_query = f"""
Find papers where {method_type} methods were successfully adapted to {target_field}.

Focus on:
- Cross-disciplinary applications
- Methodological innovations
- Practical implementation details
- Lessons learned
"""
        
        try:
            result = self.run(search_query)
            
            if result['success']:
                examples = self._extract_examples(result['output'])
                return examples
            else:
                return []
                
        except Exception as e:
            logger.error(f"Failed to find examples: {e}")
            return []
    
    def _extract_examples(self, text: str) -> List[Dict]:
        """Extract implementation examples from text"""
        examples = []
        # Simple extraction - would be more sophisticated in production
        lines = text.split('\n')
        
        current_example = {}
        for line in lines:
            if 'title:' in line.lower():
                if current_example:
                    examples.append(current_example)
                current_example = {'title': line.split(':', 1)[-1].strip()}
            elif current_example and 'lesson' in line.lower():
                current_example['lesson'] = line.split(':', 1)[-1].strip()
        
        if current_example:
            examples.append(current_example)
        
        return examples[:5]


# Example usage
if __name__ == "__main__":
    from src.config import get_settings
    from src.utils.logger import setup_logging
    
    setup_logging(level="INFO")
    config = get_settings()
    
    # Initialize agent
    print("\n=== Initializing Methodology Transfer Agent ===")
    agent = MethodologyTransferAgent(config=config)
    
    # Test method analysis
    print("\n=== Analyzing Method Requirements ===")
    test_method = {
        'title': 'Particle Image Velocimetry',
        'field': 'Fluid Dynamics',
        'method': 'Track particle motion to infer flow fields'
    }
    
    requirements = agent.analyze_method_requirements(test_method)
    print(f"Equipment needs: {len(requirements.get('equipment', []))} items")
    print(f"Expertise required: {len(requirements.get('expertise', []))} areas")
    
    # Test adaptation plan
    print("\n=== Generating Adaptation Plan ===")
    plan = agent.generate_adaptation_plan(
        test_method,
        "Track cancer cell migration in tissue",
        "Biology"
    )
    
    print(f"Implementation steps: {len(plan.get('implementation_steps', []))}")
    print(f"Timeline: {plan.get('timeline', 'N/A')}")
    
    # Test barrier identification
    print("\n=== Identifying Technical Barriers ===")
    barriers = agent.identify_technical_barriers(
        "Fluid Dynamics",
        "Biology",
        test_method
    )
    
    print(f"Found {len(barriers)} barriers:")
    for b in barriers[:3]:
        print(f"  - {b['barrier'][:80]}... (Severity: {b['severity']})")
    
    print("\n✅ All examples completed!")
