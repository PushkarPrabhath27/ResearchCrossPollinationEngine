"""
Grant Writing Assistant

Helps researchers write grant proposals by leveraging generated hypotheses
and providing structured proposal components.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from src.config import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GrantSection:
    """Grant proposal section"""
    name: str
    content: str
    word_count: int


class GrantWritingAssistant:
    """
    Assists with grant proposal writing
    
    Generates structured proposals from hypotheses and research plans.
    """
    
    SECTION_TEMPLATES = {
        'abstract': "Generate a 250-word abstract for this research project: {hypothesis}",
        'specific_aims': "List 3-4 specific aims for: {hypothesis}",
        'background': "Write background and significance (500 words) for: {hypothesis}",
        'approach': "Describe research approach and methods for: {hypothesis}",
        'timeline': "Create a 3-year timeline for: {hypothesis}",
        'budget_justification': "Justify budget needs for: {hypothesis}"
    }
    
    def __init__(self, config: Optional[Settings] = None, llm=None):
        """Initialize grant assistant"""
        self.config = config
        self.llm = llm
        self.generated_sections = {}
        logger.info("GrantWritingAssistant initialized")
    
    def generate_proposal(self, hypothesis: Dict, funding_agency: str = "NIH") -> Dict:
        """
        Generate complete grant proposal structure
        
        Args:
            hypothesis: Generated hypothesis
            funding_agency: Target agency (NIH, NSF, etc.)
        
        Returns:
            Complete proposal structure
        """
        logger.info(f"Generating proposal for {funding_agency}")
        
        proposal = {
            'title': self._generate_title(hypothesis),
            'agency': funding_agency,
            'hypothesis': hypothesis.get('title', ''),
            'sections': {},
            'created_at': datetime.utcnow().isoformat()
        }
        
        # Generate each section
        for section_name in ['abstract', 'specific_aims', 'background', 'approach', 'timeline']:
            section = self._generate_section(section_name, hypothesis)
            proposal['sections'][section_name] = section
            self.generated_sections[section_name] = section
        
        return proposal
    
    def _generate_title(self, hypothesis: Dict) -> str:
        """Generate proposal title"""
        base_title = hypothesis.get('title', 'Research Proposal')
        return f"Investigating {base_title[:100]}"
    
    def _generate_section(self, section_name: str, hypothesis: Dict) -> GrantSection:
        """Generate a specific section"""
        template = self.SECTION_TEMPLATES.get(section_name, "")
        prompt = template.format(hypothesis=hypothesis.get('description', ''))
        
        # In production, would call LLM
        content = f"[Generated {section_name} content for: {hypothesis.get('title', 'hypothesis')}]"
        
        return GrantSection(
            name=section_name,
            content=content,
            word_count=len(content.split())
        )
    
    def suggest_funding_opportunities(self, hypothesis: Dict) -> List[Dict]:
        """Suggest relevant funding opportunities"""
        field = hypothesis.get('target_field', 'general')
        
        opportunities = [
            {
                'agency': 'NIH',
                'program': 'R01 Research Project Grant',
                'max_amount': '$500,000/year',
                'relevance': 0.9
            },
            {
                'agency': 'NSF',
                'program': 'CAREER Award',
                'max_amount': '$400,000-$800,000',
                'relevance': 0.8
            }
        ]
        
        return opportunities
    
    def export_proposal(self, proposal: Dict, format: str = 'markdown') -> str:
        """Export proposal to specified format"""
        if format == 'markdown':
            lines = [f"# {proposal['title']}\n"]
            lines.append(f"**Agency**: {proposal['agency']}\n")
            
            for section_name, section in proposal.get('sections', {}).items():
                lines.append(f"\n## {section_name.replace('_', ' ').title()}\n")
                lines.append(section.content if isinstance(section, GrantSection) else str(section))
            
            return '\n'.join(lines)
        
        return str(proposal)
