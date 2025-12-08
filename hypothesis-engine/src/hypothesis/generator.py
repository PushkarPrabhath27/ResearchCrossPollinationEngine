"""
Hypothesis Generator

Synthesizes agent findings to generate novel, testable research hypotheses
combining insights from multiple domains.
"""

from typing import List, Dict, Optional
import json
from datetime import datetime
from src.config import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HypothesisGenerator:
    """
    Generates novel research hypotheses from agent findings
    
    Combines primary domain knowledge, cross-domain analogies,
    and methodology transfers to create innovative hypotheses.
    """
    
    def __init__(self, config: Optional[Settings] = None):
        """Initialize hypothesis generator"""
        self.config = config
        self.generated_hypotheses = []
        logger.info("HypothesisGenerator initialized")
    
    def generate_hypotheses(
        self,
        primary_findings: Dict,
        cross_domain_findings: Dict,
        methodology_transfers: List[Dict]
    ) -> List[Dict]:
        """
        Generate hypotheses from agent findings
        
        Args:
            primary_findings: From PrimaryDomainAgent
            cross_domain_findings: From CrossDomainAgent
            methodology_transfers: From MethodologyTransferAgent
        
        Returns:
            List of generated hypotheses
        """
        logger.info("Generating hypotheses from agent findings")
        
        hypotheses = []
        
        # Type 1: Fill knowledge gaps in primary domain
        for gap in primary_findings.get('knowledge_gaps', []):
            hypothesis = self._create_gap_filling_hypothesis(gap, primary_findings)
            if hypothesis:
                hypotheses.append(hypothesis)
        
        # Type 2: Cross-domain method transfer
        for transfer in methodology_transfers[:5]:  # Top 5
            hypothesis = self._create_transfer_hypothesis(transfer, primary_findings)
            if hypothesis:
                hypotheses.append(hypothesis)
        
        # Type 3: Analogical reasoning
        for analogy in cross_domain_findings.get('promising_analogies', [])[:3]:
            hypothesis = self._create_analogy_hypothesis(analogy, primary_findings)
            if hypothesis:
                hypotheses.append(hypothesis)
        
        # Rank hypotheses
        ranked = self._rank_hypotheses(hypotheses)
        
        self.generated_hypotheses.extend(ranked)
        
        logger.info(f"Generated {len(ranked)} hypotheses")
        return ranked
    
    def _create_gap_filling_hypothesis(self, gap: Dict, primary: Dict) -> Optional[Dict]:
        """Create hypothesis to fill identified gap"""
        return {
            "id": f"hyp_{len(self.generated_hypotheses) + 1}",
            "type": "gap_filling",
            "title": f"Addressing {gap.get('description', 'knowledge gap')[:50]}...",
            "description": gap.get('description', ''),
            "field": primary.get('field', {}).get('primary', 'unknown'),
            "novelty_score": 0.7,
            "feasibility_score": 0.8,
            "impact_potential": 0.6,
            "created_at": datetime.utcnow().isoformat()
        }
    
    def _create_transfer_hypothesis(self, transfer: Dict, primary: Dict) -> Optional[Dict]:
        """Create hypothesis from methodology transfer"""
        return {
            "id": f"hyp_{len(self.generated_hypotheses) + 1}",
            "type": "methodology_transfer",
            "title": f"Apply {transfer.get('method_summary', {}).get('name', 'method')} to {primary.get('field', {}).get('specific_area', 'problem')}",
            "description": f"Transfer methodology from {transfer.get('method_summary', {}).get('source_field', 'other field')}",
            "source_field": transfer.get('method_summary', {}).get('source_field'),
            "target_field": primary.get('field', {}).get('primary'),
            "adaptation_plan": transfer.get('adaptation_plan', {}),
            "novelty_score": 0.85,
            "feasibility_score": 0.65,
            "impact_potential": 0.8,
            "created_at": datetime.utcnow().isoformat()
        }
    
    def _create_analogy_hypothesis(self, analogy: Dict, primary: Dict) -> Optional[Dict]:
        """Create hypothesis from cross-domain analogy"""
        return {
            "id": f"hyp_{len(self.generated_hypotheses) + 1}",
            "type": "cross_domain_analogy",
            "title": f"Apply {analogy.get('field', 'cross-domain')} insights to {primary.get('field', {}).get('specific_area', 'problem')}",
            "description": analogy.get('analogy', {}).get('explanation', ''),
            "source_field": analogy.get('field'),
            "target_field": primary.get('field', {}).get('primary'),
            "similarity_score": analogy.get('analogy', {}).get('similarity_score', 0.5),
            "novelty_score": 0.9,
            "feasibility_score": 0.5,
            "impact_potential": 0.85,
            "created_at": datetime.utcnow().isoformat()
        }
    
    def _rank_hypotheses(self, hypotheses: List[Dict]) -> List[Dict]:
        """Rank hypotheses by composite score"""
        for hyp in hypotheses:
            # Composite score: weighted average
            hyp['composite_score'] = (
                hyp.get('novelty_score', 0) * 0.4 +
                hyp.get('feasibility_score', 0) * 0.3 +
                hyp.get('impact_potential', 0) * 0.3
            )
        
        # Sort by composite score
        hypotheses.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Add rank
        for i, hyp in enumerate(hypotheses):
            hyp['rank'] = i + 1
        
        return hypotheses
    
    def export_hypotheses(self, output_path: str):
        """Export generated hypotheses to JSON"""
        with open(output_path, 'w') as f:
            json.dump({
                'total_hypotheses': len(self.generated_hypotheses),
                'generated_at': datetime.utcnow().isoformat(),
                'hypotheses': self.generated_hypotheses
            }, f, indent=2)
        
        logger.info(f"Exported {len(self.generated_hypotheses)} hypotheses to {output_path}")
