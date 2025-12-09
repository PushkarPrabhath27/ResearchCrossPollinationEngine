"""
Hypothesis Scorer

Provides comprehensive scoring and ranking for generated hypotheses
based on novelty, feasibility, impact, and other dimensions.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ImpactLevel(Enum):
    """Research impact levels"""
    INCREMENTAL = 1
    MODERATE = 2
    SIGNIFICANT = 3
    HIGH = 4
    TRANSFORMATIVE = 5


@dataclass
class ScoreWeight:
    """Configurable scoring weights"""
    novelty: float = 0.30
    feasibility: float = 0.25
    impact: float = 0.25
    evidence: float = 0.10
    timeliness: float = 0.10


class HypothesisScorer:
    """
    Comprehensive hypothesis scoring system
    
    Evaluates hypotheses across multiple dimensions and provides
    composite scores for ranking.
    """
    
    def __init__(self, weights: ScoreWeight = None):
        """
        Initialize scorer with custom or default weights
        
        Args:
            weights: Custom scoring weights
        """
        self.weights = weights or ScoreWeight()
        logger.info("HypothesisScorer initialized")
    
    def score_hypothesis(self, hypothesis: Dict) -> Dict:
        """
        Score a single hypothesis across all dimensions
        
        Args:
            hypothesis: Hypothesis dictionary
        
        Returns:
            Hypothesis with detailed scores
        """
        logger.debug(f"Scoring hypothesis: {hypothesis.get('id')}")
        
        scores = {
            'novelty': self._score_novelty(hypothesis),
            'feasibility': self._score_feasibility(hypothesis),
            'impact': self._score_impact(hypothesis),
            'evidence': self._score_evidence(hypothesis),
            'timeliness': self._score_timeliness(hypothesis)
        }
        
        # Calculate composite score
        composite = self._calculate_composite(scores)
        
        # Determine confidence
        confidence = self._calculate_confidence(scores)
        
        # Add to hypothesis
        hypothesis['detailed_scores'] = scores
        hypothesis['composite_score'] = composite
        hypothesis['confidence'] = confidence
        hypothesis['grade'] = self._assign_grade(composite)
        hypothesis['recommendation'] = self._generate_recommendation(composite, confidence)
        
        return hypothesis
    
    def _score_novelty(self, hypothesis: Dict) -> Dict:
        """Score novelty dimension"""
        # Start with existing novelty score if present
        base_score = hypothesis.get('novelty_score', 0.5)
        
        # Adjust based on type
        type_adjustments = {
            'cross_domain_analogy': 0.15,  # More novel
            'methodology_transfer': 0.10,
            'gap_filling': 0.05
        }
        
        hyp_type = hypothesis.get('type', 'other')
        adjustment = type_adjustments.get(hyp_type, 0.0)
        
        final_score = min(1.0, base_score + adjustment)
        
        return {
            'score': final_score,
            'base': base_score,
            'type_adjustment': adjustment,
            'explanation': self._novelty_explanation(final_score)
        }
    
    def _novelty_explanation(self, score: float) -> str:
        """Generate explanation for novelty score"""
        if score >= 0.8:
            return "Highly novel - represents unexplored territory"
        elif score >= 0.6:
            return "Moderately novel - builds on limited existing work"
        elif score >= 0.4:
            return "Somewhat novel - extends existing approaches"
        else:
            return "Limited novelty - similar work exists"
    
    def _score_feasibility(self, hypothesis: Dict) -> Dict:
        """Score feasibility dimension"""
        base_score = hypothesis.get('feasibility_score', 0.5)
        
        # Check for barriers
        barriers = hypothesis.get('technical_barriers', [])
        barrier_penalty = len(barriers) * 0.05
        
        # Check for available resources
        resources = hypothesis.get('resources_available', False)
        resource_bonus = 0.1 if resources else 0.0
        
        final_score = max(0.0, min(1.0, base_score - barrier_penalty + resource_bonus))
        
        return {
            'score': final_score,
            'base': base_score,
            'barriers_count': len(barriers),
            'resources_available': resources,
            'explanation': self._feasibility_explanation(final_score)
        }
    
    def _feasibility_explanation(self, score: float) -> str:
        """Generate explanation for feasibility score"""
        if score >= 0.8:
            return "Highly feasible with current resources and technology"
        elif score >= 0.6:
            return "Feasible with moderate effort and resources"
        elif score >= 0.4:
            return "Challenging but achievable with significant investment"
        else:
            return "Significant barriers exist - requires breakthrough"
    
    def _score_impact(self, hypothesis: Dict) -> Dict:
        """Score potential impact dimension"""
        base_score = hypothesis.get('impact_potential', 0.5)
        
        # Adjust based on field importance
        field = hypothesis.get('target_field', '')
        high_impact_fields = ['medicine', 'biology', 'energy', 'climate']
        field_bonus = 0.1 if field in high_impact_fields else 0.0
        
        final_score = min(1.0, base_score + field_bonus)
        
        return {
            'score': final_score,
            'base': base_score,
            'field': field,
            'field_bonus': field_bonus,
            'impact_level': self._determine_impact_level(final_score).name
        }
    
    def _determine_impact_level(self, score: float) -> ImpactLevel:
        """Determine impact level from score"""
        if score >= 0.9:
            return ImpactLevel.TRANSFORMATIVE
        elif score >= 0.75:
            return ImpactLevel.HIGH
        elif score >= 0.6:
            return ImpactLevel.SIGNIFICANT
        elif score >= 0.4:
            return ImpactLevel.MODERATE
        else:
            return ImpactLevel.INCREMENTAL
    
    def _score_evidence(self, hypothesis: Dict) -> Dict:
        """Score evidence supporting the hypothesis"""
        # Count supporting papers
        supporting_papers = len(hypothesis.get('inspiration_papers', []))
        
        # Score based on evidence quantity and quality
        if supporting_papers >= 10:
            score = 0.9
        elif supporting_papers >= 5:
            score = 0.7
        elif supporting_papers >= 2:
            score = 0.5
        else:
            score = 0.3
        
        return {
            'score': score,
            'supporting_papers': supporting_papers,
            'explanation': f"Based on {supporting_papers} supporting papers"
        }
    
    def _score_timeliness(self, hypothesis: Dict) -> Dict:
        """Score timeliness/trendiness of the hypothesis"""
        # Check if it relates to trending topics
        trending_topics = ['ai', 'machine learning', 'crispr', 'quantum', 'climate']
        
        title = hypothesis.get('title', '').lower()
        description = hypothesis.get('description', '').lower()
        text = title + " " + description
        
        matches = sum(1 for topic in trending_topics if topic in text)
        score = min(1.0, 0.5 + matches * 0.1)
        
        return {
            'score': score,
            'trending_matches': matches,
            'explanation': "Aligned with current research trends" if score > 0.6 else "Standard research area"
        }
    
    def _calculate_composite(self, scores: Dict) -> float:
        """Calculate weighted composite score"""
        composite = (
            scores['novelty']['score'] * self.weights.novelty +
            scores['feasibility']['score'] * self.weights.feasibility +
            scores['impact']['score'] * self.weights.impact +
            scores['evidence']['score'] * self.weights.evidence +
            scores['timeliness']['score'] * self.weights.timeliness
        )
        return round(composite, 3)
    
    def _calculate_confidence(self, scores: Dict) -> float:
        """Calculate confidence in the scoring"""
        # Lower confidence if scores are uncertain
        score_values = [s['score'] for s in scores.values()]
        
        # Check for consistency
        spread = max(score_values) - min(score_values)
        
        # High spread = lower confidence
        confidence = 1.0 - (spread * 0.3)
        
        # Reduce confidence if evidence is low
        if scores['evidence']['score'] < 0.5:
            confidence -= 0.2
        
        return max(0.3, min(1.0, confidence))
    
    def _assign_grade(self, composite: float) -> str:
        """Assign letter grade based on composite score"""
        if composite >= 0.9:
            return 'A+'
        elif composite >= 0.85:
            return 'A'
        elif composite >= 0.8:
            return 'A-'
        elif composite >= 0.75:
            return 'B+'
        elif composite >= 0.7:
            return 'B'
        elif composite >= 0.65:
            return 'B-'
        elif composite >= 0.6:
            return 'C+'
        elif composite >= 0.55:
            return 'C'
        else:
            return 'C-'
    
    def _generate_recommendation(self, composite: float, confidence: float) -> str:
        """Generate recommendation based on scores"""
        if composite >= 0.8 and confidence >= 0.7:
            return "Highly recommended - pursue as priority research direction"
        elif composite >= 0.7 and confidence >= 0.6:
            return "Recommended - strong candidate for further investigation"
        elif composite >= 0.6 and confidence >= 0.5:
            return "Promising - worth exploring with further validation"
        elif composite >= 0.5:
            return "Interesting - requires more evidence before committing"
        else:
            return "Lower priority - consider other hypotheses first"
    
    def rank_hypotheses(self, hypotheses: List[Dict]) -> List[Dict]:
        """
        Score and rank multiple hypotheses
        
        Args:
            hypotheses: List of hypotheses
        
        Returns:
            Sorted list with scores and ranks
        """
        logger.info(f"Ranking {len(hypotheses)} hypotheses")
        
        # Score all
        scored = [self.score_hypothesis(h) for h in hypotheses]
        
        # Sort by composite score
        scored.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Add ranks
        for i, hyp in enumerate(scored):
            hyp['rank'] = i + 1
        
        return scored
    
    def compare_hypotheses(self, hyp1: Dict, hyp2: Dict) -> Dict:
        """
        Compare two hypotheses
        
        Args:
            hyp1: First hypothesis
            hyp2: Second hypothesis
        
        Returns:
            Comparison results
        """
        scored1 = self.score_hypothesis(hyp1.copy())
        scored2 = self.score_hypothesis(hyp2.copy())
        
        comparison = {
            'hypothesis1': {
                'id': scored1.get('id'),
                'composite': scored1['composite_score'],
                'grade': scored1['grade']
            },
            'hypothesis2': {
                'id': scored2.get('id'),
                'composite': scored2['composite_score'],
                'grade': scored2['grade']
            },
            'winner': scored1.get('id') if scored1['composite_score'] > scored2['composite_score'] else scored2.get('id'),
            'differential': abs(scored1['composite_score'] - scored2['composite_score'])
        }
        
        return comparison


# Example usage
if __name__ == "__main__":
    print("=== Hypothesis Scorer Examples ===\n")
    
    scorer = HypothesisScorer()
    
    # Test hypothesis
    test_hyp = {
        'id': 'hyp_1',
        'title': 'Apply deep learning to protein structure prediction',
        'description': 'Use transformer models for protein folding',
        'type': 'methodology_transfer',
        'novelty_score': 0.75,
        'feasibility_score': 0.7,
        'impact_potential': 0.9,
        'target_field': 'biology',
        'inspiration_papers': ['paper1', 'paper2', 'paper3', 'paper4', 'paper5']
    }
    
    scored = scorer.score_hypothesis(test_hyp)
    
    print(f"Hypothesis: {scored['title']}")
    print(f"Composite Score: {scored['composite_score']}")
    print(f"Grade: {scored['grade']}")
    print(f"Recommendation: {scored['recommendation']}")
    print(f"Confidence: {scored['confidence']:.2f}")
    
    print("\nDetailed Scores:")
    for dim, details in scored['detailed_scores'].items():
        print(f"  {dim}: {details['score']:.2f} - {details.get('explanation', '')[:50]}")
    
    print("\n✅ Scorer functional!")
