"""
Hypothesis Module

Core hypothesis generation, validation, and scoring functionality.
"""

from src.hypothesis.generator import HypothesisGenerator
from src.hypothesis.validator import HypothesisValidator
from src.hypothesis.scorer import HypothesisScorer, ScoreWeight, ImpactLevel
from src.hypothesis.grant_assistant import GrantWritingAssistant, GrantSection

__all__ = [
    'HypothesisGenerator',
    'HypothesisValidator',
    'HypothesisScorer',
    'ScoreWeight',
    'ImpactLevel',
    'GrantWritingAssistant',
    'GrantSection'
]
