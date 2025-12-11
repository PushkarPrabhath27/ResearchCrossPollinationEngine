"""
Keyword Extractor - LLM-Based Semantic Query Analysis

The CORE fix for irrelevant search results.

Problem: Previous approach used simple stopword removal + field keywords
- "earthquake prediction" + field="physics" → added "quantum", "qiskit" (WRONG!)

Solution: Use LLM to understand query semantics and extract proper search terms
- "earthquake prediction" → ["earthquake", "seismic", "prediction", "seismology"]
- Works for ANY query, not just manually-added topics
"""

import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from src.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedKeywords:
    """Semantic keywords extracted from user query"""
    primary_topic: str  # Main subject (e.g., "earthquake prediction")
    technical_terms: List[str]  # Key terms for API search
    methods: List[str]  # Algorithms, techniques mentioned
    domains: List[str]  # Scientific domains
    paper_query: str  # Optimized query for papers
    code_query: str  # Optimized query for GitHub
    dataset_query: str  # Optimized query for datasets
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class KeywordExtractor:
    """
    Extracts semantic keywords from ANY research query using LLM.
    
    This enables the system to work for arbitrary queries without
    hardcoded topic lists.
    """
    
    # Fallback: Simple NLP extraction if LLM fails
    STOP_WORDS = {
        "i'm", "i", "am", "are", "is", "the", "a", "an", "and", "or", "for",
        "to", "from", "that", "which", "with", "could", "would", "should",
        "can", "will", "may", "might", "must", "shall", "have", "has", "had",
        "do", "does", "did", "be", "been", "being", "what", "when", "where",
        "why", "how", "who", "whom", "whose", "this", "that", "these", "those",
        "my", "your", "our", "their", "its", "his", "her", "we", "they", "you",
        "me", "us", "them", "him", "it", "some", "any", "all", "most", "many",
        "much", "few", "little", "more", "less", "no", "not", "only", "just",
        "also", "very", "really", "quite", "rather", "too", "so", "as", "if",
        "then", "than", "but", "because", "although", "though", "while",
        "about", "into", "through", "during", "before", "after", "above",
        "below", "between", "under", "again", "further", "once", "here",
        "there", "each", "every", "either", "neither", "both", "other",
        "such", "same", "different", "new", "old", "first", "last", "next",
        "want", "need", "like", "help", "work", "make", "get", "find", "use",
        "problem", "approach", "way", "thing", "something", "anything",
        "developing", "studying", "researching", "looking", "trying"
    }
    
    # Scientific terms to preserve
    PRESERVE_TERMS = {
        "ml", "ai", "nlp", "cnn", "rnn", "lstm", "transformer", "bert", "gpt",
        "svm", "pca", "dna", "rna", "mrna", "api", "gpu", "cpu", "iot", "cv"
    }
    
    def __init__(self):
        self.config = get_settings()
    
    def extract_with_llm(self, query: str) -> Optional[ExtractedKeywords]:
        """
        Use LLM to extract semantic keywords from query.
        This is the preferred method - works for ANY query.
        """
        try:
            from langchain_groq import ChatGroq
            
            if not self.config.api.groq_api_key:
                return None
            
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                groq_api_key=self.config.api.groq_api_key,
                temperature=0.1,  # Low temp for consistent extraction
                max_tokens=500
            )
            
            prompt = f'''Extract search keywords from this research query. Be specific and technical.

QUERY: "{query}"

Return ONLY valid JSON (no markdown, no explanation):
{{
    "primary_topic": "main research subject in 2-4 words",
    "technical_terms": ["5-8 specific technical terms for API search, most important first"],
    "methods": ["specific algorithms or techniques mentioned or implied"],
    "domains": ["1-3 scientific domains like seismology, machine learning, bioinformatics"],
    "paper_query": "optimized 4-6 word query for academic paper search",
    "code_query": "optimized 3-5 word query for GitHub repo search",
    "dataset_query": "optimized 3-5 word query for dataset search"
}}

IMPORTANT: 
- technical_terms should be specific (earthquake prediction, seismic waveform, NOT generic like data, analysis)
- For earthquake query → terms like: earthquake, seismic, seismology, waveform, magnitude
- For drug discovery → terms like: drug, molecule, compound, binding, pharmacology
- For image classification → terms like: image, classification, cnn, vision, detection'''

            response = llm.invoke(prompt)
            text = response.content if hasattr(response, 'content') else str(response)
            
            # Clean up response
            text = text.strip()
            if text.startswith("```"):
                text = re.sub(r'^```\w*\n?', '', text)
                text = re.sub(r'\n?```$', '', text)
            
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                text = text[start:end]
            
            data = json.loads(text)
            
            keywords = ExtractedKeywords(
                primary_topic=data.get("primary_topic", ""),
                technical_terms=data.get("technical_terms", [])[:8],
                methods=data.get("methods", [])[:5],
                domains=data.get("domains", [])[:3],
                paper_query=data.get("paper_query", ""),
                code_query=data.get("code_query", ""),
                dataset_query=data.get("dataset_query", "")
            )
            
            logger.info(f"[KeywordExtractor] LLM extracted: {keywords.technical_terms[:5]}")
            return keywords
            
        except Exception as e:
            logger.warning(f"[KeywordExtractor] LLM extraction failed: {e}")
            return None
    
    def extract_simple(self, query: str) -> ExtractedKeywords:
        """
        Simple NLP-based extraction as fallback.
        Uses improved stopword removal and term weighting.
        """
        # Tokenize and clean
        words = re.findall(r'\b[a-zA-Z]{2,}\b', query.lower())
        
        # Remove stopwords, preserve scientific abbreviations
        filtered = []
        for w in words:
            if w in self.PRESERVE_TERMS:
                filtered.append(w.upper())
            elif w not in self.STOP_WORDS and len(w) > 2:
                filtered.append(w)
        
        # Extract potential technical terms (longer words, capitalized in original)
        technical_terms = []
        for word in filtered[:15]:
            if len(word) > 4 or word.isupper():
                technical_terms.append(word)
        
        # Build search queries
        primary_terms = filtered[:4]
        
        return ExtractedKeywords(
            primary_topic=" ".join(primary_terms[:3]),
            technical_terms=technical_terms[:8],
            methods=[],
            domains=[],
            paper_query=" ".join(primary_terms[:5]),
            code_query=" ".join(primary_terms[:4]),
            dataset_query=" ".join(primary_terms[:4])
        )
    
    def extract(self, query: str, use_llm: bool = True) -> ExtractedKeywords:
        """
        Main extraction method.
        Tries LLM first, falls back to simple NLP.
        """
        if use_llm:
            result = self.extract_with_llm(query)
            if result and result.technical_terms:
                return result
        
        logger.info("[KeywordExtractor] Using simple NLP extraction")
        return self.extract_simple(query)


# Global instance
keyword_extractor = KeywordExtractor()


def extract_keywords(query: str, use_llm: bool = True) -> Dict[str, Any]:
    """
    Main function to extract semantic keywords from ANY query.
    
    Returns dict with:
    - primary_topic: Main subject
    - technical_terms: For API searches
    - paper_query, code_query, dataset_query: Optimized search strings
    """
    result = keyword_extractor.extract(query, use_llm)
    return result.to_dict()
