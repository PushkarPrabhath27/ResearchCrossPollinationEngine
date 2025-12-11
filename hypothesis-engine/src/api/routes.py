"""
API Routes for Hypothesis Engine - ENHANCED RAG Implementation
Uses LIVE FETCHING + CROSS-DOMAIN SEARCH + EXPERT DISCOVERY

Data Sources:
- Papers: OpenAlex (250M+), arXiv (2M+), Semantic Scholar (200M+)
- Cross-Domain: Searches related fields for analogous problems
- Experts: OpenAlex Authors API (50M+ researchers)
- Code: GitHub API (filtered for implementations, not lists)
- Datasets: HuggingFace, Papers With Code
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
import json
import re
import time

from src.config import get_settings
from src.utils.logger import get_logger
from src.api.paper_fetcher import fetch_real_papers, paper_fetcher, format_papers_for_llm_context, get_top_cited_papers
from src.api.github_fetcher import fetch_github_repos, github_fetcher
from src.api.dataset_fetcher import fetch_datasets, dataset_fetcher
from src.api.cross_domain_engine import search_cross_domain, generate_cross_queries
from src.api.expert_fetcher import get_experts_from_papers, get_experts_with_why_contact
from src.api.keyword_extractor import extract_keywords
from src.api.hypothesis_validator import validate_hypothesis, format_quality_score_for_display

logger = get_logger(__name__)
router = APIRouter()



# ==================== REQUEST/RESPONSE MODELS ====================

class FieldEnum(str, Enum):
    BIOLOGY = "biology"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    COMPUTER_SCIENCE = "computer_science"
    MATHEMATICS = "mathematics"
    ENGINEERING = "engineering"
    MEDICINE = "medicine"


class HypothesisRequest(BaseModel):
    query: str = Field(..., min_length=10, max_length=5000)
    field: str = Field("biology")
    num_hypotheses: int = Field(1, ge=1, le=3)
    creativity: float = Field(0.7, ge=0.0, le=1.0)


# ==================== HELPER FUNCTIONS ====================

def sanitize_json_string(text: str) -> str:
    """
    COMPREHENSIVE JSON sanitizer - fixes ALL control character issues.
    
    Root Cause: LLM outputs contain actual newlines/tabs inside JSON strings.
    JSON requires these to be escaped (\\n, \\t) or they cause parsing errors.
    
    Solution: Character-by-character processing to escape control chars inside strings.
    """
    if not text:
        return text
    
    # Step 1: Remove truly invalid control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # Step 2: Process character by character to escape control chars inside JSON strings
    def escape_control_chars_in_strings(s):
        """Escape newlines, tabs, etc. ONLY inside JSON string values"""
        result = []
        in_string = False
        escape_next = False
        
        for char in s:
            if escape_next:
                result.append(char)
                escape_next = False
                continue
            
            if char == '\\' and in_string:
                result.append(char)
                escape_next = True
                continue
            
            if char == '"':
                in_string = not in_string
                result.append(char)
                continue
            
            if in_string:
                # Inside a JSON string - escape control characters
                if char == '\n':
                    result.append('\\n')
                elif char == '\r':
                    continue  # Skip carriage returns entirely
                elif char == '\t':
                    result.append('\\t')
                elif ord(char) < 32:
                    result.append(' ')  # Replace other control chars with space
                else:
                    result.append(char)
            else:
                result.append(char)
        
        return ''.join(result)
    
    try:
        text = escape_control_chars_in_strings(text)
    except Exception:
        # Fallback: remove all newlines
        text = text.replace('\n', ' ').replace('\r', '').replace('\t', ' ')
    
    return text




def call_groq(prompt: str, temperature: float = 0.7, compact: bool = False) -> str:
    """
    Call Groq API for LLM generation.
    
    Uses llama-3.3-70b-versatile (Dec 2024) for larger prompts
    Falls back through available models if one is decommissioned.
    """
    config = get_settings()
    from langchain_groq import ChatGroq
    
    # Model fallback list (ordered by preference, from Groq docs Dec 2024)
    # VERIFIED MODELS: https://console.groq.com/docs/rate-limits
    if compact:
        models_to_try = [
            "llama-3.1-8b-instant",                        # 6K TPM, fast
            "qwen/qwen3-32b",                              # 6K TPM, backup
        ]
        max_tokens = 4000
    else:
        models_to_try = [
            "meta-llama/llama-4-scout-17b-16e-instruct",   # 30K TPM - BEST!
            "llama-3.3-70b-versatile",                     # 12K TPM
            "moonshotai/kimi-k2-instruct",                 # 10K TPM
            "qwen/qwen3-32b",                              # 6K TPM fallback
            "llama-3.1-8b-instant",                        # 6K TPM last resort
        ]
        max_tokens = 8000
    
    last_error = None
    for model in models_to_try:
        try:
            logger.info(f"[LLM] Trying Groq model: {model}")
            llm = ChatGroq(
                model=model,
                groq_api_key=config.api.groq_api_key,
                temperature=temperature,
                max_tokens=max_tokens
            )
            response = llm.invoke(prompt)
            logger.info(f"[LLM] Success with Groq model: {model}")
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            error_msg = str(e)
            last_error = e
            logger.warning(f"[LLM] Groq {model} failed: {error_msg[:80]}")
            # Continue to next model
            continue
    
    # All models failed
    raise last_error or ValueError("All Groq models failed")


def call_gemini(prompt: str, temperature: float = 0.7) -> str:
    """
    Call Google Gemini API for LLM generation.
    
    Tries multiple Gemini model names to handle API changes.
    """
    config = get_settings()
    
    if not config.api.google_api_key:
        raise ValueError("GOOGLE_API_KEY not configured")
    
    import google.generativeai as genai
    
    genai.configure(api_key=config.api.google_api_key)
    
    # Model fallback list (ordered by preference)
    # Some names work with v1, some with v1beta
    models_to_try = [
        "gemini-pro",                   # Stable, always available
        "gemini-1.5-pro",               # 1M context if available
        "gemini-1.5-flash",             # Fast version
        "gemini-1.0-pro",               # Fallback
    ]
    
    last_error = None
    for model_name in models_to_try:
        try:
            logger.info(f"[LLM] Trying Gemini model: {model_name}")
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": 8192,
                }
            )
            response = model.generate_content(prompt)
            logger.info(f"[LLM] Success with Gemini model: {model_name}")
            return response.text
        except Exception as e:
            error_msg = str(e)
            last_error = e
            logger.warning(f"[LLM] Gemini {model_name} failed: {error_msg[:80]}")
            continue
    
    # All models failed
    raise last_error or ValueError("All Gemini models failed")


def call_llm_with_fallback(prompt: str, temperature: float = 0.7) -> str:
    """
    Smart LLM caller with fallback strategy.
    
    Strategy:
    1. Try Gemini first (1M token context - handles any prompt)
    2. If Gemini fails, fall back to Groq 70b (128k context)
    3. If 70b rate limited, use Groq 8b with compact prompt
    
    This ensures we can handle large prompts without token errors.
    """
    config = get_settings()
    errors = []
    
    # Strategy 1: Try Gemini (best - 1M token context)
    if config.api.google_api_key:
        try:
            logger.info("[LLM] Attempting Gemini (primary - 1M tokens)...")
            return call_gemini(prompt, temperature)
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"[LLM] Gemini failed: {error_msg[:100]}")
            errors.append(f"Gemini: {error_msg[:100]}")
    
    # Strategy 2: Try Groq (128k context with 70b, fallback to 8b)
    if config.api.groq_api_key:
        try:
            logger.info("[LLM] Attempting Groq 70b (fallback - 128k tokens)...")
            return call_groq(prompt, temperature, compact=False)
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"[LLM] Groq failed: {error_msg[:100]}")
            errors.append(f"Groq: {error_msg[:100]}")
    
    # No providers worked
    raise ValueError(f"All LLM providers failed: {'; '.join(errors)}")


def generate_hypothesis_prompt(query: str, field: str, real_papers: List[Dict], 
                               real_datasets: List[Dict], real_repos: List[Dict]) -> str:
    """
    Generate prompt that includes REAL data from APIs.
    The LLM builds hypothesis BASED ON real papers, not fabricated ones.
    """
    
    # Format real papers for context
    papers_context = ""
    for i, p in enumerate(real_papers[:8], 1):
        papers_context += f"""
Paper {i}: "{p.get('title', 'Untitled')}"
- Authors: {p.get('authors', 'Unknown')}
- Year: {p.get('year', 'N/A')}, Journal: {p.get('journal', 'N/A')}
- DOI: {p.get('doi', 'N/A')}
- Citations: {p.get('citation_count', 0)}
- Abstract: {p.get('abstract', '')[:300]}...
"""
    
    # Format real datasets for context
    datasets_context = ""
    for i, d in enumerate(real_datasets[:4], 1):
        datasets_context += f"""
Dataset {i}: {d.get('name', 'Unknown')}
- Source: {d.get('source', 'N/A')}
- URL: {d.get('url', 'N/A')}
- Size: {d.get('size', 'N/A')}
"""
    
    # Format real repos for context
    repos_context = ""
    for i, r in enumerate(real_repos[:3], 1):
        repos_context += f"""
Repo {i}: {r.get('name', 'Unknown')} ({r.get('stars', '0')} stars)
- URL: {r.get('url', 'N/A')}
- Description: {r.get('description', 'N/A')[:100]}
"""
    
    return f'''You are an expert research scientist generating a detailed research hypothesis.

## RESEARCH QUERY
"{query}"

## PRIMARY FIELD: {field}

## REAL RESEARCH PAPERS (from OpenAlex, arXiv, Semantic Scholar - CITE THESE)
{papers_context}

## REAL DATASETS AVAILABLE
{datasets_context}

## REAL CODE REPOSITORIES AVAILABLE
{repos_context}

## YOUR TASK
Generate ONE comprehensive research hypothesis that:
1. CITES the real papers above by name (e.g., "Building on Zhang et al.'s work in Paper 1...")
2. REFERENCES the real datasets above by name
3. LINKS to the real code repositories above
4. Provides SPECIFIC methodology with exact algorithms and parameters

Return a JSON object with this structure:
{{
    "title": "Specific hypothesis title",
    "description": "Detailed 3-4 paragraph description that CITES the real papers by name",
    
    "novelty_score": 8.5,
    "feasibility_score": 7.0,
    "impact_score": 8.0,
    
    "theoretical_basis": "Explain the science, citing specific papers from the list above",
    
    "methodology_steps": [
        {{
            "step_number": 1,
            "title": "Step title",
            "algorithm": "Exact algorithm name (e.g., ResNet-50, LSTM, SVM)",
            "parameters": "Exact parameters (e.g., learning_rate=0.001, epochs=100)",
            "libraries": ["numpy", "torch", "etc"],
            "code_snippet": "import torch; model = ResNet50(pretrained=True)",
            "estimated_time": "2-3 hours",
            "source_paper": "Name the paper from the list that describes this method"
        }}
    ],
    
    "why_novel": "What makes this combination of existing methods novel",
    "expected_results": "Specific expected outcomes with numbers (e.g., 85% accuracy)",
    "risks_and_mitigation": [
        {{"risk": "Description", "mitigation": "How to address it"}}
    ],
    "timeline_weeks": [
        {{"week": "1-2", "activities": ["Activity 1", "Activity 2"], "deliverable": "What will be done"}}
    ],
    "estimated_budget": "$5,000-$20,000",
    "cross_domain_insights": ["Insight connecting different fields"],
    "next_steps": ["Concrete action 1", "Concrete action 2"]
}}

CRITICAL RULES:
1. ONLY cite papers from the list above - don't invent papers
2. Use SPECIFIC algorithm names and parameters
3. Include actual code snippets that would work
4. Be scientifically accurate and realistic

Return ONLY the JSON object.'''


def generate_with_llm(query: str, field: str, num: int, creativity: float) -> Dict:
    """
    Generate hypothesis with REAL data from live APIs.
    
    KEY FIX: Uses LLM keyword extraction for query-relevant results.
    
    Flow:
    1. EXTRACT semantic keywords from query (LLM-based)
    2. Fetch REAL papers using extracted keywords
    3. Fetch REAL GitHub repos using code_query
    4. Fetch REAL datasets using dataset_query
    5. Cross-domain search
    6. Expert discovery
    7. Generate hypothesis
    """
    logger.info(f"[GENERATE] Starting ENHANCED generation for: {query[:50]}...")
    start_time = time.time()
    
    config = get_settings()
    
    # STEP 0: EXTRACT SEMANTIC KEYWORDS (THE KEY FIX!)
    # This makes search work for ANY query, not just hardcoded topics
    logger.info("[KEYWORDS] Extracting semantic keywords from query...")
    try:
        extracted = extract_keywords(query, use_llm=True)
        search_terms = extracted.get("technical_terms", [])
        paper_query = extracted.get("paper_query", query)
        code_query = extracted.get("code_query", query)
        dataset_query = extracted.get("dataset_query", query)
        logger.info(f"[KEYWORDS] Extracted: {search_terms[:5]}")
        logger.info(f"[KEYWORDS] Paper query: {paper_query}")
        logger.info(f"[KEYWORDS] Code query: {code_query}")
        logger.info(f"[KEYWORDS] Dataset query: {dataset_query}")
    except Exception as e:
        logger.warning(f"[KEYWORDS] Extraction failed, using original query: {e}")
        search_terms = []
        paper_query = query
        code_query = query
        dataset_query = query
    
    # STEP 1: Fetch REAL papers using EXTRACTED keywords
    logger.info(f"[FETCH] Getting real papers with query: {paper_query[:60]}...")
    try:
        # Use the optimized paper_query from keyword extraction
        paper_result = fetch_real_papers(paper_query, field=field, max_results=30)
        real_papers = paper_fetcher.format_papers_for_display(paper_result["papers"])
        paper_stats = paper_result["stats"]
        logger.info(f"[FETCH] Got {len(real_papers)} real papers")
    except Exception as e:
        logger.error(f"[FETCH] Paper fetch failed: {e}")
        real_papers = []
        paper_stats = {"error": str(e)}
    
    # STEP 2: Fetch REAL GitHub repos using EXTRACTED code_query
    logger.info(f"[FETCH] Getting GitHub repos with query: {code_query[:60]}...")
    try:
        # Use the optimized code_query from keyword extraction
        real_repos = fetch_github_repos(code_query, field=field, max_results=5)
        logger.info(f"[FETCH] Got {len(real_repos)} real repos")
    except Exception as e:
        logger.error(f"[FETCH] GitHub fetch failed: {e}")
        real_repos = []
    
    # STEP 3: Fetch REAL datasets using EXTRACTED dataset_query
    logger.info(f"[FETCH] Getting datasets with query: {dataset_query[:60]}...")
    try:
        # Use the optimized dataset_query from keyword extraction
        real_datasets = fetch_datasets(dataset_query, field, max_results=6)
        logger.info(f"[FETCH] Got {len(real_datasets)} real datasets")
    except Exception as e:
        logger.error(f"[FETCH] Dataset fetch failed: {e}")
        real_datasets = []
    
    # STEP 4: Cross-domain search
    logger.info(f"[CROSS-DOMAIN] Searching related fields...")
    cross_domain_results = {"cross_domain_results": [], "connections": []}
    try:
        cross_domain_results = search_cross_domain(query, field)
        logger.info(f"[CROSS-DOMAIN] Found papers across {len(cross_domain_results.get('cross_domain_results', []))} fields")
    except Exception as e:
        logger.error(f"[CROSS-DOMAIN] Search failed: {e}")
    
    # STEP 5: Fetch REAL EXPERTS from paper authors
    logger.info("[FETCH] Getting real experts from paper authors...")
    real_experts = []
    try:
        if paper_result.get("papers"):
            papers_for_experts = [
                {"first_author_id": getattr(p, 'first_author_id', None) or p.to_dict().get('first_author_id')}
                for p in paper_result["papers"][:10]
                if hasattr(p, 'to_dict')
            ]
            if not papers_for_experts:
                papers_for_experts = real_papers[:10]
            real_experts = get_experts_from_papers(papers_for_experts, max_experts=5)
        logger.info(f"[FETCH] Got {len(real_experts)} real experts")
    except Exception as e:
        logger.error(f"[FETCH] Expert fetch failed: {e}")
    
    fetch_time = time.time() - start_time
    logger.info(f"[FETCH] Total fetch time: {fetch_time:.2f}s")
    
    # STEP 6: Generate hypothesis with LLM using ALL real data as context
    if not config.api.groq_api_key and not config.api.google_api_key:
        return {
            "success": False,
            "error": "No LLM API key configured (need GROQ_API_KEY or GOOGLE_API_KEY)",
            "query": query,
            "field": field
        }
    
    # Build ENHANCED prompt with cross-domain data
    prompt = generate_enhanced_prompt(
        query, field, real_papers, real_datasets, real_repos,
        cross_domain_results, real_experts
    )
    
    try:
        logger.info("[LLM] Generating ENHANCED hypothesis with fallback...")
        llm_start = time.time()
        text = call_llm_with_fallback(prompt, creativity)
        llm_time = time.time() - llm_start
        logger.info(f"[LLM] Generated in {llm_time:.2f}s, length: {len(text)}")
        
        # Parse JSON response
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
        
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]
        
        text = sanitize_json_string(text)
        hypothesis = json.loads(text.strip())
        
        # STEP 7: VALIDATE hypothesis quality and catch hallucinations
        logger.info("[VALIDATE] Checking hypothesis quality and citations...")
        try:
            validation_result = validate_hypothesis(hypothesis, real_papers)
            quality_score = format_quality_score_for_display(validation_result)
            logger.info(f"[VALIDATE] Quality score: {validation_result.overall}/10")
            if validation_result.fabricated_citations:
                logger.warning(f"[VALIDATE] Found {len(validation_result.fabricated_citations)} fabricated citations: {validation_result.fabricated_citations[:3]}")
        except Exception as e:
            logger.warning(f"[VALIDATE] Validation failed: {e}")
            quality_score = {
                "overall": {"score": 0, "max": 10, "status": "⚠️"},
                "is_valid": False,
                "issues": [f"Validation error: {str(e)}"]
            }
        
        # STEP 8: Construct COMPREHENSIVE response with ALL real data
        total_time = time.time() - start_time
        
        return {
            "success": True,
            "query": query,
            "field": field,
            
            # Real search statistics
            "search_stats": {
                "query": query,
                "total_fetch_time_seconds": round(fetch_time, 2),
                "llm_time_seconds": round(llm_time, 2),
                "total_time_seconds": round(total_time, 2),
                "sources_searched": paper_stats.get("sources_searched", []),
                "total_papers_found": paper_stats.get("total_papers_found", 0),
                "papers_by_source": paper_stats.get("papers_by_source", {}),
                "datasets_found": len(real_datasets),
                "repos_found": len(real_repos),
                "experts_found": len(real_experts),
                "cross_domain_fields_searched": len(cross_domain_results.get("cross_domain_results", []))
            },
            
            # The generated hypothesis
            "hypothesis": hypothesis,
            
            # NEW: Quality validation score
            "quality_score": quality_score,
            
            # REAL papers with working URLs
            "real_papers": real_papers,
            
            # CROSS-DOMAIN papers and connections
            "cross_domain": cross_domain_results,
            
            # REAL experts with profiles
            "real_experts": real_experts,
            
            # REAL datasets with working URLs  
            "real_datasets": real_datasets,
            
            # REAL code repos with working URLs
            "real_repos": real_repos,
            
            "error": None
        }
        
    except Exception as e:
        logger.error(f"[ERROR] Generation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "field": field,
            "search_stats": paper_stats,
            "real_papers": real_papers,
            "cross_domain": cross_domain_results,
            "real_experts": real_experts,
            "real_datasets": real_datasets,
            "real_repos": real_repos
        }


def generate_enhanced_prompt(query: str, field: str, real_papers: List[Dict],
                              real_datasets: List[Dict], real_repos: List[Dict],
                              cross_domain: Dict, real_experts: List[Dict]) -> str:
    """
    COMPLETE REWRITE v2 based on updated updatesprompt.md requirements.
    
    Enforces 6 CRITICAL RULES:
    1. ZERO-TOLERANCE FABRICATION: Only cite papers in retrieved list
    2. QUANTITATIVE SPECIFICITY: Every claim needs concrete numbers
    3. MECHANISM EXPLANATION: Explain HOW, not just WHAT
    4. CROSS-DOMAIN AUTHENTICITY: Genuine technique transfer with both domain citations
    5. USE HIGH-IMPACT PAPERS: Top 3 cited papers MUST be used
    6. REALISTIC METHODOLOGY: Code snippets and literature-justified parameters
    
    Also includes 6 NEW SECTIONS (A-F):
    A. Why This Hasn't Been Done Before
    B. Alternative Approaches Rejected
    C. Preliminary Data / Proof of Concept
    D. Broader Impact
    E. Funding Opportunities
    F. IP Landscape
    """
    
    # Sort papers by citations (highest first)
    sorted_papers = sorted(real_papers, key=lambda p: p.get('citation_count', 0), reverse=True)
    
    # Build the paper context with [MUST USE] markers for top 3
    papers_context = ""
    for i, p in enumerate(sorted_papers[:15], 1):
        if i <= 3:
            marker = "🔴 [MUST USE - TOP CITED] "
        elif i <= 5:
            marker = "[TOP CITED] "
        else:
            marker = ""
        
        papers_context += f"""
{marker}Paper {i}:
- Title: "{p.get('title', 'Untitled')}"
- Authors: {p.get('authors', 'Unknown')}
- Year: {p.get('year', 'N/A')}, Journal: {p.get('journal', 'N/A')}
- DOI: {p.get('doi', 'N/A')}
- Citations: {p.get('citation_count', 0)}
- Abstract: {p.get('abstract', 'No abstract available')[:600]}
"""

    # Format CROSS-DOMAIN papers
    cross_domain_context = ""
    for cr in cross_domain.get("cross_domain_results", [])[:3]:
        cross_domain_context += f"\n### Cross-Domain Field: {cr.get('field', 'Unknown')}\n"
        cross_domain_context += f"Connection Reasoning: {cr.get('reasoning', 'N/A')}\n"
        for j, cp in enumerate(cr.get("papers", [])[:3], 1):
            cross_domain_context += f"""
Cross-Domain Paper {j}:
- Title: "{cp.get('title', 'Untitled')}"
- Authors: {cp.get('authors', 'Unknown')}
- Year: {cp.get('year', 'N/A')}, Citations: {cp.get('citation_count', 0)}
- DOI: {cp.get('doi', 'N/A')}
- Abstract: {cp.get('abstract', '')[:400]}
"""

    # Format experts with full details
    experts_context = ""
    for e in real_experts[:5]:
        recent_works = e.get('recent_works', [])[:3]
        works_str = "; ".join(recent_works) if recent_works else "N/A"
        experts_context += f"""
Expert: {e.get('name', 'Unknown')}
- Institution: {e.get('affiliation', 'Unknown')}
- Total Citations: {e.get('citations', 'N/A')}, h-index: {e.get('h_index', 'N/A')}
- Research Topics: {', '.join(e.get('research_topics', [])[:4])}
- Recent Works: {works_str}
- ORCID: {e.get('orcid', 'N/A')}
"""

    # Format datasets
    datasets_context = ""
    for d in real_datasets[:5]:
        datasets_context += f"""
Dataset: {d.get('name', 'Unknown')}
- Source: {d.get('source', 'N/A')}
- URL: {d.get('url', 'N/A')}
- Size: {d.get('size', 'N/A')}, Format: {d.get('format', 'N/A')}
- Description: {d.get('description', 'N/A')[:250]}
"""

    # Format repos
    repos_context = ""
    for r in real_repos[:5]:
        repos_context += f"""
Repository: {r.get('name', 'Unknown')} ⭐ {r.get('stars', 0)} stars
- URL: {r.get('url', 'N/A')}
- Language: {r.get('language', 'N/A')}
- Description: {str(r.get('description', 'N/A'))[:200]}
"""

    return f'''You are an elite research scientist with expertise across multiple domains. Generate a RIGOROUS research hypothesis based ONLY on the retrieved papers below.

═══════════════════════════════════════════════════════════════════
🚨 6 CRITICAL RULES (VIOLATION = AUTOMATIC REJECTION)
═══════════════════════════════════════════════════════════════════

RULE 1: ZERO-TOLERANCE FABRICATION POLICY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• You may ONLY cite papers in <RETRIEVED_PAPERS> and <CROSS_DOMAIN_PAPERS>
• Check each citation against the list before including
• If a paper is not retrieved, you CANNOT mention it - NO EXCEPTIONS
• Include full metadata: Author name, Year, Journal, DOI, Citation count
• ❌ NEVER: "Johnson et al. (2020) showed..."
• ✅ ALWAYS: "Johnson, Smith, Lee et al. (2020) 'Title' [Journal, DOI, 1,234 citations] showed..."

RULE 2: QUANTITATIVE SPECIFICITY REQUIREMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• EVERY claim must include concrete numbers
• BANNED WORDS (never use): significant, substantial, considerable, notable,
  high, low, large, small, many, few, some, several, better, worse, improved
• ❌ NEVER: "significant improvement in accuracy"
• ✅ ALWAYS: "accuracy improved from 60% to 85% (42% relative improvement)"
• Required numbers per section:
  - Problem statement: 3+ quantitative claims
  - Methodology: 5+ specific parameters with values
  - Comparison table: ALL cells must contain numbers
  - Risk assessment: Exact probabilities (e.g., "65% based on...")

RULE 3: MECHANISM EXPLANATION REQUIREMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• NEVER just state WHAT, always explain HOW and WHY
• Include molecular/physical/computational mechanisms
• Connect cause → effect with intermediate steps
• ❌ NEVER: "Use PINNs to improve tracking"
• ✅ ALWAYS: "PINNs enforce conservation of mass (∂ρ/∂t + ∇·v = 0) as a soft 
   constraint during training. This reduces overfitting when training data is 
   sparse (<100 trajectories) because the physics constraint acts as regularization.
   Expected: 15% improvement in generalization error vs unconstrained networks."

RULE 4: CROSS-DOMAIN AUTHENTICITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Cross-domain connections must be GENUINELY non-obvious
• Must cite papers from BOTH source AND target domains
• Must explain specific technique transfer mechanism
• Required structure for each connection:
  1. Source domain + specific technique + paper citation
  2. Target domain + specific problem + paper citation  
  3. HOW to adapt technique (3+ concrete steps)
  4. WHY this connection is non-obvious
  5. Expected quantitative improvement

RULE 5: USE HIGH-IMPACT RETRIEVED PAPERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• The top 3 cited papers are marked with 🔴 [MUST USE]
• These papers MUST appear in your hypothesis
• Extract specific findings from their abstracts
• If you cannot use a top-3 paper, explicitly justify why

RULE 6: REALISTIC METHODOLOGY DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each methodology step, you MUST include:
• Algorithm name + version number
• Parameters with literature justification (not default values)
• Input/output formats with sizes
• Expected compute time + cost
• Success criteria with thresholds
• Working code snippet (5-10 lines)
• Week-by-week time breakdown

═══════════════════════════════════════════════════════════════════
📥 INPUT DATA
═══════════════════════════════════════════════════════════════════

<USER_QUERY>
{query}
</USER_QUERY>

<PRIMARY_FIELD>
{field}
</PRIMARY_FIELD>

<RETRIEVED_PAPERS>
{papers_context}
</RETRIEVED_PAPERS>

<CROSS_DOMAIN_PAPERS>
{cross_domain_context}
</CROSS_DOMAIN_PAPERS>

<EXPERTS>
{experts_context}
</EXPERTS>

<DATASETS>
{datasets_context}
</DATASETS>

<CODE_REPOS>
{repos_context}
</CODE_REPOS>

═══════════════════════════════════════════════════════════════════
📤 REQUIRED OUTPUT FORMAT (JSON)
═══════════════════════════════════════════════════════════════════

Return ONLY valid JSON with this EXACT structure:

{{
  "executive_summary_new": {{
    "one_sentence": "Problem + Solution + Impact in <50 words with numbers",
    "target_audience": "Who should care (e.g., cancer researchers, ML engineers)",
    "key_innovation": "What's novel in <30 words"
  }},

  "hypothesis_title": "Descriptive title with key innovation",
  
  "problem_context": {{
    "current_sota": {{
      "method": "Best current method name",
      "performance": "Metric with numbers (e.g., '90% in 10 hours')",
      "source": "Author (Year) - Paper Title - DOI [from retrieved papers]",
      "limitation": "Specific weakness with numbers"
    }},
    "failed_attempts": [
      {{
        "approach": "What was tried",
        "researchers": "Who tried it (from papers)",
        "result": "What happened with numbers",
        "why_failed": "Root cause",
        "source": "Full paper citation from list"
      }}
    ],
    "unmet_need": "Specific gap with quantified impact"
  }},

  "hypothesis": {{
    "main_claim": "Clear 2-3 sentence statement",
    "theoretical_basis": "Mechanism explanation citing 3-5 papers WITH NUMBERS",
    "novelty": "What has NOT been done + why new",
    "expected_improvement": "Quantitative prediction (e.g., '10x faster')"
  }},

  "novelty_analysis": {{
    "what_has_not_been_done": "Specific combination/approach that's new",
    "why_not_done_before": "Barrier that prevented it",
    "why_possible_now": "What changed recently to make this possible",
    "literature_search": {{
      "query_used": "Search terms we would use",
      "papers_found": 0,
      "closest_work": "Most similar paper and how ours differs"
    }}
  }},

  "cross_domain_connections": [
    {{
      "source_domain": "Field A",
      "source_technique": "Specific method name",
      "source_paper": "Full citation from list with DOI",
      "source_finding": "What they found WITH NUMBERS",
      "target_domain": "Field B",
      "target_problem": "Problem being solved",
      "transfer_mechanism": "Specific steps (minimum 100 words) to adapt technique",
      "why_nonobvious": "Why experts haven't connected these (cultural/technical barriers)"
    }}
  ],

  "methodology": [
    {{
      "step_number": 1,
      "step_name": "Action description",
      "algorithm": "Exact name and version",
      "parameters": {{
        "param1": "value (justification: Author Year found X)",
        "param2": "value (justification: standard for Y)"
      }},
      "source_papers": ["Full citation 1", "Full citation 2"],
      "input_spec": "Format, size, source",
      "output_spec": "Expected values with numbers",
      "success_criteria": "Threshold (e.g., error < 0.1)",
      "time_estimate": "X weeks: Week 1: task, Week 2: task",
      "resources_needed": "Compute/equipment with costs ($)",
      "code_snippet": "import torch\\nmodel = Model()\\n..."
    }}
  ],

  "comparison_table": {{
    "methods": [
      {{
        "name": "Method 1 from literature",
        "performance": "Number with unit (from paper X)",
        "cost": "$X/unit (source)",
        "advantages": ["Pro 1 with number", "Pro 2"],
        "limitations": ["Con 1 with number", "Con 2"],
        "source": "Full paper citation"
      }},
      {{
        "name": "Our proposed method",
        "performance": "Target number with unit",
        "cost": "Estimated $/unit with breakdown",
        "advantages": ["Pro 1 with number", "Pro 2"],
        "limitations": ["Con 1 with mitigation"]
      }}
    ],
    "when_to_use": {{
      "use_method_1_when": "Condition with specifics",
      "use_our_method_when": "Condition with specifics"
    }},
    "recommendation": "Use X when Y, use ours when Z"
  }},

  "enhanced_experts": [
    {{
      "name": "Dr. Name from experts list",
      "institution": "University/Institute",
      "email": "estimated email (e.g., lastname@institution.edu)",
      "expertise_summary": "What they're expert in",
      "relevant_papers": ["Their paper titles from list"],
      "contributions": [
        {{
          "contribution_type": "Data Sharing / Validation / Consultation",
          "description": "What exactly they could contribute",
          "value_to_project": "Saves X weeks / provides Y data"
        }}
      ],
      "collaboration_likelihood": {{
        "likelihood": "VERY HIGH/HIGH/MEDIUM/LOW (X%)",
        "evidence_for": ["Reason 1", "Reason 2"],
        "evidence_against": ["Concern 1"]
      }},
      "email_template": {{
        "subject": "Collaboration: [Title]",
        "body": "Dear Dr. X, I am working on... Would you be open to... In exchange..."
      }},
      "priority": "HIGHEST/SECONDARY/FOUNDATIONAL"
    }}
  ],

  "risk_assessment": [
    {{
      "risk": "What could go wrong",
      "probability": "X% (based on Author Year who found Y)",
      "impact": "HIGH/MEDIUM/LOW",
      "evidence": "Citation showing others hit this",
      "mitigation": "How to reduce risk",
      "contingency": "Backup plan if it happens"
    }}
  ],

  "validation_metrics": {{
    "primary_metrics": [
      {{
        "metric": "What to measure",
        "current_sota": "Best value (Author Year)",
        "your_target": "Your goal",
        "measurement_method": "How to measure (citation)",
        "success_threshold": "Minimum for publication"
      }}
    ]
  }},

  "why_not_done_before": {{
    "temporal_barrier": "When biological/domain knowledge became available (Year: Event)",
    "technical_barrier": "When tools/methods became available (Year: Tool)",
    "cultural_barrier": "Why fields don't talk (conf A vs conf B)",
    "data_barrier": "When sufficient data became available",
    "opportunity_now": "Why timing is perfect NOW"
  }},

  "alternatives_rejected": {{
    "alternatives": [
      {{
        "option_name": "Alternative approach",
        "what_is_it": "Brief description",
        "pros": ["Advantage 1", "Advantage 2"],
        "cons": ["Disadvantage 1 with numbers", "Disadvantage 2"],
        "why_rejected": "Specific reason we chose not to use this"
      }}
    ],
    "why_our_approach_best": "Summary comparing to all alternatives"
  }},

  "preliminary_data": {{
    "pilot_studies": [
      {{
        "study_name": "Synthetic Data Validation",
        "date": "Current",
        "method": "What we would test first",
        "results": ["Expected result 1 with numbers"],
        "conclusion": "What we would learn",
        "next_steps": "What to do after"
      }}
    ],
    "overall_readiness": "How ready to proceed"
  }},

  "broader_impact": {{
    "clinical_impact": {{
      "area": "Clinical",
      "description": "How this affects patients",
      "quantitative_benefit": "X lives saved or $Y reduced"
    }},
    "economic_impact": {{
      "area": "Economic", 
      "description": "Cost implications",
      "quantitative_benefit": "$X/year savings or X% cost reduction"
    }},
    "scientific_impact": {{
      "area": "Scientific",
      "description": "How this advances science",
      "quantitative_benefit": "Enables X new experiments"
    }},
    "sdg_alignment": ["SDG 3: Good Health", "SDG 9: Innovation"]
  }},

  "funding_plan": {{
    "opportunities": [
      {{
        "agency": "NIH/NSF/CZI/etc",
        "program": "Specific program name",
        "amount": "$X over Y years",
        "duration": "Y years",
        "deadline": "Month Year",
        "fit_score": 8,
        "why_good_fit": ["Reason 1", "Reason 2"],
        "success_rate": "X%",
        "application_strategy": "How to approach",
        "recent_examples": ["Similar funded project"]
      }}
    ],
    "application_timeline": "Month 1: X, Month 2: Y",
    "total_potential": "$X-Y over Z years",
    "recommended_first": "Apply to X first because Y"
  }},

  "ip_landscape": {{
    "search_terms": ["term1", "term2"],
    "patents_found": 0,
    "relevant_patents": [],
    "recommendation": "Open Science / File Provisional / Full Patent",
    "estimated_cost": "$X for recommended approach"
  }},

  "relevant_datasets": [
    {{
      "name": "Dataset from list",
      "source": "Provider",
      "url": "URL",
      "relevance": "HOW it helps this research",
      "specific_use": "What you'll do with it"
    }}
  ],

  "relevant_code": [
    {{
      "repo_name": "From list",
      "url": "URL",
      "relevance": "HOW it helps",
      "specific_use": "What to adapt"
    }}
  ],

  "quality_checks": {{
    "all_citations_verified": true,
    "top3_papers_used": true,
    "vague_words_found": [],
    "numbers_count": 25,
    "cross_domain_has_both_citations": true,
    "methodology_has_code": true,
    "experts_have_email": true,
    "overall_compliance": 100.0
  }}
}}

═══════════════════════════════════════════════════════════════════
✅ QUALITY CHECKS (VERIFY BEFORE SUBMITTING)
═══════════════════════════════════════════════════════════════════

Before outputting, verify ALL of these:
✓ EVERY paper cited exists in RETRIEVED_PAPERS or CROSS_DOMAIN_PAPERS (check each one!)
✓ Top 3 cited papers (marked 🔴 [MUST USE]) appear in your hypothesis
✓ At least 20 specific numbers included across all sections
✓ ZERO banned vague words (significant, substantial, high, low, many, few, etc.)
✓ Cross-domain connections cite papers from BOTH domains
✓ Each methodology step has code_snippet with actual code
✓ Comparison table has numbers in EVERY performance/cost cell
✓ Risks include exact probability % with paper evidence
✓ Experts have estimated email addresses
✓ All 6 new sections (why_not_done_before, alternatives_rejected, etc.) are filled

If ANY check fails, fix it before outputting.

Return ONLY the JSON object, no markdown formatting, no explanation.'''




# ==================== API ENDPOINTS ====================

@router.post("/generate", tags=["Hypotheses"])
async def generate_hypotheses(request: HypothesisRequest):
    """
    Generate research hypothesis with REAL data from live APIs.
    
    This endpoint:
    1. Searches OpenAlex, arXiv, Semantic Scholar for real papers
    2. Searches HuggingFace, Papers With Code for real datasets
    3. Searches GitHub for real code repositories
    4. Generates hypothesis using LLM with real data as context
    5. Returns hypothesis with all real, working links
    """
    logger.info(f"[API] Request: {request.query[:50]}...")
    
    try:
        result = generate_with_llm(
            request.query,
            request.field,
            request.num_hypotheses,
            request.creativity
        )
        return result
    except Exception as e:
        logger.error(f"[API] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fields", tags=["Metadata"])
async def get_fields() -> List[str]:
    """Get available research fields"""
    return [f.value for f in FieldEnum]


@router.get("/stats", tags=["Metadata"])
async def get_stats():
    """Get system statistics"""
    return {
        "version": "2.0-production",
        "features": [
            "Real papers from OpenAlex (250M+)",
            "Real papers from arXiv (2M+)",
            "Real papers from Semantic Scholar (200M+)",
            "Real datasets from HuggingFace",
            "Real datasets from Papers With Code",
            "Real code from GitHub"
        ],
        "storage": "NONE - All live fetching",
        "status": "operational"
    }


@router.get("/search-papers", tags=["Search"])
async def search_papers(query: str, max_results: int = 20):
    """
    Search for papers directly without hypothesis generation.
    Useful for exploring literature.
    """
    try:
        result = fetch_real_papers(query, max_results)
        papers = paper_fetcher.format_papers_for_display(result["papers"])
        return {
            "success": True,
            "query": query,
            "stats": result["stats"],
            "papers": papers
        }
    except Exception as e:
        logger.error(f"[SEARCH] Error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/search-datasets", tags=["Search"])
async def search_datasets(query: str, field: str = "", max_results: int = 10):
    """Search for datasets directly"""
    try:
        datasets = fetch_datasets(query, field, max_results)
        return {
            "success": True,
            "query": query,
            "datasets": datasets
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/search-repos", tags=["Search"])
async def search_repos(query: str, max_results: int = 10):
    """Search for GitHub repositories directly"""
    try:
        repos = fetch_github_repos(query, max_results)
        return {
            "success": True,
            "query": query,
            "repos": repos
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
