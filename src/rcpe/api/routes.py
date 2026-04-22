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

from src.rcpe.config import get_settings
from src.rcpe.utils.logger import get_logger
from src.rcpe.api.paper_fetcher import fetch_real_papers, paper_fetcher, format_papers_for_llm_context, get_top_cited_papers
from src.rcpe.api.github_fetcher import fetch_github_repos, github_fetcher
from src.rcpe.api.dataset_fetcher import fetch_datasets, dataset_fetcher
from src.rcpe.api.cross_domain_engine import search_cross_domain, generate_cross_queries
from src.rcpe.api.expert_fetcher import get_experts_from_papers, get_experts_with_why_contact
from src.rcpe.api.keyword_extractor import extract_keywords
from src.rcpe.api.hypothesis_validator import validate_hypothesis, format_quality_score_for_display

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
    COMPREHENSIVE JSON sanitizer - fixes ALL common LLM JSON issues.
    
    Issues fixed:
    1. Trailing commas before ] or } (multiple patterns)
    2. Unescaped control characters inside strings
    3. Invalid control characters
    4. Single quotes instead of double quotes (outside strings)
    """
    if not text:
        return text
    
    # Step 1: Remove truly invalid control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # Step 2: Remove trailing commas - MULTIPLE PASSES for nested structures
    # Apply 5 times to handle deeply nested structures
    for _ in range(5):
        old_text = text
        # Remove , before ] with any whitespace/newlines between
        text = re.sub(r',\s*\]', ']', text)
        # Remove , before } with any whitespace/newlines between  
        text = re.sub(r',\s*\}', '}', text)
        # Also handle case where there's a newline before the bracket
        text = re.sub(r',[\s\n\r]*\]', ']', text)
        text = re.sub(r',[\s\n\r]*\}', '}', text)
        if old_text == text:
            break  # No more changes
    
    # Step 3: Fix common LLM issues - double commas, empty objects
    text = re.sub(r',,+', ',', text)  # ,, -> ,
    text = re.sub(r'\[\s*,', '[', text)  # [, -> [
    text = re.sub(r'{\s*,', '{', text)  # {, -> {
    
    # Step 4: Process character by character to escape control chars inside JSON strings
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
    
    # Step 5: Final cleanup - one more pass for trailing commas
    text = re.sub(r',\s*\]', ']', text)
    text = re.sub(r',\s*\}', '}', text)
    
    return text


def robust_json_parse(text: str) -> dict:
    """
    Parse JSON with multiple fallback strategies.
    
    Tries:
    1. Direct parse
    2. Parse after sanitization
    3. Parse after aggressive cleanup
    4. Return error dict if all fail
    """
    # Log what we received for debugging
    logger.info(f"[JSON] Attempting to parse response, length: {len(text)}, preview: {text[:500]}...")
    
    # Try 1: Direct parse
    try:
        result = json.loads(text)
        logger.info("[JSON] Direct parse SUCCESS")
        return result
    except json.JSONDecodeError as e:
        logger.warning(f"[JSON] Direct parse failed at position {e.pos}: {e.msg}")
    
    # Try 2: Sanitize and parse
    try:
        sanitized = sanitize_json_string(text)
        result = json.loads(sanitized)
        logger.info("[JSON] Sanitized parse SUCCESS")
        return result
    except json.JSONDecodeError as e:
        logger.warning(f"[JSON] Sanitized parse failed at position {e.pos}: {e.msg}")
    
    # Try 3: Aggressive cleanup - remove all whitespace between tokens
    try:
        # Replace multiple spaces/newlines with single space
        cleaned = re.sub(r'\s+', ' ', text)
        # Remove trailing commas again
        cleaned = re.sub(r',\s*\]', ']', cleaned)
        cleaned = re.sub(r',\s*\}', '}', cleaned)
        result = json.loads(cleaned)
        logger.info("[JSON] Aggressive cleanup SUCCESS")
        return result
    except json.JSONDecodeError as e:
        logger.warning(f"[JSON] Aggressive cleanup failed at position {e.pos}: {e.msg}")
    
    # Try 4: Find JSON object boundaries and extract
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            extracted = text[start:end]
            extracted = sanitize_json_string(extracted)
            result = json.loads(extracted)
            logger.info("[JSON] Extracted JSON block SUCCESS")
            return result
    except json.JSONDecodeError as e:
        logger.error(f"[JSON] All parse attempts FAILED. Last error at position {e.pos}: {e.msg}")
        logger.error(f"[JSON] Response preview: {text[:1000]}...")
    
    # All attempts failed - return error structure
    logger.error(f"[JSON] CRITICAL: Could not parse LLM response. Full response:\n{text[:2000]}")
    return {
        "error": "JSON parse failed after all attempts",
        "hypothesis_title": "Error: Could not parse LLM response",
        "hypothesis": {"main_claim": "LLM response was not valid JSON. Please try again."}
    }




# ==================== POST-PROCESSING FUNCTIONS ====================
# These fix LLM output that doesn't follow prompt rules

def build_author_lookup(papers: list) -> dict:
    """
    Build a lookup table mapping any author mention to the correct first-author LAST NAME format.
    
    Example: "Frances R. Balkwill" paper -> maps "Frances et al." -> "Balkwill et al."
    Example: "Weiying Zhou" paper -> maps "Weiying et al." -> "Zhou et al."
    """
    lookup = {}
    for p in papers:
        authors_raw = p.get('authors', '')
        year = str(p.get('year', ''))
        if not authors_raw or not year:
            continue
        
        # Get first author's LAST NAME (the CORRECT citation format)
        # Author formats: "Frances R. Balkwill" or "Weiying Zhou" - last name is LAST word
        if ',' in authors_raw:
            first_author_full = authors_raw.split(',')[0].strip()
        elif ' and ' in authors_raw.lower():
            first_author_full = authors_raw.split(' and ')[0].strip()
        else:
            first_author_full = authors_raw.strip()
        
        # Extract LAST NAME = last word that's not an initial (like "R.")
        name_parts = first_author_full.split()
        first_author_lastname = first_author_full  # fallback
        for i in range(len(name_parts) - 1, -1, -1):
            part = name_parts[i]
            # Skip if it's an initial like "R." or "Jr." or "III"
            if len(part) > 2 and not part.endswith('.') and part not in ['Jr', 'Jr.', 'Sr', 'Sr.', 'III', 'II', 'IV']:
                first_author_lastname = part
                break
        
        correct_citation = f"{first_author_lastname} et al. ({year})"
        
        # Map ALL name parts (first name, middle name, etc.) to the correct citation
        for author_part in authors_raw.replace(' and ', ',').split(','):
            author_part = author_part.strip()
            for name in author_part.split():
                name = name.strip()
                # Skip initials like "J." or "K." or "R."
                if len(name) > 2 and not name.endswith('.'):
                    wrong_citation = f"{name} et al. ({year})"
                    if wrong_citation.lower() != correct_citation.lower():
                        lookup[wrong_citation] = correct_citation
    
    return lookup


def correct_citation_authors(hypothesis: dict, papers: list) -> dict:
    """
    Fix wrong author names in citations.
    
    LLM often uses wrong author (e.g., "Weiying et al." instead of "Zhou et al.")
    This function corrects them based on the actual paper metadata.
    """
    lookup = build_author_lookup(papers)
    
    # Convert hypothesis to string, replace wrong citations, convert back
    text = json.dumps(hypothesis, ensure_ascii=False)
    
    for wrong, correct in lookup.items():
        text = text.replace(wrong, correct)
    
    try:
        return json.loads(text)
    except:
        return hypothesis  # Return original if JSON parse fails


def fill_missing_sections(hypothesis: dict) -> dict:
    """
    Fill empty sections with required default values per elevenprompt.md rules.
    Uses proper data structures that frontend expects.
    """
    # Skip if this is an error response
    if hypothesis.get('error'):
        return hypothesis
    
    # Fix preliminary_data - must use list for pilot_studies!
    prelim = hypothesis.get('preliminary_data', {})
    if not prelim or not isinstance(prelim, dict):
        hypothesis['preliminary_data'] = {
            'status': 'None yet. This is a proposed hypothesis requiring validation.',
            'pilot_studies': [],  # Empty list, not string!
            'overall_readiness': 'Requires experimental validation'
        }
    elif isinstance(prelim, dict):
        # Ensure pilot_studies is a list
        if not isinstance(prelim.get('pilot_studies'), list):
            prelim['pilot_studies'] = []
        if not prelim.get('status'):
            prelim['status'] = 'None yet. This is a proposed hypothesis requiring validation.'
    
    # Fix funding_plan
    funding = hypothesis.get('funding_plan', {})
    if not funding or not isinstance(funding, dict):
        hypothesis['funding_plan'] = {
            'status': 'Funding: Requires manual search of NIH Reporter, NSF FastLane for current calls.',
            'opportunities': []  # Empty list, not inventing grants
        }
    
    # Fix ip_landscape
    ip = hypothesis.get('ip_landscape', {})
    if not ip or not isinstance(ip, dict):
        hypothesis['ip_landscape'] = {
            'status': 'IP Landscape: Requires USPTO search. Recommend provisional filing if novel.',
            'search_terms_to_use': [],
            'recommendation': 'Consult IP attorney.'
        }
    
    return hypothesis


def post_process_hypothesis(hypothesis: dict, papers: list) -> dict:
    """
    Main post-processing function that applies all fixes to LLM output.
    
    Fixes applied:
    1. Correct wrong citation author names
    2. Fill missing/empty sections with required defaults
    """
    logger.info("[POST-PROCESS] Applying citation corrections and section fills...")
    
    # Step 1: Correct citation authors
    hypothesis = correct_citation_authors(hypothesis, papers)
    
    # Step 2: Fill missing sections
    hypothesis = fill_missing_sections(hypothesis)
    
    logger.info("[POST-PROCESS] Post-processing complete")
    return hypothesis


def call_groq(prompt: str, temperature: float = 0.7, compact: bool = False) -> str:
    """
    Call Groq API for LLM generation with JSON enforcement.
    
    Uses llama-3.3-70b-versatile (Dec 2024) for larger prompts.
    ENHANCED: Added JSON system message for more reliable JSON output.
    """
    config = get_settings()
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage
    
    # Model fallback list (ordered by preference, from Groq docs Dec 2024)
    if compact:
        models_to_try = [
            "llama-3.1-8b-instant",                        # 6K TPM, fast
            "llama3-8b-8192",                              # Alternative 8b
        ]
        max_tokens = 4000
    else:
        models_to_try = [
            "llama-3.3-70b-versatile",                     # 12K TPM - BEST for JSON
            "llama-3.1-70b-versatile",                     # Alternative 70b
            "llama3-70b-8192",                             # Older 70b
            "mixtral-8x7b-32768",                          # Mixtral fallback
            "llama-3.1-8b-instant",                        # 8b last resort
        ]
        max_tokens = 8000
    
    # Lower temperature for more deterministic JSON output
    json_temperature = min(temperature, 0.3)
    
    # System message to enforce JSON output
    system_prompt = """You are a JSON generator. You MUST respond with ONLY valid JSON.
Do NOT include any text before or after the JSON.
Do NOT use markdown code blocks.
Do NOT add explanations.
Start your response with { and end with }.
Ensure all strings are properly quoted and escaped."""
    
    last_error = None
    for model in models_to_try:
        try:
            logger.info(f"[LLM] Trying Groq model: {model} (JSON mode, temp={json_temperature})")
            llm = ChatGroq(
                model=model,
                groq_api_key=config.api.groq_api_key,
                temperature=json_temperature,
                max_tokens=max_tokens
            )
            
            # Use messages with system prompt for better JSON adherence
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ]
            
            response = llm.invoke(messages)
            result = response.content if hasattr(response, 'content') else str(response)
            logger.info(f"[LLM] Success with Groq model: {model}, response length: {len(result)}")
            return result
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
    Call Google Gemini API for LLM generation with JSON mode.
    
    ENHANCED: Uses JSON mode to GUARANTEE valid JSON output.
    Uses Gemini 1.5 Pro as primary (1M token context, best for complex outputs).
    """
    config = get_settings()
    
    if not config.api.google_api_key:
        raise ValueError("GOOGLE_API_KEY not configured")
    
    import google.generativeai as genai
    
    genai.configure(api_key=config.api.google_api_key)
    
    # Model fallback list - EXACT names from genai.list_models() API (Dec 2024)
    # These are the ONLY models that work with this API key - verified by API call
    # Note: SDK adds 'models/' prefix automatically if not present
    models_to_try = [
        "gemini-2.0-flash",                  # FREE! Stable 2.0 flash
        "gemini-2.0-flash-exp",              # Experimental 2.0 flash  
        "gemini-2.0-flash-lite",             # Lightweight 2.0
        "gemini-2.5-flash",                  # Newest 2.5 flash (if available)
    ]
    
    # Lower temperature for more deterministic JSON output
    json_temperature = min(temperature, 0.4)
    
    last_error = None
    rate_limit_retry_done = False  # Only retry once globally, not per model
    
    for model_name in models_to_try:
        try:
            logger.info(f"[LLM] Trying Gemini model: {model_name} (JSON mode, temp={json_temperature})")
            
            # Create model with JSON mode enabled
            generation_config = {
                "temperature": json_temperature,
                "max_output_tokens": 8192,
                "top_p": 0.95,
                "response_mime_type": "application/json"
            }
            
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config
            )
            
            response = model.generate_content(prompt)
            
            # Validate we got text back
            if not response.text:
                raise ValueError("Empty response from Gemini")
            
            logger.info(f"[LLM] Success with Gemini model: {model_name}, response length: {len(response.text)}")
            return response.text
            
        except Exception as e:
            error_msg = str(e)
            last_error = e
            
            # Check for rate limit (429) - only retry ONCE globally
            if ("429" in error_msg or "quota" in error_msg.lower() or "resource exhausted" in error_msg.lower()):
                if not rate_limit_retry_done:
                    logger.warning(f"[LLM] Rate limited on {model_name}, waiting 10 seconds then trying next model...")
                    import time
                    time.sleep(10)  # Wait 10 seconds before trying next model
                    rate_limit_retry_done = True
                else:
                    logger.warning(f"[LLM] Still rate limited on {model_name}, skipping...")
            else:
                logger.warning(f"[LLM] Gemini {model_name} failed: {error_msg[:100]}")
            
            continue  # Try next model
    
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
        
        # COMPREHENSIVE JSON EXTRACTION - Handle ALL formats LLMs return
        # Step 1: Log what we received for debugging
        logger.info(f"[PARSE] Raw response first 200 chars: {text[:200]}")
        
        # Step 2: Strip markdown code blocks (many formats)
        original_text = text
        
        # Pattern 1: ```json ... ``` or ```JSON ... ```
        if "```json" in text.lower():
            text = text.lower().split("```json", 1)[1]
            if "```" in text:
                text = text.split("```")[0]
                
        # Pattern 2: ``` ... ``` (plain backticks)
        elif "```" in text:
            parts = text.split("```")
            # Find the part that contains JSON (has '{')
            for part in parts:
                if '{' in part and '}' in part:
                    text = part
                    break
        
        # Step 3: Find JSON boundaries - required
        start = text.find("{")
        end = text.rfind("}")
        
        if start >= 0 and end > start:
            text = text[start:end + 1]
            logger.info(f"[PARSE] Extracted JSON, length: {len(text)}")
        else:
            # Fallback: try original text for JSON
            start = original_text.find("{")
            end = original_text.rfind("}")
            if start >= 0 and end > start:
                text = original_text[start:end + 1]
                logger.info(f"[PARSE] Fallback extraction from original, length: {len(text)}")
            else:
                logger.error(f"[PARSE] No JSON found in response! First 500 chars: {original_text[:500]}")
        
        hypothesis = robust_json_parse(text.strip())
        
        # STEP 6.5: POST-PROCESS to fix LLM errors (citations, empty sections)
        hypothesis = post_process_hypothesis(hypothesis, real_papers)
        
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
    
    # Build the paper context with FIRST AUTHOR extracted and CORRECT CITATION format
    # This is CRITICAL - the LLM must know exactly how to cite each paper
    papers_context = """
════════════════════════════════════════════════════════════════
RETRIEVED_PAPERS (YOU MAY ONLY CITE FROM THIS LIST)
════════════════════════════════════════════════════════════════

CITATION FORMAT RULE:
- Always use FIRST AUTHOR'S LAST NAME + "et al." + (YEAR)
- Example: If authors are "Balkwill F, Capasso A, Hagemann T"
  ✅ CORRECT: "Balkwill et al. (2012)"
  ❌ WRONG: "Hagemann (2012)" or "Capasso et al. (2012)"

"""
    
    for i, p in enumerate(sorted_papers[:15], 1):
        # CRITICAL: Extract first author's LAST NAME for citation format
        authors_raw = p.get('authors', 'Unknown')
        
        # Handle different author formats (e.g., "Frances R. Balkwill" -> "Balkwill")
        if ',' in authors_raw:
            first_author_full = authors_raw.split(',')[0].strip()
        elif ' and ' in authors_raw.lower():
            first_author_full = authors_raw.split(' and ')[0].strip()
        else:
            first_author_full = authors_raw.strip()
        
        # Extract LAST NAME = last word that's not an initial
        name_parts = first_author_full.split()
        first_author_lastname = first_author_full  # fallback
        for j in range(len(name_parts) - 1, -1, -1):
            part = name_parts[j]
            # Skip initials like "R." and suffixes like "Jr."
            if len(part) > 2 and not part.endswith('.') and part not in ['Jr', 'Jr.', 'Sr', 'Sr.', 'III', 'II', 'IV']:
                first_author_lastname = part
                break
        
        year = p.get('year', 'N/A')
        citations = p.get('citation_count', 0)
        title = p.get('title', 'Untitled')
        journal = p.get('journal', 'N/A')
        doi = p.get('doi', 'N/A')
        abstract = p.get('abstract', 'No abstract available')[:1000]  # Increased to 1000 chars
        
        # Mark papers by citation impact with MUST CITE requirements
        if citations >= 1000:
            impact_marker = f"🔴 [MUST CITE 3+ TIMES - {citations:,} citations]"
            cite_requirement = f"""⚠️ MANDATORY: This is a HIGH-IMPACT paper. You MUST:
  1. Cite this paper AT LEAST 3 times
  2. Extract AT LEAST 3 different specific findings with NUMBERS
  3. Use this paper to establish baselines and context"""
        elif citations >= 500:
            impact_marker = f"🟠 [MUST CITE 2+ TIMES - {citations:,} citations]"
            cite_requirement = f"""⚠️ MANDATORY: You MUST cite this paper at least 2 times with specific findings"""
        elif citations >= 100:
            impact_marker = f"🟡 [SHOULD CITE - {citations:,} citations]"
            cite_requirement = ""
        else:
            impact_marker = f"[{citations:,} citations]"
            cite_requirement = ""
        
        papers_context += f"""
══════════════════════════════════════════════════════════════
PAPER {i}: {first_author_lastname} et al. ({year}) {impact_marker}
══════════════════════════════════════════════════════════════
CORRECT CITATION FORMAT: "{first_author_lastname} et al. ({year})"
❌ WRONG citations: Any other author name
Title: "{title}"
Journal: {journal} | DOI: {doi} | Citations: {citations:,}
Authors: {authors_raw}
{cite_requirement}

📄 ABSTRACT - YOU MUST EXTRACT SPECIFIC NUMBERS FROM THIS:
{abstract}

📝 EXTRACTION CHECKLIST for this paper:
  □ What specific metric/result is reported? (with numbers)
  □ What method did they use?
  □ What limitation did they identify?
  □ How does this connect to YOUR hypothesis?
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

    # Format experts with full details - CRITICAL: Extract REAL authors from papers
    # to prevent LLM from inventing fake names like "Dr. John Smith"
    
    # First, extract all first authors from retrieved papers as the ONLY valid expert names
    paper_authors = []
    for p in sorted_papers[:15]:
        authors_raw = p.get('authors', 'Unknown')
        if ',' in authors_raw:
            first_author = authors_raw.split(',')[0].strip()
        elif ' and ' in authors_raw.lower():
            first_author = authors_raw.split(' and ')[0].strip()
        else:
            first_author = authors_raw.split()[0] if authors_raw.split() else "Unknown"
        
        paper_authors.append({
            "name": first_author,
            "paper_title": p.get('title', 'Unknown'),
            "year": p.get('year', 'N/A'),
            "citations": p.get('citation_count', 0),
            "journal": p.get('journal', 'N/A')
        })
    
    experts_context = """
════════════════════════════════════════════════════════════════
EXPERT COLLABORATORS (ONLY SUGGEST THESE REAL AUTHORS)
════════════════════════════════════════════════════════════════

🚨 CRITICAL RULE: You MUST NOT invent expert names!
❌ NEVER suggest: "Dr. John Smith", "Dr. Jane Doe", or ANY name not listed below
✅ ONLY suggest authors from this list (extracted from RETRIEVED_PAPERS):

"""
    
    # Add paper authors as valid expert candidates
    for i, author in enumerate(paper_authors[:10], 1):
        experts_context += f"""
REAL AUTHOR {i}: {author['name']}
  - Paper: "{author['paper_title']}" ({author['year']})
  - Citations: {author['citations']:,}
  - Journal: {author['journal']}
  - Email: [Search institutional directory for {author['name']}]
"""
    
    # Also add experts from the real_experts list if they match paper authors
    experts_context += """

ADDITIONAL EXPERT INFO (from researcher databases):
"""
    for e in real_experts[:5]:
        experts_context += f"""
Expert: {e.get('name', 'Unknown')}
- Institution: {e.get('affiliation', 'Unknown')}
- h-index: {e.get('h_index', 'N/A')}
- Research Topics: {', '.join(e.get('research_topics', [])[:4])}
- Email: [Search institutional directory] (DO NOT INVENT EMAILS)
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

    return f'''You are a meticulous research scientist generating a novel research hypothesis. You MUST follow these rules with ZERO exceptions:

═══════════════════════════════════════════════════════════════
🚨 12 CRITICAL RULES (VIOLATION OF ANY = AUTOMATIC REJECTION)
═══════════════════════════════════════════════════════════════

RULE 1: CITATION DISCIPLINE - MANDATORY HIERARCHY (MOST CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**CITATION PRIORITY (YOU WILL BE REJECTED IF YOU VIOLATE THIS):**

| Citation Count | Minimum Times Must Cite | Purpose |
|----------------|-------------------------|---------|
| 1000+ citations | 3+ times EACH | Landmark papers - establish context |
| 500-999 citations | 2+ times EACH | Key papers - methodology/validation |
| <500 citations | 1+ time EACH | Supporting papers - specific details |

**VIOLATION = AUTOMATIC FAILURE:**
❌ WRONG: Paper with 5000 citations cited 0 times
❌ WRONG: Paper with 100 citations cited more than paper with 3000 citations
✅ CORRECT: Paper with 5000 citations cited 5+ times across hypothesis

**CITATION FORMAT (use EVERY time you cite):**
Author et al. (Year) "Title" [Journal, DOI: xxx, N citations]

Example: Meinshausen et al. (2011) "The RCP greenhouse gas concentrations" [Climatic Change, DOI: 10.1007/s10584-011-0156-z, 3,690 citations] found that...

**FOR EVERY CITATION YOU MUST:**
1. Use FIRST AUTHOR'S LAST NAME from RETRIEVED_PAPERS list
   - Check the "CORRECT CITATION FORMAT" line for each paper
2. Extract SPECIFIC NUMBERS from the abstract
   - ❌ WRONG: "Balkwill (2012) discusses tumor microenvironment"
   - ✅ CORRECT: "Balkwill et al. (2012) identified 5 stages of carcinogenesis where non-malignant cells promote tumors"
3. Connect citation to YOUR hypothesis
   - Why does this paper matter for your approach?

4. NEVER cite papers not in RETRIEVED_PAPERS
   - If you need info not in papers, write: "This requires additional literature search on [topic]"

5. Papers marked 🔴 [MUST CITE 3+ TIMES] MUST appear 3+ times with DIFFERENT findings each time

RULE 2: SPECIFICITY DISCIPLINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BANNED WORDS - Never use without immediate quantification:
"significant", "substantial", "considerable", "large", "small", "high", "low",
"better", "worse", "improved", "enhanced", "reduced", "many", "few", "several", "various"

EVERY claim must have:
1. Concrete numbers FROM retrieved paper abstracts:
   ❌ WRONG: "Current methods have high error rates"
   ✅ CORRECT: "Zhou et al. (2014) reported 2D imaging achieves 60% accuracy with 40% error rate"

2. Specific mechanisms from retrieved papers:
   ❌ WRONG: "Cancer cells migrate through blood vessels"
   ✅ CORRECT: "Zhou et al. (2014): cancer-secreted miR-105 downregulates ZO-1, increasing permeability 2.5-fold (p<0.001)"

3. Quantified improvements with evidence:
   ❌ WRONG: "Our method will improve accuracy"
   ✅ CORRECT: "Based on Balkwill et al. (2012)'s 60% baseline, adding 3D depth could reach 85%"

4. Costs with breakdown:
   ❌ WRONG: "High cost"
   ✅ CORRECT: "$50,000 equipment + $200/sample + $100/hour = $500/experiment"

RULE 3: CROSS-DOMAIN CONNECTION - REAL MECHANISMS ONLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**BANNED PHRASES (NEVER USE - TOO VAGUE):**
❌ "Bio-inspired design principles"
❌ "Self-organization and adaptability"
❌ "Nature-inspired optimization"
❌ "Biomimicry approaches"
❌ "Adaptive algorithms from biology"

**REQUIRED: SPECIFIC mechanism with QUANTITATIVE details**

1. **SOURCE MECHANISM (be SPECIFIC):**
   - WHICH organism: Not "biology" but "Morpho butterfly wing scales" or "photosystem II in purple bacteria"
   - WHAT mechanism with NUMBERS: "quantum coherence lasting 660 femtoseconds enables 95% energy transfer"
   - Key dimensions/parameters: "ridges with 200nm spacing", "5nm Förster radius"
   - Source paper from RETRIEVED_PAPERS or CROSS_DOMAIN_PAPERS

2. **TARGET PROBLEM (be QUANTITATIVE):**
   - Exactly what fails in current approach
   - Performance numbers: "current method achieves only 30% efficiency"
   - WHY it fails mechanistically
   - Source paper from RETRIEVED_PAPERS

3. **ADAPTATION MECHANISM (MINIMUM 5 CONCRETE STEPS with parameters):**
   Step 1: [Specific modification] - parameter = [value] because [source paper]
   Step 2: [Technical change] - from [A] to [B] (improvement: X%)
   Step 3: [Integration approach] - materials: [list with costs]
   Step 4: [Validation method] - success if [metric] > [threshold]
   Step 5: [Scale-up plan] - from [small] to [large] with [timeline]

   ❌ WRONG: "Step 1: Identify bio-inspired principles"
   ✅ CORRECT: "Step 1: Synthesize quantum dots with 3-5nm diameter (matching chlorophyll spacing) using hot-injection at 280°C"

4. **WHY NON-OBVIOUS (explain the REAL barrier):**
   - Different publication venues (journals A vs B)
   - Different timescales (femtoseconds vs microseconds)
   - Different temperature assumptions (cryogenic vs room temp)
   - EVIDENCE: "Zero cross-citations between [field A] and [field B] in past 10 years"

5. **EXPECTED IMPROVEMENT with CALCULATION:**
   - Current: [value] (Source: Paper A)
   - Biology: [value] (Source: Paper B)
   - After adaptation: [calculated value] = current × improvement_factor
   - Show your math!


RULE 4: METHODOLOGY DISCIPLINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For EVERY methodological step, you MUST include:

**BANNED JUSTIFICATIONS (NEVER USE):**
❌ "standard practice"
❌ "commonly used"
❌ "typical value"
❌ "conventional approach"
❌ "industry standard"

**REQUIRED for EVERY parameter:**
1. Paper citation: "batch_size: 32 (Author et al. 2024 used 16-64 for similar problem)"
2. OR calculation: "batch_size: 32 = dataset_size/8 (dataset has 256 samples)"
3. OR uncertainty: "batch_size: 32 (no paper guidance; will sweep 16-64 to validate)"

**FULL STEP TEMPLATE:**
Step N: [Name]
- Algorithm: [Name] justified by [Paper citation]
- Parameters: [list with sources for EACH]
- Success criteria: [metric] > [threshold] (baseline: Author 2024 achieved X)
- Resources: GPU: Xh × $Y = $Z; Storage: X TB × $0.10/mo = $Z
- Timeline: Week 1: [task]; Week 2: [task]; Week 3: [task]
- Input: Format [X], Size [Y], Range [Z]
- Output: Format [X], Expected [Y ± Z]
- Risk: If [failure], then [contingency] at cost $X + Y weeks delay

RULE 5: COMPARISON TABLE DISCIPLINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Your comparison table MUST:
1. Include ONLY methods mentioned in retrieved papers
2. EVERY number must have "(Source: Paper et al. Year)" next to it
3. Show tradeoffs and limitations, not just advantages

RULE 6: EXPERT COLLABORATORS DISCIPLINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚨 ONLY suggest authors from RETRIEVED_PAPERS - they are listed in EXPERT section
❌ NEVER invent names like "Dr. John Smith" or "Dr. Jane Doe"
✅ ONLY use REAL AUTHORS listed in the EXPERT section below
For email: Write "[Search institutional directory]" - DO NOT INVENT EMAILS

RULE 7: PRELIMINARY DATA DISCIPLINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. ONLY include preliminary data if explicitly in retrieved paper abstracts
2. If NO preliminary data exists, write EXACTLY:
   "Preliminary Data: None yet. This is a proposed hypothesis requiring validation."
3. ❌ NEVER write: "80% accuracy on synthetic data" unless a retrieved paper says this

RULE 8: FUNDING & IP DISCIPLINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If you cannot verify real funding opportunities, write EXACTLY:
"Funding: Requires manual search of NIH Reporter, NSF FastLane for current calls matching [topic]"

For IP, write EXACTLY:
"IP Landscape: Requires USPTO search with terms: [list terms]. Recommend provisional filing if novel."

RULE 9: LITERATURE GAP - DEEP ANALYSIS REQUIRED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**1. WHAT HAS BEEN TRIED (list 3-5 approaches from papers):**
For EACH approach:
- Method name: [Exact technique, not "traditional methods"]
- Who tried it: [Author (Year) from RETRIEVED_PAPERS]
- Results: [Specific numbers from their abstract]
- Why it hit limits: [Mechanistic reason, not just "didn't work"]
- Source: [Full citation]

**2. WHAT HAS NOT BEEN TRIED (your novel contribution):**
- Specific gap: [Exact combination that's missing]
- Evidence of gap: "Zero papers in RETRIEVED_PAPERS combining [X] and [Y]"
- Why not tried: [Technical barrier, cultural barrier, cost barrier]
- Cross-citation analysis: "[Field A] papers have zero citations from [Field B]"

**3. WHY NOW IS THE RIGHT TIME (enabling factors):**
- Recent advancement: [What changed? When? Citation if possible]
- Cost reduction: [From $X to $Y - specific numbers]
- New capability: [Tool/method that didn't exist before]
- Converging fields: [What's bringing communities together]

RULE 10: RISK ASSESSMENT - EVIDENCE-BASED PROBABILITIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**BANNED RISK STATEMENTS:**
❌ "Medium probability" (what's medium? 30%? 50%?)
❌ "High impact" (impact on what? timeline? cost?)
❌ "Technical challenges" (which challenge specifically?)

**REQUIRED FORMAT for EACH risk:**
```
Risk N: [Specific failure mode]
- Probability: X% 
  - Basis: [How calculated? Paper evidence? Historical data?]
  - Evidence: [Citation or reasoning]
- Impact: 
  - Timeline delay: +X weeks/months
  - Cost increase: +$X
  - Performance reduction: -X%
- Mitigation: 
  - Action: [Specific steps]
  - Cost: $X
  - Reduces probability: X% → Y%
- Contingency:
  - Trigger: [Metric indicating failure]
  - Backup plan: [Alternative] 
  - Additional cost: $X + Y weeks
```

RULE 11: BROADER IMPACT - SHOW YOUR CALCULATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**BANNED STATEMENTS:**
❌ "Could save 10,000 lives" (no calculation)
❌ "$1 billion savings" (no derivation)
❌ "Major impact on society" (too vague)

**REQUIRED FORMAT - Calculate from retrieved papers:**
```
Impact Type: [Clinical/Economic/Scientific]
Baseline from paper: [Author (Year)] reports [X]
Calculation:
  Step 1: [formula]
  Step 2: [substitute numbers]
  Step 3: [result with units]
Result: Y ± Z (uncertainty range)
Assumptions: [list what could be wrong]
```

Example:
"Baseline: IPCC (2015) reports solar provides 750 TWh/year globally.
Calculation: 20% efficiency gain × 750 TWh = +150 TWh/year
CO2 impact: 150 TWh × 900 kg CO2/MWh = 135 million tons CO2/year avoided
Economic: 150 TWh × $50/MWh = $7.5 billion value/year"

RULE 12: COMPLETE EVERY SECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You MUST generate ALL sections. If you skip ANY section, you have FAILED.

═══════════════════════════════════════════════════════════════
VERIFICATION CHECKLIST (Check before outputting):
═══════════════════════════════════════════════════════════════
□ Every citation uses FIRST AUTHOR from RETRIEVED_PAPERS
□ Every number has "(Source: Paper et al. Year)" 
□ Zero banned words without quantification
□ Cross-domain has all 5 parts
□ Expert collaborators are REAL AUTHORS from papers
□ Preliminary data says "None yet" if no evidence
□ All sections complete

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
      "name": "MUST be FIRST AUTHOR from REAL AUTHOR list above - NO INVENTED NAMES",
      "institution": "From paper metadata OR '[Not specified in retrieved papers]'",
      "email": "[Search institutional directory] - DO NOT INVENT EMAILS",
      "paper_from_retrieved_list": "EXACT title from RETRIEVED_PAPERS the author wrote",
      "paper_year": "Year from that paper",
      "paper_citations": "Citation count from that paper",
      "expertise_summary": "Based on their paper's abstract",
      "contributions": [
        {{
          "contribution_type": "Data Sharing / Validation / Consultation",
          "description": "What exactly they could contribute based on their paper",
          "value_to_project": "Saves X weeks / provides Y data"
        }}
      ],
      "collaboration_likelihood": {{
        "likelihood": "X% (reasoning based on their paper's content)",
        "evidence_for": ["Reason from their paper"],
        "evidence_against": ["Concern"]
      }},
      "email_template": {{
        "subject": "Collaboration: [Title]",
        "body": "Dear [REAL NAME from papers], I am working on... Would you be open to..."
      }},
      "priority": "HIGHEST/SECONDARY based on citation count"
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
    "status": "MUST be 'None yet. This is a proposed hypothesis requiring validation.' UNLESS retrieved papers explicitly mention pilot data",
    "pilot_studies": "Only include if a retrieved paper EXPLICITLY mentions pilot/preliminary results",
    "note": "DO NOT INVENT preliminary data like '80% accuracy on synthetic data' - this is FABRICATION"
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
    "status": "Funding: Requires manual search of NIH Reporter, NSF FastLane for current calls matching [your topic]",
    "note": "DO NOT INVENT funding deadlines or programs - link to real search portals only"
  }},

  "ip_landscape": {{
    "status": "IP Landscape: Requires USPTO search with terms: [list relevant terms]. Recommend provisional patent filing if novel.",
    "search_terms_to_use": ["term1 related to technique", "term2 related to application"],
    "recommendation": "Cannot determine without real search - recommend provisional filing ($10-15K) if approach is novel"
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
