"""
API Routes for Hypothesis Engine - PRODUCTION RAG Implementation
Uses LIVE FETCHING from real APIs - No local storage

Data Sources:
- Papers: OpenAlex (250M+), arXiv (2M+), Semantic Scholar (200M+)
- Code: GitHub API
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
from src.api.paper_fetcher import fetch_real_papers, paper_fetcher
from src.api.github_fetcher import fetch_github_repos
from src.api.dataset_fetcher import fetch_datasets

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




def call_groq(prompt: str, temperature: float = 0.7) -> str:
    """Call Groq API for LLM generation"""
    config = get_settings()
    from langchain_groq import ChatGroq
    
    model = "llama-3.1-8b-instant"
    logger.info(f"[LLM] Using Groq model: {model}")
    
    llm = ChatGroq(
        model=model,
        groq_api_key=config.api.groq_api_key,
        temperature=temperature,
        max_tokens=6000
    )
    response = llm.invoke(prompt)
    return response.content if hasattr(response, 'content') else str(response)


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
    
    Flow:
    1. Fetch REAL papers from OpenAlex, arXiv, Semantic Scholar
    2. Fetch REAL datasets from HuggingFace, Papers With Code
    3. Fetch REAL code repos from GitHub
    4. Generate hypothesis using LLM with real data as context
    5. Return hypothesis with all real links
    """
    logger.info(f"[GENERATE] Starting for: {query[:50]}...")
    start_time = time.time()
    
    config = get_settings()
    
    # STEP 1: Fetch REAL papers (live from APIs) - WITH FIELD FILTERING
    logger.info(f"[FETCH] Getting real papers from APIs for field: {field}...")
    try:
        paper_result = fetch_real_papers(query, field=field, max_results=30)
        real_papers = paper_fetcher.format_papers_for_display(paper_result["papers"])
        paper_stats = paper_result["stats"]
        logger.info(f"[FETCH] Got {len(real_papers)} real papers")
    except Exception as e:
        logger.error(f"[FETCH] Paper fetch failed: {e}")
        real_papers = []
        paper_stats = {"error": str(e)}
    
    # STEP 2: Fetch REAL datasets (live from APIs)
    logger.info("[FETCH] Getting real datasets from APIs...")
    try:
        real_datasets = fetch_datasets(query, field, max_results=6)
        logger.info(f"[FETCH] Got {len(real_datasets)} real datasets")
    except Exception as e:
        logger.error(f"[FETCH] Dataset fetch failed: {e}")
        real_datasets = []
    
    # STEP 3: Fetch REAL GitHub repos (live from API) - WITH FIELD FILTERING
    logger.info(f"[FETCH] Getting real GitHub repos for field: {field}...")
    try:
        real_repos = fetch_github_repos(query, field=field, max_results=5)
        logger.info(f"[FETCH] Got {len(real_repos)} real repos")
    except Exception as e:
        logger.error(f"[FETCH] GitHub fetch failed: {e}")
        real_repos = []
    
    fetch_time = time.time() - start_time
    logger.info(f"[FETCH] Total fetch time: {fetch_time:.2f}s")
    
    # STEP 4: Generate hypothesis with LLM using real data as context
    if not config.api.groq_api_key:
        return {
            "success": False,
            "error": "No LLM API key configured",
            "query": query,
            "field": field
        }
    
    prompt = generate_hypothesis_prompt(query, field, real_papers, real_datasets, real_repos)
    
    try:
        logger.info("[LLM] Generating hypothesis...")
        llm_start = time.time()
        text = call_groq(prompt, creativity)
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
        
        # STEP 5: Construct complete response with real data
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
                "repos_found": len(real_repos)
            },
            
            # The generated hypothesis
            "hypothesis": hypothesis,
            
            # REAL papers with working URLs
            "real_papers": real_papers,
            
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
            "real_datasets": real_datasets,
            "real_repos": real_repos
        }


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
