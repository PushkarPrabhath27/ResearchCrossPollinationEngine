"""
API Routes for Hypothesis Engine - Real LLM Implementation
Uses Groq (fast, free) as primary, with proper error handling
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
import json

from src.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


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
    num_hypotheses: int = Field(3, ge=1, le=10)
    creativity: float = Field(0.7, ge=0.0, le=1.0)


class Hypothesis(BaseModel):
    title: str
    description: str
    novelty_score: float
    feasibility_score: float
    impact_score: float
    source_fields: List[str]
    key_references: List[str]
    next_steps: List[str]


class HypothesisResult(BaseModel):
    success: bool
    query: str
    field: str
    hypotheses: List[Hypothesis]
    cross_domain_insights: List[str]
    methodology_suggestions: List[str]
    error: Optional[str] = None


def call_groq(prompt: str, temperature: float = 0.7) -> str:
    """Call Groq API directly - fast and reliable"""
    config = get_settings()
    from langchain_groq import ChatGroq
    
    # Use llama-3.1-8b-instant (current model, replaces decommissioned llama3-8b-8192)
    model = "llama-3.1-8b-instant"
    logger.info(f"Using Groq LLM model: {model}")
    llm = ChatGroq(
        model=model,
        groq_api_key=config.api.groq_api_key,
        temperature=temperature
    )
    response = llm.invoke(prompt)
    return response.content if hasattr(response, 'content') else str(response)


def call_google(prompt: str, temperature: float = 0.7) -> str:
    """Call Google Gemini API"""
    config = get_settings()
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    # Try different model names that might work
    models_to_try = ["gemini-1.5-flash", "gemini-pro", "gemini-1.0-pro"]
    
    for model in models_to_try:
        try:
            logger.info(f"Trying Google Gemini model: {model}")
            llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=config.api.google_api_key,
                temperature=temperature,
                convert_system_message_to_human=True
            )
            response = llm.invoke(prompt)
            logger.info(f"Success with model: {model}")
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.warning(f"Model {model} failed: {e}")
            continue
    
    raise Exception("All Google Gemini models failed")


def generate_with_llm(query: str, field: str, num: int, creativity: float) -> Dict:
    """Generate hypotheses using LLM with proper fallback"""
    logger.info(f"Generating {num} hypotheses for: {query[:50]}...")
    config = get_settings()
    
    prompt = f"""You are a scientific hypothesis generator that finds cross-disciplinary connections.

RESEARCH QUESTION: {query}
PRIMARY FIELD: {field}
NUMBER OF HYPOTHESES: {num}
CREATIVITY LEVEL: {creativity}

Generate {num} novel research hypotheses by finding connections from OTHER scientific fields.

For each hypothesis, provide:
- title: A clear, specific title
- description: 2-3 sentences explaining the hypothesis and why it's promising
- novelty_score: Score from 0-10
- feasibility_score: Score from 0-10  
- impact_score: Score from 0-10
- source_fields: List of other fields that inspired this
- key_references: Types of papers to look for
- next_steps: Actions to pursue this hypothesis

Also provide:
- cross_domain_insights: 3 insights from analyzing across fields
- methodology_suggestions: 3 methods from other fields that could apply

Return ONLY a valid JSON object in this exact format:
{{"hypotheses": [{{"title": "...", "description": "...", "novelty_score": 8.0, "feasibility_score": 7.0, "impact_score": 8.5, "source_fields": ["field1", "field2"], "key_references": ["ref1", "ref2"], "next_steps": ["step1", "step2"]}}], "cross_domain_insights": ["insight1", "insight2", "insight3"], "methodology_suggestions": ["method1", "method2", "method3"]}}"""

    # Try Groq first (fast and reliable), then Google as fallback
    providers = []
    
    # Add Groq if API key exists
    if config.api.groq_api_key:
        providers.append(("groq", lambda p: call_groq(p, creativity)))
    
    # Add Google if API key exists  
    if config.api.google_api_key:
        providers.append(("google", lambda p: call_google(p, creativity)))
    
    if not providers:
        return {
            "success": False, "query": query, "field": field,
            "hypotheses": [], "cross_domain_insights": [],
            "methodology_suggestions": [], 
            "error": "No LLM API keys configured. Set GROQ_API_KEY or GOOGLE_API_KEY in .env"
        }
    
    last_error = None
    for provider_name, call_fn in providers:
        try:
            logger.info(f"Trying LLM provider: {provider_name}")
            text = call_fn(prompt)
            logger.info(f"Got response from {provider_name}, length: {len(text)}")
            
            # Parse JSON from response
            try:
                # Clean up response - extract JSON
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    parts = text.split("```")
                    if len(parts) >= 2:
                        text = parts[1]
                
                # Find JSON object in text
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    text = text[start:end]
                
                result = json.loads(text.strip())
                
                # Validate required fields
                if "hypotheses" not in result:
                    result["hypotheses"] = []
                if "cross_domain_insights" not in result:
                    result["cross_domain_insights"] = []
                if "methodology_suggestions" not in result:
                    result["methodology_suggestions"] = []
                
                return {
                    "success": True, 
                    "query": query, 
                    "field": field,
                    "hypotheses": result["hypotheses"],
                    "cross_domain_insights": result["cross_domain_insights"],
                    "methodology_suggestions": result["methodology_suggestions"],
                    "error": None
                }
                
            except json.JSONDecodeError as je:
                logger.warning(f"JSON parse error: {je}")
                # Return raw text as single hypothesis
                return {
                    "success": True, "query": query, "field": field,
                    "hypotheses": [{
                        "title": f"AI-Generated Hypothesis for {field}",
                        "description": text[:800] if len(text) > 800 else text,
                        "novelty_score": 7.5, "feasibility_score": 7.0, "impact_score": 8.0,
                        "source_fields": ["cross-disciplinary"],
                        "key_references": ["See description for insights"],
                        "next_steps": ["Review the AI response", "Extract testable elements", "Search related literature"]
                    }],
                    "cross_domain_insights": ["AI analysis complete - see hypothesis"],
                    "methodology_suggestions": ["Review hypothesis for methodology ideas"],
                    "error": None
                }
                
        except Exception as e:
            logger.error(f"Provider {provider_name} failed: {e}")
            last_error = str(e)
            continue
    
    # All providers failed
    return {
        "success": False, "query": query, "field": field,
        "hypotheses": [], "cross_domain_insights": [],
        "methodology_suggestions": [], 
        "error": f"All LLM providers failed. Last error: {last_error}"
    }


@router.post("/generate", response_model=HypothesisResult, tags=["Hypotheses"])
async def generate_hypotheses(request: HypothesisRequest):
    """Generate hypotheses using AI"""
    logger.info(f"API Request received: {request.query[:50]}...")
    
    try:
        result = generate_with_llm(
            request.query, request.field,
            request.num_hypotheses, request.creativity
        )
        return HypothesisResult(**result)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fields", tags=["Metadata"])
async def get_fields() -> List[str]:
    return [f.value for f in FieldEnum]


@router.get("/stats", tags=["Metadata"])
async def get_stats():
    config = get_settings()
    providers = []
    if config.api.groq_api_key:
        providers.append("groq")
    if config.api.google_api_key:
        providers.append("google")
    return {
        "llm_provider": config.agent.llm_provider,
        "available_providers": providers,
        "status": "operational"
    }
