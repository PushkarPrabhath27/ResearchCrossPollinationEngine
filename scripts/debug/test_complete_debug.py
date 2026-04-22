"""
COMPLETE DEBUG SCRIPT - Test the EXACT code path used in production
Tests: Gemini models → Groq fallback → JSON parsing → Everything
"""
import os
import sys
import json
from dotenv import load_dotenv

load_dotenv('.env')

print("=" * 60)
print("COMPLETE DEBUG - Testing EXACT production code path")
print("=" * 60)

# Check API keys
groq_key = os.getenv('GROQ_API_KEY')
google_key = os.getenv('GOOGLE_API_KEY')

print(f"\n[CONFIG] GROQ_API_KEY present: {bool(groq_key)}")
print(f"[CONFIG] GOOGLE_API_KEY present: {bool(google_key)}")

if google_key:
    print(f"[CONFIG] Google key (masked): {google_key[:8]}...{google_key[-4:]}")

# ===================== TEST 1: Gemini with NEW model names =====================
print("\n" + "=" * 60)
print("TEST 1: Gemini API with UPDATED December 2024 models")
print("=" * 60)

import google.generativeai as genai
genai.configure(api_key=google_key)

# These are the EXACT models from routes.py after update
gemini_models = [
    "gemini-2.5-pro-preview-06-05",
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-1.5-pro-latest",
    "gemini-1.5-flash-latest",
]

simple_prompt = 'Return ONLY this JSON: {"test": "hello", "number": 42}'

gemini_success = False
for model_name in gemini_models:
    print(f"\n[GEMINI] Trying: {model_name}")
    try:
        model = genai.GenerativeModel(
            model_name,
            generation_config={
                'temperature': 0.3,
                'max_output_tokens': 500,
                'response_mime_type': 'application/json'  # JSON mode
            }
        )
        response = model.generate_content(simple_prompt)
        print(f"[GEMINI] SUCCESS! Response: {response.text[:100]}...")
        
        # Try to parse
        parsed = json.loads(response.text)
        print(f"[GEMINI] JSON PARSED: {parsed}")
        gemini_success = True
        print(f"[GEMINI] *** {model_name} WORKS! ***")
        break
    except Exception as e:
        print(f"[GEMINI] FAILED: {type(e).__name__}: {str(e)[:100]}")

if not gemini_success:
    print("\n[GEMINI] !!! ALL GEMINI MODELS FAILED !!!")

# ===================== TEST 2: Groq =====================
print("\n" + "=" * 60)
print("TEST 2: Groq API")
print("=" * 60)

if groq_key:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage
    
    groq_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "llama-3.1-8b-instant",
    ]
    
    system_prompt = "You are a JSON generator. Respond with ONLY valid JSON. No markdown."
    
    groq_success = False
    for model_name in groq_models:
        print(f"\n[GROQ] Trying: {model_name}")
        try:
            llm = ChatGroq(
                model=model_name,
                groq_api_key=groq_key,
                temperature=0.3,
                max_tokens=500
            )
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=simple_prompt)
            ]
            response = llm.invoke(messages)
            result = response.content
            print(f"[GROQ] SUCCESS! Response: {result[:100]}...")
            
            # Try to parse
            parsed = json.loads(result)
            print(f"[GROQ] JSON PARSED: {parsed}")
            groq_success = True
            print(f"[GROQ] *** {model_name} WORKS! ***")
            break
        except Exception as e:
            print(f"[GROQ] FAILED: {type(e).__name__}: {str(e)[:100]}")
    
    if not groq_success:
        print("\n[GROQ] !!! ALL GROQ MODELS FAILED !!!")
else:
    print("[GROQ] No GROQ_API_KEY - skipping")

# ===================== TEST 3: Complex prompt simulation =====================
print("\n" + "=" * 60)
print("TEST 3: Complex JSON prompt (simulating production)")
print("=" * 60)

complex_prompt = '''Generate a hypothesis in JSON format:
{
  "hypothesis_title": "string",
  "hypothesis": {
    "main_claim": "string",
    "theoretical_basis": "string",
    "novelty": "string"
  },
  "methodology": [{"step": "string"}],
  "cross_domain_connections": [{"source": "string", "target": "string"}]
}

Topic: Antibiotic resistance prediction using machine learning.
Return ONLY valid JSON.'''

print("\n[COMPLEX] Testing with complex prompt...")

# Try Gemini first for complex
if gemini_success:
    print("[COMPLEX] Using working Gemini model...")
    try:
        response = model.generate_content(complex_prompt)
        print(f"[COMPLEX] Response length: {len(response.text)}")
        print(f"[COMPLEX] First 500 chars: {response.text[:500]}...")
        parsed = json.loads(response.text)
        print(f"[COMPLEX] JSON PARSED SUCCESSFULLY!")
        print(f"[COMPLEX] Keys: {list(parsed.keys())}")
    except Exception as e:
        print(f"[COMPLEX] FAILED: {type(e).__name__}: {str(e)[:200]}")
elif groq_success:
    print("[COMPLEX] Using working Groq model...")
    try:
        messages = [
            SystemMessage(content="Return ONLY valid JSON"),
            HumanMessage(content=complex_prompt)
        ]
        response = llm.invoke(messages)
        result = response.content
        print(f"[COMPLEX] Response length: {len(result)}")
        print(f"[COMPLEX] First 500 chars: {result[:500]}...")
        parsed = json.loads(result)
        print(f"[COMPLEX] JSON PARSED SUCCESSFULLY!")
        print(f"[COMPLEX] Keys: {list(parsed.keys())}")
    except Exception as e:
        print(f"[COMPLEX] FAILED: {type(e).__name__}: {str(e)[:200]}")

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)
