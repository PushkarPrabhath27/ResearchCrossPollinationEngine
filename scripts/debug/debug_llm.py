"""
Debug script to reproduce EXACT production issue
This will call the LLM with the same prompt structure and show exactly what's failing
"""
import os
import sys
import json
import time
from dotenv import load_dotenv

load_dotenv('.env')

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import google.generativeai as genai

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

print("="*80)
print("STEP 1: Testing which models are available")
print("="*80)

available_models = []
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"  ✓ {m.name}")
            available_models.append(m.name.replace("models/", ""))
except Exception as e:
    print(f"  ERROR listing models: {e}")

print("\n" + "="*80)
print("STEP 2: Testing gemini-2.0-flash with a SIMPLE JSON prompt")
print("="*80)

simple_prompt = """Return a JSON object with exactly this structure:
{"hypothesis_title": "Test", "hypothesis": {"main_claim": "Test claim"}}"""

try:
    model = genai.GenerativeModel(
        'gemini-2.0-flash',
        generation_config={
            'temperature': 0.4,
            'response_mime_type': 'application/json'
        }
    )
    response = model.generate_content(simple_prompt)
    print(f"✓ Simple prompt SUCCESS")
    print(f"  Response: {response.text[:200]}")
    parsed = json.loads(response.text)
    print(f"  ✓ JSON parsed successfully")
except Exception as e:
    print(f"✗ Simple prompt FAILED: {type(e).__name__}: {e}")

print("\n" + "="*80)
print("STEP 3: Testing with a LONG prompt (simulating real use)")
print("="*80)

# Simulate the real prompt length - this is the key test
long_prompt = """You are a research scientist. Generate a hypothesis.

Here are 15 papers:
""" + "\n".join([
    f"""
Paper {i}:
Title: "Example Paper About Plastic Degradation Number {i}"
Authors: Author A, Author B
Year: 2024
Citations: {i*100}
Abstract: This is a detailed abstract with numbers like 45% efficiency and $500,000 cost.
""" for i in range(1, 16)
]) + """

Return JSON with this structure:
{
  "hypothesis_title": "Your hypothesis title here",
  "hypothesis": {
    "main_claim": "Your main claim with specific numbers"
  },
  "methodology": [
    {"step_number": 1, "step_name": "First step"}
  ]
}

Return ONLY valid JSON, no markdown."""

print(f"Prompt length: {len(long_prompt)} characters")

try:
    model = genai.GenerativeModel(
        'gemini-2.0-flash',
        generation_config={
            'temperature': 0.4,
            'max_output_tokens': 8192,
            'response_mime_type': 'application/json'
        }
    )
    
    start = time.time()
    response = model.generate_content(long_prompt)
    elapsed = time.time() - start
    
    print(f"✓ Long prompt SUCCESS in {elapsed:.1f}s")
    print(f"  Response length: {len(response.text)} chars")
    print(f"  Response preview: {response.text[:500]}...")
    
    try:
        parsed = json.loads(response.text)
        print(f"  ✓ JSON parsed successfully")
        print(f"  Keys: {list(parsed.keys())}")
    except json.JSONDecodeError as je:
        print(f"  ✗ JSON PARSE FAILED at position {je.pos}: {je.msg}")
        print(f"  Content around error: {response.text[max(0, je.pos-50):je.pos+50]}")
        
except Exception as e:
    print(f"✗ Long prompt FAILED: {type(e).__name__}")
    print(f"  Full error: {str(e)[:500]}")

print("\n" + "="*80)  
print("STEP 4: Test with fallback models")
print("="*80)

models_to_try = ["gemini-2.0-flash", "gemini-2.0-flash-exp", "gemini-2.0-flash-lite"]

for model_name in models_to_try:
    print(f"\nTrying {model_name}...")
    try:
        model = genai.GenerativeModel(
            model_name,
            generation_config={
                'temperature': 0.4,
                'response_mime_type': 'application/json'
            }
        )
        response = model.generate_content('Return JSON: {"test": true}')
        print(f"  ✓ {model_name} works! Response: {response.text[:100]}")
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "quota" in error_str.lower():
            print(f"  ✗ {model_name} - RATE LIMITED (429)")
        else:
            print(f"  ✗ {model_name} - {type(e).__name__}: {str(e)[:100]}")

print("\n" + "="*80)
print("DONE - Check output above to find root cause")
print("="*80)
