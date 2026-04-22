"""Quick test to debug Gemini API response"""
import os
import sys
from dotenv import load_dotenv

# Load environment
load_dotenv('.env')

api_key = os.getenv('GOOGLE_API_KEY')
print(f"API Key present: {bool(api_key)}")

if not api_key:
    print("ERROR: No GOOGLE_API_KEY found in .env!")
    print("Please set GOOGLE_API_KEY in your .env file")
    sys.exit(1)

print(f"API Key (masked): {api_key[:8]}...{api_key[-4:]}")

import google.generativeai as genai

genai.configure(api_key=api_key)

# List available models first
print("\n=== Listing Available Models ===")
try:
    models_list = list(genai.list_models())
    for m in models_list[:10]:
        if 'generateContent' in getattr(m, 'supported_generation_methods', []):
            print(f"  - {m.name}")
except Exception as e:
    print(f"Could not list models: {e}")

# Try different models - VERIFIED Dec 2024/Jan 2025
models_to_try = [
    "gemini-2.0-flash-exp",              # NEW! Best free model (Dec 2024)
    "gemini-1.5-flash-latest",           # Latest flash
    "gemini-1.5-flash",                  # Stable flash
    "gemini-1.5-pro-latest",             # Pro with updates
]

for model_name in models_to_try:
    print(f"\n=== Testing {model_name} ===")
    try:
        model = genai.GenerativeModel(
            model_name,
            generation_config={
                'temperature': 0.4,
                'max_output_tokens': 500
            }
        )
        
        response = model.generate_content('Return a JSON object: {"status": "ok", "value": 42}')
        print(f"SUCCESS! Response: {response.text[:200]}")
        
        # Try to parse
        import json
        parsed = json.loads(response.text)
        print(f"Parsed JSON: {parsed}")
        print(f"*** {model_name} WORKS! ***")
        break
        
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {str(e)[:100]}")
else:
    print("\n!!! ALL MODELS FAILED !!!")
