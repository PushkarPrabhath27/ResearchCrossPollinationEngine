"""Quick test to check what Gemini returns"""
import os
import json
from dotenv import load_dotenv

load_dotenv('.env')

import google.generativeai as genai

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Test with the exact model we're using
print("Testing gemini-2.0-flash with JSON mode...")

model = genai.GenerativeModel(
    'gemini-2.0-flash',
    generation_config={
        'temperature': 0.4,
        'response_mime_type': 'application/json'
    }
)

prompt = """Return a simple JSON object with this exact structure:
{
  "hypothesis_title": "Test Hypothesis",
  "hypothesis": {"main_claim": "This is a test claim"},
  "methodology": [{"step_number": 1, "step_name": "Test step"}]
}"""

print(f"Prompt length: {len(prompt)}")

try:
    response = model.generate_content(prompt)
    print(f"\nResponse type: {type(response.text)}")
    print(f"Response length: {len(response.text)}")
    print(f"\nRaw response:\n{response.text[:1000]}")
    
    # Try to parse
    parsed = json.loads(response.text)
    print(f"\n✅ JSON parsed successfully!")
    print(f"Keys: {list(parsed.keys())}")
    
except Exception as e:
    print(f"\n❌ Error: {type(e).__name__}: {e}")
