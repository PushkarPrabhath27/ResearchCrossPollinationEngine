"""Test gemini-2.5-flash specifically"""
import os
from dotenv import load_dotenv
load_dotenv('.env')

import google.generativeai as genai
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

print("Testing gemini-2.5-flash...")
try:
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content('Return JSON: {"message": "hello"}')
    print(f"SUCCESS: {response.text[:200]}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")

print("\nTesting gemini-2.5-pro...")
try:
    model = genai.GenerativeModel('gemini-2.5-pro')
    response = model.generate_content('Return JSON: {"message": "hello"}')
    print(f"SUCCESS: {response.text[:200]}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
