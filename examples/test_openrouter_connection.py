"""
Test script to connect to OpenRouter.ai API server.

This script tests the connection using the OpenAI client library
configured for OpenRouter.ai.
"""

from openai import OpenAI

# OpenRouter.ai configuration
# IMPORTANT: Replace with your own API key from https://openrouter.ai
import os
api_key = os.getenv('API_KEY', 'your-api-key-here')
if api_key == 'your-api-key-here':
    raise ValueError("Please set API_KEY environment variable or update this script with your API key")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# Optional: Set your site URL and name for rankings on openrouter.ai
# Replace <YOUR_SITE_URL> and <YOUR_SITE_NAME> with your actual values
SITE_URL = ""  # Optional: e.g., "https://your-site.com"
SITE_NAME = ""  # Optional: e.g., "Your Site Name"

# Prepare extra headers (only if site URL/name are provided)
extra_headers = {}
if SITE_URL:
    extra_headers["HTTP-Referer"] = SITE_URL
if SITE_NAME:
    extra_headers["X-Title"] = SITE_NAME

# Model to use (get from environment or use default)
model = os.getenv('MODEL', 'your-model-name-here')
if model == 'your-model-name-here':
    raise ValueError("Please set MODEL environment variable or update this script with your model name")

# Test the connection
print("Testing OpenRouter.ai connection...")
print(f"Model: {model}")
print(f"Base URL: https://openrouter.ai/api/v1")
print("-" * 50)

try:
    completion = client.chat.completions.create(
        extra_headers=extra_headers if extra_headers else None,
        extra_body={},
        model=model,
        messages=[
            {
                "role": "user",
                "content": "What is the meaning of life?"
            }
        ]
    )
    
    print("✅ Connection successful!")
    print("\nResponse:")
    print(completion.choices[0].message.content)
    print("\n" + "-" * 50)
    print("Full response object:")
    print(f"Model used: {completion.model}")
    print(f"Finish reason: {completion.choices[0].finish_reason}")
    
except Exception as e:
    print(f"❌ Connection failed: {str(e)}")
    print("\nTroubleshooting:")
    print("1. Check that your API key is correct")
    print("2. Verify you have credits on OpenRouter.ai")
    print("3. Ensure the model name is correct")
    print("4. Check your internet connection")

