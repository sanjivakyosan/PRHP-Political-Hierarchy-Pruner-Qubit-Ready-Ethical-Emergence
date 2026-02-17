#!/bin/bash

# Setup script for OpenRouter.ai API configuration
# This script sets environment variables for the PRHP framework

# OpenRouter.ai API Key
# IMPORTANT: Replace with your own API key from https://openrouter.ai
export API_KEY="your-api-key-here"

# AI Provider
export AI_PROVIDER="openrouter"

# OpenRouter.ai Base URL
export BASE_URL="https://openrouter.ai/api/v1"

# Model to use (change to your preferred model)
export MODEL="your-model-name-here"

# Optional: For rankings on openrouter.ai (set these if you have a site)
# export SITE_URL="https://your-site.com"
# export SITE_NAME="Your Site Name"

# API Parameters
export MAX_TOKENS=1000
export TEMPERATURE=0.7

echo "✅ OpenRouter.ai environment variables configured!"
echo ""
echo "Configuration:"
echo "  Provider: $AI_PROVIDER"
echo "  Base URL: $BASE_URL"
echo "  Model: $MODEL"
echo "  API Key: ${API_KEY:0:20}... (hidden)"
echo ""
echo "To use these settings, run:"
echo "  source setup_openrouter.sh"
echo ""
echo "Or to run the Flask app with these settings:"
echo "  source setup_openrouter.sh && python app.py"

