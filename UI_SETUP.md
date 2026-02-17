# AI Chat UI Setup Guide

This guide will help you set up and run the local AI Chat Interface.

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The UI supports multiple AI providers. Configure it using environment variables or a `.env` file.

### Option 1: Environment Variables

Set these environment variables before running:

```bash
export AI_PROVIDER=openrouter
export API_KEY=your-api-key-here
export BASE_URL=https://openrouter.ai/api/v1
export MODEL=x-ai/grok-4-fast
export MAX_TOKENS=1000
export TEMPERATURE=0.7
```

### Option 2: .env File (Recommended)

Create a `.env` file in the project root with:

```env
AI_PROVIDER=openrouter
API_KEY=your-api-key-here
BASE_URL=https://openrouter.ai/api/v1
MODEL=x-ai/grok-4-fast
MAX_TOKENS=1000
TEMPERATURE=0.7
SITE_URL=https://your-site.com
SITE_NAME=Your Site Name
```

## Supported Providers

### OpenRouter.ai (Default)

OpenRouter.ai provides access to multiple AI models through a single API.

```env
AI_PROVIDER=openrouter
API_KEY=your-api-key-here
BASE_URL=https://openrouter.ai/api/v1
MODEL=x-ai/grok-4-fast
SITE_URL=https://your-site.com  # Optional, for rankings
SITE_NAME=Your Site Name        # Optional, for rankings
```

**Note:** Replace `your-api-key-here` with your actual API key from OpenRouter.ai.

### OpenAI

```env
AI_PROVIDER=openai
API_KEY=your-openai-api-key-here
API_URL=https://api.openai.com/v1/chat/completions
MODEL=gpt-3.5-turbo
```

### Anthropic (Claude)

```env
AI_PROVIDER=anthropic
API_KEY=your-anthropic-api-key-here
MODEL=claude-3-opus-20240229
```

### Custom API

```env
AI_PROVIDER=custom
API_KEY=your-api-key
API_URL=https://your-api-endpoint.com/chat
MODEL=your-model-name
```

For custom APIs, the backend expects a JSON response with one of these fields:
- `response`
- `text`
- `content`
- `message`

Or it will display the full JSON response.

## Running the UI

Start the Flask server:

```bash
python app.py
```

Then open your browser to:
```
http://localhost:5000
```

## Usage

1. Enter your prompt in the input box
2. Click "Send" or press Enter (Shift+Enter for new line)
3. View the AI response in the output box
4. Use "Clear" to clear the output
5. Use "Clear Input" to clear the input box

## Features

- ✅ Modern, responsive UI
- ✅ Real-time API communication
- ✅ Support for multiple AI providers
- ✅ Error handling and status indicators
- ✅ Keyboard shortcuts (Enter to send, Shift+Enter for new line)
- ✅ Loading states and visual feedback

