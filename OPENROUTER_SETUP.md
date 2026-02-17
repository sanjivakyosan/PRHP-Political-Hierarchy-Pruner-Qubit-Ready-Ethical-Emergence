# OpenRouter.ai Connection Setup

## ✅ Connection Status

Your OpenRouter.ai connection has been successfully configured and tested!

**API Key**: Set via `API_KEY` environment variable  
**Model**: Set via `MODEL` environment variable  
**Base URL**: `https://openrouter.ai/api/v1`

## Quick Start

### Option 1: Test Connection (Standalone)

Run the test script:
```bash
python test_openrouter_connection.py
```

### Option 2: Use in Flask App

Set environment variables and run the Flask app:
```bash
source setup_openrouter.sh
python app.py
```

Or set them manually:
```bash
export API_KEY="your-api-key-here"
export AI_PROVIDER="openrouter"
export MODEL="your-model-name-here"
python app.py
```

### Option 3: Use as Python Module

```python
from src.openrouter_client import OpenRouterClient
import os

client = OpenRouterClient(
    api_key=os.getenv('API_KEY', 'your-api-key-here'),
    model=os.getenv('MODEL', 'your-model-name-here')
)

response = client.chat("What is the meaning of life?")
print(response)
```

## Files Created

1. **`test_openrouter_connection.py`** - Standalone test script
2. **`setup_openrouter.sh`** - Environment variable setup script
3. **`src/openrouter_client.py`** - Reusable Python module

## Integration with Existing App

The Flask app (`app.py`) already supports OpenRouter.ai! It will automatically use:
- `API_KEY` environment variable for authentication
- `AI_PROVIDER=openrouter` to use OpenRouter
- `MODEL` environment variable for model selection
- `SITE_URL` and `SITE_NAME` for OpenRouter rankings (optional)

## Example Usage

### Basic Chat
```python
from src.openrouter_client import OpenRouterClient

client = OpenRouterClient(
    api_key="your-api-key",
    model="your-model-name-here"
)

response = client.chat("Hello!")
print(response)
```

### With Conversation History
```python
messages = [
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI is..."},
    {"role": "user", "content": "Tell me more"}
]

response = client.chat_with_history(messages)
print(response)
```

### With Custom Parameters
```python
response = client.chat(
    "Explain quantum computing",
    max_tokens=2000,
    temperature=0.8
)
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_KEY` | Your OpenRouter.ai API key | Required |
| `AI_PROVIDER` | Provider to use | `openrouter` |
| `BASE_URL` | OpenRouter base URL | `https://openrouter.ai/api/v1` |
| `MODEL` | Model to use | Required (no default) |
| `SITE_URL` | Your site URL (optional) | - |
| `SITE_NAME` | Your site name (optional) | - |
| `MAX_TOKENS` | Max tokens in response | `1000` |
| `TEMPERATURE` | Temperature (0.0-2.0) | `0.7` |

### Using the Setup Script

```bash
# Load environment variables
source setup_openrouter.sh

# Now run your app
python app.py
```

## Testing

Test the connection:
```bash
python test_openrouter_connection.py
```

Test the module:
```bash
python src/openrouter_client.py
```

## Troubleshooting

### Connection Failed
1. Verify your API key is correct
2. Check you have credits on OpenRouter.ai
3. Ensure the model name is correct
4. Check your internet connection

### Module Import Error
```bash
pip install openai
```

### Environment Variables Not Working
Make sure to `source` the setup script:
```bash
source setup_openrouter.sh  # Note: use 'source', not './'
```

## Next Steps

1. ✅ Connection tested and working
2. Set up environment variables for Flask app
3. Start using the API in your code
4. (Optional) Configure `SITE_URL` and `SITE_NAME` for rankings

## Security Note

⚠️ **Important**: Never commit your API key to version control. The API key in the setup script is for local use only. For production, use environment variables or a secure secrets manager.

