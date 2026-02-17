"""
OpenRouter.ai API Client Module

A simple wrapper for connecting to OpenRouter.ai using the OpenAI client library.

Usage:
    from src.openrouter_client import OpenRouterClient
    
    client = OpenRouterClient(
        api_key="your-api-key",
        model="your-model-name-here",
        site_url="https://your-site.com",  # Optional
        site_name="Your Site Name"  # Optional
    )
    
    response = client.chat("What is the meaning of life?")
    print(response)
"""

from openai import OpenAI
from typing import Optional, List, Dict, Any


class OpenRouterClient:
    """Client for OpenRouter.ai API using OpenAI-compatible interface."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "your-model-name-here",
        base_url: str = "https://openrouter.ai/api/v1",
        site_url: Optional[str] = None,
        site_name: Optional[str] = None
    ):
        """
        Initialize OpenRouter.ai client.
        
        Args:
            api_key: Your OpenRouter.ai API key
            model: Model to use (must be specified)
            base_url: OpenRouter.ai base URL
            site_url: Optional site URL for rankings on openrouter.ai
            site_name: Optional site name for rankings on openrouter.ai
        """
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = model
        self.site_url = site_url
        self.site_name = site_name
        
        # Prepare extra headers
        self.extra_headers = {}
        if site_url:
            self.extra_headers["HTTP-Referer"] = site_url
        if site_name:
            self.extra_headers["X-Title"] = site_name
    
    def chat(
        self,
        message: str,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Send a chat message and get a response.
        
        Args:
            message: The user message (if messages is None)
            messages: Optional list of message dicts with 'role' and 'content'
            max_tokens: Optional max tokens for response
            temperature: Optional temperature (0.0-2.0)
            **kwargs: Additional parameters for chat.completions.create
        
        Returns:
            The assistant's response text
        """
        if messages is None:
            messages = [
                {
                    "role": "user",
                    "content": message
                }
            ]
        
        params = {
            "model": self.model,
            "messages": messages,
        }
        
        if self.extra_headers:
            params["extra_headers"] = self.extra_headers
        
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        
        if temperature is not None:
            params["temperature"] = temperature
        
        # Add any additional kwargs
        params.update(kwargs)
        
        completion = self.client.chat.completions.create(**params)
        return completion.choices[0].message.content
    
    def chat_with_history(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Send a conversation with history.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Optional max tokens for response
            temperature: Optional temperature (0.0-2.0)
            **kwargs: Additional parameters
        
        Returns:
            The assistant's response text
        """
        return self.chat(
            message="",  # Not used when messages is provided
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )


# Convenience function for quick usage
def create_client(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> OpenRouterClient:
    """
    Create an OpenRouterClient with optional API key from environment.
    
    Args:
        api_key: API key (if None, tries to get from environment)
        model: Model to use
        **kwargs: Additional arguments for OpenRouterClient
    
    Returns:
        OpenRouterClient instance
    """
    import os
    
    if api_key is None:
        api_key = os.getenv('API_KEY', '')
        if not api_key:
            raise ValueError("API key must be provided or set in API_KEY environment variable")
    
    if model is None:
        model = os.getenv('MODEL', '')
        if not model:
            raise ValueError("Model must be provided or set in MODEL environment variable")
    
    return OpenRouterClient(api_key=api_key, model=model, **kwargs)


if __name__ == "__main__":
    # Example usage
    import os
    
    api_key = os.getenv('API_KEY', '')
    model = os.getenv('MODEL', 'your-model-name-here')
    
    if not api_key:
        print("Please set API_KEY environment variable")
        exit(1)
    
    if model == 'your-model-name-here':
        print("Please set MODEL environment variable")
        exit(1)
    
    client = OpenRouterClient(
        api_key=api_key,
        model=model
    )
    
    print("Testing OpenRouter.ai connection...")
    response = client.chat("What is the meaning of life?")
    print("\nResponse:")
    print(response)

