"""
LLM Tester - LlamaIndex Framework Implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini
from llama_index.core.llms import ChatMessage

from utils import get_api_key, get_default_model, BaseLLMManager, interactive_cli

class LlamaIndexLLMManager(BaseLLMManager):
    """LlamaIndex implementation - only LlamaIndex-specific code"""
    
    def __init__(self):
        super().__init__("LlamaIndex")
    
    def _test_provider(self, provider: str):
        """Test LlamaIndex provider initialization"""
        self._create_client(provider, temperature=0.7, max_tokens=1000)
    
    def _create_client(self, provider: str, temperature: float, max_tokens: int):
        """Create LlamaIndex client - the only LlamaIndex-specific logic"""
        
        if provider == "anthropic":
            return Anthropic(
                api_key=get_api_key(provider),
                model=get_default_model(provider),
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        elif provider == "openai":
            return OpenAI(
                api_key=get_api_key(provider),
                model=get_default_model(provider),
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        elif provider == "google":
            return Gemini(
                api_key=get_api_key(provider),
                model=get_default_model(provider),
                temperature=temperature,
                max_output_tokens=max_tokens  # Gemini's parameter name
            )
        
        elif provider == "xai":
            return OpenAI(
                api_key=get_api_key(provider),
                api_base="https://api.x.ai/v1",
                model=get_default_model(provider),
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def ask_question(self, topic: str, provider: str = None, template: str = "{topic}", 
                     max_tokens: int = 1000, temperature: float = 0.7) -> Dict:
        """LlamaIndex-specific question asking"""
        
        prompt = template.format(topic=topic)
        
        # Use first available provider if none specified
        available_providers = self.get_available_providers()
        if not provider or provider not in available_providers:
            if not available_providers:
                return {
                    "success": False,
                    "error": "No providers available",
                    "provider": "none",
                    "model": "none",
                    "prompt": prompt,
                    "response": None
                }
            provider = available_providers[0]
        
        model = get_default_model(provider)
        
        try:
            # Check if we're in web mode (no sys.stdout) to avoid print statements
            web_mode = hasattr(sys.stdout, 'getvalue')  # StringIO has getvalue
            
            if not web_mode:
                print(f"Creating LlamaIndex client for {provider} (temp={temperature}, max_tokens={max_tokens})")
            
            # LlamaIndex-specific: Create client
            client = self._create_client(provider, temperature=temperature, max_tokens=max_tokens)
            
            if not web_mode:
                print(f"Making LlamaIndex call to {provider}...")
            
            # LlamaIndex-specific: Make the call
            # LlamaIndex supports both complete() and chat() methods
            # Use chat() for better conversation handling
            if hasattr(client, 'chat'):
                messages = [ChatMessage(role="user", content=prompt)]
                response = client.chat(messages)
                result = response.message.content
            else:
                # Fallback to complete() for models that don't support chat
                response = client.complete(prompt)
                result = str(response)
            
            if not web_mode:
                print(f"LlamaIndex call completed for {provider}")
            
            return {
                "success": True,
                "provider": provider,
                "model": model,
                "prompt": prompt,
                "response": result,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
        except Exception as e:
            return {
                "success": False,
                "provider": provider,
                "model": model,
                "prompt": prompt,
                "error": str(e),
                "response": None,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        try:
            from web import run_web_server
            run_web_server(LlamaIndexLLMManager)
        except ImportError:
            print("Error: web.py not found or FastAPI not installed.")
            print("Install FastAPI: pip install fastapi uvicorn")
            sys.exit(1)
    else:
        # CLI mode - all generic logic is in utils.interactive_cli()
        manager = LlamaIndexLLMManager()
        interactive_cli(manager)

if __name__ == "__main__":
    main()
