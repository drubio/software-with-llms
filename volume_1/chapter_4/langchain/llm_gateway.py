"""
LLM Tester - LangChain Framework Implementation
ONLY LangChain-specific logic, inherits all generic functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from utils import get_api_key, get_default_model, BaseLLMManager, interactive_cli

class LangChainLLMManager(BaseLLMManager):
    """LangChain implementation - only LangChain-specific code"""
    
    def __init__(self):
        super().__init__("LangChain")
    
    def _test_provider(self, provider: str):
        """Test LangChain provider initialization"""
        self._create_client(provider, temperature=0.7, max_tokens=1000)
    
    def _create_client(self, provider: str, temperature: float, max_tokens: int):
        """Create LangChain client - the only LangChain-specific logic"""
        
        if provider == "anthropic":
            return ChatAnthropic(
                anthropic_api_key=get_api_key(provider),
                model=get_default_model(provider),
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        elif provider == "openai":
            return ChatOpenAI(
                openai_api_key=get_api_key(provider),
                model=get_default_model(provider),
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        elif provider == "google":
            return ChatGoogleGenerativeAI(
                google_api_key=get_api_key(provider),
                model=get_default_model(provider),
                temperature=temperature,
                max_output_tokens=max_tokens  # Google's parameter name
            )
        
        elif provider == "xai":
            return ChatOpenAI(
                openai_api_key=get_api_key(provider),
                openai_api_base="https://api.x.ai/v1",
                model=get_default_model(provider),
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def ask_question(self, topic: str, provider: str = None, template: str = "{topic}", 
                     max_tokens: int = 1000, temperature: float = 0.7) -> Dict:
        """LangChain-specific question asking"""
        
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
                print(f"Creating LangChain client for {provider} (temp={temperature}, max_tokens={max_tokens})")
            
            # LangChain-specific: Create client
            client = self._create_client(provider, temperature=temperature, max_tokens=max_tokens)
            
            # LangChain-specific: Create messages
            messages = [
                SystemMessage(content="You are a helpful AI assistant."),
                HumanMessage(content=prompt)
            ]
            
            if not web_mode:
                print(f"Making LangChain invoke() call to {provider}...")
            
            # LangChain-specific: Make the call
            result = client.invoke(messages)
            
            if not web_mode:
                print(f"LangChain call completed for {provider}")
            
            return {
                "success": True,
                "provider": provider,
                "model": model,
                "prompt": prompt,
                "response": str(result.content),
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
            run_web_server(LangChainLLMManager)
        except ImportError:
            print("Error: web.py not found or FastAPI not installed.")
            print("Install FastAPI: pip install fastapi uvicorn")
            sys.exit(1)
    else:
        # CLI mode - all generic logic is in utils.interactive_cli()
        manager = LangChainLLMManager()
        interactive_cli(manager)

if __name__ == "__main__":
    main()
