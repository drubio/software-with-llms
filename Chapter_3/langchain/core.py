"""
LLM Tester - Interactive CLI tool to test different LLM providers
This script lets you interactively query different LLM providers and compare their responses.
"""

import os
import json
from typing import Dict, Any, Optional, List, Union
from dotenv import load_dotenv
import anthropic
import openai
from google.generativeai import configure, GenerativeModel

# Load environment variables from .env file
load_dotenv()

class LLMProviderInterface:
    """Interface for LLM providers."""
    def call(self, prompt: str, model: str, max_tokens: int, temperature: float) -> str:
        """Calls the LLM provider."""
        raise NotImplementedError

class AnthropicProvider(LLMProviderInterface):
    """Handles Anthropic Claude API calls."""
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def call(self, prompt: str, model: str, max_tokens: int, temperature: float) -> str:
        message = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

class OpenAIProvider(LLMProviderInterface):
    """Handles OpenAI API calls."""
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)

    def call(self, prompt: str, model: str, max_tokens: int, temperature: float) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content

class GoogleProvider(LLMProviderInterface):
    """Handles Google Gemini API calls."""
    def __init__(self, api_key: str):
        configure(api_key=api_key)

    def call(self, prompt: str, model_name: str, max_tokens: int, temperature: float) -> str:
        model = GenerativeModel(model_name=model_name)
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature
            }
        )
        return response.text

class XAIProvider(LLMProviderInterface):
    """Handles xAI Grok API calls."""
    def __init__(self, api_key: str):
        # xAI uses OpenAI-compatible API
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )

    def call(self, prompt: str, model: str, max_tokens: int, temperature: float) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are Grok, a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content

class LLMManager:
    """A modular manager for interacting with various LLM providers."""

    def __init__(self):
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.xai_api_key = os.getenv("XAI_API_KEY")        

        self.available_providers = {}
        self.initialization_messages = {}

        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Initialize LLM providers based on API keys."""
        if self.anthropic_api_key:
            try:
                self.available_providers["anthropic"] = AnthropicProvider(self.anthropic_api_key)
                self.initialization_messages["anthropic"] = "✓ Anthropic Claude initialized successfully"
            except Exception as e:
                self.initialization_messages["anthropic"] = f"✗ Failed to initialize Anthropic client: {str(e)}"
        else:
            self.initialization_messages["anthropic"] = "✗ Anthropic API key not found in .env file"

        if self.openai_api_key:
            try:
                self.available_providers["openai"] = OpenAIProvider(self.openai_api_key)
                self.initialization_messages["openai"] = "✓ OpenAI client initialized successfully"
            except Exception as e:
                self.initialization_messages["openai"] = f"✗ Failed to initialize OpenAI client: {str(e)}"
        else:
            self.initialization_messages["openai"] = "✗ OpenAI API key not found in .env file"

        if self.google_api_key:
            try:
                self.available_providers["google"] = GoogleProvider(self.google_api_key)
                self.initialization_messages["google"] = "✓ Google Gemini initialized successfully"
            except Exception as e:
                self.initialization_messages["google"] = f"✗ Failed to initialize Google client: {str(e)}"
        else:
            self.initialization_messages["google"] = "✗ Google API key not found in .env file"

        if self.xai_api_key:
            try:
                self.available_providers["xai"] = XAIProvider(self.xai_api_key)
                self.initialization_messages["xai"] = "✓ xAI Grok initialized successfully"
            except Exception as e:
                self.initialization_messages["xai"] = f"✗ Failed to initialize xAI client: {str(e)}"
        else:
            self.initialization_messages["xai"] = "✗ xAI API key not found in .env file"


        if not self.available_providers:
            print("Warning: No LLM providers initialized. Make sure API keys are in your .env file.")

    def display_initialization_status(self) -> None:
        """Display the initialization status of all LLM providers."""
        print("\n=== LLM Providers Status ===")
        for provider, message in self.initialization_messages.items():
            print(f"{provider.capitalize()}: {message}")
        print("=" * 30 + "\n")

    def get_available_providers(self) -> List[str]:
        """Return list of available LLM providers."""
        return list(self.available_providers.keys())

    def ask_question(self, topic: str, provider: str = None, model: str = None, template: str = "{topic}", max_tokens: int = 1000, temperature: float = 0.7) -> Dict[str, Any]:
        """Ask a question to the selected LLM provider using a template string."""
        prompt = template.format(topic=topic)

        if not provider or provider not in self.available_providers:
            if not self.available_providers:
                return {"success": False, "error": "No LLM providers available. Check your API keys.", "provider": None, "model": None, "prompt": prompt, "response": None}
            provider = list(self.available_providers.keys())[0]
            print(f"Using default provider: {provider}")

        model = model or self._get_default_model(provider)

        try:
            result = self.available_providers[provider].call(prompt, model, max_tokens, temperature)
            return {"success": True, "provider": provider, "model": model, "prompt": prompt, "response": result}
        except Exception as e:
            return {"success": False, "error": str(e), "provider": provider, "model": model, "prompt": prompt, "response": None}

    def _get_default_model(self, provider: str) -> str:
        """Get the default model for a provider."""
        defaults = {
            "anthropic": "claude-3-5-sonnet-20241022", 
            "openai": "gpt-4o", 
            "google": "gemini-2.0-flash",
            "xai": "grok-3"
        }
        return defaults.get(provider, "")

    def query_all_providers(self, topic: str, template: str = "{topic}", max_tokens: int = 1000, temperature: float = 0.7) -> Dict[str, Any]:
        """Query all available LLM providers with the same question."""
        if not self.available_providers:
            return {"success": False, "error": "No LLM providers available. Check your API keys.", "prompt": template.format(topic=topic), "responses": {}}

        responses = {}
        for provider in self.available_providers:
            print(f"Querying {provider}...")
            result = self.ask_question(topic=topic, provider=provider, template=template, max_tokens=max_tokens, temperature=temperature)
            responses[provider] = result

        return {"success": True, "prompt": template.format(topic=topic), "responses": responses}

def save_response_to_file(response: Dict[str, Any], filename: str) -> None:
    """Save response data to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(response, f, indent=2)
    print(f"Response saved to {filename}")

def display_provider_response(provider: str, response: Dict[str, Any]) -> None:
    """Display a provider's response with appropriate formatting."""
    provider_names = {
        "anthropic": "Anthropic Claude answered:", 
        "openai": "OpenAI GPT answered:", 
        "google": "Google Gemini answered:",
        "xai": "xAI Grok answered:"
    }
    provider_display = provider_names.get(provider, f"{provider.capitalize()} knows:")

    print(f"\n=== {provider_display} ===")
    if response["success"]:
        print(response["response"])
    else:
        print(f"Error: {response['error']}")
    print("=" * 50)

def interactive_cli():
    """Run the interactive command-line interface."""
    print("=" * 50)
    print("Welcome to the LLM Tester!")
    print("=" * 50)

    llm_manager = LLMManager()
    llm_manager.display_initialization_status()

    available_providers = llm_manager.get_available_providers()

    if not available_providers:
        print("No LLM providers available. Please check your .env file and API keys.")
        return

    question = input("What topic do you want to ask the LLM about? ")

    if len(available_providers) > 1:
        print("\nAvailable providers:", ", ".join(available_providers))
        query_all = input("Do you want to ask ALL available LLMs or just one? (all/one): ").lower()

        if query_all in ["all", "a", ""]:
            results = llm_manager.query_all_providers(question)
            if results["success"]:
                for provider, response in results["responses"].items():
                    display_provider_response(provider, response)
            else:
                print(f"Error: {results['error']}")

            save_option = input("\nDo you want to save these results to a file? (y/n): ").lower()
            if save_option in ["y", "yes"]:
                filename = f"llm_responses_{question[:20].replace(' ', '_')}.json"
                save_response_to_file(results, filename)
        else:
            print("\nAvailable providers:")
            for i, provider in enumerate(available_providers, 1):
                print(f"{i}. {provider.capitalize()}")

            choice = input(f"Select a provider (1-{len(available_providers)}): ")
            try:
                index = int(choice) - 1
                if 0 <= index < len(available_providers):
                    provider = available_providers[index]
                    result = llm_manager.ask_question(question, provider=provider)
                    display_provider_response(provider, result)
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    else:
        provider = available_providers[0]
        print(f"\nOnly one provider available: {provider}")
        result = llm_manager.ask_question(question, provider=provider)
        display_provider_response(provider, result)

    print("\nThank you for using the LLM Tester!")

if __name__ == "__main__":
    interactive_cli()
