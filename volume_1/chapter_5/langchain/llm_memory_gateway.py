"""
LLM Memory Gateway - LangChain with session-based memory (in-memory only)
"""

import sys
import os
from typing import Dict, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'chapter_4')))

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from utils import get_api_key, get_default_model, BaseLLMManager, interactive_cli


class LangChainLLMManager(BaseLLMManager):
    """LangChain implementation with optional in-memory session support"""

    def __init__(self, memory_enabled: bool = False):
        self.memory_enabled = memory_enabled
        self.chains: Dict[Tuple[str, str], RunnableWithMessageHistory] = {}
        self.histories: Dict[Tuple[str, str], InMemoryChatMessageHistory] = {}
        super().__init__("LangChain+Memory")

    def _get_history(self, provider: str, session_id: str) -> InMemoryChatMessageHistory:
        key = (provider, session_id)
        if key not in self.histories:
            self.histories[key] = InMemoryChatMessageHistory()
        return self.histories[key]

    def _test_provider(self, provider: str):
        if self.memory_enabled:
            self._get_chain(provider, "test-session", temperature=0.7, max_tokens=1000)
        else:
            self._create_client(provider, temperature=0.7, max_tokens=1000)

    def _create_client(self, provider: str, temperature: float, max_tokens: int):
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
                max_output_tokens=max_tokens
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

    def _get_chain(self, provider: str, session_id: str, temperature: float, max_tokens: int):
        key = (provider, session_id)
        if key not in self.chains:
            client = self._create_client(provider, temperature, max_tokens)
            history = self._get_history(provider, session_id)

            chain = RunnableWithMessageHistory(
                RunnableLambda(lambda input: client.invoke([HumanMessage(content=input["input"])])),
                get_session_history=lambda _: history,
                input_messages_key="input",
                history_messages_key="history"
            )
            self.chains[key] = chain
        return self.chains[key]

    def ask_question(self, topic: str, provider: str = None, template: str = "{topic}",
                     max_tokens: int = 1000, temperature: float = 0.7,
                     session_id: str = "default") -> Dict:

        prompt = template.format(topic=topic)
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
            if self.memory_enabled:
                chain = self._get_chain(provider, session_id, temperature, max_tokens)
                result = chain.invoke({"input": prompt}, config={"configurable": {"session_id": session_id}})
                response_text = str(result)
            else:
                client = self._create_client(provider, temperature, max_tokens)
                result = client.invoke([HumanMessage(content=prompt)])
                response_text = str(result.content)

            return {
                "success": True,
                "provider": provider,
                "model": model,
                "prompt": prompt,
                "response": response_text,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "session_id": session_id
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
                "max_tokens": max_tokens,
                "session_id": session_id
            }

    def get_history(self, provider: str, session_id: str) -> Dict:
        history = self._get_history(provider, session_id).messages
        return {
            "provider": provider,
            "session_id": session_id,
            "turns": [{"role": msg.type, "content": msg.content} for msg in history if hasattr(msg, "content")],
            "count": len(history)
        }

    def reset_memory(self, provider: str = None, session_id: str = None) -> Dict:
        removed = []

        if provider and session_id:
            key = (provider, session_id)
            self.chains.pop(key, None)
            self.histories.pop(key, None)
            removed.append(key)

        elif provider:
            for key in list(self.histories.keys()):
                if key[0] == provider:
                    self.chains.pop(key, None)
                    self.histories.pop(key, None)
                    removed.append(key)

        elif session_id:
            for key in list(self.histories.keys()):
                if key[1] == session_id:
                    self.chains.pop(key, None)
                    self.histories.pop(key, None)
                    removed.append(key)

        else:
            self.chains.clear()
            self.histories.clear()
            removed = ["ALL"]

        return {
            "status": "cleared",
            "removed_sessions": removed
        }


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        from web import run_web_server
        run_web_server(lambda: LangChainLLMManager(memory_enabled=True))
    else:
        manager = LangChainLLMManager(memory_enabled=True)
        interactive_cli(manager)


if __name__ == "__main__":
    main()
