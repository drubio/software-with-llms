"""
LLM Memory History Gateway - LlamaIndex with persistent session memory
"""

import sys
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'chapter_4')))

from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini
from llama_index.core.chat_engine.types import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store.simple_chat_store import SimpleChatStore

from utils import get_api_key, get_default_model, BaseLLMManager, interactive_cli


class LlamaIndexLLMManager(BaseLLMManager):
    """LlamaIndex implementation with persistent session memory"""

    def __init__(self, memory_enabled: bool = True):
        self.memory_enabled = memory_enabled
        self.clients: Dict[Tuple[str, str], Tuple] = {}
        self.histories: Dict[Tuple[str, str], ChatMemoryBuffer] = {}
        super().__init__("LlamaIndex+History")

    def _session_file_path(self, provider: str, session_id: str) -> Path:
        path = Path("sessions")
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{provider}__{session_id}.json"

    def _get_history(self, provider: str, session_id: str) -> ChatMemoryBuffer:
        key = (provider, session_id)
        if key not in self.histories:
            file_path = self._session_file_path(provider, session_id)
            store = (SimpleChatStore.from_persist_path(str(file_path))
                     if file_path.exists()
                     else SimpleChatStore())
            self.histories[key] = ChatMemoryBuffer.from_defaults(chat_store=store, chat_store_key=session_id)
        return self.histories[key]

    def _test_provider(self, provider: str):
        self._get_client(provider, "test-session", temperature=0.7, max_tokens=1000)

    def _create_client(self, provider: str, temperature: float, max_tokens: int):
        if provider == "anthropic":
            return Anthropic(api_key=get_api_key(provider),
                             model=get_default_model(provider),
                             temperature=temperature,
                             max_tokens=max_tokens)
        elif provider == "openai":
            return OpenAI(api_key=get_api_key(provider),
                          model=get_default_model(provider),
                          temperature=temperature,
                          max_tokens=max_tokens)
        elif provider == "google":
            return Gemini(api_key=get_api_key(provider),
                          model=get_default_model(provider),
                          temperature=temperature,
                          max_output_tokens=max_tokens)
        elif provider == "xai":
            return OpenAI(api_key=get_api_key(provider),
                          api_base="https://api.x.ai/v1",
                          model=get_default_model(provider),
                          temperature=temperature,
                          max_tokens=max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _get_client(self, provider: str, session_id: str, temperature: float, max_tokens: int):
        key = (provider, session_id)
        if key not in self.clients:
            client = self._create_client(provider, temperature, max_tokens)
            memory = self._get_history(provider, session_id) if self.memory_enabled else None
            self.clients[key] = (client, memory)
        return self.clients[key]

    def ask_question(self, topic: str, provider: Optional[str] = None,
                     template: str = "{topic}", max_tokens: int = 1000,
                     temperature: float = 0.7, session_id: str = "default") -> Dict:

        prompt = template.format(topic=topic)
        available = self.get_available_providers()
        if not provider or provider not in available:
            if not available:
                return {"success": False, "error": "No providers"}
            provider = available[0]

        model = get_default_model(provider)
        client, memory = self._get_client(provider, session_id, temperature, max_tokens)

        try:
            if self.memory_enabled and memory:
                memory.put_messages([ChatMessage(role="user", content=prompt)])
                response = client.chat([ChatMessage(role="user", content=prompt)])
                assistant_content = response.message.content
                memory.put_messages([ChatMessage(role="assistant", content=assistant_content)])

                # Persist session
                file_path = self._session_file_path(provider, session_id)
                memory.chat_store.persist(persist_path=str(file_path))
            else:
                response = client.chat([ChatMessage(role="user", content=prompt)]) \
                    if hasattr(client, "chat") else client.complete(prompt)
                assistant_content = response.message.content if hasattr(response, "message") else response

            return {
                "success": True,
                "provider": provider,
                "model": model,
                "prompt": prompt,
                "response": assistant_content,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "session_id": session_id
            }

        except Exception as e:
            return {"success": False, "error": str(e), "provider": provider}

    def get_history(self, provider: str, session_id: str) -> Dict:
        memory = self._get_history(provider, session_id)
        messages = memory.get_all()
        return {
            "provider": provider,
            "session_id": session_id,
            "turns": [{"role": m.role, "content": m.content} for m in messages],
            "count": len(messages)
        }

    def reset_memory(self, provider: Optional[str] = None, session_id: Optional[str] = None) -> Dict:
        removed = []

        if provider and session_id:
            key = (provider, session_id)
            self.clients.pop(key, None)
            self.histories.pop(key, None)
            path = self._session_file_path(provider, session_id)
            path.unlink(missing_ok=True)
            removed.append(key)

        elif provider:
            for key in list(self.histories):
                if key[0] == provider:
                    self.clients.pop(key, None)
                    self.histories.pop(key, None)
                    self._session_file_path(*key).unlink(missing_ok=True)
                    removed.append(key)

        elif session_id:
            for key in list(self.histories):
                if key[1] == session_id:
                    self.clients.pop(key, None)
                    self.histories.pop(key, None)
                    self._session_file_path(*key).unlink(missing_ok=True)
                    removed.append(key)

        else:
            for key in list(self.histories):
                self._session_file_path(*key).unlink(missing_ok=True)
            self.clients.clear()
            self.histories.clear()
            removed = ["ALL"]

        return {
            "status": "cleared",
            "removed_sessions": removed
        }


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        from web import run_web_server
        run_web_server(lambda: LlamaIndexLLMManager(memory_enabled=True))
    else:
        manager = LlamaIndexLLMManager(memory_enabled=True)
        interactive_cli(manager)


if __name__ == "__main__":
    main()
