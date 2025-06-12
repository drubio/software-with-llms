import sys
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'chapter_4')))

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages.ai import AIMessage
from langchain_community.chat_message_histories import FileChatMessageHistory

from utils import get_api_key, get_default_model, BaseLLMManager, interactive_cli


class LangChainLLMManager(BaseLLMManager):
    """LangChain implementation with persistent session memory and structured output"""

    def __init__(self, memory_enabled: bool = True):
        self.memory_enabled = memory_enabled
        self.chains: Dict[Tuple[str, str], RunnableWithMessageHistory] = {}
        self.histories: Dict[Tuple[str, str], FileChatMessageHistory] = {}
        super().__init__("LangChain+History")

    def _session_file_path(self, provider: str, session_id: str) -> Path:
        base = Path(__file__).resolve().parent
        path = base / "sessions"
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{provider}__{session_id}.json"

    def _get_history(self, provider: str, session_id: str) -> FileChatMessageHistory:
        key = (provider, session_id)
        if key not in self.histories:
            file_path = self._session_file_path(provider, session_id)
            self.histories[key] = FileChatMessageHistory(file_path=str(file_path))
        return self.histories[key]

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

            prompt = PromptTemplate.from_template(
                """
                Given the topic below, provide:

                1. A direct factual answer (if possible)
                2. A summary of what the question is about
                3. Relevant keywords

                Respond in the following JSON format:
                {{
                    "answer": "...",
                    "summary": "...",
                    "keywords": ["...", "..."]
                }}

                Topic: {topic}
                """
            )
            parser = JsonOutputParser()

            def to_ai_message(parsed: dict) -> AIMessage:
                raw_text = parsed.get("answer", "")
                return AIMessage(content=raw_text, additional_kwargs={"parsed": parsed})

            pipeline: Runnable = prompt | client | parser | to_ai_message

            chain = RunnableWithMessageHistory(
                pipeline,
                get_session_history=lambda _: history,
                input_messages_key="topic",
                history_messages_key="history"
            )
            self.chains[key] = chain
        return self.chains[key]

    def _test_provider(self, provider: str):
        self._get_chain(provider, "test-session", temperature=0.7, max_tokens=1000)

    def ask_question(self, topic: str, provider: Optional[str] = None,
                     template: str = "{topic}", max_tokens: int = 1000,
                     temperature: float = 0.7, session_id: str = "default") -> Dict:

        available = self.get_available_providers()
        if not provider or provider not in available:
            if not available:
                return {"success": False, "error": "No providers available"}
            provider = available[0]

        model = get_default_model(provider)
        try:
            chain = self._get_chain(provider, session_id, temperature, max_tokens)
            result = chain.invoke({"topic": topic}, config={"configurable": {"session_id": session_id}})

            parsed = result.additional_kwargs.get("parsed", {})
            return {
                "success": True,
                "provider": provider,
                "model": model,
                "prompt": topic,
                "response": parsed,
                "raw_answer": result.content,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "session_id": session_id
            }
        except OutputParserException as e:
            return {
                "success": False,
                "provider": provider,
                "model": model,
                "prompt": topic,
                "error": f"Parsing failed: {str(e)}",
                "response": None,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "session_id": session_id
            }
        except Exception as e:
            return {
                "success": False,
                "provider": provider,
                "model": model,
                "prompt": topic,
                "error": str(e),
                "response": None,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "session_id": session_id
            }

    def get_history(self, provider: str, session_id: str) -> Dict:
        history = self._get_history(provider, session_id)
        messages = history.messages
        turns = []

        for m in messages:
            role = m.type
            content = m.content
            if role == "ai":
                parsed = getattr(m, "additional_kwargs", {}).get("parsed", {})
                answer = parsed.get("answer", content)
                summary = parsed.get("summary")
                keywords = parsed.get("keywords", [])

                combined = answer
                if summary:
                    combined += f"\n📘 Summary: {summary}"
                if keywords:
                    combined += f"\n🔑 Keywords: {', '.join(keywords)}"
                content = combined

            turns.append({"role": role, "content": content})

        return {
            "provider": provider,
            "session_id": session_id,
            "turns": turns,
            "count": len(messages)
        }

    def reset_memory(self, provider: Optional[str] = None, session_id: Optional[str] = None) -> Dict:
        removed = []
        if provider and session_id:
            key = (provider, session_id)
            self.chains.pop(key, None)
            self.histories.pop(key, None)
            self._session_file_path(provider, session_id).unlink(missing_ok=True)
            removed.append(key)
        elif provider:
            for key in list(self.histories):
                if key[0] == provider:
                    self.chains.pop(key, None)
                    self.histories.pop(key, None)
                    self._session_file_path(*key).unlink(missing_ok=True)
                    removed.append(key)
        elif session_id:
            for key in list(self.histories):
                if key[1] == session_id:
                    self.chains.pop(key, None)
                    self.histories.pop(key, None)
                    self._session_file_path(*key).unlink(missing_ok=True)
                    removed.append(key)
        else:
            for key in list(self.histories):
                self._session_file_path(*key).unlink(missing_ok=True)
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
