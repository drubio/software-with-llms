"""
web.py - Clean web interface for LLM testers
Now supports optional memory endpoints and session-based tracking.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import sys
import io
from contextlib import redirect_stdout


# -------------------------
# Pydantic request schemas
# -------------------------

class QueryRequest(BaseModel):
    topic: str
    provider: Optional[str] = None
    template: str = "{topic}"
    max_tokens: int = 1000
    temperature: float = 0.7
    session_id: Optional[str] = "default"


class QueryAllRequest(BaseModel):
    topic: str
    template: str = "{topic}"
    max_tokens: int = 1000
    temperature: float = 0.7
    session_id: Optional[str] = "default"


# -------------------------
# Web API generator
# -------------------------

def create_web_api(manager_class):
    app = FastAPI(
        title="LLM Service API",
        version="1.0.0",
        description="Universal API for LLM framework testing"
    )

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize manager
    manager = manager_class()

    # -------------------------
    # Routes
    # -------------------------

    @app.get("/")
    async def get_status():
        return {
            "framework": manager.framework,
            "available_providers": manager.get_available_providers(),
            "total_available": len(manager.get_available_providers()),
            "initialization_status": manager.initialization_messages,
            "status": "healthy" if manager.get_available_providers() else "no_providers"
        }

    @app.get("/providers")
    async def get_providers():
        from utils import get_display_name, get_default_model
        providers = manager.get_available_providers()
        return {
            "framework": manager.framework,
            "providers": [
                {
                    "name": p,
                    "display_name": get_display_name(p),
                    "model": get_default_model(p),
                    "status": manager.initialization_messages.get(p, "Unknown")
                } for p in providers
            ]
        }

    @app.post("/query")
    async def query_single(request: QueryRequest):
        try:
            with redirect_stdout(io.StringIO()) as f:
                args = {
                    "topic": request.topic,
                    "provider": request.provider,
                    "template": request.template,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature
                }
                # Conditionally pass session_id if manager supports it
                if hasattr(manager, "memory_enabled") and manager.memory_enabled:
                    args["session_id"] = request.session_id

                result = manager.ask_question(**args)

            if not result.get("success"):
                raise HTTPException(status_code=400, detail=result.get("error", "Query failed"))

            # Clean response
            raw = result.get("response")
            content = raw.content if hasattr(raw, "content") else raw

            return {
                "success": True,
                "framework": manager.framework,
                "provider": result["provider"],
                "model": result["model"],
                "response": content,
                "parameters": {
                    "temperature": result["temperature"],
                    "max_tokens": result["max_tokens"],
                    "template": request.template
                },
                "prompt": result["prompt"],
                "session_id": result.get("session_id", "default")
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/query-all")
    async def query_all(request: QueryAllRequest):
        try:
            with redirect_stdout(io.StringIO()) as f:
                result = manager.query_all_providers(
                    topic=request.topic,
                    template=request.template,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                )

            if not result.get("success"):
                raise HTTPException(status_code=400, detail=result.get("error", "Query failed"))

            clean_responses = {}
            for provider, res in result["responses"].items():
                raw = res.get("response")
                content = raw.content if hasattr(raw, "content") else raw
                clean_responses[provider] = {
                    "success": res["success"],
                    "model": res.get("model", ""),
                    "response": content,
                    "parameters": {
                        "temperature": res.get("temperature"),
                        "max_tokens": res.get("max_tokens")
                    }
                }

            return {
                "success": True,
                "framework": manager.framework,
                "prompt": result["prompt"],
                "responses": clean_responses
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    if (
            hasattr(manager, "memory_enabled")
            and manager.memory_enabled
            and hasattr(manager, "get_history")
            and hasattr(manager, "reset_memory")
    ):        
        @app.get("/history")
        async def get_history(provider: str, session_id: str = "default"):
            return manager.get_history(provider, session_id)
        
        @app.post("/reset-memory")
        async def reset_memory(provider: Optional[str] = None, session_id: Optional[str] = None):
            return manager.reset_memory(provider, session_id)

    return app


# -------------------------
# Entrypoint
# -------------------------

def run_web_server(manager_class, host: str = "0.0.0.0", port: int = 8000):
    app = create_web_api(manager_class)
    try:
        framework_name = manager_class().framework
    except:
        framework_name = "Unknown"

    print(f"Starting web server for {framework_name}")
    print(f"Docs: http://{host}:{port}/docs")
    print(f"Health: http://{host}:{port}/")
    uvicorn.run(app, host=host, port=port)


def main():
    print("Universal LLM Web API")
    print("Run using `run_web_server(manager_class)`")


if __name__ == "__main__":
    main()
