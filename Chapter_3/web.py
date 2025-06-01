"""
web.py - Clean web interface for LLM testers
Provides clean JSON responses without CLI artifacts
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import sys
import io
from contextlib import redirect_stdout

# Pydantic models for API
class QueryRequest(BaseModel):
    topic: str
    provider: Optional[str] = None
    template: str = "{topic}"
    max_tokens: int = 1000
    temperature: float = 0.7

class QueryAllRequest(BaseModel):
    topic: str
    template: str = "{topic}"
    max_tokens: int = 1000
    temperature: float = 0.7

def create_web_api(manager_class):
    """Create a web API for any LLM manager class"""
    
    app = FastAPI(
        title="LLM Service API", 
        version="1.0.0",
        description="Universal API for LLM framework testing"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize the manager
    manager = manager_class()
    
    @app.get("/")
    async def get_status():
        """Get service status"""
        available_providers = manager.get_available_providers()
        
        return {
            "framework": manager.framework,
            "available_providers": available_providers,
            "total_available": len(available_providers),
            "initialization_status": manager.initialization_messages,
            "status": "healthy" if available_providers else "no_providers"
        }
    
    @app.get("/providers")
    async def get_providers():
        """Get available providers with details"""
        from utils import get_display_name, get_default_model
        
        available_providers = manager.get_available_providers()
        
        providers_detail = []
        for provider in available_providers:
            providers_detail.append({
                "name": provider,
                "display_name": get_display_name(provider),
                "model": get_default_model(provider),
                "status": manager.initialization_messages.get(provider, "Unknown")
            })
        
        return {
            "framework": manager.framework,
            "providers": providers_detail,
            "count": len(available_providers)
        }
    
    @app.post("/query")
    async def query_single(request: QueryRequest):
        """Query a single provider - returns clean JSON"""
        try:
            # Capture any print statements from the manager
            captured_output = io.StringIO()
            
            with redirect_stdout(captured_output):
                result = manager.ask_question(
                    topic=request.topic,
                    provider=request.provider,
                    template=request.template,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                )
            
            # Get any debug output for optional inclusion
            debug_output = captured_output.getvalue().strip()
            
            if not result.get("success", False):
                raise HTTPException(
                    status_code=400, 
                    detail={
                        "error": result.get("error", "Query failed"),
                        "provider": result.get("provider"),
                        "debug": debug_output if debug_output else None
                    }
                )
            
            # Clean response for web
            clean_result = {
                "success": True,
                "framework": manager.framework,
                "provider": result["provider"],
                "model": result["model"],
                "response": result["response"],
                "parameters": {
                    "temperature": result["temperature"],
                    "max_tokens": result["max_tokens"],
                    "template": request.template
                },
                "prompt": result["prompt"]
            }
            
            # Optionally include debug info
            if debug_output:
                clean_result["debug"] = debug_output
            
            return clean_result
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail={
                    "error": str(e),
                    "framework": manager.framework
                }
            )
    
    @app.post("/query-all")
    async def query_all(request: QueryAllRequest):
        """Query all available providers - returns clean JSON"""
        try:
            # Capture any print statements
            captured_output = io.StringIO()
            
            with redirect_stdout(captured_output):
                result = manager.query_all_providers(
                    topic=request.topic,
                    template=request.template,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                )
            
            debug_output = captured_output.getvalue().strip()
            
            if not result.get("success", False):
                raise HTTPException(
                    status_code=400, 
                    detail={
                        "error": result.get("error", "Query failed"),
                        "framework": manager.framework,
                        "debug": debug_output if debug_output else None
                    }
                )
            
            # Clean up the responses for web
            clean_responses = {}
            for provider, response in result["responses"].items():
                if response.get("success"):
                    clean_responses[provider] = {
                        "success": True,
                        "response": response["response"],
                        "model": response["model"],
                        "parameters": {
                            "temperature": response["temperature"],
                            "max_tokens": response["max_tokens"]
                        }
                    }
                else:
                    clean_responses[provider] = {
                        "success": False,
                        "error": response.get("error", "Unknown error"),
                        "model": response.get("model", "unknown")
                    }
            
            clean_result = {
                "success": True,
                "framework": manager.framework,
                "prompt": result["prompt"],
                "responses": clean_responses,
                "summary": {
                    "total_providers": len(result["responses"]),
                    "successful": len([r for r in clean_responses.values() if r["success"]]),
                    "failed": len([r for r in clean_responses.values() if not r["success"]])
                },
                "parameters": {
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "template": request.template
                }
            }
            
            # Optionally include debug info
            if debug_output:
                clean_result["debug"] = debug_output
            
            return clean_result
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail={
                    "error": str(e),
                    "framework": manager.framework
                }
            )
    
    @app.get("/health")
    async def health_check():
        """Simple health check"""
        available_providers = manager.get_available_providers()
        return {
            "status": "healthy" if available_providers else "unhealthy",
            "framework": manager.framework,
            "providers_available": len(available_providers)
        }
    
    return app

def run_web_server(manager_class, host: str = "0.0.0.0", port: int = 8000):
    """Run the web server with the given manager class"""
    app = create_web_api(manager_class)
    
    # Get framework name for display
    framework_name = "Unknown"
    try:
        framework_name = manager_class().framework
    except:
        pass
    
    print(f"Starting web server for {framework_name} framework...")
    print(f"API documentation available at: http://{host}:{port}/docs")
    print(f"Health check: http://{host}:{port}/health")
    print(f"Status: http://{host}:{port}/")
    
    uvicorn.run(app, host=host, port=port)

def main():
    """
    Main function for standalone web server
    Usage: python web.py
    """
    print("Universal LLM Web API")
    print("This should be called from an LLM tester script like:")
    print("python llm_tester.py web")

if __name__ == "__main__":
    main()
