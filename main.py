# main.py
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any
from src.main_pipeline import MultimodalRAGPipeline
from langfuse import get_client

langfuse = get_client()

# Initialize FastAPI Framework
app = FastAPI(
    title="Production Multimodal RAG API",
    description="Hybrid Search + Reranking + Hallucination Mitigation",
    version="1.0.0"
)

# Initialize our core engine
pipeline = MultimodalRAGPipeline()

# Define strictly typed API payload contracts
class AskRequest(BaseModel):
    query: str
    user_id: str = "anonymous_frontend_user"

class AskResponse(BaseModel):
    answer: str
    confident: bool
    context: List[Dict[str, Any]]

@app.post("/v1/ask", response_model=AskResponse)
async def ask_question(request: AskRequest, background_tasks: BackgroundTasks):
    """
    Primary endpoint for querying the RAG system.
    Runs the synchronous pipeline in a background thread to prevent blocking the async event loop.
    """
    try:
        # Offload the heavy compute and synchronous network calls to a background thread
        result = await asyncio.to_thread(
            pipeline.execute_query, 
            query=request.query, 
            user_id=request.user_id
        )
        
        # Ensure Langfuse telemetry uploads gracefully after the request completes
        background_tasks.add_task(langfuse.flush)

        return AskResponse(
            answer=result["response"],
            confident=result["confident"],
            context=result.get("raw_context_used", [])
        )
        
    except Exception as e:
        print(f"🚨 API Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal RAG System Error")

@app.get("/health")
def health_check():
    return {"status": "operational", "engines": ["qdrant", "bm25", "groq"]}