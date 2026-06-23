from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from uchi.omni_router import OmniRouter
from uchi.cli import load_brain, save_brain
import logging

app = FastAPI(title="Uchi Deterministic API", description="Dual-mode architecture programmatic interface.")
router = load_brain()
if router is None:
    router = OmniRouter(use_bpe=False, memory_window=5)
    save_brain(router)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    entropy: float = 0.0

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Programmatic endpoint that hits the exact same OmniRouter core as the TUI.
    Triggers web search fallbacks, offline dreaming, etc.
    """
    if not request.message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
        
    try:
        reply = router.chat(request.message)
        
        # We can extract entropy if we want, but for now we'll just return the reply
        
        # Save after every chat turn to persist active learning and web searches
        save_brain(router)
        
        return ChatResponse(reply=reply)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        logging.error(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics_endpoint():
    """
    Returns internal state metrics.
    """
    memory_records = len(router.memory.cpu_mem.records) if hasattr(router.memory, 'cpu_mem') else 0
    return {
        "status": "online",
        "memory_records": memory_records,
        "mode": "deterministic"
    }

@app.get("/debug/walk")
async def debug_walk_endpoint():
    """
    Returns the topological walk data from the last prediction.
    """
    if not hasattr(router, 'last_walk_data'):
        return {"error": "No walk data available"}
    return router.last_walk_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
