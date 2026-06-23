from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from uchi.omni_router import OmniRouter
from uchi.cli import load_brain, save_brain
import logging

_router = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _router
    _router = load_brain()
    if _router is None:
        _router = OmniRouter(use_bpe=False, memory_window=5)
        save_brain(_router)
    # Start background RL daemon and any other background jobs
    _router.start_background_jobs()
    yield
    # Persist on shutdown
    if _router is not None:
        save_brain(_router)


app = FastAPI(
    title="Uchi ODUSP API",
    description="Deterministic Universal Sequence Predictor — programmatic interface.",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    entropy: float = 0.0


class SkillResponse(BaseModel):
    reply: str
    skill: str


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Send a message to the OmniRouter.

    Messages starting with `/name args` are dispatched to the skill registry.
    All other messages go through the standard chat pipeline.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        msg = request.message.strip()
        if msg.startswith("/"):
            parts = msg[1:].split(None, 1)
            name = parts[0]
            args = parts[1] if len(parts) > 1 else ""
            reply = _router.skills.dispatch(name, args)
        else:
            reply = _router.chat(msg)

        save_brain(_router)
        return ChatResponse(reply=reply)

    except Exception as e:
        import traceback
        traceback.print_exc()
        logging.error(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/skill/{name}", response_model=SkillResponse)
async def skill_endpoint(name: str, request: ChatRequest):
    """Invoke a named skill directly."""
    if not _router.skills.has(name):
        raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")
    reply = _router.skills.dispatch(name, request.message)
    save_brain(_router)
    return SkillResponse(reply=reply, skill=name)


@app.get("/skills")
async def list_skills():
    """List all registered skills (built-in + user-installed)."""
    return {
        "skills": [
            {
                "name": s.name,
                "description": s.description,
                "args": s.args_hint,
                "mode": s.mode,
                "source": s.source_path,
            }
            for s in _router.skills.list_skills()
        ]
    }


@app.get("/metrics")
async def metrics_endpoint():
    memory_records = (
        len(_router.memory.cpu_mem.records)
        if hasattr(_router.memory, "cpu_mem")
        else 0
    )
    return {
        "status": "online",
        "memory_records": memory_records,
        "ssm_baseline_mean": round(_router.baseline.mean, 4),
        "skills_loaded": len(_router.skills.list_skills()),
        "mode": "deterministic",
    }


@app.get("/debug/walk")
async def debug_walk_endpoint():
    if not hasattr(_router, "last_walk_data"):
        return {"error": "No walk data available"}
    return _router.last_walk_data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
