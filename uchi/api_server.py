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
        _router.stop_background_jobs()
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


class BootstrapRequest(BaseModel):
    text: str | None = None
    url: str | None = None


class BootstrapResponse(BaseModel):
    tokens_ingested: int
    source: str


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
        from uchi.response_normalizer import normalize
        msg = request.message.strip()
        if msg.startswith("/"):
            parts = msg[1:].split(None, 1)
            name = parts[0]
            args = parts[1] if len(parts) > 1 else ""
            reply = normalize(_router.skills.dispatch(name, args) or "")
        else:
            reply = normalize(_router.chat(msg) or "")

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


@app.get("/telemetry")
async def telemetry_endpoint():
    """
    Exposes deep internal engine telemetry for the TUI and Cognitive Debugger.
    Pulls data from the central telemetry singleton if available.
    """
    try:
        import uchi.telemetry as _tel
        return _tel.dump_all()
    except Exception as e:
        return {"error": f"Telemetry not available: {e}"}


@app.post("/bootstrap", response_model=BootstrapResponse)
async def bootstrap_endpoint(request: BootstrapRequest):
    """
    Ingest raw text or a URL into Uchi's trie and AssociativeMemory.

    Accepts JSON body with one of:
      - `{"text": "raw text to learn"}` — streams the text directly
      - `{"url": "https://..."}` — fetches the page, strips HTML, then streams

    Once Uchi has tool-routing, it can call this endpoint autonomously to
    permanently memorise content it discovers via web search.
    """
    if not request.text and not request.url:
        raise HTTPException(status_code=400, detail="Provide either 'text' or 'url'.")

    raw_text = request.text or ""
    source = "text"

    if request.url:
        source = request.url
        try:
            import requests as _req
            from bs4 import BeautifulSoup
            resp = _req.get(request.url, timeout=10, headers={"User-Agent": "Uchi/1.0"})
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            raw_text = soup.get_text(separator=" ", strip=True)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Failed to fetch URL: {exc}")

    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="No usable text found.")

    tokens = _router.tokenizer.tokenize(raw_text.split(), is_inference=False)
    _router.stream(tokens)
    save_brain(_router)
    return BootstrapResponse(tokens_ingested=len(tokens), source=source)


@app.get("/debug/walk")
async def debug_walk_endpoint():
    if not hasattr(_router, "last_walk_data"):
        return {"error": "No walk data available"}
    return _router.last_walk_data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
