import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from uchi.simple import Core
import logging

_router = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _router
    _router = Core()
    yield
    if _router is not None:
        pass



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


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str


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
            reply = normalize(_router.ask(msg) or "")

        return ChatResponse(reply=reply)

    except Exception as e:
        import traceback
        traceback.print_exc()
        logging.error(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: AskRequest):
    """Ask Uchi a question (mirrors the SDK's `Uchi.ask`).

    A `/name args` query is dispatched to the skill registry; anything else goes
    through FLUX (Proposer) + Uchi (Verifier). Returns the grounded answer, or an
    honest abstention when it cannot be grounded. Same contract as the SDK & TUI.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        from uchi.response_normalizer import normalize
        q = request.query.strip()
        if q.startswith("/"):
            parts = q[1:].split(None, 1)
            name = parts[0]
            args = parts[1] if len(parts) > 1 else ""
            answer = normalize(_router.skills.dispatch(name, args) or "")
        else:
            answer = normalize(_router.ask(q) or "")
        return AskResponse(answer=answer)
    except Exception as e:
        import traceback
        traceback.print_exc()
        logging.error(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/stream")
async def ask_stream_endpoint(request: AskRequest):
    """SSE endpoint streaming Uchi's Observable Monologue (0.4.0 Item
    16.9): ``thought`` events as the pipeline actually produces them,
    followed by one final ``speech`` event — Thought-vs-Speech
    separation over Server-Sent Events, masking perceived latency by
    showing the user what Uchi is doing rather than a blank wait.
    """
    if _router is None:
        raise HTTPException(status_code=503, detail="Uchi is still starting up")
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    def event_generator():
        from uchi.streaming import stream_ask
        for event in stream_ask(_router, request.query.strip()):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


class OpenAIMessage(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    model: str = "uchi"
    messages: List[OpenAIMessage]
    stream: bool = False
    temperature: float = 0.0


class OpenAIChoice(BaseModel):
    index: int
    message: OpenAIMessage
    finish_reason: str = "stop"


class OpenAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[OpenAIChoice]
    usage: OpenAIUsage = OpenAIUsage()


def _build_conversation_context(messages: List["OpenAIMessage"]) -> str:
    """Format prior turns as conversation history, reusing EpisodicMemory's
    own formatting (via a throwaway instance) rather than duplicating it.
    Pairs up consecutive user/assistant messages; a dangling, unanswered
    user message at the end (no assistant reply yet in *messages*) is
    dropped, since get_context_string only ever renders complete turns.
    """
    from uchi.episodic_memory import EpisodicMemory
    scratch = EpisodicMemory(max_history=max(1, len(messages)))
    pending_user = None
    for m in messages:
        if m.role == "user":
            pending_user = m.content
        elif m.role == "assistant" and pending_user is not None:
            scratch.add_interaction(pending_user, m.content)
            pending_user = None
    return scratch.get_context_string(n_turns=len(scratch.history))


@app.post("/v1/chat/completions", response_model=OpenAIChatResponse)
async def openai_chat_completions(request: OpenAIChatRequest):
    """OpenAI-compatible chat completions endpoint (0.4.0 Item 16.1).

    Lets Uchi plug directly into off-the-shelf OpenAI-API-compatible
    frontends (Open-WebUI, etc.) with zero custom UI code. Streaming
    (``stream=True``) isn't implemented on this endpoint — see the SSE
    endpoint (Item 16.9) for streamed responses.

    0.4.0 Item 17: the client's full ``messages`` history is threaded
    through as conversation context — previously everything but the last
    message was silently discarded. Built per-request from the client's
    own supplied messages, never through ``_router.episodic_memory``:
    ``_router`` is one global ``Core`` instance shared by every caller of
    this server, so folding history through its shared memory would mix
    different clients' conversations together.
    """
    if _router is None:
        raise HTTPException(status_code=503, detail="Uchi is still starting up")

    user_indices = [i for i, m in enumerate(request.messages) if m.role == "user"]
    if not user_indices:
        raise HTTPException(status_code=400, detail="At least one user message is required")
    last_user_idx = user_indices[-1]
    question = request.messages[last_user_idx].content
    conversation_context = _build_conversation_context(request.messages[:last_user_idx])

    try:
        # ask() already normalizes its own output — don't double-normalize.
        answer = _router.ask(question, conversation_context=conversation_context) or ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return OpenAIChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=request.model,
        choices=[OpenAIChoice(index=0, message=OpenAIMessage(role="assistant", content=answer))],
        usage=OpenAIUsage(
            prompt_tokens=len(question.split()),
            completion_tokens=len(answer.split()),
            total_tokens=len(question.split()) + len(answer.split()),
        ),
    )


@app.get("/health")
async def health_endpoint():
    """Liveness probe."""
    return {"status": "ok", "ready": _router is not None}


@app.post("/skill/{name}", response_model=SkillResponse)
async def skill_endpoint(name: str, request: ChatRequest):
    """Invoke a named skill directly."""
    if not _router.skills.has(name):
        raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")
    reply = _router.skills.dispatch(name, request.message)
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
    return {
        "status": "online",
        "memory_records": 0,
        "indexed_passages": len(_router.index.passages),
        "skills_loaded": len(_router.skills.list_skills()),
        "mode": "grounded",
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

    _router.learn(raw_text)
    return BootstrapResponse(tokens_ingested=len(raw_text.split()), source=source)


@app.get("/debug/walk")
async def debug_walk_endpoint():
    if not hasattr(_router, "last_walk_data"):
        return {"error": "No walk data available"}
    return _router.last_walk_data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
