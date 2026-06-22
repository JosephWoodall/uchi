from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from .omni_router import OmniRouter
import os
import pickle

ASCII_LOGO = r"""
      |\_/\_/\_/|
      |         |
      |  O   O  |
      |    ^    |
       \  ___  /
        \_____/
ODUSP Daemon v0.2.0
"""

app = FastAPI(title="ODUSP Brain API")

# Mount UI Harness static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def serve_ui():
    if os.path.exists(os.path.join(STATIC_DIR, "index.html")):
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))
    return {"message": "ODUSP API is running, but UI is missing."}
@app.get("/style.css")
async def serve_css():
    return FileResponse(os.path.join(STATIC_DIR, "style.css"))
@app.get("/app.js")
async def serve_js():
    return FileResponse(os.path.join(STATIC_DIR, "app.js"))

router = None

class StreamRequest(BaseModel):
    tokens: list[str]

class QueryRequest(BaseModel):
    tokens: list[str]

class PredictRequest(BaseModel):
    context: list[str] = []
    steps: int = 5

def load_brain(path: str = "brain.uchi") -> OmniRouter:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return OmniRouter(use_bpe=True, memory_window=5)

@app.on_event("startup")
async def startup_event():
    global router
    print(ASCII_LOGO)
    print("[*] Booting ODUSP daemon...")
    router = load_brain()
    print("[+] Brain loaded and active.")

@app.post("/stream")
async def stream_data(req: StreamRequest):
    if not req.tokens:
        raise HTTPException(status_code=400, detail="Empty token list")
    router.stream(req.tokens)
    return {"status": "success", "processed": len(req.tokens)}

@app.post("/query")
async def query_memory(req: QueryRequest):
    ans = router.query(req.tokens)
    return {"answer": ans}

@app.post("/predict")
async def predict_future(req: PredictRequest):
    pred = router.predict_future(req.context, steps=req.steps)
    return {"prediction": pred}
