from fastapi import FastAPI, HTTPException
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
