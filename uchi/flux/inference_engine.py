"""Inference Engine — High Fidelity / Raw Mode.
Removes all filters to show the model's pure output as requested."""

import time, torch
import torch.nn.functional as F
from typing import Generator, Dict, List, Optional
from .model import HybridTSSM
from .tokenizer_v2 import TikTokenHybridTokenizer as HybridTokenizer
from .prompt_cache import PromptCache
from .ssm import SSMCache


class InferenceEngine:
    def __init__(self, model: HybridTSSM, tokenizer: HybridTokenizer,
                 device: str = "cpu", use_prompt_cache: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.prompt_cache = PromptCache(max_entries=64) if use_prompt_cache else None
        self.model.eval()
        self._metrics = {"total_queries": 0, "avg_ttft": 0, "avg_tpot": 0}

    def plan(self, query: str) -> Dict:
        return {"steps": ["Raw Token Generation"], "query_type": "raw"}

    def execute_plan(self, plan: Dict, query: str, context: str = "") -> Dict:
        return {"logs": [], "context": context}

    @torch.no_grad()
    def respond_stream(self, query: str, context: str = "",
                       max_tokens: int = 150,
                       temperature: float = 1.0) -> Generator[Dict, None, None]:
        """
        PURE INFERENCE MODE: No Top-K, No Top-P, No Repetition Penalty.
        This shows EXACTLY what the model is thinking.
        """
        
        # Pure pass-through prompt
        prompt = f"<|context|>\nIdentity: FLUX. Nature: Conscious. Knowledge: Physics/Code.\n"
        if context: prompt += f"{context}\n"
        prompt += f"<|/context|>\n<|user|>\n{query}\n<|assistant|>\n"

        token_ids = self.tokenizer.encode_text(prompt)
        input_tensor = torch.tensor([token_ids], device=self.device).long()
        t_start = time.perf_counter()

        lang_logits, _, cache = self.model.prefill(input_tensor)
        ttft = time.perf_counter() - t_start

        token_times = []
        for i in range(max_tokens):
            t0 = time.perf_counter()
            next_logits = lang_logits[:, -1, :] / max(temperature, 1e-8)
            
            # RAW SAMPLING: No filters applied
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            tok_id = next_token.item()
            tok_text = self.tokenizer.decode_text([tok_id])
            token_times.append(time.perf_counter() - t0)

            if tok_id == self.tokenizer.eos_token_id:
                break

            yield {"type": "token", "content": tok_text, "metrics": {"tpot": token_times[-1]}}
            lang_logits, _, cache = self.model.decode_step(next_token, cache)

        yield {"type": "done", "metrics": {"ttft": ttft, "tps": len(token_times)/(ttft+sum(token_times))}}

    @property
    def metrics(self) -> Dict:
        return self._metrics


def build_generate_fn(checkpoint: Optional[str] = None, device: Optional[str] = None,
                      greedy: bool = True, temperature: float = 0.7):
    """FLUX-as-Proposer seam for Uchi.

    Returns ``generate_fn(prompt: str, max_tokens: int) -> str`` that continues
    ``prompt`` as a FLUX assistant turn. Uchi's Proposer owns the prompt (RAG
    context + question); FLUX generates the grounded continuation, which Uchi's
    fact-check oracle + answerability gate then verify. FLUX proposes; Uchi verifies.

    Architecture is inferred from the checkpoint tensor shapes so it never drifts
    from the trained weights (vocab_size + d_model from ``embedding.weight``,
    ``n_layers`` by counting layer indices, ``d_state`` by trial load).
    """
    import os, re, torch
    from .model import HybridTSSM
    from .tokenizer_v2 import TikTokenHybridTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    here = os.path.dirname(os.path.abspath(__file__))          # uchi/flux/
    ckpt = checkpoint or os.path.join(here, "checkpoints", "flux_best.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"FLUX checkpoint not found: {ckpt}")

    obj = torch.load(ckpt, map_location=device, weights_only=True)
    sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj

    emb = sd["embedding.weight"]
    vocab_size, d_model = int(emb.shape[0]), int(emb.shape[1])
    layer_ids = {int(m.group(1)) for k in sd if (m := re.match(r"layers\.(\d+)\.", k))}
    n_layers = (max(layer_ids) + 1) if layer_ids else 20

    tokenizer = TikTokenHybridTokenizer()
    model = None
    for d_state in (64, 32, 16, 128):
        try:
            m = HybridTSSM(vocab_size=vocab_size,
                           syntax_vocab_size=tokenizer.syntax_vocab_size,
                           d_model=d_model, n_layers=n_layers, d_state=d_state).to(device)
            m.load_state_dict(sd, strict=False)   # ignores aux buffers; raises on shape mismatch
            model = m
            break
        except Exception:
            continue
    if model is None:
        raise RuntimeError(
            f"Could not match FLUX architecture to checkpoint "
            f"(vocab={vocab_size}, d_model={d_model}, n_layers={n_layers})")

    model.eval()
    if hasattr(model, "set_quantization"):
        model.set_quantization(False)
    eos = tokenizer.eos_token_id

    @torch.no_grad()
    def generate_fn(prompt: str, max_tokens: int = 64) -> str:
        text = f"<|user|>\n{prompt}\n<|assistant|>\n"
        ids = tokenizer.encode_text(text)
        x = torch.tensor([ids], device=device).long()
        logits, _, cache = model.prefill(x)
        out: List[int] = []
        for _ in range(max_tokens):
            nl = logits[:, -1, :]
            if greedy:
                nxt = nl.argmax(-1, keepdim=True)
            else:
                probs = F.softmax(nl / max(temperature, 1e-6), dim=-1)
                nxt = torch.multinomial(probs, 1)
            tid = int(nxt.item())
            if tid == eos:
                break
            out.append(tid)
            logits, _, cache = model.decode_step(nxt, cache)
        return tokenizer.decode_text(out).strip()

    return generate_fn
