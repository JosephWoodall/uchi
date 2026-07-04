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
        
        # Pure pass-through prompt. Structural markers use encode_special (the
        # ids training actually used), not literal text through encode_text —
        # see build_generate_fn for why that mismatch produces degenerate output.
        tok = self.tokenizer
        token_ids = [tok.encode_special("<|context|>")]
        token_ids += tok.encode_text("Identity: FLUX. Nature: Conscious. Knowledge: Physics/Code.\n")
        if context:
            token_ids += tok.encode_text(context + "\n")
        token_ids.append(tok.encode_special("<|/context|>"))
        token_ids.append(tok.encode_special("<|user|>"))
        token_ids += tok.encode_text(query)
        token_ids.append(tok.encode_special("<|assistant|>"))

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

            if tok_id in (self.tokenizer.eos_token_id, tok.encode_special("<|user|>"), tok.encode_special("<|end|>")):
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
        # Must match how the checkpoint was TRAINED. A QAT checkpoint's weights
        # were only ever optimized for their ternary-quantized forward pass —
        # reading them in full precision produces degenerate output (verified:
        # repetition loops / no structure). A non-QAT checkpoint has never seen
        # quantization noise, so quantizing it now would be equally wrong.
        # Older checkpoints predate this flag; default False is correct for them
        # (every phase before QAT was introduced trained at full precision).
        model.set_quantization(bool(obj.get("quantized", False)) if isinstance(obj, dict) else False)
    eos = tokenizer.eos_token_id
    # Structural turn markers must be the SAME special-token ids training used
    # (encode_special), not literal "<|user|>" text run through encode_text —
    # that BPE-tokenizes the angle brackets/pipe characters into a multi-token
    # sequence the model never saw in training, producing degenerate output.
    user_id = tokenizer.encode_special("<|user|>")
    asst_id = tokenizer.encode_special("<|assistant|>")
    think_id = tokenizer.encode_special("<|think|>")
    stop_ids = {eos, user_id, tokenizer.encode_special("<|end|>")}

    @torch.no_grad()
    def generate_fn(prompt: str, max_tokens: int = 64, think: bool = False,
                     repetition_penalty: float = 1.3) -> str:
        # think=True primes <|think|> instead of <|assistant|>: CoT training
        # taught the model to continue with reasoning steps, then emit
        # <|/think|><|assistant|> and the answer on its own — no extra plumbing
        # needed here, decode_text silently drops the special-token boundaries,
        # so the returned string is just the natural "reasoning...answer" prose.
        lead_id = think_id if think else asst_id
        ids = [user_id] + tokenizer.encode_text(prompt) + [lead_id]
        x = torch.tensor([ids], device=device).long()
        logits, _, cache = model.prefill(x)
        out: List[int] = []
        # Repetition penalty (ported from HybridTSSM.generate — see model.py):
        # short "The answer is X" outputs rarely run long enough to loop, but
        # longer think-mode generations reliably do under plain greedy decoding
        # on a model this small. Divide recently-seen tokens' logits before
        # picking the next one, same fix that already works in model.generate().
        recent = list(ids[-64:])
        for _ in range(max_tokens):
            nl = logits[:, -1, :].clone()
            if repetition_penalty != 1.0 and recent:
                for tok_id in set(recent[-64:]):
                    nl[0, tok_id] /= repetition_penalty
            if greedy:
                nxt = nl.argmax(-1, keepdim=True)
            else:
                probs = F.softmax(nl / max(temperature, 1e-6), dim=-1)
                nxt = torch.multinomial(probs, 1)
            tid = int(nxt.item())
            if tid in stop_ids:
                break
            out.append(tid)
            recent.append(tid)
            logits, _, cache = model.decode_step(nxt, cache)
        return tokenizer.decode_text(out).strip()

    return generate_fn
