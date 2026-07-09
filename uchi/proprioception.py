"""proprioception.py -- FLUX's own sense of question familiarity (0.4.0
follow-on, experimental).

NOT a replacement for the verifier. See tasks/proprioception_experiment.md
for the full writeup and the decisive empirical test showing why: this
answers "is this question's topic/shape familiar to FLUX," the verifier
answers "does this specific claim match this specific evidence" -- two
orthogonal questions, confirmed by testing a correct and an incorrect
claim about the same familiar topic and finding proprioception can't tell
them apart (as expected -- it was never trying to).

Corrected methodology, found while building this: FLUX's actual
evidence-grounded generation path (GenerateAndGround._candidates(),
via FluxProposer.propose(..., think=True)) sends the RAW QUESTION ALONE
to FLUX -- no context/evidence text in the prompt at all (see
uchi/proposer.py's propose() docstring: "Grounding against evidence
still happens downstream via the oracle regardless of prompt shape").
So proprioception's reference set and any live check must also use the
raw question alone, matching CoT training's exact shape
([user_id] + encode_text(question)) -- NOT a "Context:...Question:...
Answer:" wrapped template, which is what an earlier pilot in this
session mistakenly used and needs to be considered superseded by this.

This is a pre-generation gate: "before FLUX even attempts an answer, is
this question's shape/topic familiar at all?" -- not a post-generation
claim check, which stays the verifier's job entirely.
"""
from __future__ import annotations

import torch

from .flux.verifier_model import OODDetector


def load_flux_for_proprioception(checkpoint: str, device: str = "cpu"):
    """Loads a standalone FLUX model+tokenizer for proprioception's own
    hidden-state access -- FluxProposer only exposes a generate_fn
    closure, not the raw model, so this loads a second instance rather
    than risk refactoring that well-tested seam. Same shape-inference
    pattern as inference_engine.py's build_generate_fn, just stopping
    before the projection head instead of after. Returns (None, None) on
    any failure, never raises -- graceful degradation, same contract as
    every other optional component in this codebase."""
    from .flux.model import HybridTSSM
    from .flux.tokenizer_v2 import TikTokenHybridTokenizer
    from .flux.vocab_prune import load_pruned_tokenizer
    import os

    try:
        base_tok = TikTokenHybridTokenizer()
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        sd = ckpt["model"]
        emb = sd["embedding.weight"]
        vocab_size, d_model = int(emb.shape[0]), int(emb.shape[1])
        layer_ids = [int(k.split(".")[1]) for k in sd if k.startswith("layers.")]
        n_layers = (max(layer_ids) + 1) if layer_ids else 20

        tokenizer = base_tok
        if vocab_size != base_tok.vocab_size:
            pruned_path = os.path.join(os.path.dirname(__file__), "flux", "checkpoints",
                                        "pruned_vocab_32k.json")
            tokenizer = load_pruned_tokenizer(pruned_path, base_tok)
            if tokenizer.vocab_size != vocab_size:
                return None, None

        model = None
        for d_state in (64, 32, 16, 128):
            try:
                m = HybridTSSM(vocab_size=vocab_size, syntax_vocab_size=tokenizer.syntax_vocab_size,
                               d_model=d_model, n_layers=n_layers, d_state=d_state).to(device)
                m.load_state_dict(sd, strict=False)
                model = m
                break
            except Exception:
                continue
        if model is None:
            return None, None
        model.eval()
        return model, tokenizer
    except Exception:
        return None, None


def pooled_hidden_for_question(model, tokenizer, question: str) -> torch.Tensor:
    """Mean-pooled hidden state for a raw question, matching the exact
    shape FLUX is actually prompted with at inference (think=True path):
    no context wrapper, no answer template -- just the question."""
    ids = tokenizer.encode_text(question, max_length=256) or [0]
    x = torch.tensor([ids]).long()
    with torch.no_grad():
        hidden = model.embedding(x)
        for layer in model.layers:
            hidden, _ = layer(hidden)
        hidden = model.norm_f(hidden)
    return hidden[0].mean(dim=0)


class FluxProprioception:
    """Wraps OODDetector (same class the verifier's own OOD gate uses --
    not a new algorithm) over FLUX's own pooled hidden state for a raw
    question. Fit on real questions sampled from FLUX's actual CoT
    training sources, not synthetic examples.
    """

    def __init__(self, threshold: float | None = None):
        # threshold=None until calibrate_threshold() sets it from real
        # percentile data -- no reused default from a different model's
        # config, which is exactly the mistake found in both the pilot
        # experiment and the verifier's own OOD gate this session.
        self.detector = OODDetector(threshold=threshold if threshold is not None else float("inf"))

    def fit(self, model, tokenizer, reference_questions: list[str]) -> None:
        vectors = torch.stack([
            pooled_hidden_for_question(model, tokenizer, q) for q in reference_questions
        ])
        self.detector.fit(vectors)

    def calibrate_threshold(self, model, tokenizer, held_out_in_distribution_questions: list[str],
                             percentile: float = 95.0) -> float:
        """Sets the threshold from the actual distribution of distances on
        real held-out in-distribution questions (not the reference set
        itself), at the given percentile -- so `percentile`% of genuinely
        familiar questions pass, by construction, rather than an arbitrary
        reused default. Returns the threshold set."""
        distances = sorted(
            self.detector.distance(pooled_hidden_for_question(model, tokenizer, q))
            for q in held_out_in_distribution_questions
        )
        idx = min(len(distances) - 1, int(len(distances) * percentile / 100.0))
        self.detector.threshold = distances[idx]
        return self.detector.threshold

    def is_unfamiliar(self, model, tokenizer, question: str) -> bool:
        if not self.detector.is_fitted:
            return False  # ungated by design until properly fit -- never a false positive from an empty detector
        v = pooled_hidden_for_question(model, tokenizer, question)
        return self.detector.is_ood(v)

    def state_dict(self) -> dict:
        return self.detector.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.detector.load_state_dict(state)

    @classmethod
    def load(cls, path: str) -> "FluxProprioception | None":
        """Returns None (never raises) if the file is missing or malformed
        -- same graceful-degradation contract as FluxProposer.load() and
        EntailmentChecker.load()."""
        import torch
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return None
        prop = cls()
        prop.load_state_dict(state)
        return prop
