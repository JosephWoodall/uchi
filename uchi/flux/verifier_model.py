"""verifier_model.py — Entailment classifier architecture (0.4.0 Item 17).

Reuses the same TSSMBlock/AttentionBlock backbone as HybridTSSM (model.py),
but with a 3-way classification head (entailment/neutral/contradiction)
pooled over the sequence, instead of a per-position generation head.

Deliberately a SEPARATE model with its OWN embedding table, never loaded
from or tied to the main FLUX proposer checkpoint -- a structural guarantee
the verifier cannot inherit the proposer's failure modes, not just a
training-time coincidence. See uchi/oracle.py for how this plugs in as an
additional, strictly additive veto layer on top of the existing
deterministic word-overlap check -- it can only reject a claim the
deterministic check already accepted, never accept one the deterministic
check rejected.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .tssm_block import TSSMBlock
from .attention_block import AttentionBlock


class EntailmentClassifier(nn.Module):
    """Premise/hypothesis -> entailment | neutral | contradiction.

    Input is a single token sequence: the premise (evidence) and hypothesis
    (claim) concatenated with the tokenizer's existing <|context|>/
    <|/context|> and <|user|> special tokens marking the boundary -- no new
    special tokens added to the shared tokenizer (it's actively used by the
    in-progress FLUX retraining; inserting new entries would shift every
    downstream token ID via n_special).
    """

    NUM_CLASSES = 3
    # Matches the standard GLUE-MNLI / SNLI label convention:
    # 0 = entailment, 1 = neutral, 2 = contradiction.
    LABEL_NAMES = ("entailment", "neutral", "contradiction")

    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 8, d_state: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)

        # Same hybrid interleaving as HybridTSSM: every 4th layer is attention.
        self.layers = nn.ModuleList([
            AttentionBlock(d_model, n_heads=8) if i % 4 == 3 else TSSMBlock(d_model, d_state)
            for i in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.classifier_head = nn.Linear(d_model, self.NUM_CLASSES)

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

    def encode(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Returns the pooled (batch, d_model) representation BEFORE the
        classification head -- the "latent space" used for out-of-
        distribution detection (see OODDetector below). Exposed separately
        from forward() so OOD-fitting/scoring never needs to run the
        classification head at all.
        """
        hidden = self.embedding(x)

        for layer in self.layers:
            if getattr(self, "_gradient_checkpointing", False) and torch.is_grad_enabled():
                hidden, _ = torch.utils.checkpoint.checkpoint(layer, hidden, use_reentrant=False)
            else:
                hidden, _ = layer(hidden)

        hidden = self.norm_f(hidden)  # (batch, seq_len, d_model)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = hidden.mean(dim=1)
        return pooled

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (batch, seq_len) token ids. attention_mask: (batch, seq_len),
        1 for real tokens / 0 for padding -- used to mean-pool only over real
        tokens so padding doesn't dilute the pooled representation.
        Returns (batch, NUM_CLASSES) logits.
        """
        pooled = self.encode(x, attention_mask)
        return self.classifier_head(pooled)

    @torch.no_grad()
    def predict_label(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> list[str]:
        """Convenience wrapper: returns label names instead of raw logits."""
        self.eval()
        logits = self.forward(x, attention_mask)
        idx = logits.argmax(dim=-1).tolist()
        return [self.LABEL_NAMES[i] for i in idx]


class OODDetector:
    """Out-of-distribution gate on EntailmentClassifier's latent space
    (Mahalanobis distance -- Lee et al.'s standard technique for this,
    not a novel/unproven approach).

    Neural classifiers tend to be confidently wrong on inputs unlike
    anything in their training data -- the softmax always produces SOME
    distribution, even for a claim nothing like MNLI/SNLI. This fits a
    Gaussian over the pooled representations of real training examples,
    then flags a new input as OOD if its representation is far from that
    fitted distribution.

    Used as a GATE on the entailment veto, never an independent veto:
    if an input is OOD, the entailment classifier's judgment shouldn't be
    trusted at all for it -- fall back to whatever the deterministic +
    numeric layers already decided, don't treat "unusual" as "wrong" (an
    obscure-but-valid claim would also look unusual; vetoing on that alone
    would just create more false rejects). Gracefully inert until fit(),
    same pattern as every other component here.
    """

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self._mean: Optional[torch.Tensor] = None
        self._precision: Optional[torch.Tensor] = None  # inverse covariance

    @property
    def is_fitted(self) -> bool:
        return self._mean is not None

    def fit(self, representations: torch.Tensor) -> None:
        """representations: (N, d_model) pooled reps from real training
        examples (collected during verifier_train.py's training loop)."""
        self._mean = representations.mean(dim=0)
        centered = representations - self._mean
        n = representations.shape[0]
        cov = (centered.T @ centered) / max(1, n - 1)
        d = cov.shape[0]
        # Regularize -- a raw empirical covariance from a finite sample is
        # rarely invertible/well-conditioned on its own.
        cov = cov + torch.eye(d, device=cov.device) * 1e-3
        self._precision = torch.linalg.inv(cov)

    def distance(self, representation: torch.Tensor) -> float:
        """Mahalanobis distance of one (d_model,) representation from the
        fitted distribution. Returns 0.0 (never OOD) if unfitted."""
        if not self.is_fitted:
            return 0.0
        delta = (representation - self._mean).unsqueeze(0)  # (1, d)
        dist_sq = (delta @ self._precision @ delta.T).squeeze()
        return float(torch.sqrt(torch.clamp(dist_sq, min=0.0)).item())

    def is_ood(self, representation: torch.Tensor) -> bool:
        if not self.is_fitted:
            return False  # no opinion -- never gates anything when unfitted
        return self.distance(representation) > self.threshold

    def state_dict(self) -> dict:
        return {"mean": self._mean, "precision": self._precision, "threshold": self.threshold}

    def load_state_dict(self, state: dict) -> None:
        self._mean = state["mean"]
        self._precision = state["precision"]
        self.threshold = state.get("threshold", self.threshold)


class EntailmentChecker:
    """Adapter oracle.py actually talks to: wraps a loaded
    EntailmentClassifier + tokenizer behind a simple
    ``is_contradiction(premise, hypothesis) -> bool`` call, so the oracle
    doesn't need to know anything about tokenization or tensors.

    ``load()`` mirrors FluxProposer.load()'s graceful degradation exactly:
    returns None (not an exception) when no checkpoint exists yet, so
    ``FactCheckOracle(entailment_checker=EntailmentChecker.load(path))``
    works identically whether or not training has happened.
    """

    def __init__(self, model: EntailmentClassifier, tokenizer, device: str = "cpu",
                 max_seq_len: int = 192, contradiction_threshold: float = 0.5,
                 ood_detector: "OODDetector | None" = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_len = max_seq_len
        self.contradiction_threshold = contradiction_threshold
        # 0.4.0 Item 17 refinement: gates the contradiction verdict, never
        # an independent veto. If the input is OOD, the classifier's
        # judgment isn't trusted for it -- treated as "no opinion" for this
        # call, not as its own rejection (an obscure-but-valid claim would
        # also look OOD; vetoing on that alone would just create more
        # false rejects). None (default) = gate disabled, unchanged
        # behavior from before this refinement.
        self.ood_detector = ood_detector
        self.model.eval()

    def _encode(self, premise: str, hypothesis: str):
        context_open = self.tokenizer.encode_special("<|context|>")
        context_close = self.tokenizer.encode_special("<|/context|>")
        user_id = self.tokenizer.encode_special("<|user|>")
        pad_id = self.tokenizer.pad_token_id

        premise_ids = [context_open] + self.tokenizer.encode_text(premise) + [context_close]
        hyp_ids = [user_id] + self.tokenizer.encode_text(hypothesis)
        full_ids = (premise_ids + hyp_ids)[: self.max_seq_len]

        n_pad = self.max_seq_len - len(full_ids)
        attention_mask = [1] * len(full_ids) + [0] * n_pad
        full_ids = full_ids + [pad_id] * n_pad
        return full_ids, attention_mask

    @torch.no_grad()
    def is_contradiction(self, premise: str, hypothesis: str) -> bool:
        if not premise.strip() or not hypothesis.strip():
            return False
        ids, mask = self._encode(premise, hypothesis)
        x = torch.tensor([ids], device=self.device).long()
        m = torch.tensor([mask], device=self.device).float()

        pooled = self.model.encode(x, m)

        if self.ood_detector is not None and self.ood_detector.is_ood(pooled[0]):
            # Out of the classifier's training distribution -- its
            # judgment isn't trustworthy here, gate it to "no opinion"
            # rather than let a confident-but-baseless verdict count.
            return False

        logits = self.model.classifier_head(pooled)
        probs = torch.softmax(logits, dim=-1)[0]
        contradiction_idx = self.model.LABEL_NAMES.index("contradiction")
        return bool(probs[contradiction_idx].item() >= self.contradiction_threshold)

    @classmethod
    def load(cls, checkpoint: str, device: str | None = None) -> "EntailmentChecker | None":
        """Returns None (never raises) if *checkpoint* doesn't exist or
        fails to load -- same graceful-degradation contract as
        FluxProposer.load()."""
        import os
        if not checkpoint or not os.path.exists(checkpoint):
            return None
        try:
            from .tokenizer_v2 import TikTokenHybridTokenizer
            device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            obj = torch.load(checkpoint, map_location=device, weights_only=True)
            sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj

            emb = sd["embedding.weight"]
            vocab_size, d_model = int(emb.shape[0]), int(emb.shape[1])
            import re as _re
            layer_ids = {int(m.group(1)) for k in sd if (m := _re.match(r"layers\.(\d+)\.", k))}
            n_layers = (max(layer_ids) + 1) if layer_ids else 8

            model = None
            for d_state in (32, 16, 64, 128):
                try:
                    m = EntailmentClassifier(vocab_size=vocab_size, d_model=d_model,
                                              n_layers=n_layers, d_state=d_state).to(device)
                    m.load_state_dict(sd, strict=False)
                    model = m
                    break
                except Exception:
                    continue
            if model is None:
                return None

            tokenizer = TikTokenHybridTokenizer()

            ood_detector = None
            ood_path = checkpoint.rsplit(".", 1)[0] + ".ood.pt"
            if os.path.exists(ood_path):
                try:
                    ood_state = torch.load(ood_path, map_location=device, weights_only=True)
                    ood_detector = OODDetector()
                    ood_detector.load_state_dict(ood_state)
                except Exception:
                    ood_detector = None  # fail open -- load without the gate rather than fail entirely

            return cls(model, tokenizer, device=device, ood_detector=ood_detector)
        except Exception:
            return None
