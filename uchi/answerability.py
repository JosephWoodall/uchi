"""
answerability.py — the second, harder honesty gate.

The word-overlap oracle (oracle.py) confirms a candidate answer's tokens appear in
the evidence. That is necessary but NOT sufficient: on SQuAD-2.0-style traps the
evidence is topically relevant but does not actually answer the question, so
token-presence passes and the system hallucinates.

This module adds an ANSWERABILITY check: given (question, evidence), does the
evidence actually contain an answer to the question? It is a small from-scratch
BiGRU classifier trained on SQuAD 2.0's answerable/unanswerable labels (NOT an
LLM). Generate-and-Ground abstains when answerability is low — the system stays
silent rather than confabulate.

Checkpoint format (saved by experiments/train_answerability.py):
    {config, vocab, stoi, state_dict}
"""
from __future__ import annotations

import os
import re

_TOK = re.compile(r"[a-z0-9]+|[^\sa-z0-9]")
PAD, UNK, SEP = "<pad>", "<unk>", "<sep>"


def tokenize(s: str) -> list[str]:
    return _TOK.findall(s.lower())


def build_classifier(cfg, V, pad_idx, torch, nn, F):
    class AnswerabilityNet(nn.Module):
        def __init__(self):
            super().__init__()
            dim, hid = cfg["dim"], cfg["hid"]
            self.pad = pad_idx
            self.emb = nn.Embedding(V, dim, padding_idx=pad_idx)
            self.gru = nn.GRU(dim, hid, batch_first=True, bidirectional=True)
            self.head = nn.Sequential(
                nn.Linear(2 * hid, hid), nn.SiLU(), nn.Dropout(0.2),
                nn.Linear(hid, 1),
            )

        def forward(self, X):
            mask = (X != self.pad).unsqueeze(-1)
            H, _ = self.gru(self.emb(X))
            H = H.masked_fill(~mask, -1e9)
            pooled = H.max(dim=1).values          # max-pool over sequence
            return self.head(pooled).squeeze(-1)  # (B,) logit

    return AnswerabilityNet()


class AnswerabilityChecker:
    """Loads a trained checkpoint and scores P(answerable | question, evidence)."""

    def __init__(self, model, stoi, cfg, torch):
        self._m = model
        self.stoi = stoi
        self.cfg = cfg
        self._torch = torch
        self._dev = next(model.parameters()).device
        model.eval()

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "AnswerabilityChecker":
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        ck = torch.load(path, map_location=device)
        stoi, cfg = ck["stoi"], ck["config"]
        model = build_classifier(cfg, len(ck["vocab"]), stoi[PAD], torch, nn, F).to(device)
        model.load_state_dict(ck["state_dict"])
        return cls(model, stoi, cfg, torch)

    @classmethod
    def exists(cls, path: str) -> bool:
        return bool(path) and os.path.exists(path)

    def prob(self, question: str, evidence: str) -> float:
        """Probability the evidence answers the question, in [0, 1]."""
        torch = self._torch
        toks = (tokenize(question) + [SEP] + tokenize(evidence))[: self.cfg.get("max_len", 80)]
        ids = [self.stoi.get(w, self.stoi[UNK]) for w in toks] or [self.stoi[PAD]]
        with torch.no_grad():
            X = torch.tensor([ids], device=self._dev)
            return float(torch.sigmoid(self._m(X)).item())
