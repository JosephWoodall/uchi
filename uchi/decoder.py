"""
decoder.py — Uchi's from-scratch retrieval-conditioned answer generator.

The *generator* half of Generate-and-Ground: given a question and the evidence
retrieved from the brain, it produces a candidate answer string. It is a small
BiGRU encoder + attention decoder (~18M params) trained from scratch on
(question + evidence → answer) triples — NOT a pretrained LLM. Its embeddings are
warm-started from the brain's own skip-gram vectors.

It hallucinates freely; that is expected and safe, because every answer it
produces is fact-checked by ``oracle.FactCheckOracle`` before it can reach the
user. This module only *proposes*; honesty is enforced downstream.

Checkpoints are saved by ``uchi.decoder.train`` (or the build pipeline) as a dict
{config, vocab, stoi, state_dict} and loaded here for inference.
"""
from __future__ import annotations

import os
import re
from typing import Optional

_TOK = re.compile(r"[a-z0-9]+|[^\sa-z0-9]")
PAD, SOS, EOS, UNK, SEP = "<pad>", "<sos>", "<eos>", "<unk>", "<sep>"


def tokenize(s: str) -> list[str]:
    return _TOK.findall(s.lower())


def _build_model(cfg, V, pad_idx, torch, nn, F):
    class Seq2Seq(nn.Module):
        def __init__(self):
            super().__init__()
            dim, hid = cfg["dim"], cfg["hid"]
            self.pad = pad_idx
            self.emb = nn.Embedding(V, dim, padding_idx=pad_idx)
            self.enc = nn.GRU(dim, hid, batch_first=True, bidirectional=True)
            self.bridge = nn.Linear(2 * hid, hid)
            self.dec = nn.GRU(dim + 2 * hid, hid, batch_first=True)
            self.attn = nn.Linear(hid + 2 * hid, 1)
            self.out = nn.Linear(hid, V)

        def encode(self, S):
            mask = (S != self.pad)
            H, h = self.enc(self.emb(S))
            h = torch.tanh(self.bridge(torch.cat([h[0], h[1]], -1))).unsqueeze(0)
            return H, h, mask

        def step(self, y, h, H, mask):
            e = self.emb(y).unsqueeze(1)
            hd = h[-1].unsqueeze(1).expand(-1, H.size(1), -1)
            sc = self.attn(torch.cat([hd, H], -1)).squeeze(-1).masked_fill(~mask, -1e9)
            a = F.softmax(sc, -1).unsqueeze(1)
            ctx = torch.bmm(a, H)
            o, h = self.dec(torch.cat([e, ctx], -1), h)
            return self.out(o.squeeze(1)), h

        def forward(self, S, T):
            H, h, mask = self.encode(S)
            logits = []
            for t in range(T.size(1) - 1):
                lo, h = self.step(T[:, t], h, H, mask)
                logits.append(lo)
            return torch.stack(logits, 1)

    return Seq2Seq()


class NeuralDecoder:
    """Retrieval-conditioned answer generator (inference wrapper).

    Load a trained checkpoint with ``NeuralDecoder.load(path)``, then call
    ``generate(question, evidence)`` to get a candidate answer string.
    """

    def __init__(self, model, vocab, stoi, cfg, torch):
        self._m = model
        self.vocab = vocab
        self.stoi = stoi
        self.cfg = cfg
        self._torch = torch
        self._dev = next(model.parameters()).device
        model.eval()

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "NeuralDecoder":
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        ck = torch.load(path, map_location=device)
        vocab, stoi, cfg = ck["vocab"], ck["stoi"], ck["config"]
        model = _build_model(cfg, len(vocab), stoi[PAD], torch, nn, F).to(device)
        model.load_state_dict(ck["state_dict"])
        return cls(model, vocab, stoi, cfg, torch)

    @classmethod
    def exists(cls, path: str) -> bool:
        return bool(path) and os.path.exists(path)

    def _enc(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(w, self.stoi[UNK]) for w in tokens]

    def generate(self, question: str, evidence: list[str], max_len: int = 16) -> str:
        """Produce a candidate answer conditioned on question + retrieved evidence.

        The output is a *proposal* — it must be fact-checked before use.
        """
        torch = self._torch
        ev = " ".join(evidence[:2])
        src = tokenize(question) + [SEP] + tokenize(ev)
        src = src[: self.cfg.get("max_src", 60)]
        with torch.no_grad():
            S = torch.tensor([self._enc(src)], device=self._dev)
            H, h, mask = self._m.encode(S)
            y = torch.tensor([self.stoi[SOS]], device=self._dev)
            out: list[str] = []
            for _ in range(max_len):
                lo, h = self._m.step(y, h, H, mask)
                y = lo.argmax(-1)
                w = self.vocab[int(y.item())]
                if w == EOS:
                    break
                out.append(w)
        return _detok(out)


def _detok(tokens: list[str]) -> str:
    """Join word/punctuation tokens back into a readable string."""
    s = ""
    for t in tokens:
        if re.match(r"[^\sa-z0-9]", t) and t not in "([{":
            s += t
        else:
            s += (" " if s and not s.endswith(("(", "[", "{")) else "") + t
    return s.strip()
