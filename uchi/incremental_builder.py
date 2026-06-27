"""Incremental offline brain building pipeline.

Adds new knowledge to an existing brain without wiping it. Supports
checkpoint/resume so it can be interrupted and restarted safely.
Runs SSM training only on new (delta) tokens — not full retraining.

Usage:
    python -m uchi.incremental_builder
    python -m uchi.incremental_builder --limit 200
    python -m uchi.incremental_builder --sources wikipedia,mmlu
    python -m uchi.incremental_builder --brain path/to/brain.uchi

Sources (default: all):
    openhermes  — conversational (teknium/OpenHermes-2.5)
    wikipedia   — encyclopedic (wikipedia 20220301.en)
    mmlu        — academic reasoning (cais/mmlu)
    gsm8k       — math (gsm8k)
    swebench    — code/bugs (princeton-nlp/SWE-bench)
    humaneval   — algorithm completion (openai/openai_humaneval)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from typing import Iterator, List, Optional

import torch
from tqdm import tqdm

_log = logging.getLogger(__name__)

KNOWLEDGE_LIMIT = 200
ALL_SOURCES = ["openhermes", "wikipedia", "mmlu", "gsm8k", "swebench", "humaneval"]

# Maximum new token sequences to accumulate for SSM delta training.
_MAX_DELTA_SEQS = 2000


class IncrementalBrainBuilder:
    """Extends an existing brain with new knowledge.

    Workflow:
      Phase 1 — Load existing brain (no wipe).
      Phase 2 — Ingest new documents, deduplicating against already-seen content.
                 Checkpoint after each source completes so the run is resumable.
      Phase 3 — Run SSM contrastive training on delta tokens only.
      Phase 4 — Persist updated brain.
    """

    def __init__(self, brain_path: str = "brain.uchi",
                 checkpoint_path: Optional[str] = None):
        self.brain_path = brain_path
        self.checkpoint_path = (
            checkpoint_path
            or brain_path.replace(".uchi", "_incremental_checkpoint.json")
        )
        self._checkpoint = self._load_checkpoint()

    # ── checkpoint ────────────────────────────────────────────────────────────

    def _load_checkpoint(self) -> dict:
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path) as f:
                    data = json.load(f)
                _log.info("Resuming from checkpoint: %s", self.checkpoint_path)
                return data
            except Exception as e:
                _log.warning("Checkpoint unreadable (%s), starting fresh.", e)
        return {
            "completed_sources": [],
            "current_source":    None,
            "current_offset":    0,
            "total_ingested":    0,
            "started_at":        datetime.now(timezone.utc).isoformat(),
            "last_updated":      None,
        }

    def _save_checkpoint(self, *, current_source: Optional[str] = None,
                         offset: int = 0, total: int = 0) -> None:
        self._checkpoint["current_source"] = current_source
        self._checkpoint["current_offset"] = offset
        self._checkpoint["total_ingested"]  = total
        self._checkpoint["last_updated"]    = datetime.now(timezone.utc).isoformat()
        try:
            with open(self.checkpoint_path, "w") as f:
                json.dump(self._checkpoint, f, indent=2)
        except Exception as e:
            _log.warning("Checkpoint save failed: %s", e)

    def _mark_source_complete(self, source: str) -> None:
        if source not in self._checkpoint["completed_sources"]:
            self._checkpoint["completed_sources"].append(source)
        self._checkpoint["current_source"]  = None
        self._checkpoint["current_offset"]  = 0
        self._checkpoint["last_updated"]    = datetime.now(timezone.utc).isoformat()
        try:
            with open(self.checkpoint_path, "w") as f:
                json.dump(self._checkpoint, f, indent=2)
        except Exception as e:
            _log.warning("Checkpoint save failed: %s", e)

    def _clear_checkpoint(self) -> None:
        try:
            os.remove(self.checkpoint_path)
        except FileNotFoundError:
            pass

    # ── source loaders ────────────────────────────────────────────────────────

    def _iter_source(self, source: str, limit: int,
                     start_offset: int) -> Iterator[str]:
        """Yield text strings from the named source, starting at start_offset."""
        from uchi.builder import _safe_load_dataset

        if source == "openhermes":
            ds = _safe_load_dataset("teknium/OpenHermes-2.5", f"train[:{limit}]")
            if ds is None:
                return
            for i, item in enumerate(ds):
                if i < start_offset:
                    continue
                turns = item.get("conversations", [])
                text = "".join(
                    f"{'<|user|>' if t.get('from') == 'human' else '<|assistant|>'} "
                    f"{t.get('value', '')} "
                    for t in turns
                )
                if text.strip():
                    yield text.strip() + " <|end|>"

        elif source == "wikipedia":
            ds = _safe_load_dataset("wikipedia", "20220301.en",
                                    split=f"train[:{limit}]")
            if ds is None:
                return
            for i, item in enumerate(ds):
                if i < start_offset:
                    continue
                title = item.get("title", "")
                body  = item.get("text", "")[:1200]
                yield f"<|user|> Tell me about {title}. <|assistant|> {body} <|end|>"

        elif source == "mmlu":
            ds = _safe_load_dataset("cais/mmlu", "all", split="test")
            if ds is None:
                return
            for i, item in enumerate(list(ds)[:limit]):
                if i < start_offset:
                    continue
                q, choices, ans_idx = (
                    item.get("question", ""),
                    item.get("choices", []),
                    item.get("answer", -1),
                )
                if 0 <= ans_idx < len(choices):
                    yield (
                        f"<|user|> Question: {q} Choices: {', '.join(choices)} "
                        f"<|assistant|> {choices[ans_idx]} <|end|>"
                    )

        elif source == "gsm8k":
            ds = _safe_load_dataset("gsm8k", "main", split="test")
            if ds is None:
                return
            for i, item in enumerate(list(ds)[:limit]):
                if i < start_offset:
                    continue
                yield (
                    f"<|user|> {item.get('question', '')} "
                    f"<|assistant|> {item.get('answer', '')} <|end|>"
                )

        elif source == "swebench":
            ds = _safe_load_dataset("princeton-nlp/SWE-bench", split="test")
            if ds is None:
                return
            for i, item in enumerate(list(ds)[:limit]):
                if i < start_offset:
                    continue
                issue = item.get("problem_statement", "")[:600]
                patch = item.get("patch", "")[:600]
                yield f"<|user|> Fix issue:\n{issue} <|assistant|> {patch} <|end|>"

        elif source == "humaneval":
            ds = _safe_load_dataset("openai/openai_humaneval", split="test")
            if ds is None:
                return
            for i, item in enumerate(list(ds)[:limit]):
                if i < start_offset:
                    continue
                yield (
                    f"<|user|> Complete Python code:\n{item.get('prompt', '')} "
                    f"<|assistant|> {item.get('canonical_solution', '')} <|end|>"
                )

    # ── SSM delta training ────────────────────────────────────────────────────

    def _train_ssm_delta(self, router, delta_seqs: list) -> None:
        """Run a short SSM training pass over new-token sequences only."""
        if not delta_seqs:
            return
        from uchi.neuro_symbolic import get_ssm

        ssm = get_ssm()
        optimizer = torch.optim.Adam(ssm.parameters(), lr=5e-4)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        print(f"\n[*] Phase 3: SSM delta training on {len(delta_seqs)} new sequences…")
        for tokens in tqdm(delta_seqs, desc="SSM delta train"):
            if len(tokens) < 2:
                continue
            optimizer.zero_grad()
            try:
                v_loss = ssm.update_value(tokens, reward=1.0)
                d_loss = ssm.train_dynamics(tokens)
                (v_loss + d_loss).backward()
                optimizer.step()
            except Exception as e:
                _log.debug("SSM step skipped: %s", e)

        pt_path = os.path.join(project_root, "ssm_dynamics.pt")
        torch.save(ssm.state_dict(), pt_path)
        print(f"[+] SSM weights saved to {os.path.basename(pt_path)}")

    # ── main entry ────────────────────────────────────────────────────────────

    def run(self, limit: int = KNOWLEDGE_LIMIT,
            sources: Optional[List[str]] = None) -> None:
        """Execute the incremental build.

        Args:
            limit:   Max documents to ingest per source.
            sources: List of source names to process (default: ALL_SOURCES).
        """
        from uchi.cli import save_brain
        from uchi.omni_router import OmniRouter
        from uchi.deduplication import IngestionDeduplicator

        sources = sources or ALL_SOURCES

        # ── Phase 1: Load existing brain ──────────────────────────────────────
        print("\n" + "=" * 70)
        print(" Uchi Incremental Brain Builder")
        print("=" * 70)

        if os.path.exists(self.brain_path):
            print(f"\n[*] Phase 1: Loading existing brain from {self.brain_path}…")
            try:
                import gzip
                import pickle
                with gzip.open(self.brain_path, "rb") as f:
                    router = pickle.load(f)
                print("[+] Brain loaded.")
            except Exception as e:
                print(f"[!] Brain load failed ({e}). Building cold router.")
                OmniRouter._bootstrap_knowledge = lambda self, *a, **kw: None
                OmniRouter._bootstrap_persona   = lambda self, *a, **kw: None
                router = OmniRouter(use_bpe=False)
        else:
            print("[*] Phase 1: No existing brain — starting fresh router.")
            OmniRouter._bootstrap_knowledge = lambda self, *a, **kw: None
            OmniRouter._bootstrap_persona   = lambda self, *a, **kw: None
            router = OmniRouter(use_bpe=False)

        dedup = IngestionDeduplicator(threshold=0.8)
        delta_seqs: list = []   # token sequences of newly ingested docs
        total_ingested = self._checkpoint.get("total_ingested", 0)

        # ── Phase 2: Incremental ingestion ────────────────────────────────────
        print(f"\n[*] Phase 2: Ingesting from {len(sources)} sources (limit={limit}/source)…")
        if self._checkpoint["completed_sources"]:
            print(f"[*] Resuming — already completed: "
                  f"{', '.join(self._checkpoint['completed_sources'])}")

        for source in sources:
            if source in self._checkpoint["completed_sources"]:
                print(f"  [=] {source}: already complete (checkpoint), skipping.")
                continue

            start_offset = 0
            if self._checkpoint.get("current_source") == source:
                start_offset = self._checkpoint.get("current_offset", 0)
                if start_offset:
                    print(f"  [>] {source}: resuming from offset {start_offset}")

            source_count = 0
            self._save_checkpoint(current_source=source,
                                  offset=start_offset, total=total_ingested)

            for i, text in enumerate(
                    tqdm(self._iter_source(source, limit, start_offset),
                         desc=f"  {source}", total=limit - start_offset)):

                abs_offset = start_offset + i

                if dedup.check_and_add(text):
                    continue  # near-duplicate, skip

                tokens = router.tokenizer.tokenize(
                    text.split(), is_inference=False
                )
                router.stream(tokens)
                source_count += 1
                total_ingested += 1

                # Accumulate for delta SSM training (cap to avoid OOM).
                if len(delta_seqs) < _MAX_DELTA_SEQS:
                    delta_seqs.append(tokens)

                # Checkpoint every 50 unique docs ingested.
                if source_count % 50 == 0:
                    self._save_checkpoint(current_source=source,
                                          offset=abs_offset + 1,
                                          total=total_ingested)

            self._mark_source_complete(source)
            print(f"  [+] {source}: {source_count} new documents ingested.")

        dedup_stats = dedup.stats()
        print(f"\n[*] Deduplication: {dedup_stats['duplicates_blocked']} duplicates blocked "
              f"({dedup_stats['duplicate_rate']*100:.1f}%), "
              f"{total_ingested} unique documents added.")

        # ── Phase 3: SSM delta training ───────────────────────────────────────
        self._train_ssm_delta(router, delta_seqs)

        # ── Phase 4: Persist updated brain ────────────────────────────────────
        print(f"\n[*] Phase 4: Saving updated brain to {self.brain_path}…")
        save_brain(router, self.brain_path)
        print("[+] Done. Brain updated incrementally.")

        self._clear_checkpoint()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Uchi Incremental Brain Builder — extends an existing brain without wiping it."
    )
    parser.add_argument("--brain",    default="brain.uchi",
                        help="Path to brain file (default: brain.uchi)")
    parser.add_argument("--limit",    type=int, default=KNOWLEDGE_LIMIT,
                        help=f"Max docs per source (default: {KNOWLEDGE_LIMIT})")
    parser.add_argument("--sources",  default=None,
                        help=f"Comma-separated sources (default: all). "
                             f"Options: {', '.join(ALL_SOURCES)}")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint file (default: auto)")
    args = parser.parse_args()

    sources = (
        [s.strip() for s in args.sources.split(",")]
        if args.sources else None
    )

    builder = IncrementalBrainBuilder(
        brain_path=args.brain,
        checkpoint_path=args.checkpoint,
    )
    builder.run(limit=args.limit, sources=sources)


if __name__ == "__main__":
    main()
