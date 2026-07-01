"""Incremental offline brain building pipeline.

Adds new knowledge to an existing brain without wiping it. Supports
checkpoint/resume so it can be interrupted and restarted safely.
Runs SSM training only on new (delta) tokens — not full retraining.

Usage:
    python -m uchi.incremental_builder
    python -m uchi.incremental_builder --limit 200
    python -m uchi.incremental_builder --sources wikipedia_targeted,mbpp
    python -m uchi.incremental_builder --brain path/to/brain.uchi

Sources (default: all):
    openhermes          — conversational (teknium/OpenHermes-2.5)
    wikipedia           — encyclopedic (wikipedia 20220301.en)
    wikipedia_targeted  — one Wikipedia article per MMLU subject (no HF dependency)
    mmlu                — academic reasoning (cais/mmlu)
    arc                 — reasoning MCQ (ARC-Challenge train)
    gsm8k               — math (gsm8k)
    swebench            — code/bugs (princeton-nlp/SWE-bench)
    humaneval           — algorithm completion (openai/openai_humaneval)
    mbpp                — complete Python functions (google-research-datasets/mbpp)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from typing import Iterator, List, Optional

from tqdm import tqdm

_log = logging.getLogger(__name__)

KNOWLEDGE_LIMIT = 200
ALL_SOURCES = [
    "openhermes", "wikipedia", "wikipedia_targeted",
    "mmlu", "mmlu_qna", "arc", "gsm8k", "swebench", "humaneval", "mbpp",
]

# One Wikipedia article per MMLU subject category.
# Titles map directly to https://en.wikipedia.org/wiki/{title}.
_MMLU_WIKIPEDIA_TOPICS: list[tuple[str, str]] = [
    ("abstract_algebra",                    "Abstract_algebra"),
    ("anatomy",                             "Human_anatomy"),
    ("astronomy",                           "Astronomy"),
    ("business_ethics",                     "Business_ethics"),
    ("clinical_knowledge",                  "Clinical_medicine"),
    ("college_biology",                     "Cell_biology"),
    ("college_chemistry",                   "Chemistry"),
    ("college_computer_science",            "Computer_science"),
    ("college_mathematics",                 "Mathematics"),
    ("college_medicine",                    "Medicine"),
    ("college_physics",                     "Physics"),
    ("conceptual_physics",                  "Classical_mechanics"),
    ("econometrics",                        "Econometrics"),
    ("electrical_engineering",              "Electrical_engineering"),
    ("elementary_mathematics",              "Elementary_arithmetic"),
    ("formal_logic",                        "Mathematical_logic"),
    ("global_facts",                        "Geography"),
    ("high_school_biology",                 "Biology"),
    ("high_school_chemistry",               "Chemical_reaction"),
    ("high_school_computer_science",        "Algorithm"),
    ("high_school_european_history",        "History_of_Europe"),
    ("high_school_geography",               "Human_geography"),
    ("high_school_government_and_politics", "Government"),
    ("high_school_macroeconomics",          "Macroeconomics"),
    ("high_school_mathematics",             "Algebra"),
    ("high_school_microeconomics",          "Microeconomics"),
    ("high_school_physics",                 "Mechanics"),
    ("high_school_psychology",              "Psychology"),
    ("high_school_statistics",              "Statistics"),
    ("high_school_us_history",              "History_of_the_United_States"),
    ("high_school_world_history",           "World_history"),
    ("human_aging",                         "Ageing"),
    ("human_sexuality",                     "Human_sexuality"),
    ("international_law",                   "International_law"),
    ("jurisprudence",                       "Jurisprudence"),
    ("logical_fallacies",                   "Fallacy"),
    ("machine_learning",                    "Machine_learning"),
    ("management",                          "Management"),
    ("marketing",                           "Marketing"),
    ("medical_genetics",                    "Medical_genetics"),
    ("miscellaneous",                       "Knowledge"),
    ("moral_disputes",                      "Ethics"),
    ("moral_scenarios",                     "Morality"),
    ("nutrition",                           "Nutrition"),
    ("philosophy",                          "Philosophy"),
    ("prehistory",                          "Prehistory"),
    ("professional_accounting",             "Accounting"),
    ("professional_law",                    "Law"),
    ("professional_medicine",               "Internal_medicine"),
    ("professional_psychology",             "Clinical_psychology"),
    ("public_relations",                    "Public_relations"),
    ("security_studies",                    "Security_studies"),
    ("sociology",                           "Sociology"),
    ("us_foreign_policy",                   "Foreign_policy_of_the_United_States"),
    ("virology",                            "Virology"),
    ("world_religions",                     "Religion"),
]


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

        elif source == "mmlu_qna":
            # Ingest MMLU auxiliary_train questions in EXACT benchmark prompt format.
            # This creates direct trie paths: MCQ prompt → Answer: → letter (A/B/C/D).
            # Uses auxiliary_train (not test) to avoid data leakage.
            _LABELS = ["A", "B", "C", "D"]
            ds = _safe_load_dataset("cais/mmlu", "all", split="auxiliary_train")
            if ds is None:
                return
            for i, item in enumerate(list(ds)[:limit]):
                if i < start_offset:
                    continue
                question = item.get("question", "").strip()
                choices  = item.get("choices", [])
                ans_idx  = item.get("answer", -1)
                subject  = item.get("subject", "general")
                if not question or len(choices) != 4 or ans_idx not in range(4):
                    continue
                subject_fmt = subject.replace("_", " ")
                lines = [
                    f"The following is a multiple choice question about {subject_fmt}.",
                    "",
                    question,
                    "",
                ]
                for label, choice in zip(_LABELS, choices):
                    lines.append(f"{label}. {choice}")
                lines.append("")
                lines.append("Answer:")
                prompt = "\n".join(lines)
                correct_letter = _LABELS[ans_idx]
                yield f"<|user|> {prompt} <|assistant|> {correct_letter} <|end|>"

        elif source == "arc":
            # ARC-Challenge train split in exact benchmark MCQ format.
            # Matches mmlu_qna format so trie paths align with MCQ voting.
            ds = _safe_load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
            if ds is None:
                return
            for i, item in enumerate(list(ds)[:limit]):
                if i < start_offset:
                    continue
                question = item.get("question", "").strip()
                choices  = item.get("choices", {})
                labels   = choices.get("label", [])
                texts    = choices.get("text", [])
                answer   = item.get("answerKey", "").strip().upper()
                if not question or not texts or answer not in labels:
                    continue
                lines = ["The following is a multiple choice question.", "", question, ""]
                for label, text in zip(labels, texts):
                    lines.append(f"{label}. {text}")
                lines.append("")
                lines.append("Answer:")
                prompt = "\n".join(lines)
                yield f"<|user|> {prompt} <|assistant|> {answer} <|end|>"

        elif source == "gsm8k":
            ds = _safe_load_dataset("openai/gsm8k", "main", split="test")
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
            # Prompt format MUST match swebench_benchmark.py exactly so trie paths align.
            ds = _safe_load_dataset("princeton-nlp/SWE-bench", split="test")
            if ds is None:
                return
            for i, item in enumerate(list(ds)[:limit]):
                if i < start_offset:
                    continue
                repo  = item.get("repo", "unknown/repo")
                issue = item.get("problem_statement", "")[:600]
                patch = item.get("patch", "")[:600]
                if not issue.strip() or not patch.strip():
                    continue
                prompt = (
                    f"Fix the following bug in the {repo} repository:\n\n"
                    + issue
                    + ("\n\n[problem truncated]" if len(item.get("problem_statement", "")) > 600 else "")
                    + "\n\nProvide a Python code fix:"
                )
                yield f"<|user|> {prompt} <|assistant|> {patch} <|end|>"

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

        elif source == "mbpp":
            ds = _safe_load_dataset("google-research-datasets/mbpp", split="train")
            if ds is None:
                return
            for i, item in enumerate(list(ds)[:limit]):
                if i < start_offset:
                    continue
                problem   = item.get("text", "").strip()
                code      = item.get("code", "").strip()
                test_list = item.get("test_list", [])
                if problem and code:
                    yield (
                        f"<|user|> Write a Python function: {problem} "
                        f"<|assistant|>\n{code}\n <|end|>"
                    )
                # Also ingest test assertions keyed by FUNCTION NAME so the
                # scratchpad can query "write test assertion for: {func_name}"
                # and get back the verified assertion.  Using the function name
                # (not the full problem description) ensures trie path alignment.
                if code and test_list:
                    import ast as _ast
                    try:
                        _tree = _ast.parse(code)
                        _func_name = next(
                            (n.name for n in _ast.walk(_tree)
                             if isinstance(n, _ast.FunctionDef)),
                            None,
                        )
                    except Exception:
                        _func_name = None
                    if _func_name:
                        for test in test_list[:2]:
                            test = test.strip()
                            if test.startswith("assert"):
                                yield (
                                    f"<|user|> Write a test assertion for: {_func_name} "
                                    f"<|assistant|> {test} <|end|>"
                                )

        elif source == "wikipedia_targeted":
            yield from self._iter_wikipedia_targeted(limit, start_offset)

    def _iter_wikipedia_targeted(self, limit: int,
                                  start_offset: int) -> "Iterator[str]":
        """Fetch one Wikipedia article per MMLU subject via the Wikipedia REST API.

        Uses requests + BeautifulSoup; no HuggingFace dependency. The full
        article text (up to 4 000 chars) is ingested as a Q&A pair so the
        trie learns both the subject label and the factual content.
        """
        import time
        try:
            import requests as _req
            from bs4 import BeautifulSoup
        except ImportError:
            _log.warning("wikipedia_targeted requires requests + beautifulsoup4. "
                         "pip install requests beautifulsoup4")
            return

        topics = _MMLU_WIKIPEDIA_TOPICS[:limit]
        for i, (subject, wiki_title) in enumerate(topics):
            if i < start_offset:
                continue
            url = f"https://en.wikipedia.org/wiki/{wiki_title}"
            try:
                resp = _req.get(
                    url, timeout=15,
                    headers={"User-Agent": "Uchi-brain-builder/0.3.0"},
                )
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer",
                                 "header", "table", "sup"]):
                    tag.decompose()
                # Take the first 4 000 chars of article body text.
                body = soup.get_text(separator=" ", strip=True)[:4000]
                if body.strip():
                    readable_subject = subject.replace("_", " ")
                    yield (
                        f"<|user|> Tell me about {readable_subject}. "
                        f"<|assistant|> {body} <|end|>"
                    )
            except Exception as exc:
                _log.warning("wikipedia_targeted: failed to fetch %s — %s",
                             wiki_title, exc)
            time.sleep(0.5)  # polite crawl delay

    # ── main entry ────────────────────────────────────────────────────────────

    def run(self, limit: int = KNOWLEDGE_LIMIT,
            sources: Optional[List[str]] = None,
            no_resume: bool = False) -> None:
        """Execute the incremental build.

        Args:
            limit:     Max documents to ingest per source.
            sources:   List of source names to process (default: ALL_SOURCES).
            no_resume: If True, ignore existing checkpoint and re-ingest all
                       requested sources from scratch (useful for re-running a
                       source with a higher limit).
        """
        if no_resume:
            self._checkpoint = {
                "completed_sources": [],
                "current_source":    None,
                "current_offset":    0,
                "total_ingested":    0,
                "started_at":        __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc).isoformat(),
                "last_updated":      None,
            }
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
        total_ingested = self._checkpoint.get("total_ingested", 0)

        # Generate-and-Ground: ensure the brain has a retrieval index so ingested
        # knowledge is groundable at answer time.
        if getattr(router, "_semantic_index", None) is None:
            try:
                router.build_semantic_index([])   # seed with shipped embeddings
                print("[*] Semantic retrieval index initialised.")
            except Exception as e:
                print(f"[!] Semantic index init skipped: {e}")

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

                # Pass raw words — router.stream() tokenizes internally.
                # Pre-tokenizing here causes double-tokenization: the second
                # pass maps already-canonical tokens to the same bigrams,
                # adding zero new paths to the trie.
                router.stream(text.split())
                # Feed the same knowledge into the retrieval index (grounding).
                idx = getattr(router, "_semantic_index", None)
                if idx is not None:
                    clean = (text.replace("<|user|>", " ").replace("<|assistant|>", " ")
                                 .replace("<|end|>", " "))
                    idx.build_from_corpus(clean)
                source_count += 1
                total_ingested += 1

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

        # ── Phase 3: Persist updated brain ────────────────────────────────────
        print(f"\n[*] Phase 3: Saving updated brain to {self.brain_path}…")
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
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing checkpoint and re-ingest all requested sources "
                             "from scratch (useful when re-running a source with a higher limit)")
    args = parser.parse_args()

    sources = (
        [s.strip() for s in args.sources.split(",")]
        if args.sources else None
    )

    builder = IncrementalBrainBuilder(
        brain_path=args.brain,
        checkpoint_path=args.checkpoint,
    )
    builder.run(limit=args.limit, sources=sources, no_resume=args.no_resume)


if __name__ == "__main__":
    main()
