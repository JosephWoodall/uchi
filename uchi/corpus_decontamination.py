"""corpus_decontamination.py — repo-level SWE-bench exclusion gate for
pretraining corpora (0.5.0 Item 1's decontamination filter).

Why this exists
----------------
0.5.0 adds a new pretraining-corpus source: real GitHub issue->diff pairs at
scale. SWE-bench (the eval benchmark Item 9 reports 0.5.0's headline number
against) draws its tasks from a small, fixed set of well-known open-source
Python repos. If the training corpus contains *any* issue from one of those
repos — even a completely different issue than the ones SWE-bench actually
tests — the model can pick up repo-specific idioms and conventions that
inflate the eventual SWE-bench score without proving real generalization.
That would make Item 9's number measure memorization, not the mechanism
stack 0.5.0 is built to prove out.

The gate is therefore **repo-level**, not instance-level: every record whose
`repo` field matches an excluded repo is dropped, regardless of which
specific issue/instance it is. Deduplicating against only the exact
instance ids SWE-bench evaluates would leave a repo's other issues in the
corpus, and those still teach repo-specific conventions.

Where the exclusion list comes from
------------------------------------
The list is resolved live from the actual SWE-bench dataset(s) on the HF
Hub (`datasets.load_dataset`), never hardcoded from memory in this module.
A hardcoded snapshot would go stale silently — SWE-bench's repo list isn't
guaranteed frozen forever, and a stale list would silently reintroduce the
exact contamination this gate exists to prevent. `resolve_excluded_repos`
is therefore the actual source of truth every time it runs; `save_repo_cache`
/ `load_repo_cache` exist only as an explicit, opt-in local snapshot for
offline reuse (e.g. a data-pipeline worker with no network egress) — never
used implicitly, and never falls back to a bundled list on network failure.
A network failure here is a real failure and is raised as one.

What was verified against the live HF Hub (2026-07-09), documented here so
a future reader doesn't have to re-derive it:
  - `SWE-bench/SWE-bench` (the maintained org; `princeton-nlp/SWE-bench` is
    the original/legacy mirror with identical `test`-split content) has a
    `test` split of 2294 instances spanning exactly 12 repos.
  - `SWE-bench/SWE-bench_Lite` (300 instances) and `SWE-bench/SWE-bench_Verified`
    (500 instances) are **filtered instance-level subsets of that same
    12-repo population** — not a different set of repos. Their `test`
    splits' unique `repo` values are identical to the full benchmark's.
    This means the repo-level exclusion set collapses to the full
    benchmark's 12 repos regardless of which of the three you load; all
    three are still resolved and unioned below in case a future SWE-bench
    revision diverges (e.g. Lite/Verified adding a repo before Full does).
  - The full dataset also ships `dev` (225 instances, 6 *different* repos —
    marshmallow, pvlib, pydicom, astroid, pyvista, sqlfluff) and `train`
    (19008 instances, ~29 repos entirely unrelated to the eval set — numpy,
    pandas, transformers, scipy, ray, etc.) splits. Neither is part of
    SWE-bench's scored eval set: `dev` is a held-out development split,
    `train` is a distinct fine-tuning pool. Both are deliberately excluded
    from `DEFAULT_SWE_BENCH_SPECS` — including them would exclude dozens of
    repos that were never actually part of the benchmark being guarded
    against, over-shrinking the corpus for no decontamination benefit.

Distinct from Item 3.5's per-repo retrieval index (`uchi/retrieval.py` /
the planned `LocalKnowledgeAnchor` port): that index is built from the
actual task repo *at eval time* and is unrelated to this training-corpus
gate — explicitly exempt per `tasks/todo.md`'s Item 3.5 entry.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_CACHE_PATH = Path(".uchi") / "corpus_decontamination" / "swebench_repos.json"


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    split: str = "test"


# Only the splits that actually constitute SWE-bench's scored eval set —
# see module docstring for why `dev`/`train` are deliberately not here.
DEFAULT_SWE_BENCH_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec("SWE-bench/SWE-bench", "test"),
    DatasetSpec("SWE-bench/SWE-bench_Lite", "test"),
    DatasetSpec("SWE-bench/SWE-bench_Verified", "test"),
)


def resolve_excluded_repos(specs: Iterable[DatasetSpec] = DEFAULT_SWE_BENCH_SPECS) -> frozenset[str]:
    """Load the given SWE-bench dataset splits from the HF Hub and return the
    union of their `repo` values. This is the actual source of truth — no
    fallback to a bundled list if the network call fails.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "the `datasets` package is required to resolve the SWE-bench repo "
            "exclusion list (pip install 'uchi-python[dev]'). There is no "
            "hardcoded fallback list — see corpus_decontamination.py's module "
            "docstring for why."
        ) from e

    repos: set[str] = set()
    for spec in specs:
        ds = load_dataset(spec.dataset_id, split=spec.split)
        if "repo" not in ds.column_names:
            raise RuntimeError(
                f"{spec.dataset_id!r} split {spec.split!r} has no 'repo' column "
                f"(columns present: {ds.column_names}) — the upstream schema "
                "changed; update DatasetSpec/field handling rather than guessing."
            )
        repos.update(ds.unique("repo"))
    return frozenset(repos)


def save_repo_cache(
    repos: frozenset[str],
    path: str | Path = DEFAULT_CACHE_PATH,
    specs: Iterable[DatasetSpec] = DEFAULT_SWE_BENCH_SPECS,
) -> Path:
    """Write an explicit, opt-in local snapshot of a resolved repo set, for
    offline reuse. Never read implicitly by `resolve_excluded_repos` — the
    caller decides when a cached snapshot is an acceptable substitute for a
    live resolution.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "resolved_at": datetime.now(timezone.utc).isoformat(),
        "specs": [{"dataset_id": s.dataset_id, "split": s.split} for s in specs],
        "repos": sorted(repos),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_repo_cache(path: str | Path = DEFAULT_CACHE_PATH) -> frozenset[str]:
    """Read a previously saved snapshot. Raises if it doesn't exist rather
    than silently returning an empty set — an empty exclusion set would
    silently disable the decontamination gate.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no cached repo list at {path} — run with --refresh (or call "
            "resolve_excluded_repos() + save_repo_cache() directly) first."
        )
    payload = json.loads(path.read_text())
    return frozenset(payload["repos"])


def is_excluded(record: Mapping[str, Any], excluded_repos: frozenset[str]) -> bool:
    """Repo-level match: excludes every record from a contaminated repo, not
    just the specific instance ids SWE-bench evaluates.
    """
    return record["repo"] in excluded_repos


@dataclass
class FilterSummary:
    total: int
    kept: int
    excluded_by_repo: dict[str, int] = field(default_factory=dict)

    @property
    def excluded(self) -> int:
        return self.total - self.kept


def filter_corpus(
    records: Iterable[Mapping[str, Any]],
    excluded_repos: frozenset[str],
) -> tuple[list[Mapping[str, Any]], FilterSummary]:
    """Split *records* into (kept, excluded-summary) against the repo-level gate."""
    kept: list[Mapping[str, Any]] = []
    excluded_by_repo: dict[str, int] = {}
    total = 0
    for record in records:
        total += 1
        if is_excluded(record, excluded_repos):
            repo = record["repo"]
            excluded_by_repo[repo] = excluded_by_repo.get(repo, 0) + 1
        else:
            kept.append(record)
    return kept, FilterSummary(total=total, kept=len(kept), excluded_by_repo=excluded_by_repo)


def _read_records(path: str) -> list[dict]:
    fh = sys.stdin if path == "-" else open(path, encoding="utf-8")
    try:
        return [json.loads(line) for line in fh if line.strip()]
    finally:
        if fh is not sys.stdin:
            fh.close()


def _write_records(records: list[Mapping[str, Any]], path: str) -> None:
    fh = sys.stdout if path == "-" else open(path, "w", encoding="utf-8")
    try:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    finally:
        if fh is not sys.stdout:
            fh.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="uchi-corpus-decontaminate",
        description=(
            "Repo-level SWE-bench decontamination gate: drop every corpus "
            "record whose 'repo' field belongs to a repo in SWE-bench's eval set."
        ),
    )
    parser.add_argument("--input", "-i", default="-", help="JSONL input path, or - for stdin")
    parser.add_argument("--output", "-o", default="-", help="JSONL output path, or - for stdout")
    parser.add_argument(
        "--cache-file",
        default=None,
        help=(
            "Optional local snapshot path. If it exists, load the exclusion "
            "set from it instead of hitting the HF Hub. If given but missing, "
            "resolve live and write it there for next time. Omit entirely to "
            "always resolve live (the default, and the safest choice)."
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force a live re-resolution from the HF Hub even if --cache-file exists.",
    )
    args = parser.parse_args(argv)

    if args.cache_file and Path(args.cache_file).exists() and not args.refresh:
        excluded_repos = load_repo_cache(args.cache_file)
    else:
        excluded_repos = resolve_excluded_repos()
        if args.cache_file:
            save_repo_cache(excluded_repos, args.cache_file)

    records = _read_records(args.input)
    kept, summary = filter_corpus(records, excluded_repos)
    _write_records(kept, args.output)

    print(
        f"[corpus_decontamination] total={summary.total} kept={summary.kept} "
        f"excluded={summary.excluded}",
        file=sys.stderr,
    )
    if summary.excluded_by_repo:
        print("[corpus_decontamination] excluded by repo:", file=sys.stderr)
        for repo, count in sorted(summary.excluded_by_repo.items(), key=lambda kv: -kv[1]):
            print(f"  {repo}: {count}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
