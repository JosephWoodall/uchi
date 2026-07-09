"""retrieval_scaling_benchmark.py — 0.5.0 Item 4: benchmark Item 3's flat
fact retrieval (`uchi/retrieval.py`'s `SemanticIndex`) and Item 3.5's code
retrieval (`uchi/code_retrieval.py`'s `CodeIndex`) **separately**, per
todo.md's explicit instruction — the two systems have different cost
profiles that don't transfer into one number:

  - Item 3's embeddings are pre-trained once (shipped in
    `uchi/data/skipgram_emb.pt`) and reused across every query forever —
    only *retrieve()* has a per-query cost.
  - Item 3.5 trains a fresh skip-gram model **per repo, at index-build
    time** (`code_retrieval.py`'s own docstring: "an unfamiliar repo's
    identifier vocabulary shares ~nothing with the brain's natural-language
    co-occurrence statistics"). Index-build latency is therefore a real,
    separate cost Item 3's numbers say nothing about.

Two independent tracks, each against real data (not synthetic queries):

  1. **Fact retrieval** (Item 3): real SQuAD 2.0 answerable questions
     indexed the same way `benchmarks/trustworthiness.py` does, but scored
     on pure retrieval hit-rate (is a gold answer string present in the
     top-k retrieved passages) + query latency — decoupled from
     `trustworthiness.py`'s end-to-end generate+oracle KPIs, which measure
     something broader than retrieval alone.
  2. **Code retrieval** (Item 3.5): real `SWE-bench_Lite` instances, each
     repo fetched via `uchi/repo_fetch.py` and indexed fresh with
     `CodeIndex.build`, scored on whether the top-k retrieved chunks land
     in a file the instance's *gold* patch actually touches (extracted
     from the real unified diff) — a genuine "would this retrieval have
     pointed `agentic_repair.py` at the right file" signal, plus separate
     build-latency and query-latency numbers.

Usage:
    python -m benchmarks.retrieval_scaling_benchmark
    python -m benchmarks.retrieval_scaling_benchmark --fact-sample 500 --code-sample 10
    python -m benchmarks.retrieval_scaling_benchmark --skip-code   # fact track only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

_EMB = os.path.join(os.path.dirname(__file__), "..", "uchi", "data", "skipgram_emb.pt")
_DIFF_FILE_RE = re.compile(r"^diff --git a/(.+?) b/", re.MULTILINE)


def _extract_patch_files(patch: str) -> set[str]:
    return set(_DIFF_FILE_RE.findall(patch))


def bench_fact_retrieval(sample: int) -> dict:
    from datasets import load_dataset

    from uchi.retrieval import SemanticIndex

    print("  [fact] loading SQuAD 2.0 ...")
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    import random
    random.seed(0)
    order = list(range(len(ds)))
    random.shuffle(order)
    rows = []
    for i in order:
        row = ds[i]
        if row["answers"]["text"]:  # only answerable questions have a gold string to hit
            rows.append(row)
        if len(rows) >= sample:
            break

    index = SemanticIndex.from_embeddings_file(_EMB)
    seen: set[str] = set()
    t0 = time.time()
    for row in rows:
        if row["context"] not in seen:
            seen.add(row["context"])
            index.build_from_corpus(row["context"])
    build_s = time.time() - t0
    print(f"  [fact] indexed {len(index):,} passages from {len(seen)} contexts in {build_s:.1f}s")

    hits = 0
    query_times: list[float] = []
    for row in rows:
        golds = [g.lower() for g in row["answers"]["text"] if g.strip()]
        t0 = time.time()
        results = index.retrieve(row["question"], k=5)
        query_times.append(time.time() - t0)
        if any(g in passage.lower() for passage, _ in results for g in golds):
            hits += 1

    n = len(rows)
    return {
        "track": "item3_fact_retrieval",
        "n_queries": n,
        "indexed_passages": len(index),
        "index_build_s": round(build_s, 2),
        "hit_rate_at_5": round(hits / n, 4) if n else 0.0,
        "avg_query_ms": round(1000 * sum(query_times) / n, 3) if n else 0.0,
    }


def bench_code_retrieval(sample: int) -> dict:
    from datasets import load_dataset

    from uchi.code_retrieval import CodeIndex
    from uchi.repo_fetch import RepoFetchError, ensure_local_clone

    print("  [code] loading SWE-bench_Lite ...")
    ds = load_dataset("SWE-bench/SWE-bench_Lite", split="test")
    import random
    random.seed(0)
    order = list(range(len(ds)))
    random.shuffle(order)

    n_hits = n_scored = 0
    build_times: list[float] = []
    query_times: list[float] = []
    indexed_repos: dict[str, CodeIndex] = {}

    for i in order:
        if n_scored >= sample:
            break
        row = ds[i]
        gold_files = _extract_patch_files(row["patch"])
        if not gold_files:
            continue

        repo, base_commit = row["repo"], row["base_commit"]
        try:
            repo_path = ensure_local_clone(repo, base_commit)
        except RepoFetchError as e:
            print(f"  [code] skip {row['instance_id']}: clone failed ({e})")
            continue

        cache_key = f"{repo}@{base_commit}"
        if cache_key not in indexed_repos:
            t0 = time.time()
            indexed_repos[cache_key] = CodeIndex.build(repo_path)
            build_times.append(time.time() - t0)
        code_index = indexed_repos[cache_key]

        t0 = time.time()
        results = code_index.retrieve(row["problem_statement"], k=10)
        query_times.append(time.time() - t0)

        retrieved_files = {chunk.file for chunk, _ in results}
        hit = bool(retrieved_files & gold_files)
        n_hits += int(hit)
        n_scored += 1
        print(f"  [code] [{n_scored}/{sample}] {row['instance_id']}: {'HIT' if hit else 'miss'}")

    return {
        "track": "item3.5_code_retrieval",
        "n_queries": n_scored,
        "n_repos_indexed": len(indexed_repos),
        "avg_index_build_s": round(sum(build_times) / len(build_times), 2) if build_times else 0.0,
        "hit_rate_at_10": round(n_hits / n_scored, 4) if n_scored else 0.0,
        "avg_query_ms": round(1000 * sum(query_times) / n_scored, 3) if n_scored else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="0.5.0 Item 4: separate retrieval benchmarks for Item 3 vs Item 3.5",
    )
    parser.add_argument("--fact-sample", type=int, default=200)
    parser.add_argument("--code-sample", type=int, default=5)
    parser.add_argument("--skip-fact", action="store_true")
    parser.add_argument("--skip-code", action="store_true")
    parser.add_argument(
        "--out", default=os.path.join(os.path.dirname(__file__), "retrieval_scaling_results.json"),
    )
    args = parser.parse_args(argv)

    print("\n" + "=" * 70)
    print(" Uchi Retrieval Scaling Validation (0.5.0 Item 4)")
    print("=" * 70 + "\n")

    results: dict[str, dict] = {}
    if not args.skip_fact:
        results["item3_fact_retrieval"] = bench_fact_retrieval(args.fact_sample)
    if not args.skip_code:
        results["item3.5_code_retrieval"] = bench_code_retrieval(args.code_sample)

    print("\n" + "─" * 70)
    print("  Results (separate tracks — do not compare hit-rates across them,")
    print("  the two systems answer structurally different query shapes)")
    print("─" * 70)
    for track, r in results.items():
        print(f"  {track}:")
        for k, v in r.items():
            if k != "track":
                print(f"    {k}: {v}")
    print("─" * 70)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
