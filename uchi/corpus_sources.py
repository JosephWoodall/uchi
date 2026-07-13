"""corpus_sources.py — 0.5.0 Item 1's two corpus streams: the general-code
portion and the new GitHub issue->diff pairs source.

Sourcing decision (todo.md: "Source (a)/(b)/(c) general-code portion — pick
and document which"): **The Stack v2 (BigCode), Python subset.** Chosen over
CodeSearchNet (too thin — docstring-paired functions only, not full-file
context) and reusing 0.4.0's Magicoder/CodeAlpaca/CommitPackFT (those are
SFT-scale sources, ~10K-50K examples fair-budgeted against three other
datasets — not sized for a pretraining corpus). The Stack is the standard
choice for from-scratch code pretraining at this scale, and — same
discipline as ``scripts/pretokenize.py``'s FineWeb-Edu source — streamed via
``datasets.load_dataset(..., streaming=True)`` rather than bulk-downloaded.

**Real schema, confirmed live 2026-07-09 with an authenticated token
(correcting two wrong assumptions made before auth was available):**
``bigcode/the-stack-v2-dedup`` is an **index over Software Heritage, not a
content dataset** — streamed rows carry ``repo_name``/``path``/``language``/
``license_type``/``is_generated``/``is_vendor``/``blob_id``/``src_encoding``,
no ``content`` field at all. File content has to be fetched separately, per
row, from SWH's public S3 mirror (anonymous/unsigned access, bucket
``softwareheritage``, key ``content/{blob_id}``, gzip-compressed, decoded
with the row's own ``src_encoding``) — ``_fetch_swh_content`` below. This is
BigCode's own documented retrieval path for v2, not a workaround. It is also
**not** pre-filtered to permissive licenses (the earlier draft of this
docstring claimed that; live sampling showed both ``"permissive"`` and
``"no_license"`` rows) — ``license_type == "permissive"`` is filtered
client-side here, alongside ``is_generated``/``is_vendor`` exclusion (both
BigCode-provided quality flags: generated and vendored code aren't
representative human-written training signal). The dataset also exposes a
per-language HF *config* (confirmed: ``load_dataset(dataset_id, "Python",
split="train", streaming=True)``), used instead of streaming every language
and filtering client-side — far cheaper, and the same reason
``the-stack-dedup`` v1 uses a ``data_dir`` per-language layout below.

A per-row SWH fetch occasionally misses (BigCode's dataset card documents a
small, expected non-zero miss rate — content that was garbage-collected or
never archived) or decodes wrong (``src_encoding`` mismatched to actual
bytes) — both are skipped with a visible stderr line, same "best-effort
ingestion, not a swallowed real failure" discipline as
``code_retrieval.py``'s unparseable-file skip. An S3 permissions/auth error,
by contrast, is not swallowed — it means the *pipeline* is broken, not that
one file was ungrabbable, so it propagates.

Primary id is ``bigcode/the-stack-v2-dedup``; ``bigcode/the-stack-dedup``
(v1, content included inline, no SWH fetch needed) is an explicit, documented
fallback if v2 is unavailable, same shape as ``pretokenize.py``'s FineWeb-Edu
-> OpenWebText fallback. **Both are gated HF datasets, under separate grants**
(confirmed live 2026-07-09): v2 auto-approved on accepting its terms; v1 was
still pending/denied for the token used to verify this module and its exact
row schema could not be live-confirmed as a result — the repo-identifier/
content/path columns for the v1 fallback path are still resolved from a
prioritized candidate list (best recollection: ``max_stars_repo_name``/
``content``/``max_stars_repo_path``) rather than a verified name, same
"raise loud rather than guess" discipline as
``corpus_decontamination.resolve_excluded_repos``. Confirm those columns
against a live authenticated pull once v1 access clears, and update
``_STACK_REPO_COLUMNS``/``_STACK_CONTENT_COLUMNS``/``_STACK_PATH_COLUMNS`` if
they differ — the fallback path is not yet exercised against real data the
way the v2 path now is.

Issue->diff source: **SWE-Gym** (``SWE-Gym/SWE-Gym``, Yuxuan Tong et al.,
"SWE-Gym: Reinforcement Learning Environments for Training Software
Engineering Agents", 2024/2025) — real GitHub issues paired with the PR
diff that resolved them, in SWE-bench's own schema
(``repo``/``problem_statement``/``patch``/``base_commit``/``FAIL_TO_PASS``/
``PASS_TO_PASS``), built explicitly from repos disjoint from SWE-bench's
eval set. Confirmed live (2026-07-09, ungated, no token needed): schema
matches exactly, first 500 streamed rows span only ``getmoto/moto`` and
``python/mypy`` — both outside SWE-bench's 12-repo eval set. SWE-Gym's own
disjointness claim is not treated as sufficient on its own, though — every
record still passes through ``corpus_decontamination.filter_corpus`` before
reaching Item 2, same defense-in-depth reasoning as the module docstring
there ("resolved live... never hardcoded from memory"). ``SWE-Gym/SWE-Gym-Raw``
(the larger, less-curated ~66K-instance release) is available via
``dataset_id=SWE_GYM_RAW_DATASET_ID`` for later scale-up; not the default,
since the curated set is the safer starting point for Item 6's curriculum.

Both ``iter_*`` functions yield plain dicts with a ``"repo"`` key, so they
compose directly with ``corpus_decontamination.filter_corpus`` — see
``main()`` below for the actual pipeline wiring.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from typing import Any

from uchi.corpus_decontamination import filter_corpus, load_repo_cache, resolve_excluded_repos

STACK_V2_DATASET_ID = "bigcode/the-stack-v2-dedup"
STACK_V1_FALLBACK_DATASET_ID = "bigcode/the-stack-dedup"
STACK_V2_LANGUAGE_CONFIG = "Python"       # confirmed live: HF dataset config name
STACK_LANGUAGE_DATA_DIR = "data/python"   # v1 fallback's per-language layout (unverified)

SWH_S3_BUCKET = "softwareheritage"

# v1 fallback only — v2's schema is confirmed and handled directly.
# Priority order: try each until one is present in the streamed row.
_STACK_REPO_COLUMNS = ("max_stars_repo_name", "repo_name", "repo")
_STACK_CONTENT_COLUMNS = ("content",)
_STACK_PATH_COLUMNS = ("max_stars_repo_path", "path")

SWE_GYM_DATASET_ID = "SWE-Gym/SWE-Gym"
SWE_GYM_RAW_DATASET_ID = "SWE-Gym/SWE-Gym-Raw"

_SWE_GYM_REQUIRED_COLUMNS = ("repo", "instance_id", "problem_statement", "patch")


class CorpusSourceError(RuntimeError):
    """Raised on a schema mismatch or an auth/gating failure that would
    otherwise surface as an opaque upstream traceback."""


def _first_present(row: Any, candidates: tuple[str, ...], what: str) -> str:
    for name in candidates:
        if name in row:
            return name
    raise CorpusSourceError(
        f"none of {candidates!r} found for {what} — upstream schema changed "
        f"or my recollection of it was wrong; actual columns: {sorted(row.keys())}. "
        "Update the candidate list rather than guessing further."
    )


def _load_streaming(dataset_id: str, **kwargs: Any):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise CorpusSourceError(
            "the `datasets` package is required (pip install 'uchi-python[dev]')"
        ) from e

    try:
        from huggingface_hub.errors import GatedRepoError
    except ImportError:
        GatedRepoError = ()  # type: ignore[assignment]
    from datasets.exceptions import DatasetNotFoundError

    try:
        return load_dataset(dataset_id, split="train", streaming=True, **kwargs)
    except (GatedRepoError, DatasetNotFoundError) as e:
        # `datasets` itself catches GatedRepoError internally and re-raises
        # as DatasetNotFoundError with "gated" in the message (confirmed
        # live 2026-07-09 against this exact datasets version) — checking
        # the message is what actually distinguishes "gated, needs auth"
        # from "typo'd dataset id", not the exception class alone.
        if "gated" not in str(e).lower():
            raise
        raise CorpusSourceError(
            f"{dataset_id!r} is a gated dataset — accept its terms at "
            f"https://huggingface.co/datasets/{dataset_id} while logged in, "
            "then authenticate this environment (`huggingface-cli login` or "
            "`HF_TOKEN=...`). Not something this module can do on your behalf."
        ) from e


def _swh_s3_client(max_pool_connections: int = 64) -> Any:
    """Anonymous/unsigned S3 client for Software Heritage's public content
    mirror — no SWH-side auth needed, separate from the HF token gating the
    dataset index itself.

    *max_pool_connections* must be raised to match (or exceed) the fetch
    thread pool's worker count -- botocore's own default is only 10,
    silently capping concurrency far below whatever `ThreadPoolExecutor`
    size `_iter_stack_v2` actually uses (found live: 150 fetch threads
    against the 10-connection default performed *worse* than 100 threads
    against it, extra threads just queuing and contending for the same 10
    connections rather than adding real parallelism).
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    return boto3.client(
        "s3", config=Config(signature_version=UNSIGNED, max_pool_connections=max_pool_connections),
    )


def _fetch_swh_content(client: Any, blob_id: str, src_encoding: str) -> str | None:
    """Fetch one file's content from Software Heritage via *client*. Returns
    ``None`` (a skip, not an error) on a missing blob or a bad decode — both
    are a known, low-rate condition per BigCode's the-stack-v2 dataset card,
    not a broken pipeline. Any other S3 failure (auth, permissions, bucket
    unreachable) propagates — that *is* a broken pipeline.
    """
    import gzip

    from botocore.exceptions import ClientError

    try:
        obj = client.get_object(Bucket=SWH_S3_BUCKET, Key=f"content/{blob_id}")
        with gzip.GzipFile(fileobj=obj["Body"]) as fin:
            return fin.read().decode(src_encoding)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404"):
            print(f"  [!] SWH blob missing, skipping: {blob_id}", file=sys.stderr)
            return None
        raise
    except (UnicodeDecodeError, OSError) as e:
        print(f"  [!] SWH blob {blob_id} failed to decode as {src_encoding!r} ({e}), skipping",
              file=sys.stderr)
        return None


def _iter_stack_v2(limit: int | None, max_workers: int = 32) -> Iterator[dict[str, Any]]:
    """Bounded-concurrency SWH fetch. Confirmed live (0.5.0 Item 1 real-scale
    pull): sequential fetches ran at ~4.1 files/s -- an I/O-bound bottleneck
    (network round-trip per file), not CPU-bound, so a thread pool helps a
    lot here despite the GIL (boto3/botocore release it during the actual
    HTTP call, and a single client is documented thread-safe for concurrent
    read calls -- standard AWS SDK usage pattern, not a hack). Order isn't
    preserved (results come back as fetches complete, not as HF streamed
    them) -- fine here, this corpus gets interleaved/shuffled downstream
    regardless (`pretokenize_0_5_0.py`).
    """
    ds = _load_streaming(STACK_V2_DATASET_ID, name=STACK_V2_LANGUAGE_CONFIG)
    client = _swh_s3_client(max_pool_connections=max_workers)

    import concurrent.futures

    def qualifies(row: dict) -> bool:
        return not (row["is_generated"] or row["is_vendor"] or row["license_type"] != "permissive")

    count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        row_iter = iter(ds)
        in_flight: dict[concurrent.futures.Future, dict] = {}

        def submit_next() -> bool:
            for row in row_iter:
                if qualifies(row):
                    fut = executor.submit(_fetch_swh_content, client, row["blob_id"], row["src_encoding"])
                    in_flight[fut] = row
                    return True
            return False

        for _ in range(max_workers):
            if not submit_next():
                break

        while in_flight:
            done, _ = concurrent.futures.wait(in_flight, return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in done:
                row = in_flight.pop(fut)
                submit_next()
                content = fut.result()
                if content is None:
                    continue
                yield {"repo": row["repo_name"], "path": row["path"], "content": content}
                count += 1
                if limit is not None and count >= limit:
                    return


def _iter_stack_v1_fallback(limit: int | None) -> Iterator[dict[str, Any]]:
    ds = _load_streaming(STACK_V1_FALLBACK_DATASET_ID, data_dir=STACK_LANGUAGE_DATA_DIR)
    count = 0
    for row in ds:
        repo_col = _first_present(row, _STACK_REPO_COLUMNS, "the-stack repo identifier")
        content_col = _first_present(row, _STACK_CONTENT_COLUMNS, "the-stack file content")
        path_col = _first_present(row, _STACK_PATH_COLUMNS, "the-stack file path")
        yield {"repo": row[repo_col], "path": row[path_col], "content": row[content_col]}
        count += 1
        if limit is not None and count >= limit:
            return


def iter_general_code_python(
    limit: int | None = None,
    use_fallback: bool = False,
    max_workers: int = 32,
) -> Iterator[dict[str, Any]]:
    """Stream Python files from The Stack, normalized to
    ``{"repo", "path", "content"}``. Falls back to the v1 dedup dataset (no
    SWH fetch needed there) if v2 raises — same visible-fallback discipline
    as ``pretokenize.py``'s FineWeb-Edu source, not a silent one. Pass
    *use_fallback=True* to go straight to v1 (e.g. once v1 access clears and
    v2's per-row SWH fetch latency isn't worth it for a given run).

    *max_workers* controls the v2 path's SWH fetch concurrency (ignored by
    the v1 fallback, which has no per-row fetch to parallelize) — see
    ``_iter_stack_v2``'s docstring for why this is a real, not premature,
    optimization at real-scale pull sizes.
    """
    if use_fallback:
        yield from _iter_stack_v1_fallback(limit)
        return
    try:
        yield from _iter_stack_v2(limit, max_workers=max_workers)
    except CorpusSourceError as e:
        print(f"  [!] {STACK_V2_DATASET_ID} unavailable ({e}); "
              f"falling back to {STACK_V1_FALLBACK_DATASET_ID}", file=sys.stderr)
        yield from _iter_stack_v1_fallback(limit)


def iter_issue_diff_pairs(
    limit: int | None = None,
    dataset_id: str = SWE_GYM_DATASET_ID,
) -> Iterator[dict[str, Any]]:
    """Stream real GitHub issue->diff pairs from SWE-Gym, normalized to a
    SWE-bench-shaped record: ``repo``, ``instance_id``, ``problem_statement``
    (the issue description), ``patch`` (the resolving diff), plus
    ``base_commit``/``fail_to_pass``/``pass_to_pass`` passthrough for
    Item 5/7's sandbox to grade against later.
    """
    ds = _load_streaming(dataset_id)
    count = 0
    for row in ds:
        for col in _SWE_GYM_REQUIRED_COLUMNS:
            if col not in row:
                raise CorpusSourceError(
                    f"{dataset_id!r} missing expected column {col!r} — upstream "
                    f"schema changed; actual columns: {sorted(row.keys())}"
                )
        yield {
            "repo": row["repo"],
            "instance_id": row["instance_id"],
            "problem_statement": row["problem_statement"],
            "patch": row["patch"],
            "base_commit": row.get("base_commit"),
            "fail_to_pass": row.get("FAIL_TO_PASS"),
            "pass_to_pass": row.get("PASS_TO_PASS"),
        }
        count += 1
        if limit is not None and count >= limit:
            return


_SOURCES = {
    "stack": lambda limit=None, max_workers=32: iter_general_code_python(limit=limit, max_workers=max_workers),
    "swe-gym": lambda limit=None, max_workers=32: iter_issue_diff_pairs(limit=limit),
    "swe-gym-raw": lambda limit=None, max_workers=32: iter_issue_diff_pairs(
        limit=limit, dataset_id=SWE_GYM_RAW_DATASET_ID,
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="uchi-corpus-source",
        description=(
            "Stream one of 0.5.0 Item 1's corpus sources, decontaminate it "
            "against SWE-bench's repo-level exclusion list, and write JSONL."
        ),
    )
    parser.add_argument("--source", choices=sorted(_SOURCES), required=True)
    parser.add_argument("--output", "-o", default="-", help="JSONL output path, or - for stdout")
    parser.add_argument("--limit", type=int, default=None, help="Cap on records streamed from the source")
    parser.add_argument(
        "--cache-file",
        default=None,
        help="Local SWE-bench repo-list snapshot; if omitted, resolves live (see corpus_decontamination.py).",
    )
    parser.add_argument(
        "--max-workers", type=int, default=32,
        help="Concurrent SWH fetch workers for --source stack (ignored by the other sources). "
             "Confirmed live: sequential fetching was the bottleneck at real-scale pull sizes.",
    )
    args = parser.parse_args(argv)

    excluded_repos = load_repo_cache(args.cache_file) if args.cache_file else resolve_excluded_repos()
    records = _SOURCES[args.source](limit=args.limit, max_workers=args.max_workers)
    kept, summary = filter_corpus(records, excluded_repos)

    fh = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")
    try:
        for record in kept:
            fh.write(json.dumps(record) + "\n")
    finally:
        if fh is not sys.stdout:
            fh.close()

    print(
        f"[corpus_sources:{args.source}] total={summary.total} kept={summary.kept} "
        f"excluded={summary.excluded}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
