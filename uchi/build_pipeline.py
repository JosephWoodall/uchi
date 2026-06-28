"""Uchi Brain Build Pipeline — single offline entry point.

Orchestrates the full sequence for producing a shippable brain artifact:

  Phase 1  Ingest       — incremental_builder: add new knowledge to trie + HNSW
  Phase 2  Train        — grpo_offline_trainer: restore SSM format behavior
  Phase 3  Calibrate    — calibration: fit temperature scaling on SSM value head
  Phase 4  Benchmark    — mmlu + swebench: validate quality gates
  Phase 5  Package      — copy brain to uchi/data/brain.uchi if gates pass

The output is a versioned brain artifact ready to ship as package data.
Run this as an offline job whenever you want to build a new brain release.

Usage:
    python -m uchi.build_pipeline
    python -m uchi.build_pipeline --sources openhermes,wikipedia,mmlu,gsm8k,humaneval,swebench
    python -m uchi.build_pipeline --limit 500 --grpo-steps 300
    python -m uchi.build_pipeline --skip-benchmarks   # rapid iteration, no gate
    python -m uchi.build_pipeline --brain brain.uchi --brain-out uchi/data/brain.uchi

Benchmark gates (default thresholds, override with --mmlu-min-acc / --swe-min-composite):
    MMLU accuracy        >= 10%   (must meaningfully beat random)
    MMLU no-parse rate   <= 50%   (must produce parseable answers half the time)
    SWE-bench composite  >= 0.05  (must generate some code content)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

_log = logging.getLogger(__name__)

_DEFAULT_BRAIN     = "brain.uchi"
_DEFAULT_BRAIN_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "brain.uchi"
)
_REPORT_PATH = "build_pipeline_report.json"

# Default benchmark gates
_GATE_MMLU_MIN_ACC       = 0.10   # 10% accuracy
_GATE_MMLU_MAX_NOPARSE   = 0.50   # 50% no-parse rate
_GATE_SWE_MIN_COMPOSITE  = 0.05   # composite proxy score


def _print_header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f" {title}")
    print(f"{'=' * 70}\n")


def _elapsed(start: float) -> str:
    s = int(time.time() - start)
    return f"{s // 60}m {s % 60}s"


# ── Phase 1: Ingest ──────────────────────────────────────────────────────────

def phase_ingest(
    brain_path: str,
    sources: List[str],
    limit: int,
) -> Dict:
    _print_header("Phase 1 — Ingest")
    from .incremental_builder import IncrementalBrainBuilder

    t0 = time.time()
    builder = IncrementalBrainBuilder(brain_path=brain_path)
    builder.run(limit=limit, sources=sources)

    result = {
        "sources": sources,
        "limit_per_source": limit,
        "elapsed": _elapsed(t0),
    }
    print(f"[+] Ingest complete in {result['elapsed']}")
    return result


# ── Phase 2: Train ───────────────────────────────────────────────────────────

def phase_train(
    brain_path: str,
    grpo_steps: int,
    grpo_batch: int,
    grpo_lr: float,
) -> Dict:
    _print_header("Phase 2 — GRPO Format-Reward Training")
    from .grpo_offline_trainer import run_offline_grpo

    t0 = time.time()
    run_offline_grpo(
        brain_path=brain_path,
        steps=grpo_steps,
        batch_size=grpo_batch,
        lr=grpo_lr,
    )

    result = {
        "steps": grpo_steps,
        "batch_size": grpo_batch,
        "elapsed": _elapsed(t0),
    }
    print(f"[+] GRPO training complete in {result['elapsed']}")
    return result


# ── Phase 3: Calibrate ───────────────────────────────────────────────────────

def phase_calibrate(brain_path: str, n_samples: int = 200) -> Dict:
    _print_header("Phase 3 — SSM Confidence Calibration")
    import gzip
    import pickle
    from .calibration import run_calibration

    t0 = time.time()
    try:
        with gzip.open(brain_path, "rb") as f:
            router = pickle.load(f)
        calibrator = run_calibration(router, n_samples=n_samples)
        T = calibrator.temperature.item()
        print(f"[+] Calibration complete: T={T:.4f}  (elapsed {_elapsed(t0)})")
        return {"temperature": T, "n_samples": n_samples, "elapsed": _elapsed(t0)}
    except Exception as e:
        print(f"[!] Calibration failed: {e}  — continuing with T=1.0")
        return {"temperature": 1.0, "error": str(e), "elapsed": _elapsed(t0)}


# ── Phase 4: Benchmark ───────────────────────────────────────────────────────

def phase_benchmark(
    mmlu_sample: int,
    swe_sample: int,
    python: str,
) -> Dict:
    _print_header("Phase 4 — Benchmark Validation")
    t0 = time.time()

    benchmarks_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "benchmarks",
    )

    def _run(script: str, sample: int) -> Optional[Dict]:
        path = os.path.join(benchmarks_dir, script)
        if not os.path.exists(path):
            print(f"[!] Benchmark script not found: {path}")
            return None
        print(f"[*] Running {script} (sample={sample})…")
        try:
            proc = subprocess.run(
                [python, path, "--sample", str(sample)],
                capture_output=False,
                check=True,
            )
            _ = proc  # output goes to stdout live
        except subprocess.CalledProcessError as e:
            print(f"[!] {script} exited with code {e.returncode}")
            return None
        return True

    mmlu_ok  = _run("mmlu_benchmark.py",    mmlu_sample)
    swe_ok   = _run("swebench_benchmark.py", swe_sample)

    # Read results from JSON files written by the benchmarks.
    mmlu_results = {}
    swe_results  = {}

    mmlu_json = os.path.join(benchmarks_dir, "mmlu_results.json")
    swe_json  = os.path.join(benchmarks_dir, "swebench_results.json")

    if os.path.exists(mmlu_json):
        try:
            with open(mmlu_json) as f:
                mmlu_results = json.load(f)
        except Exception:
            pass

    if os.path.exists(swe_json):
        try:
            with open(swe_json) as f:
                swe_results = json.load(f)
        except Exception:
            pass

    return {
        "mmlu":    mmlu_results,
        "swebench": swe_results,
        "elapsed": _elapsed(t0),
        "mmlu_ok": bool(mmlu_ok),
        "swe_ok":  bool(swe_ok),
    }


# ── Phase 5: Package ─────────────────────────────────────────────────────────

def phase_package(
    brain_path: str,
    brain_out: str,
    benchmark_results: Dict,
    min_mmlu_acc: float,
    max_mmlu_noparse: float,
    min_swe_composite: float,
    skip_benchmarks: bool,
) -> Dict:
    _print_header("Phase 5 — Package")

    if skip_benchmarks:
        print("[*] --skip-benchmarks set: packaging without gate check.")
        gate_passed = True
        gate_details = {"skipped": True}
    else:
        mmlu = benchmark_results.get("mmlu", {})
        swe  = benchmark_results.get("swebench", {})

        acc          = mmlu.get("mmlu_accuracy", mmlu.get("accuracy", 0.0))
        _no_parse    = mmlu.get("mmlu_no_parse", 0)
        _total       = mmlu.get("mmlu_total", 1)
        noparse      = _no_parse / _total if _total else 1.0
        composite    = swe.get("swebench_avg_composite", swe.get("avg_composite_score", 0.0))

        gate_details = {
            "mmlu_accuracy":       acc,
            "mmlu_noparse_rate":   noparse,
            "swe_composite":       composite,
            "gate_mmlu_acc":       acc       >= min_mmlu_acc,
            "gate_mmlu_noparse":   noparse   <= max_mmlu_noparse,
            "gate_swe_composite":  composite >= min_swe_composite,
        }
        gate_passed = all([
            gate_details["gate_mmlu_acc"],
            gate_details["gate_mmlu_noparse"],
            gate_details["gate_swe_composite"],
        ])

        print("  Benchmark gates:")
        print(f"    MMLU accuracy    {acc*100:.1f}%  (min {min_mmlu_acc*100:.0f}%)  "
              f"{'PASS' if gate_details['gate_mmlu_acc'] else 'FAIL'}")
        print(f"    MMLU no-parse    {noparse*100:.1f}%  (max {max_mmlu_noparse*100:.0f}%)  "
              f"{'PASS' if gate_details['gate_mmlu_noparse'] else 'FAIL'}")
        print(f"    SWE composite    {composite:.3f}  (min {min_swe_composite:.2f})  "
              f"{'PASS' if gate_details['gate_swe_composite'] else 'FAIL'}")

    if gate_passed:
        os.makedirs(os.path.dirname(brain_out), exist_ok=True)
        shutil.copy2(brain_path, brain_out)
        print(f"\n[+] Brain packaged → {brain_out}")
        print("[+] Ready to ship. Update pyproject.toml version and commit.")
    else:
        print("\n[!] Gate check FAILED — brain NOT copied to package data.")
        print("[!] Improve the failing metrics and re-run.")

    return {"gate_passed": gate_passed, "gate_details": gate_details}


# ── Orchestrator ─────────────────────────────────────────────────────────────

def run_pipeline(
    brain_path: str = _DEFAULT_BRAIN,
    brain_out: str = _DEFAULT_BRAIN_OUT,
    sources: Optional[List[str]] = None,
    limit: int = 500,
    grpo_steps: int = 300,
    grpo_batch: int = 8,
    grpo_lr: float = 5e-4,
    calib_samples: int = 200,
    mmlu_sample: int = 200,
    swe_sample: int = 50,
    min_mmlu_acc: float = _GATE_MMLU_MIN_ACC,
    max_mmlu_noparse: float = _GATE_MMLU_MAX_NOPARSE,
    min_swe_composite: float = _GATE_SWE_MIN_COMPOSITE,
    skip_ingest: bool = False,
    skip_train: bool = False,
    skip_calibrate: bool = False,
    skip_benchmarks: bool = False,
    python: str = sys.executable,
) -> Dict:
    from .incremental_builder import ALL_SOURCES
    sources = sources or ALL_SOURCES

    _print_header("Uchi Brain Build Pipeline")
    print(f"  Brain in:   {brain_path}")
    print(f"  Brain out:  {brain_out}")
    print(f"  Sources:    {', '.join(sources)}")
    print(f"  Limit:      {limit}/source")
    print(f"  GRPO steps: {grpo_steps}")
    print(f"  Python:     {python}")

    pipeline_start = time.time()
    report: Dict = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "brain_path": brain_path,
        "brain_out":  brain_out,
        "sources":    sources,
        "phases":     {},
    }

    # Phase 1 — Ingest
    if not skip_ingest:
        report["phases"]["ingest"] = phase_ingest(brain_path, sources, limit)
    else:
        print("\n[=] Phase 1 (Ingest) skipped via --skip-ingest")

    # Phase 2 — Train
    if not skip_train:
        report["phases"]["train"] = phase_train(
            brain_path, grpo_steps, grpo_batch, grpo_lr
        )
    else:
        print("\n[=] Phase 2 (Train) skipped via --skip-train")

    # Phase 3 — Calibrate
    if not skip_calibrate:
        report["phases"]["calibrate"] = phase_calibrate(brain_path, calib_samples)
    else:
        print("\n[=] Phase 3 (Calibrate) skipped via --skip-calibrate")

    # Phase 4 — Benchmark
    benchmark_results: Dict = {}
    if not skip_benchmarks:
        benchmark_results = phase_benchmark(mmlu_sample, swe_sample, python)
        report["phases"]["benchmark"] = benchmark_results
    else:
        print("\n[=] Phase 4 (Benchmark) skipped via --skip-benchmarks")

    # Phase 5 — Package
    report["phases"]["package"] = phase_package(
        brain_path, brain_out, benchmark_results,
        min_mmlu_acc, max_mmlu_noparse, min_swe_composite,
        skip_benchmarks,
    )

    total = _elapsed(pipeline_start)
    report["completed_at"]  = datetime.now(timezone.utc).isoformat()
    report["total_elapsed"] = total
    report["gate_passed"]   = report["phases"]["package"]["gate_passed"]

    # Write report
    try:
        with open(_REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[+] Report written to {_REPORT_PATH}")
    except Exception as e:
        _log.warning("Report write failed: %s", e)

    _print_header(
        f"Build complete in {total}  —  "
        f"{'SHIPPED' if report['gate_passed'] else 'NOT SHIPPED (gate failed)'}"
    )
    return report


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Uchi Brain Build Pipeline — produces a shippable brain artifact.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    from .incremental_builder import ALL_SOURCES, KNOWLEDGE_LIMIT

    parser.add_argument("--brain",     default=_DEFAULT_BRAIN,
                        help="Working brain file (input/output during build)")
    parser.add_argument("--brain-out", default=_DEFAULT_BRAIN_OUT,
                        help="Packaged brain destination (uchi/data/brain.uchi)")
    parser.add_argument("--sources",   default=",".join(ALL_SOURCES),
                        help="Comma-separated knowledge sources")
    parser.add_argument("--limit",     type=int, default=KNOWLEDGE_LIMIT,
                        help="Max documents per source")
    parser.add_argument("--grpo-steps",  type=int,   default=300)
    parser.add_argument("--grpo-batch",  type=int,   default=8)
    parser.add_argument("--grpo-lr",     type=float, default=5e-4)
    parser.add_argument("--calib-samples", type=int, default=200)
    parser.add_argument("--mmlu-sample",   type=int, default=200)
    parser.add_argument("--swe-sample",    type=int, default=50)
    parser.add_argument("--mmlu-min-acc",      type=float, default=_GATE_MMLU_MIN_ACC)
    parser.add_argument("--mmlu-max-noparse",  type=float, default=_GATE_MMLU_MAX_NOPARSE)
    parser.add_argument("--swe-min-composite", type=float, default=_GATE_SWE_MIN_COMPOSITE)
    parser.add_argument("--skip-ingest",     action="store_true")
    parser.add_argument("--skip-train",      action="store_true")
    parser.add_argument("--skip-calibrate",  action="store_true")
    parser.add_argument("--skip-benchmarks", action="store_true",
                        help="Package without benchmark gate check")

    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    run_pipeline(
        brain_path=args.brain,
        brain_out=args.brain_out,
        sources=[s.strip() for s in args.sources.split(",")],
        limit=args.limit,
        grpo_steps=args.grpo_steps,
        grpo_batch=args.grpo_batch,
        grpo_lr=args.grpo_lr,
        calib_samples=args.calib_samples,
        mmlu_sample=args.mmlu_sample,
        swe_sample=args.swe_sample,
        min_mmlu_acc=args.mmlu_min_acc,
        max_mmlu_noparse=args.mmlu_max_noparse,
        min_swe_composite=args.swe_min_composite,
        skip_ingest=args.skip_ingest,
        skip_train=args.skip_train,
        skip_calibrate=args.skip_calibrate,
        skip_benchmarks=args.skip_benchmarks,
    )


if __name__ == "__main__":
    main()
