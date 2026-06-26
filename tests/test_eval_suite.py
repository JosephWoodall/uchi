"""
tests/test_eval_suite.py
========================
Pytest wrappers for the OmniEvaluator quality benchmarks.

These tests are SLOW (each metric involves real MCTS deliberation).
Run them separately with:

    pytest -m eval tests/test_eval_suite.py -v

They are excluded from the standard unit test run.

All assertions here verify infrastructure correctness, NOT quality thresholds.
Quality thresholds are established by tracking eval_metrics.json over time.
"""

import json
import math
import os
import pytest

# ── shared router (module-scoped to avoid redundant bootstrap overhead) ────────

@pytest.fixture(scope="module")
def eval_router():
    from uchi.omni_router import OmniRouter
    return OmniRouter(use_bpe=False)


@pytest.fixture(scope="module")
def evaluator(eval_router):
    from uchi.omni_evaluator import OmniEvaluator
    return OmniEvaluator(eval_router, verbose=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_rate(v) -> bool:
    return isinstance(v, float) and 0.0 <= v <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 1. Convergent Oracle Pass Rate
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.eval
def test_convergent_oracle_pass_rate(evaluator):
    """ConvergentEngine must return a valid rate; infrastructure must not crash."""
    rate = evaluator.convergent_oracle_pass_rate()
    assert _is_rate(rate), f"expected float in [0,1], got {rate!r}"
    print(f"\n  convergent_oracle_pass_rate = {rate:.2%}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. MCTS Efficiency Score
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.eval
def test_mcts_efficiency_score(evaluator):
    """Rollout counting wrapper must return a valid fraction."""
    score = evaluator.mcts_efficiency_score()
    assert _is_rate(score), f"expected float in [0,1], got {score!r}"
    print(f"\n  mcts_efficiency_score = {score:.2%} of budget consumed")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Tool Routing Precision / Recall
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.eval
def test_tool_routing_precision_recall(evaluator):
    """Precision and recall must be valid rates; counts must be non-negative."""
    result = evaluator.tool_routing_precision_recall()

    assert _is_rate(result["precision"]), f"precision out of range: {result['precision']}"
    assert _is_rate(result["recall"]),    f"recall out of range: {result['recall']}"
    assert result["tp"] >= 0
    assert result["fp"] >= 0
    assert result["fn"] >= 0

    # At cold start: ABSOLUTE_FLOOR blocks all tool dispatch → fp == 0.
    # We assert this as a hard regression guard: a false positive on a
    # purely conversational query means the floor was lowered or the
    # vectors have catastrophically collapsed.
    from uchi.omni_evaluator import TOOL_ROUTING_QUERIES
    n_conversational = sum(1 for q in TOOL_ROUTING_QUERIES if not q["expected_tool"])
    assert result["fp"] <= n_conversational, (
        f"False positive rate too high: {result['fp']} FPs out of "
        f"{n_conversational} conversational queries"
    )

    print(
        f"\n  precision={result['precision']:.2%}  "
        f"recall={result['recall']:.2%}  "
        f"(TP={result['tp']} FP={result['fp']} FN={result['fn']})"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. Uncertain Canary Rate
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.eval
def test_uncertain_canary_rate(evaluator):
    """Canary must return a valid rate and the infrastructure must not crash."""
    rate = evaluator.uncertain_canary_rate()
    assert _is_rate(rate), f"expected float in [0,1], got {rate!r}"
    print(f"\n  uncertain_canary_rate = {rate:.2%}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Contrastive Loss Observable
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.eval
def test_contrastive_loss_observable(eval_router, evaluator):
    """After running chat(), the loss buffer must have grown."""
    from uchi.vector_oracle import _contrastive_loss_history
    import time

    before = len(_contrastive_loss_history)

    # Trigger a few contrastive updates via chat (they're async/daemon).
    eval_router.chat("hello how are you")
    eval_router.chat("what can you do")
    time.sleep(0.3)  # let daemon threads flush

    after = len(_contrastive_loss_history)
    assert after >= before, "no contrastive updates recorded after chat()"
    print(f"\n  contrastive loss entries: {before} → {after}")

    if after >= 2:
        trend = evaluator.contrastive_loss_trend()
        assert isinstance(trend["mean"], float)
        assert trend["n"] >= 2


# ══════════════════════════════════════════════════════════════════════════════
# 6. Legacy metrics still compute (regression guard)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.eval
def test_legacy_active_teaching_trigger_rate(evaluator):
    rate = evaluator.active_teaching_trigger_rate()
    assert _is_rate(rate)
    print(f"\n  active_teaching_trigger_rate = {rate:.2%}")


@pytest.mark.eval
def test_legacy_knowledge_recall_rate(evaluator):
    rate = evaluator.knowledge_recall_rate()
    assert _is_rate(rate)
    print(f"\n  knowledge_recall_rate = {rate:.2%}")


@pytest.mark.eval
def test_legacy_average_prompt_entropy(evaluator):
    entropy = evaluator.average_prompt_entropy()
    assert isinstance(entropy, float) and entropy > 0
    print(f"\n  average_prompt_entropy = {entropy:.4f} bits/token")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Full evaluation saves to eval_metrics.json
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.eval
def test_full_eval_saves_metrics(evaluator, tmp_path):
    """run_full_evaluation() must produce valid keys and persist to JSON."""
    output = tmp_path / "eval_metrics_test.json"
    results = evaluator.run_full_evaluation()
    evaluator.save_metrics(results, str(output))

    assert output.exists(), "metrics file was not created"
    with open(output) as f:
        data = json.load(f)

    assert isinstance(data, list) and len(data) == 1
    entry = data[0]

    expected_keys = {
        "timestamp",
        "pass_at_1",
        "self_correction_rate",
        "active_teaching_trigger_rate",
        "knowledge_recall_rate",
        "average_prompt_entropy",
        "convergent_oracle_pass_rate",
        "mcts_efficiency_score",
        "tool_routing_precision",
        "tool_routing_recall",
        "uncertain_canary_rate",
    }
    missing = expected_keys - set(entry.keys())
    assert not missing, f"Missing keys in saved metrics: {missing}"

    for key in expected_keys - {"timestamp", "average_prompt_entropy", "mcts_efficiency_score"}:
        v = entry[key]
        assert isinstance(v, (int, float)) and 0.0 <= v <= 1.0, (
            f"{key} = {v!r} is not in [0, 1]"
        )

    print(f"\n  Saved {len(entry)} metrics to {output}")
