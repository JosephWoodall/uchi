import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmarks")))

from ops_benchmark import run_ops_benchmark


def test_ops_benchmark_reports_expected_shape():
    result = run_ops_benchmark(n_steps=3, warmup=1)
    assert result["n_steps"] == 3
    assert result["successful_tool_calls"] == 3
    assert result["ops"] > 0
    assert result["elapsed_seconds"] > 0
    assert result["context_intact"] is True


def test_ops_benchmark_all_steps_succeed():
    result = run_ops_benchmark(n_steps=5, warmup=0)
    assert result["successful_tool_calls"] == result["n_steps"]
