"""Tests for benchmarks/humaneval_benchmark.py (0.5.0 Item 9).

Tests `grade_candidate` directly against hand-written correct/incorrect
candidates for a synthetic prompt/test/entry_point triple -- no network
or dataset download needed. Real sandboxed execution either way (isolated
if bwrap is available in this environment, plain subprocess otherwise --
`grade_candidate`'s own `isolate` flag exercises both paths explicitly).
"""
from benchmarks.humaneval_benchmark import _build_prompt, grade_candidate

ENTRY_POINT = "add"
TEST_CODE = (
    "def check(candidate):\n"
    "    assert candidate(2, 3) == 5\n"
    "    assert candidate(0, 0) == 0\n"
    "    assert candidate(-1, 1) == 0\n"
)
CORRECT_CANDIDATE = "def add(a, b):\n    return a + b\n"
WRONG_CANDIDATE = "def add(a, b):\n    return a - b\n"
CRASHING_CANDIDATE = "def add(a, b):\n    raise ValueError('boom')\n"
SYNTAX_ERROR_CANDIDATE = "def add(a, b):\n    return a +\n"


def test_build_prompt_includes_instruction_and_signature():
    prompt = _build_prompt("def add(a, b):\n    \"\"\"docstring\"\"\"\n")
    assert "Complete the following Python function" in prompt
    assert "def add(a, b):" in prompt


def test_grade_candidate_correct_passes():
    passed, output = grade_candidate(CORRECT_CANDIDATE, TEST_CODE, ENTRY_POINT, timeout=10.0)
    assert passed is True


def test_grade_candidate_wrong_fails_with_real_assertion():
    passed, output = grade_candidate(WRONG_CANDIDATE, TEST_CODE, ENTRY_POINT, timeout=10.0)
    assert passed is False
    assert "AssertionError" in output


def test_grade_candidate_crashing_candidate_fails():
    passed, output = grade_candidate(CRASHING_CANDIDATE, TEST_CODE, ENTRY_POINT, timeout=10.0)
    assert passed is False
    assert "ValueError" in output


def test_grade_candidate_syntax_error_fails():
    passed, output = grade_candidate(SYNTAX_ERROR_CANDIDATE, TEST_CODE, ENTRY_POINT, timeout=10.0)
    assert passed is False


def test_grade_candidate_non_isolated_path_also_works():
    """isolate=False must produce the same real verdicts -- confirms the
    fallback path isn't just a stub."""
    passed, _ = grade_candidate(CORRECT_CANDIDATE, TEST_CODE, ENTRY_POINT, timeout=10.0, isolate=False)
    assert passed is True
    passed, _ = grade_candidate(WRONG_CANDIDATE, TEST_CODE, ENTRY_POINT, timeout=10.0, isolate=False)
    assert passed is False


def test_grade_candidate_cleans_up_sandbox_dir():
    import os

    from uchi.workspace import DEFAULT_ROOT

    sandbox_root = os.path.join(DEFAULT_ROOT, "humaneval_sandbox")
    before = set(os.listdir(sandbox_root)) if os.path.isdir(sandbox_root) else set()
    grade_candidate(CORRECT_CANDIDATE, TEST_CODE, ENTRY_POINT, timeout=10.0)
    after = set(os.listdir(sandbox_root)) if os.path.isdir(sandbox_root) else set()
    assert after == before, "grade_candidate should remove its per-run sandbox dir, not accumulate them"
