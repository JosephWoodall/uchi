from uchi.code_engine import CodeEngine


class _FakeBadPredictor:
    """Always proposes the exact same invalid Python, regardless of temperature."""

    def generate(self, n_tokens, seed, temperature, use_mcts, stop_tokens):
        return ["def", "broken(", ":", "pass"]  # invalid syntax, deterministic


class _FakeGoodPredictor:
    """Always proposes the exact same valid Python."""

    def generate(self, n_tokens, seed, temperature, use_mcts, stop_tokens):
        return ["def", "run():\n    return", "1"]


def test_repeated_failing_candidate_is_skipped_on_retry():
    engine = CodeEngine(_FakeBadPredictor(), n_workers=2)

    verify_calls = []
    _orig_verify = engine.oracle.verify
    def _spy_verify(code, timeout=3.0):
        verify_calls.append(code)
        return _orig_verify(code, timeout)
    engine.oracle.verify = _spy_verify

    code1, reward1, passed1 = engine.generate_code(["seed"], max_tokens=10)
    assert not passed1
    first_call_count = len(verify_calls)
    assert first_call_count > 0  # verified at least once on the first attempt

    # Retry with the identical seed -> workers propose the identical bad
    # candidate again. It should be skipped (loop_guard), not re-verified.
    code2, reward2, passed2 = engine.generate_code(["seed"], max_tokens=10)
    assert not passed2
    # No new verify() calls for the already-penalized candidate.
    assert len(verify_calls) == first_call_count


def test_successful_candidate_is_not_penalized_and_reused_fine():
    engine = CodeEngine(_FakeGoodPredictor(), n_workers=2)
    code1, reward1, passed1 = engine.generate_code(["seed"], max_tokens=10)
    assert passed1
    assert not engine.loop_guard.is_penalized(code1)

    # Calling again should succeed again (not blocked, since it succeeded).
    code2, reward2, passed2 = engine.generate_code(["seed"], max_tokens=10)
    assert passed2
