from uchi.iq_router import estimate_complexity, should_decompose


def test_simple_lookup_scores_low():
    assert estimate_complexity("What is the boiling point of water?") < 0.3
    assert not should_decompose("What is the boiling point of water?")


def test_empty_question_scores_zero():
    assert estimate_complexity("") == 0.0
    assert estimate_complexity("   ") == 0.0


def test_multi_part_question_scores_higher():
    q = "What is the capital of France and also what is its population?"
    assert estimate_complexity(q) >= 0.3
    assert should_decompose(q)


def test_comparison_question_scores_higher():
    q = "Compare the GDP of Germany versus the GDP of France."
    assert should_decompose(q)


def test_multiple_question_marks_scores_higher():
    q = "What is X? What is Y?"
    assert should_decompose(q)


def test_enumerated_question_scores_higher():
    q = "Answer these: 1) what is X 2) what is Y"
    assert should_decompose(q)


def test_long_question_scores_higher():
    q = " ".join(["word"] * 30) + "?"
    assert estimate_complexity(q) >= 0.3


def test_score_is_capped_at_one():
    q = "Compare and also contrast: 1) what is X? 2) what is Y versus Z, also, what about W?"
    assert estimate_complexity(q) <= 1.0


def test_threshold_is_configurable():
    q = "What is the capital of France and also its population?"
    score = estimate_complexity(q)
    assert should_decompose(q, threshold=score)  # exactly at threshold -> True
    assert not should_decompose(q, threshold=score + 0.01)
