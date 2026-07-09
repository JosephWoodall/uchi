"""Tests for TaskConfigCache (0.4.0 follow-on to Item 17): ODUSP-backed
recall of a recommended self-consistency vote count, keyed by a
question's structural shape rather than its exact text.
"""
from uchi.task_config_cache import TaskConfigCache, _feature_signature, _nearest_bucket, N_BUCKETS


def test_unfitted_cache_returns_no_recommendation():
    cache = TaskConfigCache()
    n, conf = cache.recall_n("What is the capital of Spain?")
    assert n is None
    assert conf == 0.0


def test_learns_to_distinguish_simple_from_complex_patterns():
    cache = TaskConfigCache()
    simple = ["What is the capital of France?", "How tall is the Eiffel Tower?"]
    complex_ = [
        "Compare the GDP of Germany and France, and also their population?",
        "What is the difference between Python and Java, and also which is faster?",
    ]
    for _ in range(15):
        for q in simple:
            cache.record_outcome(q, 1)
        for q in complex_:
            cache.record_outcome(q, 8)

    # Never-seen-verbatim questions, same structural shape as the ones trained on.
    n_simple, conf_simple = cache.recall_n("What is the capital of Italy?")
    n_complex, conf_complex = cache.recall_n(
        "Compare the size of the Sun versus the Moon, and also their distance?"
    )
    assert n_simple == 1
    assert n_complex == 8
    assert conf_simple > 0.0
    assert conf_complex > 0.0


def test_nearest_bucket_rounds_correctly():
    assert _nearest_bucket(1) == 1
    assert _nearest_bucket(4) in (3, 5)  # equidistant, either is a valid bucket
    assert _nearest_bucket(100) == max(N_BUCKETS)
    assert _nearest_bucket(0) == min(N_BUCKETS)


def test_feature_signature_is_structural_not_lexical():
    """Two totally different topics with the same structural shape must
    produce the same signature -- that's the whole point."""
    sig_a = _feature_signature("Compare the GDP of Germany and France?")
    sig_b = _feature_signature("Compare the height of Everest and K2?")
    assert sig_a == sig_b

    sig_simple = _feature_signature("What is the capital of France?")
    assert sig_simple != sig_a


def test_record_outcome_does_not_raise_on_empty_question():
    cache = TaskConfigCache()
    cache.record_outcome("", 3)  # must not raise


def test_recall_n_does_not_raise_on_empty_question():
    cache = TaskConfigCache()
    n, conf = cache.recall_n("")
    assert n is None or isinstance(n, int)
