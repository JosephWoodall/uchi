from unittest.mock import patch

from uchi.flux.sft_train import load_sft_examples
from uchi.flux.tokenizer_v2 import TikTokenHybridTokenizer


def _counting_gen(rows, counter, key):
    for row in rows:
        counter[key] += 1
        yield row


def _squad_rows(n):
    for i in range(n):
        yield {
            "question": f"squad question {i}",
            "context": "some context",
            "answers": {"text": [f"squad answer {i}"]},
        }


def _dolly_rows(n):
    for i in range(n):
        yield {"instruction": f"dolly instruction {i}", "context": "", "response": f"dolly response {i}"}


def _codealpaca_rows(n):
    for i in range(n):
        yield {"prompt": f"code prompt {i}", "completion": f"code completion {i}"}


def _ultrachat_rows(n):
    for i in range(n):
        yield {
            "messages": [
                {"role": "user", "content": f"chat question {i}"},
                {"role": "assistant", "content": f"chat answer {i}"},
            ]
        }


def test_every_source_gets_a_fair_independent_share():
    """Regression test: previously all 4 sources shared one cumulative
    len(examples) break check. SQuAD alone reached max_examples (10,000
    available rows vs. a 400 cap) before exhausting its pool, so
    Dolly/CodeAlpaca/UltraChat -- loaded after it -- each contributed only
    ~1 example (the shared counter was already at the cap when their
    loops started). Each source must now be consumed close to its own
    fair share of the budget, not starved to ~1.
    """
    tokenizer = TikTokenHybridTokenizer()
    consumed = {"squad": 0, "dolly": 0, "codealpaca": 0, "ultrachat": 0}

    def loader(name, split=None, **kwargs):
        if "squad" in name:
            return _counting_gen(_squad_rows(10_000), consumed, "squad")
        if "dolly" in name.lower():
            return _counting_gen(_dolly_rows(10_000), consumed, "dolly")
        if "codealpaca" in name.lower():
            return _counting_gen(_codealpaca_rows(10_000), consumed, "codealpaca")
        if "ultrachat" in name.lower():
            return _counting_gen(_ultrachat_rows(10_000), consumed, "ultrachat")
        raise AssertionError(f"unexpected dataset requested: {name}")

    with patch("datasets.load_dataset", side_effect=loader):
        formatted = load_sft_examples(tokenizer, max_seq_len=128, max_examples=400)

    print(consumed)
    assert len(formatted) > 0

    # The old bug: only the FIRST source (SQuAD) would show meaningful
    # consumption; every source after it would show ~1.
    fair_share = 400 // 4  # 100
    for name in ("dolly", "codealpaca", "ultrachat"):
        assert consumed[name] >= fair_share * 0.5, (
            f"{name} was starved: only consumed {consumed[name]} rows "
            f"(expected roughly {fair_share}) -- the shared-counter bug is back"
        )


def test_total_pool_respects_max_examples_cap():
    tokenizer = TikTokenHybridTokenizer()

    def loader(name, split=None, **kwargs):
        if "squad" in name:
            return _squad_rows(10_000)
        if "dolly" in name.lower():
            return _dolly_rows(10_000)
        if "codealpaca" in name.lower():
            return _codealpaca_rows(10_000)
        if "ultrachat" in name.lower():
            return _ultrachat_rows(10_000)
        return []

    with patch("datasets.load_dataset", side_effect=loader):
        formatted = load_sft_examples(tokenizer, max_seq_len=128, max_examples=400)

    assert len(formatted) <= 400
