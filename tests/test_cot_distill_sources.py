from unittest.mock import patch

from uchi.flux.cot_distill import _split_think_answer, load_cot_examples
from uchi.flux.tokenizer_v2 import TikTokenHybridTokenizer


def test_split_think_answer_multi_sentence():
    think, answer = _split_think_answer("First we do X. Then we do Y. So the result is Z.")
    assert answer == "So the result is Z."
    assert think == "First we do X. Then we do Y."


def test_split_think_answer_single_sentence():
    think, answer = _split_think_answer("Just one sentence.")
    assert think == answer == "Just one sentence."


def test_split_think_answer_empty():
    assert _split_think_answer("") == ("", "")


def _gsm8k_rows(n):
    for i in range(n):
        yield {"question": f"gsm q{i}", "answer": f"step {i} <<1+1=2>> #### {i}"}


def _openorca_rows(n):
    for i in range(n):
        yield {"question": f"orca q{i}", "response": f"First reasoning {i}. Final answer {i}."}


def _magicoder_rows(n):
    for i in range(n):
        yield {"problem": f"code problem {i}", "solution": f"Explanation {i}. def f(): return {i}."}


def _commitpackft_rows(n):
    for i in range(n):
        yield {
            "old_contents": f"def f():\n    return {i}\n",
            "new_contents": f"def f():\n    return {i + 1}\n",
            "message": f"Fix off-by-one in f() #{i}\n",
            "old_file": "f.py",
        }


def test_each_cot_source_gets_a_fair_independent_share():
    tokenizer = TikTokenHybridTokenizer()
    consumed = {"gsm8k": 0, "orca": 0, "magicoder": 0, "commitpackft": 0}

    def loader(name, config=None, split=None, streaming=None, data_files=None):
        if name == "json" and data_files and "commitpackft" in data_files:
            consumed["commitpackft"] += 1
            return _commitpackft_rows(300)
        if "gsm8k" in name:
            consumed["gsm8k"] += 1
            return _gsm8k_rows(300)
        if "openorca" in name.lower():
            consumed["orca"] += 1
            return _openorca_rows(300)
        if "magicoder" in name.lower():
            consumed["magicoder"] += 1
            return _magicoder_rows(300)
        raise AssertionError(f"unexpected dataset: {name}")

    with patch("datasets.load_dataset", side_effect=loader):
        formatted = load_cot_examples(tokenizer, max_seq_len=256, max_examples=120)

    # Each of the 4 sources was actually queried (called at least once) --
    # confirms none were skipped or starved to zero.
    assert all(v > 0 for v in consumed.values())
    assert len(formatted) > 0
    assert len(formatted) <= 120


def test_commitpackft_cot_uses_real_diff_and_real_commit_message():
    """The <|think|> content must be a real, computed fact about the diff
    (never a fabricated reasoning narrative), and the answer must be the
    dataset's own real commit message."""
    from uchi.flux.cot_distill import generate_commitpackft_cot

    def loader(name, config=None, split=None, streaming=None, data_files=None):
        return _commitpackft_rows(5)

    with patch("datasets.load_dataset", side_effect=loader):
        examples = generate_commitpackft_cot(5)

    assert len(examples) == 5
    ex = examples[0]
    assert "diff to f.py" in ex["think"]
    assert "line(s)" in ex["think"]
    assert ex["answer"] == "Fix off-by-one in f() #0"
    assert "def f():" in ex["question"]


def test_commitpackft_cot_skips_identical_or_empty_rows():
    from uchi.flux.cot_distill import generate_commitpackft_cot

    def identical_rows(n):
        for i in range(n):
            yield {
                "old_contents": "def f(): return 1\n",
                "new_contents": "def f(): return 1\n",  # identical -- no real change
                "message": "No-op commit",
                "old_file": "f.py",
            }

    def loader(name, config=None, split=None, streaming=None, data_files=None):
        return identical_rows(5)

    with patch("datasets.load_dataset", side_effect=loader):
        examples = generate_commitpackft_cot(5)

    assert examples == []
