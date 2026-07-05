import json

from uchi.flux.tokenizer_v2 import TikTokenHybridTokenizer
from uchi.flux.vocab_prune import (
    PrunedTikTokenTokenizer,
    compute_used_vocab,
    load_pruned_tokenizer,
    load_pruned_vocab,
)


def _build_pruned_from_text(base, corpus_texts, max_vocab):
    sequences = []
    for t in corpus_texts:
        shifted = base.encode_text(t)
        sequences.append([i - base.n_special for i in shifted if i >= base.n_special])
    return compute_used_vocab(sequences, max_vocab=max_vocab)


def test_pruned_tokenizer_vocab_size_is_reduced():
    base = TikTokenHybridTokenizer()
    pruned = _build_pruned_from_text(base, ["hello world, this is a test."], max_vocab=50)
    wrapped = PrunedTikTokenTokenizer(base, pruned)
    assert wrapped.vocab_size < base.vocab_size
    assert wrapped.vocab_size == pruned.size + base.n_special


def test_special_tokens_pass_through_untouched():
    base = TikTokenHybridTokenizer()
    pruned = _build_pruned_from_text(base, ["some corpus text"], max_vocab=50)
    wrapped = PrunedTikTokenTokenizer(base, pruned)
    for name, tid in base.SPECIAL_TOKENS.items():
        assert wrapped.encode_special(name) == tid


def test_in_vocab_tokens_roundtrip():
    base = TikTokenHybridTokenizer()
    text = "the quick brown fox jumps over the lazy dog"
    # build the pruned vocab FROM this exact text, so every token in it survives
    pruned = _build_pruned_from_text(base, [text], max_vocab=1000)
    wrapped = PrunedTikTokenTokenizer(base, pruned)

    encoded = wrapped.encode_text(text)
    decoded = wrapped.decode_text(encoded)
    assert decoded.strip() == text.strip()


def test_out_of_vocab_tokens_fall_back_to_unk_not_a_crash():
    base = TikTokenHybridTokenizer()
    pruned = _build_pruned_from_text(base, ["a b c"], max_vocab=3)  # tiny vocab
    wrapped = PrunedTikTokenTokenizer(base, pruned)

    # Encode something with tokens definitely NOT in the tiny pruned vocab.
    encoded = wrapped.encode_text("a completely different sentence about quantum physics")
    assert all(0 <= tid < wrapped.vocab_size for tid in encoded)
    # Must not raise, even though most tokens map to the UNK slot.
    wrapped.decode_text(encoded)


def test_save_and_load_pruned_vocab_roundtrip(tmp_path):
    base = TikTokenHybridTokenizer()
    pruned = _build_pruned_from_text(base, ["hello world"], max_vocab=50)

    path = str(tmp_path / "pruned.json")
    with open(path, "w") as f:
        json.dump(
            {
                "size": pruned.size,
                "old_to_new": pruned.old_to_new,
                "new_to_old": {str(k): v for k, v in pruned.new_to_old.items()},
            },
            f,
        )

    loaded = load_pruned_vocab(path)
    assert loaded.size == pruned.size
    assert loaded.old_to_new == pruned.old_to_new
    assert loaded.new_to_old == pruned.new_to_old


def test_load_pruned_tokenizer_end_to_end(tmp_path):
    base = TikTokenHybridTokenizer()
    text = "grounded retrieval beats hallucination"
    pruned = _build_pruned_from_text(base, [text], max_vocab=1000)

    path = str(tmp_path / "pruned.json")
    with open(path, "w") as f:
        json.dump(
            {
                "size": pruned.size,
                "old_to_new": pruned.old_to_new,
                "new_to_old": {str(k): v for k, v in pruned.new_to_old.items()},
            },
            f,
        )

    wrapped = load_pruned_tokenizer(path, base_tokenizer=base)
    encoded = wrapped.encode_text(text)
    assert wrapped.decode_text(encoded).strip() == text.strip()
    assert max(encoded) < wrapped.vocab_size
