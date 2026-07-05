from uchi.flux.vocab_prune import compute_used_vocab, coverage


def test_compute_used_vocab_keeps_most_frequent():
    # token 5 appears most, then 7, then 9 -- with max_vocab=3 (2 real + UNK)
    # only the top-2 most frequent should survive.
    sequences = [[5, 5, 5, 7, 7, 9], [5, 7]]
    pruned = compute_used_vocab(sequences, max_vocab=3)
    assert pruned.size == 3
    assert 5 in pruned.old_to_new
    assert 7 in pruned.old_to_new
    assert 9 not in pruned.old_to_new


def test_remap_maps_known_tokens_and_falls_back_to_unk():
    sequences = [[5, 5, 7]]
    pruned = compute_used_vocab(sequences, max_vocab=10)
    remapped = pruned.remap([5, 7, 999])
    assert remapped[0] == pruned.old_to_new[5]
    assert remapped[1] == pruned.old_to_new[7]
    assert remapped[2] == 0  # UNK fallback for a token never seen


def test_new_to_old_is_a_valid_inverse():
    sequences = [[1, 2, 3, 2, 1, 1]]
    pruned = compute_used_vocab(sequences, max_vocab=10)
    for old_id, new_id in pruned.old_to_new.items():
        assert pruned.new_to_old[new_id] == old_id


def test_compact_ids_are_dense_starting_at_1():
    sequences = [[100, 200, 300]]
    pruned = compute_used_vocab(sequences, max_vocab=10)
    assert sorted(pruned.old_to_new.values()) == [1, 2, 3]


def test_coverage_is_full_when_vocab_isnt_truncated():
    sequences = [[1, 2, 3], [2, 3, 4]]
    pruned = compute_used_vocab(sequences, max_vocab=100)
    assert coverage(pruned, sequences) == 1.0


def test_coverage_drops_when_vocab_is_truncated():
    # 10 occurrences of token 1, 1 occurrence each of 2..11 -- keeping only
    # the single most frequent token should cover just the majority token.
    sequences = [[1] * 10 + list(range(2, 12))]
    pruned = compute_used_vocab(sequences, max_vocab=2)  # 1 real slot + UNK
    cov = coverage(pruned, sequences)
    assert 0.0 < cov < 1.0
    assert pruned.new_to_old[1] == 1  # the majority token survives


def test_empty_corpus_yields_empty_vocab():
    pruned = compute_used_vocab([], max_vocab=100)
    assert pruned.size == 1  # just the UNK slot
    assert pruned.remap([1, 2, 3]) == [0, 0, 0]
