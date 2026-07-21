"""Tests for uchi/world_model_train.py (0.5.0 Item 8 launcher).

Covers `load_trajectories` -- the file missing/empty checks and the JSONL
round-trip. Real dynamics/value training itself needs a real live
checkpoint's trajectories, exercised via a live dry-run, not here.
"""
import json

import pytest

from uchi.world_model_train import load_trajectories


def test_load_trajectories_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.jsonl"
    with pytest.raises(FileNotFoundError, match="uchi/grpo_train.py"):
        load_trajectories(str(missing))


def test_load_trajectories_empty_file_raises(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_trajectories(str(empty))


def test_load_trajectories_round_trip(tmp_path):
    records = [
        {"instance_id": "a", "reward": 1.0, "resolved": True, "tokens": [1, 2, 3]},
        {"instance_id": "b", "reward": 0.0, "resolved": False, "tokens": [4, 5]},
    ]
    path = tmp_path / "trajectories.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    loaded = load_trajectories(str(path))
    assert loaded == records


def test_load_trajectories_skips_blank_lines(tmp_path):
    path = tmp_path / "trajectories.jsonl"
    path.write_text('{"instance_id": "a", "reward": 1.0, "resolved": true, "tokens": [1]}\n\n\n')

    loaded = load_trajectories(str(path))
    assert len(loaded) == 1
