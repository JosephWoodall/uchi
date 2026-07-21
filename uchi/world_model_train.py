"""world_model_train.py — 0.5.0 Item 8: real WorldModel training launcher.

`uchi/world_model.py`'s `WorldModel.train_dynamics`/`update_value_from_state`
and `uchi/mcts.py`'s `MCTS` are fully built and unit-tested
(`tests/test_world_model.py`, `tests/test_mcts.py`) against synthetic
tensors / a stub dynamics-value function, per both modules' own docstrings:
"real training waits for [the Phase 1->4] gate." Real training needs real
trajectories with real rewards, which `uchi/grpo_train.py` (Item 6) now
produces into `.uchi/corpus/item6_trajectories.jsonl`. This script is the
launcher that consumes them.

The base FLUX model stays **frozen** throughout -- only `WorldModel`'s two
small heads (`DynamicsHead`, `ValueHead`) train, exactly matching
`train_dynamics`'s own `model.eval()`/`torch.no_grad()` internals. No
optimizer touches the FLUX backbone here.

Usage:
    .venv/bin/python -m uchi.world_model_train --base uchi/flux/checkpoints/flux_best.pt \\
        --pruned-vocab uchi/flux/checkpoints/pruned_vocab_0_5_0_32k.json
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from uchi.flux.inference_engine import _load_flux_for_inference, build_generate_fn_from_model
from uchi.mcts import MCTS
from uchi.world_model import WorldModel, extract_state

DEFAULT_TRAJECTORY_LOG = ".uchi/corpus/item6_trajectories.jsonl"


def load_trajectories(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trajectory log at {path} -- run uchi/grpo_train.py (Item 6) first to "
            f"produce real trajectories. Training WorldModel against nothing would be a "
            f"silent no-op, not a smaller run."
        )
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"{path} exists but is empty -- same problem as a missing file.")
    return records


def main():
    parser = argparse.ArgumentParser(description="FLUX 0.5.0 Item 8 — real WorldModel training")
    parser.add_argument("--base", type=str, required=True, help="Path to the (frozen) FLUX checkpoint")
    parser.add_argument("--pruned-vocab", type=str, default=None,
                        help="Path to the PrunedVocab JSON --base was trained with.")
    parser.add_argument("--trajectory-log", type=str, default=DEFAULT_TRAJECTORY_LOG)
    parser.add_argument("--dynamics-steps", type=int, default=300)
    parser.add_argument("--dynamics-lr", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--value-lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default=None, help="Defaults to WorldModel.CKPT_PATH")
    parser.add_argument("--smoke-prompt", type=str, default="What is 2+2?",
                         help="Prompt used for the post-training MCTS.select_action smoke-check")
    parser.add_argument("--device", default=None,
                        help="Force cpu/cuda (default: auto-detect). Useful to avoid VRAM "
                             "contention with a concurrent training run on the same GPU.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("FLUX 0.5.0 Item 8 — Real WorldModel Training")
    print("=" * 72)

    records = load_trajectories(args.trajectory_log)
    print(f"  Loaded {len(records):,} real trajectories from {args.trajectory_log}")

    model, tokenizer, device, user_id, asst_id, think_id, stop_ids = _load_flux_for_inference(
        args.base, device, args.pruned_vocab,
    )
    model.eval()  # frozen throughout -- only WorldModel's heads train
    d_model = model.embedding.weight.shape[1]
    print(f"  Base:         {args.base}  (frozen, d_model={d_model})")

    world_model = WorldModel(d_model=d_model).to(device)

    # ── Dynamics head: supervise next-hidden-state prediction from real
    # trajectories ──
    token_sequences = [r["tokens"] for r in records if r.get("tokens")]
    world_model.train_dynamics(
        model, token_sequences, device,
        steps=args.dynamics_steps, lr=args.dynamics_lr, seq_len=args.seq_len,
    )

    # ── Value head: supervise V(final_state) against each trajectory's real
    # collected reward ──
    print(f"  [WorldModel] Value head: {len(records)} real (trajectory, reward) pairs")
    value_losses = []
    for r in records:
        tokens = r.get("tokens")
        if not tokens:
            continue
        seq = torch.tensor([tokens], dtype=torch.long, device=device)
        loss = world_model.update_value(model, seq, r["reward"], device, lr=args.value_lr)
        value_losses.append(loss)
    if value_losses:
        print(f"  [WorldModel] Value head done. Avg loss: {sum(value_losses) / len(value_losses):.5f}")

    out_path = args.out or WorldModel.CKPT_PATH
    world_model.save(out_path)
    print(f"  Saved WorldModel checkpoint: {out_path}")

    # ── Smoke-check: real MCTS.select_action against the trained world
    # model + a live generate_fn. Confirms the wiring works end-to-end --
    # search *quality* is Item 9's job, not asserted here. ──
    print("\n  Smoke-checking MCTS.select_action ...")
    generate_fn = build_generate_fn_from_model(
        model, tokenizer, device, user_id, asst_id, think_id, stop_ids, greedy=False,
    )
    initial_ids = torch.tensor([[user_id] + tokenizer.encode_text(args.smoke_prompt)], device=device)
    state_vec = extract_state(model, initial_ids)
    mcts = MCTS(world_model)
    best_action = mcts.select_action(state_vec, model, tokenizer, generate_fn, args.smoke_prompt)
    print(f"  MCTS smoke-check OK. Selected action text (truncated): {best_action.text[:200]!r}")


if __name__ == "__main__":
    main()
