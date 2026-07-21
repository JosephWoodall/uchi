"""grpo_train.py — 0.5.0 Item 6: real GRPO curriculum training launcher.

`uchi/grpo.py`'s `GRPOTrainer`/`load_curriculum`/`run_curriculum` are fully
built and unit-tested (`tests/test_grpo.py`) but only ever exercised against
a scripted fake `generate_fn` or smoke-tested against a live checkpoint --
nothing drives a real, checkpointed, logged training run the way
`uchi/flux/{train_v2,sft_train,cot_distill,qat_train}.py` do for the FLUX
phases. This script is that launcher.

Does NOT call `GRPOTrainer.run_curriculum()` directly -- that method holds
every instance's `Branch` list (each carrying a `log_prob` tensor) in memory
for the whole run and has no checkpointing/logging hook, both fine for its
own tests' small N but wrong for a real multi-day run. This script re-walks
`load_curriculum()`'s buckets itself (identical field access to
`run_curriculum`'s own loop), calling `GRPOTrainer.train_step()` per
instance directly so it can checkpoint, log, and persist trajectories
incrementally instead.

**Generation must come from the SAME model instance the optimizer updates**,
not a second frozen copy loaded from the checkpoint path -- that's exactly
why `uchi/flux/inference_engine.py` grew `build_generate_fn_from_model`
(factored out of `build_generate_fn` this session) rather than this script
just calling `build_generate_fn(checkpoint=...)`, which would load an
independent, never-updated model.

**Sampling must not be greedy.** GRPO's advantage is zero-variance (no
gradient) when every branch in a group gets the same reward -- the module
docstring's whole reason curriculum bucketing exists. Identical/near-identical
branches from greedy decoding make that worse, not better (the exact bug
`build_batch_generate_fn`'s docstring documents finding in the self-consistency
voting path). `--temperature` defaults to 0.8, `greedy=False` always.

**Trajectory persistence for Item 8**: `Branch` (grpo.py) exposes
`transcript: list[str]` (the parsed/formatted ReAct transcript) and
`reward`/`eval_result`, but never the raw prompt+response strings
`_sample_branch` scores internally (deliberately -- see grpo.py's own
docstring on why the abbreviated transcript isn't what log-prob scoring
uses). For Item 8's `WorldModel.train_dynamics`, re-tokenizing the joined
transcript text is a reasonable, honest substitute -- it doesn't need to
match GRPO's own log-prob computation exactly, just realistic real
trajectories with real rewards attached. Documented here rather than
silently treated as identical.

Usage:
    # Calibrate: measure real per-instance wall-clock before committing to
    # a full run (this project's standing practice -- see Phase 1's token-
    # budget calibration, the corpus-fetch parallelization benchmarking).
    .venv/bin/python -m uchi.grpo_train --base uchi/flux/checkpoints/flux_best.pt \\
        --pruned-vocab uchi/flux/checkpoints/pruned_vocab_0_5_0_32k.json \\
        --max-instances 20

    # Real run, once calibration gives you a --max-instances that fits your
    # wall-clock budget:
    .venv/bin/python -m uchi.grpo_train --base uchi/flux/checkpoints/flux_best.pt \\
        --pruned-vocab uchi/flux/checkpoints/pruned_vocab_0_5_0_32k.json \\
        --max-instances 500
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from uchi.execution_sandbox import ExecutionSandbox
from uchi.flux.inference_engine import _load_flux_for_inference, build_generate_fn_from_model
from uchi.flux.train_v2 import save_checkpoint
from uchi.grpo import GRPOTrainer, load_curriculum
from uchi.repo_fetch import RepoFetchError, ensure_local_clone

DEFAULT_CURRICULUM = ".uchi/corpus/swe_gym_full.jsonl"
DEFAULT_CHECKPOINT_DIR = "uchi/flux/checkpoints/v050_item6_grpo"
DEFAULT_TRAJECTORY_LOG = ".uchi/corpus/item6_trajectories.jsonl"
FULL_CURRICULUM_SIZE = 2438  # .uchi/corpus/swe_gym_full.jsonl's real instance count (0.5.0 Item 1)


def _iter_curriculum(jsonl_path: str, max_instances: int | None):
    """Same bucket-order iteration as `GRPOTrainer.run_curriculum`, exposed
    as a generator here so the caller can act between instances (log,
    checkpoint, persist trajectories) instead of only after everything."""
    buckets = load_curriculum(jsonl_path)
    n_done = 0
    for bucket in buckets:
        for record in bucket:
            if max_instances is not None and n_done >= max_instances:
                return
            yield record
            n_done += 1


def _branch_trajectory_lines(branches, tokenizer, problem_statement: str, instance_id: str) -> list[str]:
    """One JSONL line per branch: real token ids (re-tokenized from the
    branch's own transcript, see module docstring) + its real reward --
    Item 8's `WorldModel.train_dynamics`/`update_value_from_state` input."""
    prompt_text = f"Issue:\n{problem_statement}\n\n"
    lines = []
    for branch in branches:
        full_text = prompt_text + "\n".join(branch.transcript)
        token_ids = tokenizer.encode_text(full_text, max_length=1024)
        resolved = bool(branch.eval_result and branch.eval_result.resolved)
        lines.append(json.dumps({
            "instance_id": instance_id,
            "reward": branch.reward,
            "resolved": resolved,
            "tokens": token_ids,
        }))
    return lines


def main():
    parser = argparse.ArgumentParser(description="FLUX 0.5.0 Item 6 — real GRPO curriculum training")
    parser.add_argument("--base", type=str, required=True, help="Path to the FLUX checkpoint to train")
    parser.add_argument("--pruned-vocab", type=str, default=None,
                        help="Path to the PrunedVocab JSON --base was trained with (REQUIRED if "
                             "--base uses a pruned vocab -- see inference_engine.py's own warning "
                             "about the wrong-vocab-same-size trap found this session).")
    parser.add_argument("--curriculum", type=str, default=DEFAULT_CURRICULUM)
    parser.add_argument("--max-instances", type=int, required=True,
                         help="Hard cap, no silent full-corpus default -- calibrate first.")
    parser.add_argument("--n-branches", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=5, help="ReAct steps per episode")
    parser.add_argument("--max-tokens", type=int, default=300, help="Tokens per ReAct turn")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature -- must be >0 (never greedy, see module docstring)")
    parser.add_argument("--lr", type=float, default=3e-6,
                        help="RL fine-tune step on an already instruction-tuned model -- "
                             "gentler than SFT's 3e-5")
    parser.add_argument("--checkpoint-interval", type=int, default=50,
                        help="Save a model checkpoint + flush trajectories every N instances")
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--trajectory-log", type=str, default=DEFAULT_TRAJECTORY_LOG,
                        help="Appended, not overwritten -- accumulates across runs for Item 8")
    parser.add_argument("--sandbox-timeout", type=float, default=120.0)
    parser.add_argument("--device", default=None,
                        help="Force cpu/cuda (default: auto-detect). Useful to avoid VRAM "
                             "contention with a concurrent training run on the same GPU.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("FLUX 0.5.0 Item 6 — Real GRPO Curriculum Training")
    print("=" * 72)

    model, tokenizer, device, user_id, asst_id, think_id, stop_ids = _load_flux_for_inference(
        args.base, device, args.pruned_vocab,
    )
    model.train()
    model._gradient_checkpointing = True
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Base:         {args.base}")
    print(f"  Parameters:   {n_params:,} ({n_params / 1e6:.1f}M)")
    print(f"  Curriculum:   {args.curriculum}")
    print(f"  Max instances:{args.max_instances}")
    print(f"  n_branches:   {args.n_branches}  temperature: {args.temperature}")
    print("=" * 72)

    generate_fn = build_generate_fn_from_model(
        model, tokenizer, device, user_id, asst_id, think_id, stop_ids,
        greedy=False, temperature=args.temperature,
    )
    sandbox = ExecutionSandbox(timeout=args.sandbox_timeout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    trainer = GRPOTrainer(
        model, tokenizer, sandbox, generate_fn, optimizer,
        n_branches=args.n_branches, max_iterations=args.max_iterations,
        max_tokens=args.max_tokens, device=device,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.trajectory_log)), exist_ok=True)

    n_done = 0
    n_skipped_clone_failed = 0
    reward_sum = 0.0
    resolved_count = 0
    t0 = time.time()

    with open(args.trajectory_log, "a") as traj_f:
        for record in _iter_curriculum(args.curriculum, args.max_instances):
            instance_id = record.get("instance_id", "?")
            try:
                repo_path = ensure_local_clone(record["repo"], record["base_commit"])
            except RepoFetchError as e:
                n_skipped_clone_failed += 1
                print(f"  [{n_done + n_skipped_clone_failed}] {instance_id}: clone failed ({e}), skipping")
                continue

            branches, loss = trainer.train_step(
                repo_path=repo_path,
                problem_statement=record["problem_statement"],
                fail_to_pass=record["fail_to_pass"],
                pass_to_pass=record["pass_to_pass"],
                base_commit=record.get("base_commit"),
            )

            for line in _branch_trajectory_lines(branches, tokenizer, record["problem_statement"], instance_id):
                traj_f.write(line + "\n")

            group_rewards = [b.reward for b in branches]
            group_resolved = sum(1 for b in branches if b.eval_result and b.eval_result.resolved)
            reward_sum += sum(group_rewards)
            resolved_count += group_resolved
            n_done += 1

            elapsed = time.time() - t0
            sec_per_instance = elapsed / n_done
            mean_reward = reward_sum / (n_done * args.n_branches)
            print(
                f"  [{n_done}/{args.max_instances}] {instance_id}: "
                f"loss={loss.item():.4f} rewards={[f'{r:.2f}' for r in group_rewards]} "
                f"resolved={group_resolved}/{args.n_branches} "
                f"running_mean_reward={mean_reward:.4f} "
                f"({sec_per_instance:.1f}s/instance)"
            )

            if n_done % args.checkpoint_interval == 0:
                ckpt_path = os.path.join(args.checkpoint_dir, f"grpo_{n_done:06d}.pt")
                save_checkpoint(model, optimizer, n_done, mean_reward, ckpt_path)
                traj_f.flush()
                print(f"    Saved checkpoint: {ckpt_path}")

    elapsed = time.time() - t0
    final_path = os.path.join(args.checkpoint_dir, "grpo_final.pt")
    mean_reward = reward_sum / max(n_done * args.n_branches, 1)
    save_checkpoint(model, optimizer, n_done, mean_reward, final_path)

    print(f"\n  GRPO training complete. {n_done} instances "
          f"({n_skipped_clone_failed} skipped -- clone failed) in {elapsed / 60:.1f}m")
    print(f"  Mean reward: {mean_reward:.4f}  Resolved: {resolved_count}/{n_done * args.n_branches} branches")
    if n_done:
        sec_per_instance = elapsed / n_done
        projected_full_hours = sec_per_instance * FULL_CURRICULUM_SIZE / 3600
        print(f"  {sec_per_instance:.1f}s/instance -- projected full curriculum "
              f"({FULL_CURRICULUM_SIZE} instances): {projected_full_hours:.1f}h")
    print(f"  Final checkpoint: {final_path}")
    print(f"  Trajectories appended to: {args.trajectory_log}")


if __name__ == "__main__":
    main()
