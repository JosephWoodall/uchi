"""Offline GRPO format-reward training job.

Generates response candidates from the trie for a structured set of seed
queries, scores each response with format_reward(), and trains the SSM
value + dynamics heads using those real scores as rewards.

Unlike the incremental builder's old Phase 3 (blanket reward=1.0), this job:
  - Uses format_reward() for discriminative, intent-aware scoring
  - Trains on real (query, response) pairs with calibrated feedback
  - Updates the replay buffer with difficulty-weighted priorities so future
    online steps continue training on the hardest examples

Intended to be run after any incremental brain build that changes trie
structure, or whenever MMLU no-parse rate exceeds ~40%.

Usage:
    python -m uchi.grpo_offline_trainer
    python -m uchi.grpo_offline_trainer --steps 300 --batch 8
    python -m uchi.grpo_offline_trainer --brain brain.uchi --steps 500
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import pickle
from typing import List, Tuple

import torch
from tqdm import tqdm

from .grpo import format_reward

_log = logging.getLogger(__name__)

# Seed queries: (text, intent_key).  Broad enough to exercise general, math,
# and code format-reward paths.  Web search is disabled during the run so
# these queries exercise only the trie + SSM path.
_SEED_QUERIES: List[Tuple[str, str]] = [
    # general
    ("What is photosynthesis?", "general"),
    ("Who wrote Romeo and Juliet?", "general"),
    ("What is the capital of France?", "general"),
    ("Explain the water cycle.", "general"),
    ("What causes earthquakes?", "general"),
    ("What are the three branches of US government?", "general"),
    ("Name the planets in the solar system.", "general"),
    ("What is DNA?", "general"),
    ("What is the speed of light?", "general"),
    ("Who was Albert Einstein?", "general"),
    ("What is gravity?", "general"),
    ("What is the boiling point of water?", "general"),
    ("What is the French Revolution?", "general"),
    ("What is supply and demand?", "general"),
    ("What is the Pythagorean theorem?", "math"),
    # math
    ("What is 12 times 15?", "math"),
    ("Solve for x: 2x + 4 = 10.", "math"),
    ("What is the square root of 144?", "math"),
    ("What is the derivative of x squared?", "math"),
    ("What is the area of a circle with radius 5?", "math"),
    # code
    ("Write a Python function to reverse a string.", "code"),
    ("How do you open a file in Python?", "code"),
    ("Write a Python for loop that prints 1 to 10.", "code"),
    ("How do you define a class in Python?", "code"),
    ("How do you sort a list in Python?", "code"),
    ("What is a dictionary in Python?", "code"),
    ("How do you handle exceptions in Python?", "code"),
]


def _load_router(brain_path: str):
    if os.path.exists(brain_path):
        try:
            with gzip.open(brain_path, "rb") as f:
                router = pickle.load(f)
            print(f"[+] Brain loaded from {brain_path}")
            return router
        except Exception as e:
            print(f"[!] Brain load failed: {e}")
    from .omni_router import OmniRouter
    print("[*] No brain found — starting cold router.")
    OmniRouter._bootstrap_knowledge = lambda self, *a, **kw: None
    OmniRouter._bootstrap_persona   = lambda self, *a, **kw: None
    return OmniRouter(use_bpe=False)


def _disable_web(router) -> None:
    """Patch router to skip web search so training stays fast."""
    try:
        import uchi.web_search as _ws
        _ws.perform_web_search = lambda *a, **kw: []
        # Also patch any bound reference the router holds.
        if hasattr(router, "_web_search"):
            router._web_search = lambda *a, **kw: []
    except Exception:
        pass


def run_offline_grpo(
    brain_path: str = "brain.uchi",
    steps: int = 300,
    batch_size: int = 8,
    lr: float = 5e-4,
) -> None:
    print("\n" + "=" * 70)
    print(" Uchi Offline GRPO Format-Reward Trainer")
    print("=" * 70)

    router = _load_router(brain_path)
    _disable_web(router)

    from .neuro_symbolic import get_ssm
    from .cli import save_brain

    ssm = get_ssm()
    optimizer = torch.optim.Adam(ssm.parameters(), lr=lr)

    # ── Phase 1: seed replay buffer with format-scored interactions ───────────
    print(f"\n[*] Phase 1: Seeding replay buffer ({len(_SEED_QUERIES)} queries)…")
    seed_rewards: List[float] = []

    for query_text, intent_key in tqdm(_SEED_QUERIES, desc="Seeding"):
        try:
            response = router.chat(query_text)
            if not response:
                continue

            q_tokens = router.tokenizer.tokenize(
                query_text.split(), is_inference=True
            )
            r_tokens = router.tokenizer.tokenize(
                response.split(), is_inference=True
            )

            reward = format_reward(r_tokens, q_tokens, intent_key=intent_key)
            seed_rewards.append(reward)

            # Push to replay buffer with real priority (abs reward = difficulty).
            router.replay_buffer.push(
                query_tokens=q_tokens,
                positive_tokens=r_tokens,
                priority=max(abs(reward), 0.1),
            )

            # Immediate SSM training step with discriminative reward.
            if len(q_tokens) + len(r_tokens) >= 2:
                seq = ["<|user|>"] + q_tokens + ["<|assistant|>"] + r_tokens
                ssm.train()
                optimizer.zero_grad()
                try:
                    v_loss = ssm.update_value(seq, reward=reward)
                    d_loss = ssm.train_dynamics(seq)
                    (v_loss + d_loss).backward()
                    optimizer.step()
                except Exception as e:
                    _log.debug("SSM seed step skipped: %s", e)

        except Exception as e:
            _log.debug("Seed query failed: %s", e)
            continue

    if seed_rewards:
        avg_seed = sum(seed_rewards) / len(seed_rewards)
        print(f"[+] Seed complete — avg format reward: {avg_seed:+.3f} "
              f"(n={len(seed_rewards)})")
    else:
        print("[!] No seed responses generated — replay buffer empty. "
              "Check that the brain is loaded and the trie has content.")

    # ── Phase 2: replay training loop ─────────────────────────────────────────
    print(f"\n[*] Phase 2: Replay training — {steps} steps, batch={batch_size}…")
    buf_size = len(router.replay_buffer)
    if buf_size == 0:
        print("[!] Replay buffer empty after seeding — skipping replay loop.")
    else:
        print(f"[*] Buffer size: {buf_size} experiences")
        total_loss = 0.0
        trained = 0

        for step in tqdm(range(steps), desc="Replay train"):
            batch = router.replay_buffer.sample(batch_size=batch_size)
            if not batch:
                break

            ssm.train()
            for exp in batch:
                q_toks = exp.get("query", [])
                r_toks = exp.get("positive", [])
                if not q_toks or not r_toks:
                    continue

                # Re-score with format_reward (real signal, not stored priority).
                reward = format_reward(r_toks, q_toks)
                seq = ["<|user|>"] + q_toks + ["<|assistant|>"] + r_toks

                optimizer.zero_grad()
                try:
                    v_loss = ssm.update_value(seq, reward=reward)
                    d_loss = ssm.train_dynamics(seq)
                    loss = v_loss + d_loss
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.item())
                    trained += 1

                    mem_id = exp.get("id")
                    if mem_id is not None:
                        router.replay_buffer.update_priority(
                            mem_id, float(loss.item())
                        )
                except Exception as e:
                    _log.debug("Replay step skipped: %s", e)

        if trained:
            print(f"[+] Replay complete — {trained} updates, "
                  f"avg loss: {total_loss / trained:.4f}")

    # ── Phase 3: persist ──────────────────────────────────────────────────────
    print(f"\n[*] Phase 3: Saving updated brain to {brain_path}…")
    save_brain(router, brain_path)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pt_path = os.path.join(project_root, "ssm_dynamics.pt")
    torch.save(ssm.state_dict(), pt_path)
    print(f"[+] SSM weights saved to {os.path.basename(pt_path)}")
    print("[+] Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline GRPO format-reward trainer for Uchi's SSM."
    )
    parser.add_argument("--brain",  default="brain.uchi",
                        help="Path to brain file (default: brain.uchi)")
    parser.add_argument("--steps",  type=int, default=300,
                        help="Replay training steps (default: 300)")
    parser.add_argument("--batch",  type=int, default=8,
                        help="Replay batch size (default: 8)")
    parser.add_argument("--lr",     type=float, default=5e-4,
                        help="SSM optimizer learning rate (default: 5e-4)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    run_offline_grpo(
        brain_path=args.brain,
        steps=args.steps,
        batch_size=args.batch,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
