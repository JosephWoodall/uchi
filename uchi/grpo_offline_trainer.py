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
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

from .grpo import format_reward, dpo_preference_signal, execution_reward

_log = logging.getLogger(__name__)

# Seed queries: (text, intent_key) or (text, intent_key, assertion).
# Code seeds include an assertion string that is executed in a sandbox after
# the response is generated — pass/fail becomes the primary reward signal for
# those seeds (execution_reward), with format_reward as a secondary signal.
_SEED_QUERIES: List[Tuple] = [
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
    # mcq — MMLU-style
    ("The following is a multiple choice question about mathematics. What is 7 times 8? A. 54 B. 56 C. 64 D. 72 Answer:", "mcq"),
    ("The following is a multiple choice question about biology. What is the powerhouse of the cell? A. Nucleus B. Ribosome C. Mitochondria D. Vacuole Answer:", "mcq"),
    ("The following is a multiple choice question about physics. What is the unit of force? A. Joule B. Watt C. Newton D. Pascal Answer:", "mcq"),
    ("The following is a multiple choice question about history. Who was the first US president? A. Thomas Jefferson B. John Adams C. George Washington D. Benjamin Franklin Answer:", "mcq"),
    ("The following is a multiple choice question about chemistry. What is the chemical symbol for water? A. WA B. HO C. H2O D. W2O Answer:", "mcq"),
    ("The following is a multiple choice question about geography. What is the capital of France? A. Berlin B. Madrid C. Rome D. Paris Answer:", "mcq"),
    ("The following is a multiple choice question about mathematics. What is the square root of 81? A. 7 B. 8 C. 9 D. 10 Answer:", "mcq"),
    ("The following is a multiple choice question about science. What planet is closest to the Sun? A. Venus B. Mercury C. Earth D. Mars Answer:", "mcq"),
    ("The following is a multiple choice question about language. What part of speech is 'quickly'? A. Noun B. Verb C. Adjective D. Adverb Answer:", "mcq"),
    ("The following is a multiple choice question about mathematics. How many sides does a hexagon have? A. 5 B. 6 C. 7 D. 8 Answer:", "mcq"),
    # naturalness — trains the policy to produce clean, human-readable output
    # rather than raw trie vocabulary (synset tokens, control markers, etc.)
    ("What is the boiling point of water?", "factual"),
    ("What is the capital of Japan?", "factual"),
    ("Who wrote Romeo and Juliet?", "factual"),
    ("What is photosynthesis?", "factual"),
    ("How many planets are in the solar system?", "factual"),
    ("What causes thunder?", "factual"),
    ("What is the speed of sound?", "factual"),
    ("What is DNA?", "factual"),
    ("What is gravity?", "factual"),
    ("Who invented the telephone?", "factual"),
    # mcq — weak MMLU subjects (scored 0% in baseline): anatomy, astronomy,
    # prehistory, professional_accounting, government_and_politics, human_sexuality,
    # college_biology, professional_medicine, global_facts, elementary_mathematics
    ("The following is a multiple choice question about anatomy. How many bones are in the adult human body? A. 106 B. 206 C. 306 D. 406 Answer:", "mcq"),
    ("The following is a multiple choice question about anatomy. What is the largest organ in the human body? A. Heart B. Liver C. Skin D. Lung Answer:", "mcq"),
    ("The following is a multiple choice question about astronomy. What is the closest star to Earth? A. Betelgeuse B. Sirius C. Proxima Centauri D. Vega Answer:", "mcq"),
    ("The following is a multiple choice question about astronomy. How many moons does Mars have? A. 0 B. 1 C. 2 D. 4 Answer:", "mcq"),
    ("The following is a multiple choice question about prehistory. Which period came first? A. Bronze Age B. Iron Age C. Stone Age D. Copper Age Answer:", "mcq"),
    ("The following is a multiple choice question about prehistory. What species of human first used fire? A. Homo sapiens B. Homo erectus C. Neanderthal D. Homo habilis Answer:", "mcq"),
    ("The following is a multiple choice question about accounting. What does GAAP stand for? A. General Accounting and Auditing Principles B. Generally Accepted Accounting Principles C. Global Accounting Assessment Procedures D. General Audit and Assurance Practices Answer:", "mcq"),
    ("The following is a multiple choice question about accounting. Which financial statement shows a company's revenues and expenses? A. Balance sheet B. Cash flow statement C. Income statement D. Statement of equity Answer:", "mcq"),
    ("The following is a multiple choice question about government and politics. What are the three branches of the US government? A. Federal, State, Local B. Executive, Legislative, Judicial C. Senate, House, President D. Law, Order, Justice Answer:", "mcq"),
    ("The following is a multiple choice question about biology. What is the basic unit of life? A. Atom B. Molecule C. Cell D. Organ Answer:", "mcq"),
    ("The following is a multiple choice question about biology. Which organelle is known as the powerhouse of the cell? A. Nucleus B. Ribosome C. Mitochondria D. Golgi apparatus Answer:", "mcq"),
    ("The following is a multiple choice question about medicine. What does the acronym CPR stand for? A. Cardiac Pulmonary Resuscitation B. Cardiopulmonary Resuscitation C. Critical Patient Recovery D. Chest Pressure Response Answer:", "mcq"),
    ("The following is a multiple choice question about global facts. Which country has the largest population? A. India B. United States C. China D. Russia Answer:", "mcq"),
    ("The following is a multiple choice question about mathematics. What is 15 percent of 200? A. 20 B. 25 C. 30 D. 35 Answer:", "mcq"),
    ("The following is a multiple choice question about mathematics. What is the value of pi rounded to two decimal places? A. 3.12 B. 3.14 C. 3.16 D. 3.18 Answer:", "mcq"),
    # code — MBPP-style with execution assertions.
    # Each tuple is (query, "code", assertion_script).  The assertion is run
    # in a subprocess sandbox after generation; pass = +1.0 reward, fail = -0.5.
    # Function names are specified so the assertion can call them by name.
    (
        "Write a Python function named `factorial` that returns the factorial of n using recursion.",
        "code",
        "assert factorial(0) == 1\nassert factorial(1) == 1\nassert factorial(5) == 120",
    ),
    (
        "Write a Python function named `is_prime` that returns True if n is prime and False otherwise.",
        "code",
        "assert is_prime(2) == True\nassert is_prime(7) == True\nassert is_prime(1) == False\nassert is_prime(4) == False",
    ),
    (
        "Write a Python function named `find_max` that returns the maximum element in a list.",
        "code",
        "assert find_max([3, 1, 4, 1, 5, 9]) == 9\nassert find_max([1]) == 1\nassert find_max([-3, -1, -2]) == -1",
    ),
    (
        "Write a Python function named `count_occurrences` that counts how many times an element appears in a list.",
        "code",
        "assert count_occurrences([1, 2, 2, 3, 2], 2) == 3\nassert count_occurrences([], 1) == 0\nassert count_occurrences([5], 5) == 1",
    ),
    (
        "Write a Python function named `fibonacci` that returns a list of the first n Fibonacci numbers starting from 0.",
        "code",
        "assert fibonacci(1) == [0]\nassert fibonacci(5) == [0, 1, 1, 2, 3]\nassert fibonacci(7) == [0, 1, 1, 2, 3, 5, 8]",
    ),
    (
        "Write a Python function named `is_palindrome` that returns True if a string reads the same forwards and backwards.",
        "code",
        "assert is_palindrome('racecar') == True\nassert is_palindrome('hello') == False\nassert is_palindrome('level') == True\nassert is_palindrome('') == True",
    ),
    (
        "Write a Python function named `flatten` that takes a nested list and returns a single flat list.",
        "code",
        "assert flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]\nassert flatten([[1, [2, 3]], [4]]) == [1, 2, 3, 4]\nassert flatten([]) == []",
    ),
    (
        "Write a Python function named `remove_duplicates` that removes duplicate values from a list while preserving the original order.",
        "code",
        "assert remove_duplicates([1, 2, 2, 3, 3]) == [1, 2, 3]\nassert remove_duplicates([]) == []\nassert remove_duplicates([1]) == [1]",
    ),
    (
        "Write a Python function named `reverse_string` that takes a string and returns it reversed.",
        "code",
        "assert reverse_string('hello') == 'olleh'\nassert reverse_string('') == ''\nassert reverse_string('a') == 'a'",
    ),
    (
        "Write a Python function named `sum_list` that returns the sum of all numbers in a list.",
        "code",
        "assert sum_list([1, 2, 3, 4]) == 10\nassert sum_list([]) == 0\nassert sum_list([-1, 1]) == 0",
    ),
    # code modification seeds — "fix/update existing function" pattern.
    # The query contains a broken function in a fenced code block; the
    # assertion tests the corrected version.  These train the code-context-
    # conditioned path: the code block in the query becomes MCTS bias_context.
    (
        "Fix this Python function so it returns the correct factorial.\n"
        "```python\ndef factorial(n):\n    return 0\n```\n"
        "Write only the corrected function named `factorial`.",
        "code",
        "assert factorial(0) == 1\nassert factorial(1) == 1\nassert factorial(5) == 120",
    ),
    (
        "This palindrome check has a bug — fix it.\n"
        "```python\ndef is_palindrome(s):\n    return s == s[::2]\n```\n"
        "Write only the corrected function named `is_palindrome`.",
        "code",
        "assert is_palindrome('racecar') == True\nassert is_palindrome('hello') == False\nassert is_palindrome('level') == True",
    ),
    (
        "Update this function to return None when the list is empty instead of raising an error.\n"
        "```python\ndef find_max(lst):\n    return max(lst)\n```\n"
        "Write only the updated function named `find_max`.",
        "code",
        "assert find_max([3, 1, 4]) == 4\nassert find_max([-1, -5, -2]) == -1\nassert find_max([]) is None",
    ),
    (
        "Fix this remove_duplicates function — it does not preserve the original order.\n"
        "```python\ndef remove_duplicates(lst):\n    return list(set(lst))\n```\n"
        "Write only the corrected function named `remove_duplicates`.",
        "code",
        "assert remove_duplicates([1, 2, 2, 3, 3]) == [1, 2, 3]\nassert remove_duplicates([3, 1, 2, 1]) == [3, 1, 2]\nassert remove_duplicates([]) == []",
    ),
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
    """Disable web search so training stays fast and offline."""
    router.web_search_enabled = False
    try:
        import uchi.web_search as _ws
        _ws.perform_web_search = lambda *a, **kw: ""
    except Exception:
        pass


def _cap_mcts_for_training() -> None:
    """Cap MCTS rollout budget during seeding — 5 rollouts vs 20-40 normal.

    With a large brain (7MB+) each rollout is expensive. Training only needs
    *a* response, not the optimal one — 5 rollouts is sufficient.
    """
    import uchi.convergent_engine as _ce
    _ce.MAX_BUDGET = 5
    _ce.MIN_ROLLOUTS = 2
    _ce.BUDGET_SCALE_UNCERTAIN = 1  # no budget expansion for uncertain queries during training


def run_offline_grpo(
    brain_path: str = "brain.uchi",
    steps: int = 500,
    batch_size: int = 8,
    lr: float = 5e-4,
) -> None:
    print("\n" + "=" * 70)
    print(" Uchi Offline GRPO Format-Reward Trainer")
    print("=" * 70)

    router = _load_router(brain_path)
    _disable_web(router)
    _cap_mcts_for_training()

    from .neuro_symbolic import get_ssm
    from .cli import save_brain

    ssm = get_ssm()
    optimizer = torch.optim.Adam(ssm.parameters(), lr=lr)

    # ── Phase 1: DPO-style seeding — preferred vs rejected pairs ─────────────
    # For each seed, generate two responses and train the SSM contrastively:
    # push toward the higher-scoring (preferred) and away from the lower-scoring
    # (rejected). This is more stable than single-response GRPO because the
    # training signal is a preference margin, not a raw reward.
    print(f"\n[*] Phase 1: DPO seeding ({len(_SEED_QUERIES)} queries × 2 responses)…")
    seed_rewards: List[float] = []

    for seed in tqdm(_SEED_QUERIES, desc="DPO seed"):
        query_text, intent_key = seed[0], seed[1]
        assertion: str = seed[2] if len(seed) > 2 else ""
        try:
            q_tokens = router.tokenizer.tokenize(
                query_text.split(), is_inference=True
            )

            # Generate two candidate responses per seed query.
            candidates: List[Tuple[List[str], float]] = []
            for _ in range(2):
                resp = router.chat(query_text)
                if not resp:
                    continue
                r_toks = router.tokenizer.tokenize(resp.split(), is_inference=True)
                fmt_rwd = format_reward(r_toks, q_tokens, intent_key=intent_key)
                if assertion and intent_key == "code":
                    exec_rwd = execution_reward(" ".join(r_toks), assertion)
                    # Format is a gate (structure); execution is the primary signal.
                    rwd = 0.4 * fmt_rwd + 0.6 * exec_rwd
                else:
                    rwd = fmt_rwd
                candidates.append((r_toks, rwd))

            if len(candidates) == 2:
                # Sort: preferred = higher format reward
                candidates.sort(key=lambda x: x[1], reverse=True)
                pref_toks, pref_rwd = candidates[0]
                rej_toks,  rej_rwd  = candidates[1]

                push_signal, pull_signal = dpo_preference_signal(pref_rwd, rej_rwd)
                seed_rewards.append(pref_rwd)

                # Push preferred to replay buffer for Phase 2.
                router.replay_buffer.push(
                    query_tokens=q_tokens,
                    positive_tokens=pref_toks,
                    priority=max(abs(pref_rwd), 0.1),
                )

                # DPO contrastive SSM step.
                pref_seq = ["<|user|>"] + q_tokens + ["<|assistant|>"] + pref_toks
                rej_seq  = ["<|user|>"] + q_tokens + ["<|assistant|>"] + rej_toks
                ssm.train()
                optimizer.zero_grad()
                try:
                    v_p = ssm.update_value(pref_seq, reward=push_signal)
                    d_p = ssm.train_dynamics(pref_seq)
                    v_r = ssm.update_value(rej_seq,  reward=pull_signal)
                    d_r = ssm.train_dynamics(rej_seq)
                    (v_p + d_p + v_r + d_r).backward()
                    optimizer.step()
                except Exception as e:
                    _log.debug("DPO seed step skipped: %s", e)

            elif len(candidates) == 1:
                # Fallback: single response → standard GRPO step
                r_toks, rwd = candidates[0]
                seed_rewards.append(rwd)
                router.replay_buffer.push(
                    query_tokens=q_tokens,
                    positive_tokens=r_toks,
                    priority=max(abs(rwd), 0.1),
                )
                seq = ["<|user|>"] + q_tokens + ["<|assistant|>"] + r_toks
                ssm.train()
                optimizer.zero_grad()
                try:
                    v_loss = ssm.update_value(seq, reward=rwd)
                    d_loss = ssm.train_dynamics(seq)
                    (v_loss + d_loss).backward()
                    optimizer.step()
                except Exception as e:
                    _log.debug("GRPO fallback step skipped: %s", e)

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
                # Infer intent from stored query tokens — replay buffer has no intent_key field.
                _last_q = next(
                    (t.lower() for t in reversed(q_toks)
                     if t not in ("<|user|>", "<|assistant|>", "<|end|>")),
                    "",
                )
                _r_intent: Optional[str] = (
                    "mcq"  if _last_q == "answer:" else
                    "code" if any(kw in " ".join(q_toks).lower()
                                  for kw in ("def ", "class ", "import ", "return")) else
                    None
                )
                reward = format_reward(r_toks, q_toks, intent_key=_r_intent)
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
    parser.add_argument("--steps",  type=int, default=500,
                        help="Replay training steps (default: 500)")
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
