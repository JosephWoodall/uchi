"""
offline_dream.py
================
Post-bootstrap SSM alignment via sequential offline dreaming.

Once the trie has Python and Q&A data from the two bootstrap scripts, this
loop drives convergent self-play: the ConvergentEngine generates candidates
that the SSM then trains on synchronously (no background threads).

Why synchronous here (not `_fire_contrastive_update`):
- No concurrent users in a dream loop; thread overhead buys nothing.
- Synchronous backward passes let us detect loss divergence immediately.
- A gradient collision during dreaming would silently corrupt the SSM;
  with the lock removed we surface errors rather than racing.

Expected effect after ~500 iterations:
- tool routing precision:  33% → >70%
- convergent_oracle_pass_rate: 37% → >60%
- avg prompt entropy: 9.86 → <7 bits/tok

Usage
-----
    python scripts/offline_dream.py
    python scripts/offline_dream.py --iterations 200  # quick smoke-test
    python scripts/offline_dream.py --brain my.uchi --iterations 2000

Ctrl-C is safe — ssm_dynamics.pt is saved on interrupt.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)


# ── seed concept pools ────────────────────────────────────────────────────────

_CODE_CONCEPTS = [
    ["write", "a", "function", "called", "add"],
    ["implement", "binary", "search", "in", "python"],
    ["write", "a", "class", "called", "Stack"],
    ["reverse", "a", "linked", "list"],
    ["compute", "fibonacci", "numbers", "iteratively"],
    ["sort", "a", "list", "using", "quicksort"],
    ["write", "a", "recursive", "factorial", "function"],
    ["parse", "a", "json", "string", "safely"],
    ["find", "the", "maximum", "in", "an", "array"],
    ["check", "if", "a", "string", "is", "a", "palindrome"],
]

_CONVO_CONCEPTS = [
    ["what", "is", "the", "capital", "of", "France"],
    ["explain", "photosynthesis"],
    ["what", "causes", "thunder"],
    ["who", "invented", "the", "telephone"],
    ["what", "is", "machine", "learning"],
    ["how", "does", "a", "hash", "table", "work"],
    ["what", "is", "the", "speed", "of", "light"],
    ["describe", "the", "water", "cycle"],
    ["what", "is", "the", "largest", "planet"],
    ["explain", "the", "concept", "of", "recursion"],
]

_ALL_CONCEPTS = _CODE_CONCEPTS + _CONVO_CONCEPTS


# ── dream loop ────────────────────────────────────────────────────────────────

def _find_hard_negative(
    router,
    ssm,
    query_tokens: list,
    positive_tokens: list,
    n_candidates: int = 4,
):
    """
    Generate n_candidates alternative responses and return the one with the
    highest SSM cosine similarity to the query — excluding positive_tokens.

    This is the "hard negative": the candidate the SSM currently (wrongly)
    scores closest to the query. Training against it is what pushes the 256D
    space to actually separate good from bad answers, rather than coasting on
    the vast empty geometry.
    """
    import torch
    _STOP = frozenset({"<|user|>", "<|assistant|>", "<|end|>"})
    temps = [0.5, 0.8, 1.1, 1.4][:n_candidates]

    candidates = []
    for temp in temps:
        try:
            raw = router.predictor.generate(
                n_tokens=20, seed=query_tokens, temperature=temp, use_mcts=False
            )
            cand = [t for t in raw if t not in _STOP]
            if cand and cand != positive_tokens:
                candidates.append(cand)
        except Exception:
            continue

    if not candidates:
        return None

    with torch.no_grad():
        q_state = ssm.get_state(query_tokens).squeeze(0)
        q_norm = q_state / (q_state.norm() + 1e-9)
        best_sim, hard_neg = -1.0, None
        for cand in candidates:
            c_state = ssm.get_state(cand).squeeze(0)
            c_norm = c_state / (c_state.norm() + 1e-9)
            sim = (q_norm * c_norm).sum().item()
            if sim > best_sim:
                best_sim = sim
                hard_neg = cand

    return hard_neg


def dream(router, *, iterations: int = 500, replay_db: str = "replay.db") -> None:
    """
    Run *iterations* sequential dream cycles against *router*.

    Each cycle:
    1. Try to sample a hard experience from the ExperienceReplayBuffer (prioritized
       by TD error / loss magnitude).  Fall back to random seed concept if buffer empty.
    2. Run ConvergentEngine.generate(bootstrapped=True) → oracle-best positive.
    3. Hard-negative mining: find the alternative the SSM currently ranks closest.
    4. Contrastive training: pull positive toward query, push hard negative away.
    5. Single compute_loss call (dynamics + value + policy actor) on the full
       positive sequence — avoids double GRU forward before .backward().
    6. Push successful experience to the replay buffer; update priority with
       resulting loss so high-loss memories get re-sampled sooner.

    Pass iterations=0 for daemon mode (loop forever — use Ctrl-C to stop).
    """
    import torch
    from uchi.neuro_symbolic import get_ssm
    from uchi.vector_oracle import contrastive_update, hard_negative_contrastive_update
    from uchi.experience_replay import ExperienceReplayBuffer
    from uchi.semantic_index import get_semantic_index

    ssm = get_ssm()
    ssm.train()
    optimizer = torch.optim.Adam(ssm.parameters(), lr=5e-4)

    engine  = router.convergent
    replay  = ExperienceReplayBuffer(replay_db)
    sem_idx = get_semantic_index()

    daemon_mode = (iterations == 0)
    if daemon_mode:
        _log.info("Starting offline dream daemon (infinite loop, replay buffer) …")
    else:
        _log.info("Starting offline dream loop (%d iterations, replay buffer) …", iterations)
    losses: list = []

    step = 0
    while daemon_mode or step < iterations:
        step += 1

        # 1. Query source: prioritized replay → seed concept fallback
        memory_id: int | None = None
        sampled_memory = None
        if len(replay) > 0:
            batch = replay.sample(batch_size=1)
            if batch:
                sampled_memory = batch[0]
                memory_id = sampled_memory["id"]
                concepts = sampled_memory["query"]
                # Strip conversation wrapper tokens if present so the engine
                # receives raw concept tokens.
                _WRAPPERS = {"<|user|>", "<|assistant|>", "<|end|>"}
                concepts = [t for t in concepts if t not in _WRAPPERS]
                if not concepts:
                    concepts = random.choice(_ALL_CONCEPTS)

        if sampled_memory is None:
            concepts = random.choice(_ALL_CONCEPTS)

        try:
            kind, payload, reward = engine.generate(concepts, bootstrapped=True)
        except Exception as exc:
            _log.debug("generate error at step %d: %s", step, exc)
            continue

        if kind == "tool":
            continue

        positive_tokens = payload if isinstance(payload, list) else []
        if not positive_tokens:
            continue

        # AST Blame Assignment (Punished Prefix Trap fix).
        # When the engine returns "uncertain", the best MCTS candidate may have
        # a valid syntactic prefix despite failing the oracle as a whole.
        # Rather than discarding the entire sequence, extract the valid prefix
        # and train on it with partial credit — teaching the value head that
        # the prefix was *good* even though the terminal token was wrong.
        if kind == "uncertain":
            from uchi.omni_evaluator import oracle_ast_blame
            blame_rewards = oracle_ast_blame(positive_tokens)
            valid_len = next(
                (i for i, r in enumerate(blame_rewards) if r == 0.0),
                len(blame_rewards),
            )
            if valid_len == 0:
                continue  # no valid prefix at all — skip
            try:
                import uchi.telemetry as _tel
                _tel.record("dreaming", "ast_blame_index", valid_len)
            except Exception:
                pass
            positive_tokens = positive_tokens[:valid_len]
            reward = 0.5  # partial credit for a syntactically valid prefix

        query_tokens = ["<|user|>"] + list(concepts) + ["<|assistant|>"]

        # Index each step of the positive sequence into the semantic k-NN index.
        # This populates the FAISS/numpy fallback used by tree_search when the
        # trie has zero children (cold-start or OOV contexts).
        try:
            with torch.no_grad():
                ssm.eval()
                _ctx = list(query_tokens)
                for _tok in positive_tokens:
                    _state = ssm.get_state(_ctx[-8:])
                    sem_idx.add(_state, _tok)
                    _ctx.append(_tok)
                ssm.train()
        except Exception:
            pass

        # 2. Hard negative mining
        hard_neg = _find_hard_negative(router, ssm, query_tokens, positive_tokens)

        if hard_neg is not None:
            try:
                import uchi.telemetry as _tel
                import torch as _torch
                with _torch.no_grad():
                    _pos = ssm.get_state(positive_tokens).squeeze(0)
                    _neg = ssm.get_state(hard_neg).squeeze(0)
                    _sim = (_pos / (_pos.norm() + 1e-9) * _neg / (_neg.norm() + 1e-9)).sum().item()
                _tel.record("dreaming", "contrastive_cosine_sim", round(_sim, 5))
            except Exception:
                pass

        # 3. Contrastive update
        if hard_neg is not None:
            try:
                hard_negative_contrastive_update(
                    query_tokens, positive_tokens, hard_neg,
                    reward, optimizer, ssm=ssm,
                )
            except Exception as exc:
                _log.debug("hard_neg_update error at step %d: %s", step, exc)
        else:
            try:
                contrastive_update(query_tokens, positive_tokens, reward, optimizer, ssm=ssm)
            except Exception as exc:
                _log.debug("contrastive_update error at step %d: %s", step, exc)

        # 4. Single compute_loss call: dynamics + value + policy actor in one pass.
        #    Avoids running two GRU forward passes before .backward(), which would
        #    invalidate the first call's gradient graph via in-place GRU weight updates.
        full_sequence = query_tokens + positive_tokens
        step_loss = 0.0
        try:
            ssm.train()
            optimizer.zero_grad()
            total = ssm.compute_loss(full_sequence, reward=max(reward, 0.0))
            total.backward()
            optimizer.step()
            step_loss = total.item()
            losses.append(step_loss)
        except Exception as exc:
            _log.debug("SSM train error at step %d: %s", step, exc)

        # 5. Replay buffer maintenance
        if memory_id is not None:
            # Update priority: high loss → stay in queue; low loss → deprioritize
            replay.update_priority(memory_id, step_loss)
        else:
            # Push new experience with current loss as initial priority
            replay.push(query_tokens, positive_tokens, hard_neg, priority=max(step_loss, 1.0))

        if step % 50 == 0:
            recent_mean = (sum(losses[-50:]) / len(losses[-50:])) if losses else 0.0
            neg_label = "hard-neg" if hard_neg is not None else "std-neg"
            buf_size = len(replay)
            if daemon_mode:
                _log.info("  step %d  kind=%s  [%s]  loss_mean(last50)=%.4f  buf=%d",
                          step, kind, neg_label, recent_mean, buf_size)
            else:
                _log.info("  step %d / %d  kind=%s  [%s]  loss_mean(last50)=%.4f  buf=%d",
                          step, iterations, kind, neg_label, recent_mean, buf_size)
            try:
                import uchi.telemetry as _tel
                _tel.record("dreaming", "step",              step)
                _tel.record("dreaming", "loss_mean_last50",  round(recent_mean, 6))
                _tel.record("dreaming", "replay_buffer_size", buf_size)
                _tel.record("dreaming", "semantic_index_size", len(sem_idx))
                _tel.flush()
            except Exception:
                pass
            try:
                sem_idx.save()
                _log.debug("Semantic index saved (%d entries)", len(sem_idx))
            except Exception:
                pass

    _log.info("Dream loop complete. %d steps executed.", step)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline SSM alignment via sequential ConvergentEngine self-play."
    )
    parser.add_argument("--brain",       default="brain.uchi",    help="Brain file path")
    parser.add_argument("--ssm-out",     default="ssm_dynamics.pt", help="Where to write the SSM checkpoint")
    parser.add_argument("--replay-db",   default="replay.db",     help="SQLite replay buffer path (default replay.db)")
    parser.add_argument("--iterations",  type=int, default=500,
                        help="Number of dream cycles (default 500). 0 = run forever (daemon mode).")
    parser.add_argument("--daemon",      action="store_true",     help="Alias for --iterations 0 (loop forever).")
    args = parser.parse_args()

    if args.daemon:
        args.iterations = 0

    from uchi.cli import load_brain
    import torch
    from uchi.neuro_symbolic import get_ssm

    router = load_brain(args.brain)
    if router is None:
        _log.warning("Brain not found at %s — run bootstrap scripts first.", args.brain)
        _log.warning("Continuing with a fresh brain (trie will be empty; dreaming uninformative).")
        from uchi.omni_router import OmniRouter
        router = OmniRouter(use_bpe=False)

    import signal

    def _shutdown(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _shutdown)

    try:
        dream(router, iterations=args.iterations, replay_db=args.replay_db)
    except KeyboardInterrupt:
        _log.info("Interrupted — saving checkpoint …")
    finally:
        torch.save(get_ssm().state_dict(), args.ssm_out)
        _log.info("SSM checkpoint saved to %s.", args.ssm_out)


if __name__ == "__main__":
    main()
