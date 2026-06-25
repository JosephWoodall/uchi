"""
cognitive_debugger.py
=====================
Three surgical probe modes for diagnosing Uchi engine failures.

Usage:
  python scripts/cognitive_debugger.py --probe-tokenizer "a complete gibberish word likepythoon"
  python scripts/cognitive_debugger.py --probe-trie "write a python function"
  python scripts/cognitive_debugger.py --probe-mcts "write a python function"
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_SEP = "─" * 60


# ── helpers ───────────────────────────────────────────────────────────────────

def _load(brain_path: str):
    from uchi.cli import load_brain
    router = load_brain(brain_path)
    if router is None:
        print(f"  ERROR: brain not found at {brain_path!r}. Run bootstrap first.")
        sys.exit(1)
    return router


def _hdr(title: str):
    print(f"\n{_SEP}")
    print(f"  {title}")
    print(_SEP)


# ── probe 1: tokenizer ────────────────────────────────────────────────────────

def probe_tokenizer(text: str, brain_path: str):
    """Show exactly how OmniTokenizer chunks *text*, revealing OOV fragmentation."""
    from uchi.omni_tokenizer import OmniTokenizer

    _hdr(f"PROBE: tokenizer  |  input: {text!r}")
    router = _load(brain_path)
    tok    = router.tokenizer

    words  = text.split()
    print(f"\n  Input words : {words}")
    print(f"  Known vocab : {len(tok._known_concepts)} concepts\n")

    all_tokens: list[str] = []
    for word in words:
        if word in tok._known_concepts:
            tokens = [word]
            path   = "trie-direct"
        else:
            subwords = OmniTokenizer._bpe_fallback(word)
            tokens   = subwords
            path     = "BPE-fallback"
        in_trie = [t for t in tokens if t in tok._known_concepts]
        oov     = [t for t in tokens if t not in tok._known_concepts]
        status  = "OK" if not oov else f"OOV({len(oov)}/{len(tokens)})"
        print(f"  {word!r:<28} → {tokens}  [{path}]  [{status}]")
        if oov:
            print(f"    ⚠  not in trie: {oov}")
        all_tokens.extend(tokens)

    print(f"\n  Full token sequence ({len(all_tokens)} tokens):")
    print(f"  {all_tokens}")
    coverage = sum(1 for t in all_tokens if t in tok._known_concepts) / max(len(all_tokens), 1)
    print(f"\n  Trie coverage: {coverage:.0%}  ({int(coverage*len(all_tokens))}/{len(all_tokens)} tokens known)")
    if coverage < 0.5:
        print("  ⚠  Low coverage — MCTS will rely heavily on neural prior rather than trie statistics.")
    print()


# ── probe 2: trie walk ────────────────────────────────────────────────────────

def probe_trie(text: str, brain_path: str, top_k: int = 5):
    """Walk the UniversalPredictor trie token-by-token and show where the path breaks."""
    router = _load(brain_path)
    pred   = router.predictor._pred
    tok    = router.tokenizer

    concepts = tok.tokenize(text)
    seed     = ["<|user|>"] + concepts + ["<|assistant|>"]

    _hdr(f"PROBE: trie  |  input: {text!r}")
    print(f"\n  Tokenized  : {concepts}")
    print(f"  Trie seed  : {seed}")
    print(f"  Trie nodes : {len(pred._nodes):,}\n")

    print(f"  {'Depth':<6} {'Token':<28} {'In-trie?':<12} {'Children'}")
    print(f"  {'─'*5}  {'─'*27}  {'─'*11}  {'─'*8}")

    stop_depth = None
    for depth, tok_str in enumerate(seed):
        dist = pred.predict(seed[:depth + 1]) if depth > 0 else {}
        # Check if this prefix exists by peeking distribution
        dist_check = {}
        try:
            dist_check = router.predictor.peek_distribution(seed[:depth + 1])
        except Exception:
            pass
        known = bool(dist_check)
        n_children = len(dist_check)
        marker = "" if known else " ← LOST"
        print(f"  {depth:<6} {tok_str!r:<28} {'yes' if known else 'NO':<12} {n_children}{marker}")
        if not known and stop_depth is None:
            stop_depth = depth
            break

    if stop_depth is None:
        print(f"\n  ✓ Full seed sequence is known to the trie.")
    else:
        print(f"\n  Path lost at depth {stop_depth} on token {seed[stop_depth]!r}")

    # Show top continuations from the last known prefix
    last_good = seed[:stop_depth] if stop_depth else seed
    try:
        top_dist = router.predictor.peek_distribution(last_good[-8:])
        if top_dist:
            top_items = sorted(top_dist.items(), key=lambda x: -x[1])[:top_k]
            print(f"\n  Top {top_k} continuations from depth {len(last_good) - 1}:")
            for rank, (t, p) in enumerate(top_items, 1):
                bar = "█" * int(p * 30 / max(v for _, v in top_items))
                print(f"    [{rank}] {t!r:<28} p={p:.4f}  {bar}")
    except Exception as e:
        print(f"  (peek failed: {e})")
    print()


# ── probe 3: MCTS tree ────────────────────────────────────────────────────────

def _collect_branches(node, seed_len: int, max_depth: int = 6, prefix=None):
    """DFS to collect all leaf paths as (tokens_generated, value, visits)."""
    if prefix is None:
        prefix = []
    gen = node.tokens[seed_len:]
    if not node.children or len(gen) >= max_depth:
        yield (list(gen), node.value, node.visits)
        return
    for child in node.children.values():
        if not child.pruned:
            yield from _collect_branches(child, seed_len, max_depth)


def probe_mcts(text: str, brain_path: str, max_nodes: int = 80):
    """Run tree search and dump the top branches with value + visit statistics."""
    from uchi.tree_search_engine import TreeSearchEngine

    router = _load(brain_path)
    tok    = router.tokenizer

    concepts = tok.tokenize(text)
    seed     = ["<|user|>"] + concepts + ["<|assistant|>"]

    _hdr(f"PROBE: MCTS  |  input: {text!r}  |  budget={max_nodes}")
    print(f"\n  Seed ({len(seed)} tokens): {seed}")
    print(f"  Running tree search...")

    # Temporarily monkey-patch search to capture root
    engine = TreeSearchEngine(router)
    _root_holder: list = []

    _orig_best = engine._best_path

    def _patched_best(root, seed_len):
        _root_holder.append(root)
        return _orig_best(root, seed_len)

    engine._best_path = _patched_best

    import time
    t0 = time.perf_counter()
    result = engine.search(seed, max_nodes=max_nodes)
    elapsed = time.perf_counter() - t0

    print(f"  Done in {elapsed:.2f}s  ({max_nodes/elapsed:.0f} nodes/s)\n")
    print(f"  Best result: {result}")

    if not _root_holder:
        print("  (tree not captured)")
        return

    root = _root_holder[0]

    # Count total nodes and pruned nodes
    total, pruned = 0, 0
    queue = [root]
    while queue:
        n = queue.pop()
        total += 1
        if n.pruned:
            pruned += 1
        queue.extend(n.children.values())

    print(f"\n  Tree stats: {total} nodes total, {pruned} pruned ({pruned/max(total,1):.0%})")

    # Collect all branches and rank by visits × value
    branches = list(_collect_branches(root, len(seed)))
    branches.sort(key=lambda b: b[2] * max(b[1], 0.0), reverse=True)

    print(f"\n  Top branches (by visits × value):")
    print(f"  {'Rank':<5} {'V':>7} {'N':>5}  Path")
    print(f"  {'─'*4}  {'─'*7}  {'─'*4}  {'─'*40}")

    _STRIP = {"<|user|>", "<|assistant|>", "<|end|>", "<|inner_monologue|>"}
    for rank, (gen, val, visits) in enumerate(branches[:10], 1):
        path_str = " ".join(t for t in gen if t not in _STRIP)
        if not path_str:
            continue
        print(f"  {rank:<5} {val:>7.3f} {visits:>5}  {path_str[:60]}")

    # Expansion distribution: how many children did each non-leaf get?
    depths: dict[int, int] = {}
    q2 = [(root, 0)]
    while q2:
        n, d = q2.pop()
        if n.children:
            depths[d] = depths.get(d, 0) + 1
        for c in n.children.values():
            q2.append((c, d + 1))

    print(f"\n  Expansions by depth:")
    for d in sorted(depths)[:8]:
        bar = "█" * min(depths[d], 40)
        print(f"    depth {d}: {depths[d]:>3}  {bar}")
    print()


# ── probe 4: telemetry snapshot ───────────────────────────────────────────────

def _fmt_section(name: str, data: dict, indent: int = 2) -> None:
    pad = " " * indent
    print(f"\n{pad}[{name}]")
    for k, v in data.items():
        if isinstance(v, list):
            print(f"{pad}  {k}:")
            for item in v[:10]:
                print(f"{pad}    {item}")
        elif isinstance(v, dict):
            print(f"{pad}  {k}:")
            for dk, dv in v.items():
                print(f"{pad}    {dk}: {dv}")
        else:
            print(f"{pad}  {k}: {v}")


def probe_telemetry(history: int = 1, telemetry_dir: str = ".uchi/telemetry"):
    """Display the latest (or last N) telemetry sessions from disk."""
    import time
    from uchi.telemetry import load_latest, load_history

    _hdr("PROBE: telemetry snapshot")

    if history <= 1:
        data = load_latest(telemetry_dir)
        if not data:
            print("\n  No telemetry found. Run a query or dream step first.\n")
            return
        ts = data.get("_ts", 0)
        age = int(time.time() - float(ts))
        print(f"\n  Snapshot from {age}s ago  ({telemetry_dir}/latest.json)\n")
        _SKIP = {"_ts"}
        for section, content in data.items():
            if section in _SKIP:
                continue
            if isinstance(content, dict):
                _fmt_section(section, content)
            else:
                print(f"  {section}: {content}")
    else:
        sessions = load_history(n=history, telemetry_dir=telemetry_dir)
        if not sessions:
            print("\n  No history found in SQLite database.\n")
            return
        for idx, data in enumerate(sessions, 1):
            ts = data.get("_ts", 0)
            import time as _time
            age = int(_time.time() - float(ts))
            print(f"\n  ── Session {idx} ({age}s ago) ──")
            for section, content in data.items():
                if section == "_ts":
                    continue
                if isinstance(content, dict):
                    _fmt_section(section, content, indent=4)
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cognitive debugger for the Uchi engine. Probe tokeniser, trie, MCTS, or telemetry."
    )
    parser.add_argument("--brain", default="brain.uchi", help="Path to brain file")
    parser.add_argument("--probe-tokenizer", metavar="TEXT",
                        help="Show how OmniTokenizer chunks TEXT, revealing OOV fragmentation")
    parser.add_argument("--probe-trie", metavar="TEXT",
                        help="Walk the trie token-by-token and show where the path breaks")
    parser.add_argument("--probe-mcts", metavar="TEXT",
                        help="Run MCTS and dump tree branches with value/visit stats")
    parser.add_argument("--telemetry", nargs="?", const=1, type=int, metavar="N",
                        help="Display latest telemetry snapshot (--telemetry N for last N sessions)")
    parser.add_argument("--telemetry-dir", default=".uchi/telemetry",
                        help="Telemetry directory (default: .uchi/telemetry)")
    parser.add_argument("--nodes", type=int, default=80,
                        help="MCTS node budget for --probe-mcts (default: 80)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Top-K continuations shown by --probe-trie (default: 5)")

    args = parser.parse_args()

    ran_any = False
    if args.probe_tokenizer:
        probe_tokenizer(args.probe_tokenizer, args.brain)
        ran_any = True
    if args.probe_trie:
        probe_trie(args.probe_trie, args.brain, top_k=args.top_k)
        ran_any = True
    if args.probe_mcts:
        probe_mcts(args.probe_mcts, args.brain, max_nodes=args.nodes)
        ran_any = True
    if args.telemetry is not None:
        probe_telemetry(history=args.telemetry, telemetry_dir=args.telemetry_dir)
        ran_any = True

    if not ran_any:
        parser.print_help()
