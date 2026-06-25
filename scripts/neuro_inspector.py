import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def inspect_query(query: str, brain_path: str = "brain.uchi"):
    from uchi.cli import load_brain
    from uchi.vector_oracle import encode, similarity
    from uchi.omni_tokenizer import OmniTokenizer

    print(f"\\n==========================================")
    print(f"🧠 NEURO-SYMBOLIC INSPECTOR")
    print(f"==========================================\\n")

    print(f"[1] Loading Brain: {brain_path}...")
    router = load_brain(brain_path)
    if not router:
        print("❌ Failed to load brain. Run bootstrap first.")
        return

    print(f"\\n[2] Tokenizing Query...")
    tokenizer = OmniTokenizer()
    concepts = tokenizer.tokenize(query)
    print(f"  Tokens: {concepts}")

    print(f"\\n[3] Extracting Intent Vector (SSM)...")
    try:
        trie_dist = router.predictor.peek_distribution(concepts[:8])
        q_vec = encode(concepts, trie_dist)
        print(f"  Vector Shape: {len(q_vec)} dimensions")
    except Exception as e:
        print(f"❌ Failed to encode: {e}")
        return

    print(f"\\n[4] Tool Routing (Cosine Similarity)...")
    tool_vecs = router.skills.get_all_vectors()
    if not tool_vecs:
        print("  No tools registered in skill registry.")
    else:
        best_name, best_score = None, -1.0
        for name, v in tool_vecs.items():
            s = similarity(q_vec, v)
            print(f"  - Tool: {name:20} Cosine: {s:.4f}")
            if s > best_score:
                best_score, best_name = s, name
        print(f"  🏆 Best Tool: {best_name} ({best_score:.4f})")

    print(f"\\n[5] MCTS Blind Hallucination (Trie)...")
    seed = ["<|user|>"] + concepts + ["<|assistant|>"]
    print(f"  Running 5 rollouts to sample the Trie's probability space:")
    
    candidates = []
    _STOP = frozenset({"<|user|>", "<|assistant|>", "<|end|>"})
    for i in range(5):
        raw = router.predictor.generate(n_tokens=45, seed=seed, temperature=0.5, use_mcts=True)
        reply = [t for t in raw if t not in _STOP]
        if not reply:
            continue
        try:
            r_dist = router.predictor.peek_distribution(reply[:8])
            r_vec = encode(reply, r_dist)
            candidates.append((reply, r_vec))
            text = " ".join(reply).replace('\\n', '\\\\n')
            if len(text) > 80: text = text[:77] + "..."
            print(f"  [{i+1}] {text}")
        except Exception:
            pass

    print(f"\\n[6] Oracle Execution (Filtering)...")
    ssm_value = None
    try:
        from uchi.neuro_symbolic import get_ssm
        import torch
        ssm = get_ssm()
        with torch.no_grad():
            state = ssm.get_state(concepts)
            ssm_value = ssm.value(state).item()
        print(f"  SSM Value (Coherence Estimate): {ssm_value:.4f}")
    except Exception:
        pass

    valid_candidates = []
    coherence = router.convergent._coherence
    for i, (reply, vec) in enumerate(candidates):
        passes = coherence.passes(reply, concepts, ssm_value)
        status = "✅ PASS" if passes else "❌ FAIL"
        print(f"  [{i+1}] {status}")
        if passes:
            valid_candidates.append((reply, vec))

    print(f"\\n[7] Geometric Selection (Final Answer)...")
    if not valid_candidates:
        print("  ❌ All candidates murdered by Oracles. Falling back to tree search.")
    else:
        best_text, best_vec = None, None
        best_score = -2.0
        for reply, vec in valid_candidates:
            s = similarity(q_vec, vec)
            if s > best_score:
                best_score, best_text, best_vec = s, reply, vec
                
        print(f"  🏆 Winner Selected (Cosine: {best_score:.4f})")
        print(f"  Final Output: {' '.join(best_text)}")
        
    print(f"\\n==========================================\\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the Neuro-Symbolic Engine for a specific query or a default suite.")
    parser.add_argument("query", type=str, nargs="?", help="The query to inspect. If omitted, runs a default test suite.")
    parser.add_argument("--brain", default="brain.uchi", help="Path to the brain file.")
    args = parser.parse_args()
    
    if args.query:
        inspect_query(args.query, args.brain)
    else:
        print("\\n==========================================")
        print("🚀 RUNNING NEURO-SYMBOLIC TEST SUITE")
        print("==========================================\\n")
        
        test_suite = [
            "What is the capital of France?",
            "Explain how a quantum computer works.",
            "Who wrote Romeo and Juliet?",
            "Write a python function to calculate the fibonacci sequence."
        ]
        
        for q in test_suite:
            inspect_query(q, args.brain)
            print("\\n" + "="*50 + "\\n")
