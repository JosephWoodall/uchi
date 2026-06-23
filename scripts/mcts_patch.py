def _mcts_generate_from_predictor(
    p,
    n_tokens: int,
    seed: list | None,
    n_sims: int = 3,
    top_k_candidates: int = 3,
    lookahead_depth: int = 3,
    stop_tokens: set | None = None,
    tokenizer=None,
    long_term_store=None,
) -> list:
    """MCTS-guided token selection for Uchi."""
    import math
    import random
    
    saved = p.history[:]
    p.history.clear()
    
    if seed:
        seed_tokens = tokenizer.tokenize(seed) if tokenizer else seed
        for tok in seed_tokens:
            p.observe(tok)
            
    generated = []
    
    for step in range(n_tokens):
        p.predict()
        
        if p._last_prediction_depth <= 1:
            break
            
        dist = dict(p._last_distribution)
        
        # Apply simple repetition penalty to dist
        for tok in set(generated[-20:]):
            if tok in dist:
                dist[tok] /= 1.5
                
        # Get top-K candidates from dist
        if not dist:
            break
            
        sorted_candidates = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:top_k_candidates]
        
        best_tok = sorted_candidates[0][0]
        best_score = float('-inf')
        
        # Base state
        base_history = p.history[:]
        
        for cand_tok, prior in sorted_candidates:
            val_sum = 0.0
            
            for sim in range(n_sims):
                p.history = base_history[:]
                p.observe(cand_tok)
                
                sim_val = 0.0
                
                # Rollout
                for d in range(lookahead_depth):
                    p.predict()
                    
                    # If we hit a dead end or hallucination, penalize heavily
                    if p._last_prediction_depth <= 1:
                        sim_val -= 5.0
                        break
                        
                    # Value is the credibility of the node we landed on
                    node_cred = p._last_max_sim
                    sim_val += node_cred
                    
                    sim_dist = dict(p._last_distribution)
                    if not sim_dist:
                        break
                        
                    # Greedy choice for simulation
                    next_tok = max(sim_dist.items(), key=lambda x: x[1])[0]
                    if stop_tokens and next_tok in stop_tokens:
                        sim_val += 5.0 # Reward reaching a natural stop token
                        break
                        
                    p.observe(next_tok)
                    
                val_sum += sim_val
                
            avg_val = val_sum / max(1, n_sims)
            
            # Combine prior log-prob with value
            lp = math.log(max(1e-9, prior))
            score = 0.4 * lp + 0.6 * avg_val
            
            if score > best_score:
                best_score = score
                best_tok = cand_tok
                
        # Restore base history and observe the chosen token
        p.history = base_history[:]
        generated.append(best_tok)
        if stop_tokens and best_tok in stop_tokens:
            break
        p.observe(best_tok)
        
    p.history = saved
    
    if tokenizer is not None:
        generated = tokenizer.detokenize(generated)
        
    return generated
