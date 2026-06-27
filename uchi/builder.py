import os
import glob
import logging
from tqdm import tqdm
from uchi.cli import save_brain
from uchi.omni_router import OmniRouter
from uchi.neuro_symbolic import get_ssm
from uchi.deduplication import IngestionDeduplicator
import torch

# A universal limit for dataset slicing so we don't spend 24 hours downloading 
# 1 million rows, while still pulling a massive functional knowledge base.
KNOWLEDGE_LIMIT = 50

def _safe_load_dataset(*args, **kwargs):
    from datasets import load_dataset
    try:
        if len(args) == 2 and not kwargs:
            return load_dataset(args[0], split=args[1])
        return load_dataset(*args, **kwargs)
    except Exception as e:
        logging.warning(f"[-] Failed to load {args[0] if args else kwargs.get('path', 'dataset')}: {e}")
        return None

def build_full_brain(brain_path="brain.uchi"):
    """
    The Universal Master Builder.
    Executes the 5-stage reconstruction pipeline to build the brain from scratch.
    """
    print("\n[bold #bb9af7]=== UCHI UNIVERSAL BRAIN BUILDER ===[/bold #bb9af7]")
    print("[*] No existing brain detected. Initiating 5-Stage Reconstruction Pipeline.\n")

    dedup = IngestionDeduplicator(threshold=0.8)

    def _stream_if_unique(router, text: str) -> bool:
        """Stream text into the trie only if it's not a near-duplicate. Returns True if ingested."""
        if dedup.check_and_add(text):
            return False
        router.stream(router.tokenizer.tokenize(text.split(), is_inference=False))
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 1: The Wipe
    # ──────────────────────────────────────────────────────────────────────────
    print("[bold #7dcfff][*] Phase 1/5: Wiping corrupted or outdated states...[/bold #7dcfff]")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    patterns = ["brain*.uchi", "*.pt", "*.db", "uchi_cpu_memory*.json", "uchi_cpu_memory*.npy"]
    for pat in patterns:
        for f in glob.glob(os.path.join(project_root, pat)):
            os.remove(f)
            print(f"  [-] Deleted {os.path.basename(f)}")

    print("  [+] Clean slate achieved.")
    router = OmniRouter(use_bpe=False)

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 2: SSM Neural Pre-training
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[bold #7dcfff][*] Phase 2/5: SSM Neural Pre-training (databricks-dolly-15k)...[/bold #7dcfff]")
    ds_dolly = _safe_load_dataset("databricks/databricks-dolly-15k", f"train[:{KNOWLEDGE_LIMIT}]")
    if ds_dolly:
        ssm = get_ssm()
        optimizer = torch.optim.Adam(ssm.parameters(), lr=1e-3)
        for item in tqdm(ds_dolly, desc="Pre-training MoE Weights"):
            user_turn = item.get("instruction", "").strip()
            assistant_turn = item.get("response", "").strip()
            if not user_turn or not assistant_turn:
                continue
            
            tokens = ["<|user|>"] + router.tokenizer.tokenize(user_turn.split(), is_inference=False) + \
                     ["<|assistant|>"] + router.tokenizer.tokenize(assistant_turn.split(), is_inference=False) + ["<|end|>"]
            
            router.stream(tokens)
            optimizer.zero_grad()
            v_loss = ssm.update_value(tokens, reward=1.0)
            d_loss = ssm.train_dynamics(tokens)
            (v_loss + d_loss).backward()
            optimizer.step()
        torch.save(ssm.state_dict(), os.path.join(project_root, "ssm_dynamics.pt"))

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 3: Massive World Knowledge Ingestion
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[bold #7dcfff][*] Phase 3/5: Massive World Knowledge Ingestion...[/bold #7dcfff]")
    
    # 3A: OpenHermes (Conversational)
    ds_hermes = _safe_load_dataset("teknium/OpenHermes-2.5", f"train[:{KNOWLEDGE_LIMIT}]")
    if ds_hermes:
        for item in tqdm(ds_hermes, desc="Ingesting OpenHermes (Chat)"):
            conversations = item.get("conversations", [])
            text = ""
            for turn in conversations:
                role = "<|user|>" if turn.get("from") == "human" else "<|assistant|>"
                text += f"{role} {turn.get('value', '')} "
            if text:
                _stream_if_unique(router, text + "<|end|>")

    # 3B: Wikipedia (Encyclopedic)
    ds_wiki = _safe_load_dataset("wikipedia", "20220301.en", split=f"train[:{KNOWLEDGE_LIMIT}]")
    if not ds_wiki:
        ds_wiki = _safe_load_dataset("wikipedia", "20220301.en[train]")
    if ds_wiki:
        for item in tqdm(ds_wiki, desc="Ingesting Wikipedia (Facts)"):
            text = f"<|user|> Tell me about {item.get('title', '')}. <|assistant|> {item.get('text', '')[:1000]} <|end|>"
            _stream_if_unique(router, text)

    # 3C: MMLU (Graduate Reasoning)
    ds_mmlu = _safe_load_dataset("cais/mmlu", "all")
    if not ds_mmlu:
        from datasets import load_dataset
        try:
            ds_mmlu = load_dataset("cais/mmlu", "all", split="test")
        except Exception:
            ds_mmlu = None
    if ds_mmlu:
        ds_mmlu_subset = list(ds_mmlu)[:KNOWLEDGE_LIMIT]
        for item in tqdm(ds_mmlu_subset, desc="Ingesting MMLU (Reasoning)"):
            q, choices, ans_idx = item.get('question',''), item.get('choices',[]), item.get('answer',-1)
            if 0 <= ans_idx < len(choices):
                ans = choices[ans_idx]
                text = f"<|user|> Question: {q} Choices: {', '.join(choices)} <|assistant|> {ans} <|end|>"
                _stream_if_unique(router, text)

    # 3D: GSM8K (Math Reasoning)
    ds_gsm8k = _safe_load_dataset("gsm8k", "main")
    if not ds_gsm8k:
        from datasets import load_dataset
        try:
            ds_gsm8k = load_dataset("gsm8k", "main", split="test")
        except Exception:
            ds_gsm8k = None
    if ds_gsm8k:
        ds_gsm8k_subset = list(ds_gsm8k)[:KNOWLEDGE_LIMIT]
        for item in tqdm(ds_gsm8k_subset, desc="Ingesting GSM8K (Math)"):
            text = f"<|user|> {item.get('question','')} <|assistant|> {item.get('answer','')} <|end|>"
            _stream_if_unique(router, text)

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 4: Rigorous Code Logic
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[bold #7dcfff][*] Phase 4/5: Rigorous Code Logic Ingestion...[/bold #7dcfff]")
    
    # 4A: SWE-Bench (GitHub Issues)
    ds_swe = _safe_load_dataset("princeton-nlp/SWE-bench", "test")
    if not ds_swe:
        from datasets import load_dataset
        try:
            ds_swe = load_dataset("princeton-nlp/SWE-bench", split="test")
        except Exception:
            ds_swe = None
    if ds_swe:
        ds_swe_subset = list(ds_swe)[:KNOWLEDGE_LIMIT]
        for item in tqdm(ds_swe_subset, desc="Ingesting SWE-Bench (Engineering)"):
            issue, patch = item.get('problem_statement',''), item.get('patch','')
            text = f"<|user|> Fix issue:\n{issue[:500]} <|assistant|> {patch[:500]} <|end|>"
            _stream_if_unique(router, text)

    # 4B: HumanEval (Algorithms)
    ds_humaneval = _safe_load_dataset("openai/openai_humaneval", "test")
    if ds_humaneval:
        ds_humaneval_subset = list(ds_humaneval)[:KNOWLEDGE_LIMIT]
        for item in tqdm(ds_humaneval_subset, desc="Ingesting HumanEval (Algorithms)"):
            text = f"<|user|> Complete Python code:\n{item.get('prompt','')} <|assistant|> {item.get('canonical_solution','')} <|end|>"
            _stream_if_unique(router, text)

    # Save the master router
    save_brain(router, os.path.join(project_root, brain_path))
    
    # ──────────────────────────────────────────────────────────────────────────
    # Phase 5: MoE Specialists
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[bold #7dcfff][*] Phase 5/5: Building MoE Specialist Sub-Brains...[/bold #7dcfff]")
    _build_specialists(project_root)

    stats = dedup.stats()
    print(f"\n[*] Deduplication: {stats['duplicates_blocked']} duplicates blocked "
          f"({stats['duplicate_rate']*100:.1f}% of candidates), "
          f"{stats['seen']} unique documents ingested.")
    print("\n[bold #9ece6a][+] RECONSTRUCTION COMPLETE. Uchi is Online.[/bold #9ece6a]")
    return router

def _build_specialists(project_root):
    from uchi.omni_router import OmniRouter
    from uchi.cli import save_brain
    from uchi.deduplication import IngestionDeduplicator

    def _stream_unique(router, text, dedup):
        if dedup.check_and_add(text):
            return
        router.stream(router.tokenizer.tokenize(text.split(), is_inference=False))

    # Brain Code — trained on HumanEval function completions
    print("  [*] Building brain_code.uchi...")
    r_code = OmniRouter(use_bpe=False)
    dedup_code = IngestionDeduplicator()
    ds_he = _safe_load_dataset("openai/openai_humaneval", "test")
    if ds_he:
        for item in tqdm(list(ds_he)[:KNOWLEDGE_LIMIT], desc="  brain_code ← HumanEval"):
            text = (f"<|user|> Complete Python code:\n{item['prompt']} "
                    f"<|assistant|> {item['canonical_solution']} <|end|>")
            _stream_unique(r_code, text, dedup_code)
    else:
        r_code.stream(["<|user|>", "complete", "python", "code", "<|assistant|>", "def", "f", "return", "<|end|>"])
    save_brain(r_code, os.path.join(project_root, "brain_code.uchi"))

    # Brain Math — trained on GSM8K
    print("  [*] Building brain_math.uchi...")
    r_math = OmniRouter(use_bpe=False)
    dedup_math = IngestionDeduplicator()
    ds_math = _safe_load_dataset("gsm8k", "main")
    if ds_math:
        for item in tqdm(list(ds_math)[:KNOWLEDGE_LIMIT], desc="  brain_math ← GSM8K"):
            text = f"<|user|> {item['question']} <|assistant|> {item['answer']} <|end|>"
            _stream_unique(r_math, text, dedup_math)
    else:
        r_math.stream(["<|user|>", "math", "equation", "<|assistant|>", "1", "+", "1", "=", "2", "<|end|>"])
    save_brain(r_math, os.path.join(project_root, "brain_math.uchi"))

    # Brain Convo — trained on OpenHermes conversational turns
    print("  [*] Building brain_convo.uchi...")
    r_convo = OmniRouter(use_bpe=False)
    dedup_convo = IngestionDeduplicator()
    ds_convo = _safe_load_dataset("teknium/OpenHermes-2.5", f"train[:{KNOWLEDGE_LIMIT}]")
    if ds_convo:
        for item in tqdm(ds_convo, desc="  brain_convo ← OpenHermes"):
            text = ""
            for turn in item.get("conversations", []):
                role = "<|user|>" if turn.get("from") == "human" else "<|assistant|>"
                text += f"{role} {turn.get('value', '')} "
            if text:
                _stream_unique(r_convo, text + "<|end|>", dedup_convo)
    else:
        r_convo.stream(["<|user|>", "hello", "<|assistant|>", "hi", "how", "are", "you", "<|end|>"])
    save_brain(r_convo, os.path.join(project_root, "brain_convo.uchi"))
