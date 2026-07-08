"""
verifier_train.py — Verifier training pipeline (0.4.0 Item 17)

Trains EntailmentClassifier (verifier_model.py) from scratch on real
MNLI + SNLI data. Same fair-budgeting discipline established this session
for sft_train.py/cot_distill.py: each source gets an independent share of
max_examples, never a shared cumulative counter (a shared counter let one
source silently starve every source loaded after it -- found and fixed
twice already this release).

This is a SEPARATE model from the main FLUX proposer -- its own embedding
table, its own checkpoint, its own training run. It does not touch or
depend on uchi/flux/checkpoints/*.pt (the proposer's checkpoints).

Usage:
    .venv/bin/python -m uchi.flux.verifier_train --max-examples 50000
"""

import os
import time
import math
import random
import argparse
import torch
import torch.nn as nn
from contextlib import nullcontext

from .verifier_model import EntailmentClassifier
from .train_v2 import save_checkpoint


DEFAULTS = dict(
    micro_batch_size=8,
    grad_accum_steps=8,        # effective batch = 64
    max_seq_len=192,           # premise+hypothesis pairs are short
    epochs=2,
    learning_rate=3e-4,
    min_lr_frac=0.1,
    warmup_frac=0.05,
    weight_decay=0.01,
    grad_clip=1.0,
    eval_frac=0.05,
    checkpoint_dir="uchi/flux/checkpoints/verifier",
    max_examples=100_000,
    d_model=256,
    n_layers=8,
    d_state=32,
)


# ==============================================================================
# Real entailment data sources (verified against the live HF Hub before use)
# ==============================================================================
_VALID_LABELS = (0, 1, 2)  # 0=entailment, 1=neutral, 2=contradiction


def generate_mnli_examples(num_examples):
    """Real MNLI (nyu-mll/glue, mnli config): premise/hypothesis/label,
    label 0/1/2 = entailment/neutral/contradiction. Verified live before
    being used here -- see the 0.4.0 CHANGELOG entry for the confirmed
    field names and a sample row.
    """
    from datasets import load_dataset

    examples = []
    try:
        ds = load_dataset("nyu-mll/glue", "mnli", split="train", streaming=True)
    except Exception as e:
        print(f"  [!] Failed to load MNLI: {e}")
        return []

    for row in ds:
        premise = (row.get("premise") or "").strip()
        hypothesis = (row.get("hypothesis") or "").strip()
        label = row.get("label")
        if not premise or not hypothesis or label not in _VALID_LABELS:
            continue
        examples.append({"premise": premise, "hypothesis": hypothesis, "label": label})
        if len(examples) >= num_examples:
            break

    print(f"  [+] Loaded {len(examples)} real MNLI examples.")
    return examples


def generate_snli_examples(num_examples):
    """Real SNLI (stanfordnlp/snli): same premise/hypothesis/label shape.
    SNLI uses label -1 for "no annotator consensus" -- explicitly excluded
    by _VALID_LABELS, not silently included as a bogus fourth class.
    """
    from datasets import load_dataset

    examples = []
    try:
        ds = load_dataset("stanfordnlp/snli", split="train", streaming=True)
    except Exception as e:
        print(f"  [!] Failed to load SNLI: {e}")
        return []

    for row in ds:
        premise = (row.get("premise") or "").strip()
        hypothesis = (row.get("hypothesis") or "").strip()
        label = row.get("label")
        if not premise or not hypothesis or label not in _VALID_LABELS:
            continue
        examples.append({"premise": premise, "hypothesis": hypothesis, "label": label})
        if len(examples) >= num_examples:
            break

    print(f"  [+] Loaded {len(examples)} real SNLI examples.")
    return examples


# ==============================================================================
# Synthetic multi-hop transitive examples (not from any external dataset --
# generated here because MNLI/SNLI are both single-premise, and neither
# teaches "combine two stated facts into a transitive conclusion" as a
# skill at all. Demonstrated gap: an entailment classifier trained only on
# single-premise NLI has no reason to learn this, and the deterministic
# RelationalTransitivityChecker (uchi/relational_reasoning.py) only covers
# phrasings its regexes recognize -- this is the broader-but-less-certain
# neural fallback layer for the cases that checker can't parse.
# ==============================================================================
_MULTIHOP_ENTITIES = [
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
    "Ivy", "Jack", "Karen", "Liam", "Mia", "Noah", "Olivia", "Peter",
    "Building A", "Building B", "Building C", "Building D",
    "the red car", "the blue car", "the green car", "the black car",
    "Mount Everest", "K2", "Denali", "Kilimanjaro",
    "Project Alpha", "Project Beta", "Project Gamma", "Project Delta",
]
# (positive_word, negative_word) -- same relation, opposite direction.
# Deliberately more entries than uchi/relational_reasoning.py's hardcoded
# _KNOWN_COMPARATIVES: the model is meant to LEARN the antonym-equivalence
# (e.g. "shorter" and "taller" describe the same relation) from varied
# examples, not need it hand-mapped the way the deterministic checker does.
_MULTIHOP_RELATIONS = [
    ("taller", "shorter"), ("older", "younger"), ("faster", "slower"),
    ("heavier", "lighter"), ("bigger", "smaller"), ("richer", "poorer"),
    ("stronger", "weaker"), ("higher", "lower"), ("more expensive", "cheaper"),
    ("earlier", "later"), ("wider", "narrower"), ("hotter", "colder"),
]
_MULTIHOP_TEMPLATES = [
    "{subj} is {comp} than {obj}.",
    "{subj} is {comp} than {obj}, based on the available data.",
    "Compared to {obj}, {subj} is {comp}.",
]


def generate_multihop_examples(num_examples, seed=1337):
    """Synthetic multi-premise transitive-chain examples (RuleTaker/
    ProofWriter-style), balanced three ways: a valid transitive conclusion
    (entailment), the reversed/contradictory conclusion (contradiction),
    and a question about an entity outside the stated chain (neutral --
    insufficient information, not a guess). Each fact is phrased with
    either its positive or negative comparative word at random (e.g. "B is
    shorter than A" as well as "A is taller than B" for the same
    underlying fact) specifically so the model has to learn these describe
    the same relation, rather than only ever seeing one direction.
    """
    rng = random.Random(seed)  # fixed, independent of the caller's global seed
    examples = []

    for i in range(num_examples):
        a, b, c, d = rng.sample(_MULTIHOP_ENTITIES, 4)
        # One relation per EXAMPLE, not per fact() call -- both premises and
        # the hypothesis must be about the same attribute for transitivity
        # to mean anything at all. Picking a fresh relation per call (the
        # original bug here, caught by cross-checking against
        # RelationalTransitivityChecker before trusting this) silently
        # produced premises about unrelated attributes -- e.g. "richer" for
        # a-vs-b and "higher" for b-vs-c -- making the derived "contradiction"
        # label simply wrong, not just unverifiable.
        comp_pos, comp_neg = rng.choice(_MULTIHOP_RELATIONS)

        def fact(higher, lower, use_positive_word):
            if use_positive_word:
                comp, s, o = comp_pos, higher, lower
            else:
                comp, s, o = comp_neg, lower, higher
            return rng.choice(_MULTIHOP_TEMPLATES).format(subj=s, comp=comp, obj=o)

        # Ground truth by construction: a > b > c on this example's relation.
        # d never appears in either premise.
        premise = fact(a, b, rng.choice([True, False])) + " " + fact(b, c, rng.choice([True, False]))

        case = i % 3
        if case == 0:
            hypothesis, label = fact(a, c, rng.choice([True, False])), 0   # valid -> entailment
        elif case == 1:
            hypothesis, label = fact(c, a, rng.choice([True, False])), 2   # reversed -> contradiction
        else:
            hypothesis, label = fact(a, d, rng.choice([True, False])), 1   # unconnected -> neutral

        examples.append({"premise": premise, "hypothesis": hypothesis, "label": label})

    print(f"  [+] Generated {len(examples)} synthetic multi-hop transitive examples.")
    return examples


def load_verifier_examples(tokenizer, max_seq_len, max_examples, seed=42, flywheel_path=None):
    """Load + tokenize MNLI+SNLI+synthetic-multihop (+ flywheel corrections,
    if any exist), fair-budgeted across all sources present. flywheel_path
    is optional real, verified data exported by uchi/verifier_flywheel.py
    from actual observed user corrections -- empty/nonexistent on a first
    run (no conversations have happened yet), naturally growing over time.
    Same independent-per-source-share discipline as every other data source
    this session -- a real source with zero rows just contributes zero,
    it never silently starves the others.
    """
    random.seed(seed)
    n_sources = 3  # MNLI, SNLI, synthetic multi-hop
    flywheel_examples = []
    if flywheel_path:
        from uchi.verifier_flywheel import load_flywheel_examples
        n_sources = 4
        per_source_estimate = max(1, max_examples // n_sources)
        flywheel_examples = load_flywheel_examples(flywheel_path, num_examples=per_source_estimate)
        print(f"  [+] Loaded {len(flywheel_examples)} real flywheel-corrected examples.")

    print(f"  Generating entailment examples from {n_sources} source(s) ...")
    per_source = max(1, max_examples // n_sources)
    examples = (
        generate_mnli_examples(per_source)
        + generate_snli_examples(per_source)
        + generate_multihop_examples(per_source)
        + flywheel_examples
    )
    random.shuffle(examples)
    examples = examples[:max_examples]
    print(f"  Total entailment examples: {len(examples):,}")

    formatted = []
    context_open = tokenizer.encode_special("<|context|>")
    context_close = tokenizer.encode_special("<|/context|>")
    user_id = tokenizer.encode_special("<|user|>")
    pad_id = tokenizer.pad_token_id

    # Batch-tokenize all premises and all hypotheses in two calls total,
    # not 2*len(examples) individual tokenizer.encode_text() calls -- the
    # per-example loop was the actual bottleneck behind the ~3-hour first
    # training run (confirmed: encode_text() calls tiktoken's encoder one
    # example at a time). encode_ordinary_batch releases the GIL and uses
    # all cores in Rust, same fast path pretokenize.py already relies on.
    # Safe here specifically because MNLI/SNLI/synthetic premise-hypothesis
    # text is ordinary sentences -- encode_text()'s allowed_special="all"
    # only matters if the text contains literal special-token substrings,
    # which real NLI sentences don't.
    raw_enc = tokenizer._enc if hasattr(tokenizer, "_enc") else tokenizer._base._enc
    shift = tokenizer.n_special
    premise_texts = [ex["premise"] for ex in examples]
    hyp_texts = [ex["hypothesis"] for ex in examples]
    premise_batches = raw_enc.encode_ordinary_batch(premise_texts)
    hyp_batches = raw_enc.encode_ordinary_batch(hyp_texts)

    def _remap(ids):
        if hasattr(tokenizer, "_pruned"):
            return tokenizer._pruned.remap(ids)
        return ids

    dropped_too_long = 0
    for ex, premise_raw, hyp_raw in zip(examples, premise_batches, hyp_batches):
        premise_ids = [context_open] + [t + shift for t in _remap(premise_raw)] + [context_close]
        hyp_ids = [user_id] + [t + shift for t in _remap(hyp_raw)]
        full_ids = premise_ids + hyp_ids

        if len(full_ids) > max_seq_len:
            dropped_too_long += 1
            continue

        n_pad = max_seq_len - len(full_ids)
        attention_mask = [1] * len(full_ids) + [0] * n_pad
        full_ids = full_ids + [pad_id] * n_pad

        formatted.append({
            "input_ids": full_ids[:max_seq_len],
            "attention_mask": attention_mask[:max_seq_len],
            "label": ex["label"],
        })

    if dropped_too_long:
        print(f"  ({dropped_too_long} example(s) dropped for exceeding max_seq_len={max_seq_len})")

    return formatted


def make_batches(data, micro_batch_size, shuffle=True):
    if shuffle:
        random.shuffle(data)
    for i in range(0, len(data) - micro_batch_size + 1, micro_batch_size):
        batch = data[i : i + micro_batch_size]
        input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
        attention_mask = torch.tensor([b["attention_mask"] for b in batch], dtype=torch.float32)
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        yield input_ids, attention_mask, labels


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Verifier — Entailment Classifier Training (0.4.0 Item 17)")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["learning_rate"])
    parser.add_argument("--micro-batch", type=int, default=DEFAULTS["micro_batch_size"])
    parser.add_argument("--grad-accum", type=int, default=DEFAULTS["grad_accum_steps"])
    parser.add_argument("--seq-len", type=int, default=DEFAULTS["max_seq_len"])
    parser.add_argument("--max-examples", type=int, default=DEFAULTS["max_examples"])
    parser.add_argument("--d-model", type=int, default=DEFAULTS["d_model"])
    parser.add_argument("--n-layers", type=int, default=DEFAULTS["n_layers"])
    parser.add_argument("--d-state", type=int, default=DEFAULTS["d_state"])
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULTS["checkpoint_dir"])
    parser.add_argument("--flywheel-path", type=str, default=None,
                         help="Path to real corrections exported by VerifierFlywheel.export_"
                              "training_examples() -- folded in as a fourth fair-budgeted "
                              "source alongside MNLI/SNLI if given and non-empty.")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    from uchi.flux.tokenizer_v2 import TikTokenHybridTokenizer
    tokenizer = TikTokenHybridTokenizer()

    print("=" * 72)
    print("Verifier — Entailment Classifier Training (0.4.0 Item 17)")
    print("Separate model from the FLUX proposer -- own embedding table,")
    print("own checkpoint, does not touch uchi/flux/checkpoints/*.pt")
    print("=" * 72)

    model = EntailmentClassifier(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model, n_layers=args.n_layers, d_state=args.d_state,
    ).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    data = load_verifier_examples(tokenizer, args.seq_len, args.max_examples, flywheel_path=args.flywheel_path)
    n_val = max(10, int(len(data) * DEFAULTS["eval_frac"]))
    val_data, train_data = data[:n_val], data[n_val:]
    print(f"  Train: {len(train_data):,}  Val: {len(val_data):,}")

    eff_batch = args.micro_batch * args.grad_accum
    steps_per_epoch = len(train_data) // eff_batch
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * DEFAULTS["warmup_frac"])
    lr_min = args.lr * DEFAULTS["min_lr_frac"]

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=DEFAULTS["weight_decay"], betas=(0.9, 0.95),
    )
    use_scaler = device == "cuda" and not use_bf16
    scaler = torch.amp.GradScaler(device, enabled=use_scaler)

    def amp_ctx():
        if device == "cuda":
            return torch.amp.autocast(device_type="cuda", dtype=dtype)
        return nullcontext()

    raw_model = model
    if device == "cuda" and not args.no_compile:
        model = torch.compile(model)

    def get_lr(step):
        if step < warmup_steps:
            return args.lr * (step + 1) / max(1, warmup_steps)
        if step >= total_steps:
            return lr_min
        progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min + coeff * (args.lr - lr_min)

    ce_loss = nn.CrossEntropyLoss()

    @torch.no_grad()
    def validate():
        model.eval()
        total_loss, correct, n = 0.0, 0, 0
        for X, mask, y in make_batches(val_data, args.micro_batch, shuffle=False):
            X, mask, y = X.to(device), mask.to(device), y.to(device)
            with amp_ctx():
                logits = model(X, mask)
                loss = ce_loss(logits, y)
            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == y).sum().item()
            n += y.size(0)
        model.train()
        n_batches = max(1, len(val_data) // args.micro_batch)
        return total_loss / n_batches, correct / max(1, n)

    ckpt_dir = args.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    model.train()
    best_val = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        batch_iter = make_batches(train_data, args.micro_batch, shuffle=True)
        epoch_loss, n_micro, accum = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)

        for X, mask, y in batch_iter:
            X, mask, y = X.to(device), mask.to(device), y.to(device)
            with amp_ctx():
                logits = model(X, mask)
                loss = ce_loss(logits, y) / args.grad_accum

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * args.grad_accum
            n_micro += 1
            accum += 1

            if accum == args.grad_accum:
                lr = get_lr(global_step)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULTS["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                global_step += 1
                if global_step % 50 == 0:
                    print(f"  Step {global_step:05d} │ Loss: {epoch_loss / n_micro:.4f} │ LR: {lr:.2e}")

        if accum > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULTS["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        val_loss, val_acc = validate()
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss

        print(f"\n  ── Epoch {epoch + 1}/{args.epochs} │ Val Loss: {val_loss:.4f} │ Val Acc: {val_acc:.1%} {'★ best' if is_best else ''}")

        save_checkpoint(raw_model, optimizer, global_step, best_val, os.path.join(ckpt_dir, f"verifier_epoch{epoch + 1}.pt"))
        if is_best:
            save_checkpoint(raw_model, optimizer, global_step, best_val, os.path.join(ckpt_dir, "verifier_best.pt"))

    print(f"\n  Verifier training complete. Best val loss: {best_val:.4f}")

    # ── Fit the OOD detector (Item 17 refinement) on the trained model's
    # own latent space, over a real sample of the training set -- nearly
    # free, since the pooled representations are already being computed. ──
    print("  Fitting OOD detector on training-set latent representations ...")
    from .verifier_model import OODDetector
    raw_model.eval()
    sample_size = min(len(train_data), 5000)
    sample = train_data[:sample_size]
    reps = []
    with torch.no_grad():
        for X, mask, _ in make_batches(sample, args.micro_batch, shuffle=False):
            X, mask = X.to(device), mask.to(device)
            reps.append(raw_model.encode(X, mask).cpu())
    if reps:
        all_reps = torch.cat(reps, dim=0)
        ood = OODDetector()
        ood.fit(all_reps)
        torch.save(ood.state_dict(), os.path.join(ckpt_dir, "verifier_best.ood.pt"))
        print(f"  OOD detector fit on {all_reps.shape[0]:,} representations -> "
              f"{os.path.join(ckpt_dir, 'verifier_best.ood.pt')}")
    else:
        print("  [!] No training data available to fit the OOD detector -- skipped.")


if __name__ == "__main__":
    main()
