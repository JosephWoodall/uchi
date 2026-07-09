"""
qat_train.py — Phase 4: Quantization-Aware Training (QAT) for Ternary Weights

Takes the best distilled model and activates Ternary (1.58-bit) quantization.
The model weights are quantized to {-1, 0, 1} during the forward pass, but
we keep high-precision gradients and optimizer states to slowly adapt the
model to the quantization noise.

Data source (fixed): a MIX of general text (FineWeb-Edu .bin, plain LM loss)
and CoT-formatted reasoning traces (GSM8K, masked loss — same format as
cot_distill.py). Recovering on general text ALONE was the original approach,
and it silently erased the reasoning behaviour CoT had just installed: QAT
would continue training on plain prose under quantization noise with nothing
in the data to remind the model of the <|think|>...<|assistant|> structure,
so 1500 steps of that pulled the weights back toward generic continuation.
Mixing macro-steps between both distributions lets quantization noise be
absorbed everywhere while the reasoning format keeps getting reinforced.

Usage:
    .venv/bin/python -m uchi.flux.qat_train --base uchi/flux/checkpoints/cot_best.pt
"""

import os
import time
import math
import random
import argparse
import re
import torch
import torch.nn as nn
from contextlib import nullcontext
from .train_v2 import make_bin_iter, chunked_cross_entropy, save_checkpoint
from .cot_distill import load_cot_examples, make_batches as make_cot_batches

# ==============================================================================
# Defaults
# ==============================================================================
DEFAULTS = dict(
    micro_batch_size=2,
    grad_accum_steps=32,       # effective batch = 64
    max_seq_len=384,           # matches CoT's format (unifies text + CoT batches)
    max_steps=600,             # macro-steps (each = grad_accum micro-batches)
    warmup_steps=60,
    learning_rate=1e-5,        # Very low LR for QAT
    min_lr_frac=0.1,
    weight_decay=0.01,
    grad_clip=1.0,
    checkpoint_interval=200,
    eval_interval=100,
    eval_steps=20,
    log_interval=10,
    checkpoint_dir="uchi/flux/checkpoints",
    cot_frac=0.5,              # probability a given macro-step trains on CoT data
    cot_examples=2000,
)


def masked_chunked_ce_loss(logits, targets, mask, chunk_size=64):
    """Cross-entropy over masked (think+answer) positions only, chunked for memory."""
    B, T, V = logits.shape
    logits_flat = logits.reshape(B * T, V)
    targets_flat = targets.reshape(B * T)
    mask_flat = mask.reshape(B * T)
    total_loss, n_chunks = 0.0, 0
    for start in range(0, B * T, chunk_size):
        end = min(start + chunk_size, B * T)
        c_logits, c_targets, c_mask = logits_flat[start:end], targets_flat[start:end], mask_flat[start:end]
        if c_mask.sum() > 0:
            loss_per_token = nn.functional.cross_entropy(c_logits, c_targets, reduction="none")
            total_loss = total_loss + (loss_per_token * c_mask).sum() / c_mask.sum()
            n_chunks += 1
    return total_loss / max(n_chunks, 1)


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="FLUX Phase 4 — Ternary QAT (mixed CoT + general text recovery)")
    parser.add_argument("--base", type=str, required=True, help="Path to Phase 3 CoT checkpoint")
    parser.add_argument("--steps", type=int, default=DEFAULTS["max_steps"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["learning_rate"])
    parser.add_argument("--micro-batch", type=int, default=DEFAULTS["micro_batch_size"])
    parser.add_argument("--grad-accum", type=int, default=DEFAULTS["grad_accum_steps"])
    parser.add_argument("--seq-len", type=int, default=DEFAULTS["max_seq_len"])
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--data-bin", type=str, required=True,
                        help="Pre-tokenized general-text train .bin (GPU-bound fast path)")
    parser.add_argument("--val-bin", type=str, default=None)
    parser.add_argument("--cot-frac", type=float, default=DEFAULTS["cot_frac"],
                        help="Probability a macro-step trains on CoT data vs general text")
    parser.add_argument("--cot-examples", type=int, default=DEFAULTS["cot_examples"])
    parser.add_argument("--pruned-vocab", type=str, default=None,
                         help="Path to the same PrunedVocab JSON the Phase 3 --base checkpoint "
                              "was trained with. REQUIRED if --base was trained with a pruned "
                              "vocab -- encoding with the full ~100K cl100k_base tokenizer "
                              "against a smaller embedding table produces out-of-range token IDs. "
                              "NOTE: --data-bin must ALSO have been pre-tokenized with this same "
                              "pruned vocab -- a .bin file tokenized with the full vocab will "
                              "produce the same out-of-range IDs for the general-text-recovery "
                              "portion of this phase even if this flag is set correctly.")
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULTS["checkpoint_dir"],
                         help="Where to write qat_*.pt. Defaults to the shared checkpoints dir -- "
                              "override for proof/experimental runs so they don't overwrite "
                              "production checkpoints.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    from uchi.flux.model import HybridTSSM

    if args.pruned_vocab:
        from uchi.flux.vocab_prune import load_pruned_tokenizer
        tokenizer = load_pruned_tokenizer(args.pruned_vocab)
        print(f"  Tokenizer:    pruned, vocab_size={tokenizer.vocab_size:,} (from {args.pruned_vocab})")
    else:
        from uchi.flux.tokenizer_v2 import TikTokenHybridTokenizer
        tokenizer = TikTokenHybridTokenizer()
        print(f"  Tokenizer:    full cl100k_base, vocab_size={tokenizer.vocab_size:,}")

    print("=" * 72)
    print("FLUX Phase 4 — Ternary QAT (mixed CoT + general-text recovery)")
    print("=" * 72)

    ckpt = torch.load(args.base, map_location=device, weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    emb_shape = sd["embedding.weight"].shape
    vocab_size, d_model = int(emb_shape[0]), int(emb_shape[1])
    layer_ids = {int(m.group(1)) for k in sd if (m := re.match(r"layers\.(\d+)\.", k))}
    n_layers = (max(layer_ids) + 1) if layer_ids else 12

    model = None
    for d_state in (32, 16, 64, 128):
        try:
            m = HybridTSSM(
                vocab_size=vocab_size,
                syntax_vocab_size=tokenizer.syntax_vocab_size,
                d_model=d_model, n_layers=n_layers, d_state=d_state,
            ).to(device)
            m.load_state_dict(sd, strict=False)
            model = m
            break
        except Exception:
            continue

    if model is None:
        raise RuntimeError("Could not match architecture to checkpoint")

    # Turn on Ternary Quantization — stays on for the whole run, including the
    # CoT-format steps, so the reasoning behaviour is reinforced UNDER the same
    # quantization noise it needs to survive at inference.
    print("  [!] Activating 1.58-bit Ternary Quantization (BitNet)")
    model.set_quantization(True)
    model._gradient_checkpointing = True
    model.to(device)

    # ── Data: general text (bin) + CoT-formatted reasoning traces ──
    print(f"  General-text bin: {args.data_bin}")
    train_text_iter = make_bin_iter(args.data_bin, args.micro_batch, args.seq_len, seed=42)
    val_text_iter = make_bin_iter(args.val_bin or args.data_bin, args.micro_batch, args.seq_len, seed=0)

    print("  Loading CoT reasoning traces (real GSM8K teacher traces)...")
    # load_cot_examples internally calls generate_synthetic_cot and tokenizes/
    # formats to the exact <|user|>/<|think|>/<|assistant|> layout CoT trained on.
    cot_data = load_cot_examples(tokenizer, args.seq_len, args.cot_examples)
    n_val = max(10, int(len(cot_data) * 0.1))
    cot_val, cot_train = cot_data[:n_val], cot_data[n_val:]
    print(f"  CoT train: {len(cot_train):,}  CoT val: {len(cot_val):,}")

    def cot_train_gen():
        while True:
            for batch in make_cot_batches(cot_train, args.micro_batch, shuffle=True):
                yield batch

    cot_train_iter = cot_train_gen()

    # ── Training setup ──
    eff_batch = args.micro_batch * args.grad_accum
    max_steps = args.steps
    warmup_steps = min(DEFAULTS["warmup_steps"], max(1, max_steps // 10))
    lr_min = args.lr * DEFAULTS["min_lr_frac"]

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=DEFAULTS["weight_decay"],
        betas=(0.9, 0.95),
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
        if step >= max_steps:
            return lr_min
        progress = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min + coeff * (args.lr - lr_min)

    @torch.no_grad()
    def evaluate_text():
        model.eval()
        total_loss, n = 0.0, 0
        for _ in range(DEFAULTS["eval_steps"]):
            X, Y = next(val_text_iter)
            X, Y = X.to(device), Y.to(device)
            with amp_ctx():
                logits, _ = model(X)
                loss = chunked_cross_entropy(logits, Y, ignore_index=tokenizer.pad_token_id)
            total_loss += loss.item(); n += 1
        model.train()
        avg = total_loss / max(n, 1)
        return avg, math.exp(min(avg, 20.0))

    @torch.no_grad()
    def evaluate_cot():
        model.eval()
        total_loss, n = 0.0, 0
        for X, Y, mask in make_cot_batches(cot_val, args.micro_batch, shuffle=False):
            X, Y, mask = X.to(device), Y.to(device), mask.to(device)
            with amp_ctx():
                logits, _ = model(X)
                loss = masked_chunked_ce_loss(logits, Y, mask)
            total_loss += loss.item(); n += 1
        model.train()
        avg = total_loss / max(n, 1)
        return avg, math.exp(min(avg, 20.0))

    ckpt_dir = args.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    model.train()
    best_cot_val = float("inf")
    rng = random.Random(123)
    t0 = time.time()

    print(f"\n  Starting mixed QAT recovery: {max_steps} macro-steps "
          f"(cot_frac={args.cot_frac}, eff_batch={eff_batch}, seq_len={args.seq_len})\n")

    for step in range(max_steps):
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        is_cot_step = rng.random() < args.cot_frac
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for _ in range(args.grad_accum):
            if is_cot_step:
                X, Y, mask = next(cot_train_iter)
                X, Y, mask = X.to(device), Y.to(device), mask.to(device)
                with amp_ctx():
                    logits, _ = model(X)
                    loss = masked_chunked_ce_loss(logits, Y, mask)
                    scaled_loss = loss / args.grad_accum
            else:
                X, Y = next(train_text_iter)
                X, Y = X.to(device), Y.to(device)
                with amp_ctx():
                    logits, _ = model(X)
                    loss = chunked_cross_entropy(logits, Y, ignore_index=tokenizer.pad_token_id)
                    scaled_loss = loss / args.grad_accum

            scaler.scale(scaled_loss).backward()
            loss_accum += loss.item() / args.grad_accum

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULTS["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        if (step + 1) % DEFAULTS["log_interval"] == 0:
            elapsed = time.time() - t0
            ms_per_step = (elapsed * 1000) / DEFAULTS["log_interval"]
            t0 = time.time()
            kind = "cot " if is_cot_step else "text"
            print(f"  Step {step + 1:05d}/{max_steps} │ [{kind}] Loss: {loss_accum:.4f} │ LR: {lr:.2e} │ {ms_per_step:.0f}ms/step")

        if (step + 1) % DEFAULTS["eval_interval"] == 0:
            text_loss, text_ppl = evaluate_text()
            cot_loss, cot_ppl = evaluate_cot()
            is_best = cot_loss < best_cot_val   # CoT val loss is the "best" criterion —
            if is_best:                         # preserving reasoning is the point of this fix.
                best_cot_val = cot_loss
            print(f"\n  ──── VAL step {step + 1:05d} │ "
                  f"text loss {text_loss:.4f} (PPL {text_ppl:.1f}) │ "
                  f"CoT loss {cot_loss:.4f} (PPL {cot_ppl:.1f}) "
                  f"{'★ best (CoT)' if is_best else ''}\n")
            if is_best:
                save_checkpoint(raw_model, optimizer, step + 1, best_cot_val, os.path.join(ckpt_dir, "qat_best.pt"), quantized=True)

        if (step + 1) % DEFAULTS["checkpoint_interval"] == 0:
            save_checkpoint(raw_model, optimizer, step + 1, best_cot_val, os.path.join(ckpt_dir, f"qat_{step + 1:05d}.pt"), quantized=True)

    print(f"\n  QAT complete. Best CoT val loss: {best_cot_val:.4f}")

if __name__ == "__main__":
    main()
