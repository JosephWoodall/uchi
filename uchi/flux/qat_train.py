"""
qat_train.py — Phase 4: Quantization-Aware Training (QAT) for Ternary Weights

Takes the best distilled model and activates Ternary (1.58-bit) quantization.
The model weights are quantized to {-1, 0, 1} during the forward pass, but
we keep high-precision gradients and optimizer states to slowly adapt the
model to the quantization noise.

This ensures the 30M model is extremely fast and efficient for the Proposer.

Data source:
    - We use a mix of TinyStories (for language modeling) and SFT/CoT data.
      For simplicity in this script, we'll just run on the TinyStories
      data to recover any lost perplexity from the harsh quantization.

Usage:
    .venv/bin/python -m uchi.flux.qat_train --base uchi/flux/checkpoints/cot_best.pt
"""

import os
import time
import math
import argparse
import re
import torch
import torch.nn as nn
from contextlib import nullcontext
from .train_v2 import make_data_iter, make_bin_iter, chunked_cross_entropy, save_checkpoint, evaluate

# ==============================================================================
# Defaults
# ==============================================================================
DEFAULTS = dict(
    micro_batch_size=2,
    grad_accum_steps=32,       # effective batch = 64
    max_seq_len=256,
    max_steps=5000,            # QAT needs far fewer steps than pre-training
    warmup_steps=500,
    learning_rate=1e-5,        # Very low LR for QAT
    min_lr_frac=0.1,
    weight_decay=0.01,
    grad_clip=1.0,
    checkpoint_interval=1000,
    eval_interval=200,
    eval_steps=20,
    log_interval=10,
    checkpoint_dir="uchi/flux/checkpoints",
)

# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="FLUX Phase 4 — Ternary QAT")
    parser.add_argument("--base", type=str, required=True, help="Path to Phase 3 CoT checkpoint")
    parser.add_argument("--steps", type=int, default=DEFAULTS["max_steps"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["learning_rate"])
    parser.add_argument("--micro-batch", type=int, default=DEFAULTS["micro_batch_size"])
    parser.add_argument("--grad-accum", type=int, default=DEFAULTS["grad_accum_steps"])
    parser.add_argument("--seq-len", type=int, default=DEFAULTS["max_seq_len"])
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--data-bin", type=str, default=None,
                        help="Pre-tokenized train .bin (GPU-bound fast path)")
    parser.add_argument("--val-bin", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    from uchi.flux.tokenizer_v2 import TikTokenHybridTokenizer
    from uchi.flux.model import HybridTSSM

    tokenizer = TikTokenHybridTokenizer()

    print("=" * 72)
    print("FLUX Phase 4 — Ternary Quantization-Aware Training (QAT)")
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

    # 🔥 Turn on Ternary Quantization
    print("  [!] Activating 1.58-bit Ternary Quantization (BitNet)")
    model.set_quantization(True)
    model._gradient_checkpointing = True
    model.to(device)

    # ── Load data ──
    if args.data_bin:
        print(f"  Using pre-tokenized bin (GPU-bound): {args.data_bin}")
        train_iter = make_bin_iter(args.data_bin, args.micro_batch, args.seq_len, seed=42)
        val_iter = make_bin_iter(args.val_bin or args.data_bin, args.micro_batch, args.seq_len, seed=0)
    else:
        print("  Loading streaming data for QAT recovery...")
        train_iter = make_data_iter(tokenizer, "train", args.micro_batch, args.seq_len)
        val_iter = make_data_iter(tokenizer, "validation", args.micro_batch, args.seq_len, seed=42)

    # ── Training setup ──
    eff_batch = args.micro_batch * args.grad_accum
    max_steps = args.steps
    warmup_steps = min(DEFAULTS["warmup_steps"], max_steps // 10)
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

    ckpt_dir = DEFAULTS["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    model.train()
    best_val = float("inf")
    t0 = time.time()
    
    print(f"\n  Starting QAT from step 0 to {max_steps} ...\n")

    for step in range(max_steps):
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for micro_step in range(args.grad_accum):
            X, Y = next(train_iter)
            X, Y = X.to(device), Y.to(device)

            with amp_ctx():
                lang_logits, _ = model(X)
                loss = chunked_cross_entropy(lang_logits, Y, ignore_index=tokenizer.pad_token_id)
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
            print(f"  Step {step + 1:05d}/{max_steps} │ Loss: {loss_accum:.4f} │ LR: {lr:.2e} │ {ms_per_step:.0f}ms/step")

        if (step + 1) % DEFAULTS["eval_interval"] == 0:
            val_loss, val_ppl = evaluate(
                model, val_iter,
                DEFAULTS["eval_steps"], device, dtype, amp_ctx,
                tokenizer_pad_id=tokenizer.pad_token_id,
            )
            is_best = val_loss < best_val
            if is_best:
                best_val = val_loss
                
            print(f"\n  ──── VAL step {step + 1:05d} │ Loss: {val_loss:.4f} │ PPL: {val_ppl:.1f} {'★ best' if is_best else ''}\n")
            
            if is_best:
                save_checkpoint(raw_model, optimizer, step + 1, best_val, os.path.join(ckpt_dir, "qat_best.pt"))

        if (step + 1) % DEFAULTS["checkpoint_interval"] == 0:
            save_checkpoint(raw_model, optimizer, step + 1, best_val, os.path.join(ckpt_dir, f"qat_{step + 1:05d}.pt"))

    print(f"\n  QAT complete. Best val loss: {best_val:.4f}")

if __name__ == "__main__":
    main()
