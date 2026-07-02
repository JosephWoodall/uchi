"""
train_v2.py — Corrected FLUX Phase 1 pre-training on TinyStories.

Fixes over the original training:
 1. Fixed LR schedule (warmup → cosine decay over FIXED horizon, not ∞)
 2. Real effective batch via gradient accumulation (eff_batch=64)
 3. Ternary OFF, BF16 ON — de-risk the recipe before adding quantization
 4. GradScaler for fp16 fallback (prevents NaN on non-bf16 hardware)
 5. Checkpoint resume with optimizer state, step counter, and best loss
 6. torch.compile-safe state_dict saving (unwraps _orig_mod prefix)
 7. Validation loop every EVAL_INTERVAL steps on held-out TinyStories
 8. Data reshuffles via buffer-based sampling (not sequential restart)
 9. Off-by-one fix (step < MAX_STEPS, not <=)
10. Best-model tracking by validation loss

Usage:
    # Fresh start
    .venv/bin/python -m uchi.flux.train_v2

    # Resume from checkpoint
    .venv/bin/python -m uchi.flux.train_v2 --resume uchi/flux/checkpoints/ckpt_latest.pt

    # Custom config
    .venv/bin/python -m uchi.flux.train_v2 --max-steps 50000 --seq-len 512
"""

import os
import sys
import time
import math
import argparse
import torch
import torch.nn as nn
from contextlib import nullcontext

# ==============================================================================
# Default hyperparameters (overridable via CLI)
# ==============================================================================
DEFAULTS = dict(
    micro_batch_size=2,       # kept small: 100K vocab → huge logits tensor
    grad_accum_steps=32,      # effective batch = 64 (2 × 32)
    max_seq_len=256,          # fits 12GB VRAM with 100K vocab logits
    max_steps=30_000,         # fixed horizon
    warmup_steps=2_000,
    learning_rate=5e-4,
    min_lr_frac=0.05,         # min LR = 5% of peak
    weight_decay=0.1,
    grad_clip=1.0,
    checkpoint_interval=5_000,
    eval_interval=500,        # validation every N steps
    eval_steps=20,            # number of micro-batches per validation
    log_interval=10,
    checkpoint_dir="uchi/flux/checkpoints",
    gradient_checkpointing=True,  # trade compute for memory on 12GB GPU
    # Model config — Path B: 30M proof model
    d_model=256,
    n_layers=12,
    d_state=32,
)


# ==============================================================================
# LR schedule
# ==============================================================================
def get_lr(step, max_steps, warmup_steps, lr_max, lr_min):
    """Cosine decay with linear warmup."""
    if step < warmup_steps:
        return lr_max * (step + 1) / warmup_steps
    if step >= max_steps:
        return lr_min
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min + coeff * (lr_max - lr_min)


# ==============================================================================
# Data pipeline
# ==============================================================================
def make_data_iter(tokenizer, split, micro_batch_size, max_seq_len, seed=42):
    """Stream a mixture of OpenWebText, Wikipedia, and Code, encode on the fly, yield (input, target) batches.

    Uses a shuffle buffer so re-iterations don't replay in the same order.
    """
    from datasets import load_dataset, interleave_datasets

    # Load OpenWebText
    ds_owt = load_dataset("Skylion007/openwebtext", split=split, streaming=True)
    
    # Load Wikipedia (general knowledge for MMLU)
    # Note: Wikipedia only has a 'train' split by default, so if split is 'validation', fallback to openwebtext for it.
    if split == "train":
        ds_wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        # Load The Stack Smol (Python for SWE-bench)
        ds_code = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train", streaming=True)
        
        # We need to map all to a common 'text' field. OpenWebText and Wikipedia already have 'text'.
        # The Stack has 'content'.
        def map_code(x): return {"text": x["content"]}
        ds_code = ds_code.map(map_code)
        
        # Interleave: 50% OpenWebText, 25% Wikipedia, 25% Code
        dataset = interleave_datasets([ds_owt, ds_wiki, ds_code], probabilities=[0.5, 0.25, 0.25], seed=seed)
    else:
        dataset = ds_owt

    # Shuffle buffer gives pseudo-random order on each pass
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
    pad_id = tokenizer.pad_token_id

    def _token_gen():
        """Yield token sequences, looping over the dataset forever."""
        while True:
            for item in dataset:
                tokens = tokenizer.encode_text(item["text"], max_length=max_seq_len + 1)
                if len(tokens) > 10:
                    yield tokens

    gen = _token_gen()

    while True:
        batch = []
        for _ in range(micro_batch_size):
            toks = next(gen)
            # Pad or truncate to exactly max_seq_len + 1
            if len(toks) < max_seq_len + 1:
                toks = toks + [pad_id] * (max_seq_len + 1 - len(toks))
            else:
                toks = toks[: max_seq_len + 1]
            batch.append(toks)

        t = torch.tensor(batch, dtype=torch.long)
        yield t[:, :-1], t[:, 1:]  # (input, target)


# ==============================================================================
# Checkpoint helpers
# ==============================================================================
def _unwrap_state_dict(model):
    """Get state_dict without torch.compile's _orig_mod. prefix."""
    sd = model.state_dict()
    clean = {}
    for k, v in sd.items():
        # torch.compile wraps the model; strip the prefix for clean checkpoints
        key = k.replace("_orig_mod.", "")
        clean[key] = v
    return clean


def save_checkpoint(model, optimizer, step, best_val_loss, path):
    """Save a resumable checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": _unwrap_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    """Load a checkpoint, returning (step, best_val_loss)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    if optimizer is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            print(f"  [!] Could not restore optimizer state: {e}")
    step = ckpt.get("step", 0)
    best = ckpt.get("best_val_loss", float("inf"))
    return step, best


# ==============================================================================
# Chunked cross-entropy (avoids materializing full [B*T, vocab] logits)
# ==============================================================================
def chunked_cross_entropy(logits, targets, ignore_index=-100, chunk_size=64):
    """Compute CE loss in chunks along the sequence dimension.

    With 100K vocab, a full [B*T, V] logits tensor is ~400MB+ in float32.
    Processing in chunks of `chunk_size` tokens reduces peak memory by T/chunk_size.
    """
    B, T, V = logits.shape
    logits_flat = logits.reshape(B * T, V)
    targets_flat = targets.reshape(B * T)

    total_loss = 0.0
    n_chunks = 0
    for start in range(0, B * T, chunk_size):
        end = min(start + chunk_size, B * T)
        chunk_logits = logits_flat[start:end]
        chunk_targets = targets_flat[start:end]
        # Skip chunks that are all padding
        valid = (chunk_targets != ignore_index)
        if valid.any():
            loss = nn.functional.cross_entropy(
                chunk_logits, chunk_targets,
                ignore_index=ignore_index, reduction="mean",
            )
            total_loss = total_loss + loss
            n_chunks += 1

    return total_loss / max(n_chunks, 1)


# ==============================================================================
# Validation
# ==============================================================================
@torch.no_grad()
def evaluate(model, val_iter, eval_steps, device, dtype, amp_ctx_fn, tokenizer_pad_id=0):
    """Run eval_steps micro-batches and return mean loss + perplexity."""
    model.eval()
    total_loss = 0.0
    for _ in range(eval_steps):
        X, Y = next(val_iter)
        X, Y = X.to(device), Y.to(device)
        with amp_ctx_fn():
            lang_logits, _ = model(X)
            loss = chunked_cross_entropy(lang_logits, Y, ignore_index=tokenizer_pad_id)
        total_loss += loss.item()
    model.train()
    avg_loss = total_loss / eval_steps
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="FLUX Phase 1 — TinyStories pre-training")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--max_steps", type=int, default=10000, help="Total training steps")
    parser.add_argument("--seq-len", type=int, default=DEFAULTS["max_seq_len"])
    parser.add_argument("--micro-batch", type=int, default=DEFAULTS["micro_batch_size"])
    parser.add_argument("--grad-accum", type=int, default=DEFAULTS["grad_accum_steps"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["learning_rate"])
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--d_state", type=int, default=64, help="SSM state dimension")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    args = parser.parse_args()

    # Derived config
    max_steps = args.max_steps
    seq_len = args.seq_len
    micro_bs = args.micro_batch
    grad_accum = args.grad_accum
    eff_batch = micro_bs * grad_accum
    lr_max = args.lr
    lr_min = lr_max * DEFAULTS["min_lr_frac"]
    warmup = DEFAULTS["warmup_steps"]
    ckpt_dir = DEFAULTS["checkpoint_dir"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    print("=" * 72)
    print("FLUX Phase 1 — Corrected Pre-training")
    print("=" * 72)
    print(f"  Device:       {device} ({'bf16' if use_bf16 else 'fp16'})")
    print(f"  Model:        d_model={args.d_model}, n_layers={args.n_layers}, d_state={args.d_state}")
    print(f"  Batch:        {micro_bs} × {grad_accum} = {eff_batch} effective")
    print(f"  Seq len:      {seq_len}")
    print(f"  LR:           {lr_max:.1e} → {lr_min:.1e} (cosine, {warmup} warmup)")
    print(f"  Steps:        {max_steps:,}")
    print(f"  Tokens/step:  {eff_batch * seq_len:,}")
    print(f"  Total tokens: ~{max_steps * eff_batch * seq_len / 1e6:.0f}M")
    print("=" * 72)

    # ── Tokenizer ──
    from uchi.flux.tokenizer_v2 import TikTokenHybridTokenizer
    tokenizer = TikTokenHybridTokenizer()
    print(f"  Vocab size:   {tokenizer.vocab_size:,}")

    # ── Model ──
    from uchi.flux.model import HybridTSSM
    model = HybridTSSM(
        vocab_size=tokenizer.vocab_size,
        syntax_vocab_size=tokenizer.syntax_vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
    )
    # Phase 1: ternary OFF — prove convergence in full precision first
    model.set_quantization(False)
    # Gradient checkpointing: recompute activations during backward to save VRAM
    if DEFAULTS["gradient_checkpointing"]:
        model._gradient_checkpointing = True
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters:   {n_params:,} ({n_params / 1e6:.1f}M)")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr_max, weight_decay=DEFAULTS["weight_decay"],
        betas=(0.9, 0.95),  # slightly lower β2 for stability with small models
        fused=device == "cuda",  # fused AdamW is faster on CUDA
    )

    # ── GradScaler (fp16 only — bf16 doesn't need it) ──
    use_scaler = device == "cuda" and not use_bf16
    scaler = torch.amp.GradScaler(device, enabled=use_scaler)

    # ── AMP context ──
    def amp_ctx():
        if device == "cuda":
            return torch.amp.autocast(device_type="cuda", dtype=dtype)
        return nullcontext()

    # ── Loss ──
    # (CE loss is computed via chunked_cross_entropy, not nn.CrossEntropyLoss)

    # ── Resume ──
    start_step = 0
    best_val_loss = float("inf")
    if args.resume and os.path.exists(args.resume):
        print(f"  Resuming from {args.resume} ...")
        start_step, best_val_loss = load_checkpoint(args.resume, model, optimizer, device)
        print(f"  Resumed at step {start_step}, best_val_loss={best_val_loss:.4f}")

    # ── Compile (after loading weights, before training) ──
    raw_model = model  # keep reference for state_dict saving
    if device == "cuda" and not args.no_compile:
        print("  Compiling model with torch.compile ...")
        model = torch.compile(model)

    # ── Data ──
    print("  Loading OpenWebText (streaming) ...")
    train_iter = make_data_iter(tokenizer, "train", micro_bs, seq_len, seed=42)
    val_iter = make_data_iter(tokenizer, "validation", micro_bs, seq_len, seed=0)

    # ── Training loop ──
    os.makedirs(ckpt_dir, exist_ok=True)
    model.train()

    ema_loss = None
    ema_alpha = 0.02  # smoothing factor for logging
    t0 = time.time()
    step = start_step

    print(f"\n  Starting training from step {step} ...\n")

    while step < max_steps:
        # ── Set LR for this step ──
        lr = get_lr(step, max_steps, warmup, lr_max, lr_min)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Forward + backward with gradient accumulation ──
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for micro_step in range(grad_accum):
            X, Y = next(train_iter)
            X, Y = X.to(device), Y.to(device)

            with amp_ctx():
                lang_logits, _ = model(X)
                loss = chunked_cross_entropy(
                    lang_logits, Y,
                    ignore_index=tokenizer.pad_token_id,
                )
                # Scale loss by grad_accum to average (not sum) across micro-batches
                scaled_loss = loss / grad_accum

            # Backward — scaler handles fp16 overflow; is a no-op for bf16
            scaler.scale(scaled_loss).backward()
            loss_accum += loss.item() / grad_accum

        # ── Gradient clipping + optimizer step ──
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULTS["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        # ── EMA loss for smooth logging ──
        if ema_loss is None:
            ema_loss = loss_accum
        else:
            ema_loss = ema_alpha * loss_accum + (1.0 - ema_alpha) * ema_loss

        # ── Logging ──
        if step % DEFAULTS["log_interval"] == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            tok_per_sec = (eff_batch * seq_len) / max(dt, 1e-6)
            ppl = math.exp(min(ema_loss, 20.0))
            print(
                f"  Step {step:05d}/{max_steps} │ "
                f"Loss: {loss_accum:.4f} │ "
                f"EMA: {ema_loss:.4f} │ "
                f"PPL: {ppl:.1f} │ "
                f"LR: {lr:.2e} │ "
                f"Tok/s: {tok_per_sec:,.0f}"
            )

        # ── Validation ──
        if step > 0 and step % DEFAULTS["eval_interval"] == 0:
            val_loss, val_ppl = evaluate(
                model, val_iter,
                DEFAULTS["eval_steps"], device, dtype, amp_ctx,
                tokenizer_pad_id=tokenizer.pad_token_id,
            )
            print(
                f"  ──── VAL step {step:05d} │ "
                f"Loss: {val_loss:.4f} │ "
                f"PPL: {val_ppl:.1f} "
                f"{'★ new best' if val_loss < best_val_loss else ''}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    raw_model, optimizer, step, best_val_loss,
                    os.path.join(ckpt_dir, "ckpt_best.pt"),
                )

        # ── Periodic checkpoint ──
        if step > 0 and step % DEFAULTS["checkpoint_interval"] == 0:
            save_checkpoint(
                raw_model, optimizer, step, best_val_loss,
                os.path.join(ckpt_dir, "ckpt_latest.pt"),
            )
            print(f"  Saved checkpoint at step {step}")

        step += 1

    # ── Final save ──
    save_checkpoint(
        raw_model, optimizer, step, best_val_loss,
        os.path.join(ckpt_dir, "ckpt_final.pt"),
    )
    total_time = time.time() - t0
    print(f"\n  Training complete. {step} steps in {total_time / 3600:.1f}h")
    print(f"  Best val loss: {best_val_loss:.4f} (PPL {math.exp(min(best_val_loss, 20.0)):.1f})")
    print(f"  Checkpoints in: {ckpt_dir}/")


if __name__ == "__main__":
    main()
