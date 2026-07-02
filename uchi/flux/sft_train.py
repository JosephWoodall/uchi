"""
sft_train.py — Phase 2: Supervised Fine-Tuning for instruction following.

Takes a Phase 1 pre-trained FLUX checkpoint and fine-tunes it on
instruction/QA/RAG data so it learns to answer questions from context
rather than merely continuing text.

Training format (matches FluxProposer's prompt structure):
    <|user|>\n{question}\n<|context|>\n{evidence}\n<|/context|>\n<|assistant|>\n{answer}<|eos|>

Loss is computed ONLY on assistant response tokens (prompt is masked).
This teaches the model to generate grounded answers without forgetting
the language model capabilities learned in Phase 1.

Data sources (auto-downloaded via HuggingFace):
    - SQuAD 2.0 extractive QA (context + question → answer)
    - Dolly-15K instruction-following (diverse instructions)

Usage:
    # SFT from Phase 1 best checkpoint
    .venv/bin/python -m uchi.flux.sft_train --base uchi/flux/checkpoints/ckpt_best.pt

    # Custom settings
    .venv/bin/python -m uchi.flux.sft_train --base ckpt_best.pt --epochs 3 --lr 3e-5
"""

import os
import sys
import time
import math
import random
import argparse
import re
import torch
import torch.nn as nn
from contextlib import nullcontext


# ==============================================================================
# Defaults
# ==============================================================================
DEFAULTS = dict(
    micro_batch_size=4,
    grad_accum_steps=16,       # effective batch = 64
    max_seq_len=512,
    epochs=3,
    learning_rate=3e-5,        # much lower than pretraining — we're fine-tuning
    min_lr_frac=0.1,
    warmup_frac=0.05,          # 5% of total steps for warmup
    weight_decay=0.01,
    grad_clip=1.0,
    eval_frac=0.05,            # 5% held out for validation
    checkpoint_dir="uchi/flux/checkpoints",
    max_examples=50_000,       # cap dataset size for fast iteration
)


# ==============================================================================
# Data preparation
# ==============================================================================
def load_sft_examples(tokenizer, max_seq_len, max_examples, seed=42):
    """Load and format SFT examples from SQuAD + Dolly.

    Returns list of dicts: {input_ids, target_ids, loss_mask}
    where loss_mask is 1 for assistant tokens, 0 for prompt tokens.
    """
    from datasets import load_dataset

    random.seed(seed)
    examples = []

    # ── SQuAD 2.0 (extractive QA with context) ──
    print("  Loading SQuAD ...")
    try:
        ds = load_dataset("rajpurkar/squad_v2", split="train")
        for row in ds:
            answers = row["answers"]["text"]
            if not answers:
                continue  # skip unanswerable examples for now
            question = row["question"].strip()
            context = row["context"].strip()[:800]  # cap context length
            answer = answers[0].strip()
            if not answer:
                continue

            examples.append({
                "question": question,
                "context": context,
                "answer": answer,
            })
            if len(examples) >= max_examples:
                break
    except Exception as e:
        print(f"    [!] SQuAD load failed: {e}")

    # ── Dolly-15K (instruction following) ──
    print("  Loading Dolly-15K ...")
    try:
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        for row in ds:
            instruction = row.get("instruction", "").strip()
            context = row.get("context", "").strip()
            response = row.get("response", "").strip()
            if not instruction or not response:
                continue

            examples.append({
                "question": instruction,
                "context": context if context else "",
                "answer": response[:500],  # cap answer length
            })
            if len(examples) >= max_examples:
                break
    except Exception as e:
        print(f"    [!] Dolly load failed: {e}")

    random.shuffle(examples)
    examples = examples[:max_examples]
    print(f"  Total SFT examples: {len(examples):,}")

    # ── Tokenize with loss masking ──
    formatted = []
    user_id = tokenizer.encode_special("<|user|>")
    asst_id = tokenizer.encode_special("<|assistant|>")
    ctx_open = tokenizer.encode_special("<|context|>")
    ctx_close = tokenizer.encode_special("<|/context|>")
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    for ex in examples:
        # Build prompt tokens
        prompt_parts = [user_id]
        prompt_parts.extend(tokenizer.encode_text(ex["question"]))
        if ex["context"]:
            prompt_parts.append(ctx_open)
            prompt_parts.extend(tokenizer.encode_text(ex["context"], max_length=300))
            prompt_parts.append(ctx_close)
        prompt_parts.append(asst_id)
        prompt_ids = prompt_parts

        # Build answer tokens
        answer_ids = tokenizer.encode_text(ex["answer"], max_length=200)
        answer_ids.append(eos_id)

        # Combine
        full_ids = prompt_ids + answer_ids
        if len(full_ids) > max_seq_len:
            # Truncate context to fit, keep question + answer intact
            full_ids = full_ids[:max_seq_len]

        # Loss mask: 0 for prompt, 1 for answer
        n_prompt = len(prompt_ids)
        loss_mask = [0] * min(n_prompt, len(full_ids)) + [1] * max(0, len(full_ids) - n_prompt)

        # Pad to max_seq_len
        n_pad = max_seq_len - len(full_ids)
        full_ids = full_ids + [pad_id] * n_pad
        loss_mask = loss_mask + [0] * n_pad  # don't compute loss on padding

        formatted.append({
            "input_ids": full_ids[:max_seq_len],
            "loss_mask": loss_mask[:max_seq_len],
        })

    return formatted


def make_sft_batches(data, micro_batch_size, shuffle=True):
    """Yield (input, target, loss_mask) batches from formatted SFT data."""
    if shuffle:
        random.shuffle(data)
    for i in range(0, len(data) - micro_batch_size + 1, micro_batch_size):
        batch = data[i : i + micro_batch_size]
        input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
        loss_mask = torch.tensor([b["loss_mask"] for b in batch], dtype=torch.float32)
        # input = all but last token, target = all but first token
        yield input_ids[:, :-1], input_ids[:, 1:], loss_mask[:, 1:]


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="FLUX Phase 2 — SFT")
    parser.add_argument("--base", type=str, required=True,
                        help="Path to Phase 1 base checkpoint")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["learning_rate"])
    parser.add_argument("--micro-batch", type=int, default=DEFAULTS["micro_batch_size"])
    parser.add_argument("--grad-accum", type=int, default=DEFAULTS["grad_accum_steps"])
    parser.add_argument("--max-examples", type=int, default=DEFAULTS["max_examples"])
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    # ── Load tokenizer + model from Phase 1 checkpoint ──
    from uchi.flux.tokenizer_v2 import TikTokenHybridTokenizer
    from uchi.flux.model import HybridTSSM

    tokenizer = TikTokenHybridTokenizer()

    print("=" * 72)
    print("FLUX Phase 2 — Supervised Fine-Tuning")
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

    model.set_quantization(False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Base:         {args.base}")
    print(f"  Architecture: d_model={d_model}, n_layers={n_layers}, d_state={d_state}")
    print(f"  Parameters:   {n_params:,} ({n_params / 1e6:.1f}M)")

    # ── Load + format data ──
    data = load_sft_examples(tokenizer, DEFAULTS["max_seq_len"], args.max_examples)
    n_val = max(100, int(len(data) * DEFAULTS["eval_frac"]))
    val_data, train_data = data[:n_val], data[n_val:]
    print(f"  Train: {len(train_data):,}  Val: {len(val_data):,}")

    # ── Training setup ──
    eff_batch = args.micro_batch * args.grad_accum
    steps_per_epoch = len(train_data) // (eff_batch)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * DEFAULTS["warmup_frac"])
    lr_min = args.lr * DEFAULTS["min_lr_frac"]

    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch:        {args.micro_batch} × {args.grad_accum} = {eff_batch}")
    print(f"  Steps/epoch:  {steps_per_epoch}")
    print(f"  Total steps:  {total_steps}")
    print(f"  LR:           {args.lr:.1e} → {lr_min:.1e}")
    print("=" * 72)

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

    # ── LR schedule (cosine with warmup) ──
    def get_lr(step):
        if step < warmup_steps:
            return args.lr * (step + 1) / warmup_steps
        if step >= total_steps:
            return lr_min
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min + coeff * (args.lr - lr_min)

    # ── Masked loss function ──
    def masked_ce_loss(logits, targets, mask):
        """Cross-entropy loss only on masked (=1) positions."""
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)
        mask_flat = mask.reshape(-1)

        loss_per_token = nn.functional.cross_entropy(
            logits_flat, targets_flat, reduction="none"
        )
        masked_loss = (loss_per_token * mask_flat).sum() / mask_flat.sum().clamp(min=1)
        return masked_loss

    # ── Validation ──
    @torch.no_grad()
    def validate():
        model.eval()
        total_loss = 0.0
        n_batches = 0
        for X, Y, mask in make_sft_batches(val_data, args.micro_batch, shuffle=False):
            X, Y, mask = X.to(device), Y.to(device), mask.to(device)
            with amp_ctx():
                logits, _ = model(X)
                loss = masked_ce_loss(logits, Y, mask)
            total_loss += loss.item()
            n_batches += 1
        model.train()
        return total_loss / max(n_batches, 1)

    # ── Training ──
    ckpt_dir = DEFAULTS["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    model.train()
    best_val = float("inf")
    global_step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        batch_iter = make_sft_batches(train_data, args.micro_batch, shuffle=True)
        epoch_loss = 0.0
        n_epoch_batches = 0

        for X, Y, mask in batch_iter:
            # ── LR ──
            lr = get_lr(global_step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # ── Forward/backward ──
            X, Y, mask = X.to(device), Y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)

            with amp_ctx():
                logits, _ = model(X)
                loss = masked_ce_loss(logits, Y, mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULTS["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_epoch_batches += 1
            global_step += 1

            if global_step % 50 == 0:
                avg = epoch_loss / n_epoch_batches
                print(f"  Step {global_step:05d} │ Loss: {loss.item():.4f} │ Avg: {avg:.4f} │ LR: {lr:.2e}")

        # ── End-of-epoch validation ──
        val_loss = validate()
        ppl = math.exp(min(val_loss, 20.0))
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
        print(
            f"\n  ── Epoch {epoch + 1}/{args.epochs} │ "
            f"Train: {epoch_loss / max(n_epoch_batches, 1):.4f} │ "
            f"Val: {val_loss:.4f} │ PPL: {ppl:.1f} "
            f"{'★ best' if is_best else ''}"
        )

        # Save
        from uchi.flux.train_v2 import save_checkpoint
        save_checkpoint(
            raw_model, optimizer, global_step, best_val,
            os.path.join(ckpt_dir, f"sft_epoch{epoch + 1}.pt"),
        )
        if is_best:
            save_checkpoint(
                raw_model, optimizer, global_step, best_val,
                os.path.join(ckpt_dir, "sft_best.pt"),
            )

    elapsed = time.time() - t0
    print(f"\n  SFT complete. {global_step} steps in {elapsed / 60:.1f}m")
    print(f"  Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
