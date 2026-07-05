"""
cot_distill.py — Phase 3: Chain-of-Thought Distillation

Fine-tunes the SFT model on step-by-step reasoning traces to teach it
how to plan and reason before answering. This is the only reliable way
to elicit reasoning in ~30M-200M parameter models.

Training format:
    <|user|>\n{question}\n<|think|>\n{reasoning_steps}\n<|/think|>\n<|assistant|>\n{answer}<|eos|>

Loss is computed on both <|think|> tokens and <|assistant|> tokens,
but prompt is masked out.

Data source:
    - We will use an open dataset of CoT traces like Open-Orca or
      a synthesized subset (for this script, we'll build a synthetic 
      math/logic dataset generator just to prove the pipeline works).

Usage:
    .venv/bin/python -m uchi.flux.cot_distill --base uchi/flux/checkpoints/sft_best.pt
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
from .train_v2 import chunked_cross_entropy, save_checkpoint


# ==============================================================================
# Defaults
# ==============================================================================
DEFAULTS = dict(
    micro_batch_size=2,
    grad_accum_steps=32,       # effective batch = 64
    max_seq_len=384,           # CoT needs longer context
    epochs=4,
    learning_rate=2e-5,        # Very careful LR for distillation
    min_lr_frac=0.1,
    warmup_frac=0.1,
    weight_decay=0.01,
    grad_clip=1.0,
    eval_frac=0.05,
    checkpoint_dir="uchi/flux/checkpoints",
    max_examples=10_000,
)


# ==============================================================================
# Real chain-of-thought teacher traces (static distillation)
# ==============================================================================
def generate_synthetic_cot(num_examples):
    """Load REAL worked-solution CoT traces from GSM8K (static teacher distillation).

    A ~116M model cannot bootstrap reasoning by self-play / rejection-sampling its
    own outputs — that is the cold start. The reliable fix is to imitate real
    teacher traces. So the <|think|> content here is the ACTUAL GSM8K reasoning
    steps (calculator annotations like <<48/2=24>> stripped for human-readable
    I/O), and the answer is the final result. No self-generation from an untrained
    model, and no fabricated 'flywheel monologue' — training on invented
    [THINKING]: Swarm... text would teach FLUX to narrate a process it never runs.
    """
    from datasets import load_dataset

    examples = []
    try:
        gsm8k = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
    except Exception as e:
        print(f"  [!] Failed to load GSM8K teacher traces: {e}")
        return []

    for row in gsm8k:
        q = row["question"].strip()
        raw = row["answer"]
        if "####" not in raw:
            continue
        steps, final = raw.split("####")
        # Strip GSM8K's <<...>> calculator markup so the reasoning reads cleanly.
        think = re.sub(r"<<[^>]*>>", "", steps).strip()
        final = final.strip()
        if not think or not final:
            continue
        examples.append({
            "question": q,
            "think": think,
            "answer": f"The answer is {final}.",
        })
        if len(examples) >= num_examples:
            break

    print(f"  [+] Loaded {len(examples)} real GSM8K teacher CoT traces.")
    return examples


def _split_think_answer(response: str):
    """Split a real response into (reasoning, final_answer): the last
    sentence is the answer, everything before it is the reasoning trace.
    A simple heuristic, not perfect — but it splits an existing real
    response rather than fabricating one, same principle as GSM8K's
    '####' split above. Used for sources (OpenOrca, Magicoder) that don't
    ship a pre-separated reasoning/answer field the way GSM8K does.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", response.strip()) if s.strip()]
    if not sentences:
        return "", ""
    if len(sentences) == 1:
        return sentences[0], sentences[0]
    return " ".join(sentences[:-1]), sentences[-1]


def generate_openorca_cot(num_examples):
    """General step-by-step reasoning CoT from OpenOrca (0.4.0 Item 0).

    GSM8K alone only teaches math CoT. Items 3/4/6 (scratchpad, tool
    calling, self-healing loop) need FLUX to reason step-by-step about
    arbitrary problems, not just arithmetic. Real GPT-4-generated
    responses, split into reasoning/answer via _split_think_answer —
    still an actual teacher trace, not a fabricated one.
    """
    from datasets import load_dataset

    examples = []
    try:
        ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    except Exception as e:
        print(f"  [!] Failed to load OpenOrca teacher traces: {e}")
        return []

    for row in ds:
        question = row.get("question", "").strip()
        response = row.get("response", "").strip()
        if not question or not response:
            continue
        think, answer = _split_think_answer(response)
        if not think or not answer:
            continue
        examples.append({"question": question, "think": think, "answer": answer})
        if len(examples) >= num_examples:
            break

    print(f"  [+] Loaded {len(examples)} real OpenOrca teacher CoT traces.")
    return examples


def generate_magicoder_cot(num_examples):
    """Code-specific reasoning CoT from Magicoder-OSS-Instruct (0.4.0 Item 0).

    GSM8K/OpenOrca teach math/general reasoning; neither teaches
    plan-then-code structure, which is exactly what the empirical
    synthesis loop (Items 3/6) needs FLUX to do well. Real solutions from
    real OSS-derived problems, not fabricated.
    """
    from datasets import load_dataset

    examples = []
    try:
        ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", streaming=True)
    except Exception as e:
        print(f"  [!] Failed to load Magicoder teacher traces: {e}")
        return []

    for row in ds:
        problem = row.get("problem", "").strip()
        solution = row.get("solution", "").strip()
        if not problem or not solution:
            continue
        think, answer = _split_think_answer(solution)
        if not think or not answer:
            continue
        examples.append({"question": problem, "think": think, "answer": answer})
        if len(examples) >= num_examples:
            break

    print(f"  [+] Loaded {len(examples)} real Magicoder teacher CoT traces.")
    return examples


def load_cot_examples(tokenizer, max_seq_len, max_examples, seed=42):
    random.seed(seed)
    print("  Generating CoT examples from 3 real teacher-trace sources ...")
    # Each source gets its own independent share of max_examples (not a
    # shared cumulative counter — see the 0.4.0 fix in sft_train.py's
    # load_sft_examples for why a shared counter silently starves every
    # source after the first).
    per_source = max(1, max_examples // 3)
    examples = (
        generate_synthetic_cot(per_source)
        + generate_openorca_cot(per_source)
        + generate_magicoder_cot(per_source)
    )
    random.shuffle(examples)
    examples = examples[:max_examples]
    print(f"  Total CoT examples: {len(examples):,}")

    formatted = []
    user_id = tokenizer.encode_special("<|user|>")
    asst_id = tokenizer.encode_special("<|assistant|>")
    think_open = tokenizer.encode_special("<|think|>")
    think_close = tokenizer.encode_special("<|/think|>")
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    for ex in examples:
        # Prompt: <|user|>\n{question}\n
        prompt_ids = [user_id] + tokenizer.encode_text(ex["question"])
        
        # Think: <|think|>\n{reasoning}\n<|/think|>\n
        think_ids = [think_open] + tokenizer.encode_text(ex["think"]) + [think_close]
        
        # Answer: <|assistant|>\n{answer}<|eos|>
        answer_ids = [asst_id] + tokenizer.encode_text(ex["answer"]) + [eos_id]

        full_ids = prompt_ids + think_ids + answer_ids
        
        # Loss mask: 0 for prompt, 1 for think + answer
        n_prompt = len(prompt_ids)
        loss_mask = [0] * min(n_prompt, len(full_ids)) + [1] * max(0, len(full_ids) - n_prompt)

        if len(full_ids) > max_seq_len:
            # Drop examples that are too long for now to keep it simple
            continue

        n_pad = max_seq_len - len(full_ids)
        full_ids = full_ids + [pad_id] * n_pad
        loss_mask = loss_mask + [0] * n_pad

        formatted.append({
            "input_ids": full_ids[:max_seq_len],
            "loss_mask": loss_mask[:max_seq_len],
        })

    return formatted


def make_batches(data, micro_batch_size, shuffle=True):
    if shuffle:
        random.shuffle(data)
    for i in range(0, len(data) - micro_batch_size + 1, micro_batch_size):
        batch = data[i : i + micro_batch_size]
        input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
        loss_mask = torch.tensor([b["loss_mask"] for b in batch], dtype=torch.float32)
        yield input_ids[:, :-1], input_ids[:, 1:], loss_mask[:, 1:]


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="FLUX Phase 3 — CoT Distillation")
    parser.add_argument("--base", type=str, required=True, help="Path to Phase 2 SFT checkpoint")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["learning_rate"])
    parser.add_argument("--micro-batch", type=int, default=DEFAULTS["micro_batch_size"])
    parser.add_argument("--grad-accum", type=int, default=DEFAULTS["grad_accum_steps"])
    parser.add_argument("--seq-len", type=int, default=DEFAULTS["max_seq_len"])
    parser.add_argument("--max-examples", type=int, default=DEFAULTS["max_examples"])
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    from uchi.flux.tokenizer_v2 import TikTokenHybridTokenizer
    from uchi.flux.model import HybridTSSM

    tokenizer = TikTokenHybridTokenizer()

    print("=" * 72)
    print("FLUX Phase 3 — Chain-of-Thought Distillation")
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
    model._gradient_checkpointing = True
    model.to(device)

    # ── Load + format data ──
    data = load_cot_examples(tokenizer, args.seq_len, args.max_examples)
    n_val = max(10, int(len(data) * DEFAULTS["eval_frac"]))
    val_data, train_data = data[:n_val], data[n_val:]
    print(f"  Train: {len(train_data):,}  Val: {len(val_data):,}")

    # ── Training setup ──
    eff_batch = args.micro_batch * args.grad_accum
    steps_per_epoch = len(train_data) // eff_batch
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * DEFAULTS["warmup_frac"])
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
        if step >= total_steps:
            return lr_min
        progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min + coeff * (args.lr - lr_min)

    def masked_chunked_ce_loss(logits, targets, mask, chunk_size=64):
        B, T, V = logits.shape
        logits_flat = logits.reshape(B * T, V)
        targets_flat = targets.reshape(B * T)
        mask_flat = mask.reshape(B * T)

        total_loss = 0.0
        n_chunks = 0
        for start in range(0, B * T, chunk_size):
            end = min(start + chunk_size, B * T)
            c_logits = logits_flat[start:end]
            c_targets = targets_flat[start:end]
            c_mask = mask_flat[start:end]
            
            if c_mask.sum() > 0:
                loss_per_token = nn.functional.cross_entropy(c_logits, c_targets, reduction="none")
                masked_loss = (loss_per_token * c_mask).sum() / c_mask.sum()
                total_loss = total_loss + masked_loss
                n_chunks += 1
                
        return total_loss / max(n_chunks, 1)

    @torch.no_grad()
    def validate():
        model.eval()
        total_loss = 0.0
        n_batches = 0
        for X, Y, mask in make_batches(val_data, args.micro_batch, shuffle=False):
            X, Y, mask = X.to(device), Y.to(device), mask.to(device)
            with amp_ctx():
                logits, _ = model(X)
                loss = masked_chunked_ce_loss(logits, Y, mask)
            total_loss += loss.item()
            n_batches += 1
        model.train()
        return total_loss / max(n_batches, 1)

    ckpt_dir = DEFAULTS["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    model.train()
    best_val = float("inf")
    global_step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        batch_iter = make_batches(train_data, args.micro_batch, shuffle=True)
        epoch_loss = 0.0
        n_micro = 0
        accum = 0
        optimizer.zero_grad(set_to_none=True)   # zero ONCE before accumulating

        for X, Y, mask in batch_iter:
            X, Y, mask = X.to(device), Y.to(device), mask.to(device)

            with amp_ctx():
                logits, _ = model(X)
                # scale by grad_accum so summed grads average across micro-batches
                loss = masked_chunked_ce_loss(logits, Y, mask) / args.grad_accum

            scaler.scale(loss).backward()          # accumulate (no zero here)
            epoch_loss += loss.item() * args.grad_accum
            n_micro += 1
            accum += 1

            # Step only after grad_accum micro-batches have accumulated.
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
                if global_step % 10 == 0:
                    print(f"  Step {global_step:05d} │ Loss: {epoch_loss / n_micro:.4f} │ LR: {lr:.2e}")

        # Flush any partial accumulation at epoch end so no gradients are wasted.
        if accum > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULTS["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        n_epoch_batches = n_micro

        val_loss = validate()
        ppl = math.exp(min(val_loss, 20.0))
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            
        print(f"\n  ── Epoch {epoch + 1}/{args.epochs} │ Val Loss: {val_loss:.4f} │ PPL: {ppl:.1f} {'★ best' if is_best else ''}")

        save_checkpoint(raw_model, optimizer, global_step, best_val, os.path.join(ckpt_dir, f"cot_epoch{epoch + 1}.pt"))
        if is_best:
            save_checkpoint(raw_model, optimizer, global_step, best_val, os.path.join(ckpt_dir, "cot_best.pt"))

    print(f"\n  CoT Distillation complete. Best val loss: {best_val:.4f}")

if __name__ == "__main__":
    main()
