"""
validate.py — Standalone checkpoint evaluation for FLUX.

Runs validation loss + perplexity on held-out TinyStories, then generates
sample completions so you can visually judge coherence.

Usage:
    .venv/bin/python -m uchi.flux.validate uchi/flux/checkpoints/ckpt_best.pt
    .venv/bin/python -m uchi.flux.validate ckpt_best.pt --samples 5 --max-tokens 100
"""

import argparse
import math
import os
import re
import torch
import torch.nn as nn
from contextlib import nullcontext


def main():
    parser = argparse.ArgumentParser(description="Evaluate a FLUX checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint")
    parser.add_argument("--eval-steps", type=int, default=50,
                        help="Number of micro-batches for val loss")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--samples", type=int, default=3,
                        help="Number of sample completions to generate")
    parser.add_argument("--max-tokens", type=int, default=80,
                        help="Max tokens per sample completion")
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    # ── Load tokenizer ──
    from uchi.flux.tokenizer_v2 import TikTokenHybridTokenizer
    tokenizer = TikTokenHybridTokenizer()

    # ── Load model from checkpoint (auto-detect architecture) ──
    from uchi.flux.model import HybridTSSM

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # Infer architecture from tensor shapes
    emb_shape = sd["embedding.weight"].shape
    vocab_size, d_model = int(emb_shape[0]), int(emb_shape[1])
    layer_ids = {int(m.group(1)) for k in sd if (m := re.match(r"layers\.(\d+)\.", k))}
    n_layers = (max(layer_ids) + 1) if layer_ids else 12

    # Try d_state values
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
        raise RuntimeError(f"Could not match architecture to checkpoint")

    model.set_quantization(False)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    step = ckpt.get("step", "?") if isinstance(ckpt, dict) else "?"
    best_loss = ckpt.get("best_val_loss", "?") if isinstance(ckpt, dict) else "?"

    print("=" * 72)
    print("FLUX Checkpoint Evaluation")
    print("=" * 72)
    print(f"  Checkpoint:   {args.checkpoint}")
    print(f"  Step:         {step}")
    print(f"  Best loss:    {best_loss}")
    print(f"  Architecture: d_model={d_model}, n_layers={n_layers}, d_state={d_state}")
    print(f"  Parameters:   {n_params:,} ({n_params / 1e6:.1f}M)")
    print(f"  Device:       {device}")
    print("=" * 72)

    # ── Validation loss ──
    print("\n  Computing validation loss ...")
    from uchi.flux.train_v2 import make_data_iter, chunked_cross_entropy
    val_iter = make_data_iter(tokenizer, "validation", args.batch_size, args.seq_len, seed=0)

    total_loss = 0.0
    with torch.no_grad():
        for i in range(args.eval_steps):
            X, Y = next(val_iter)
            X, Y = X.to(device), Y.to(device)
            ctx = torch.amp.autocast(device_type="cuda", dtype=dtype) if device == "cuda" else nullcontext()
            with ctx:
                lang_logits, _ = model(X)
                loss = chunked_cross_entropy(lang_logits, Y, ignore_index=tokenizer.pad_token_id)
            total_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"    [{i + 1}/{args.eval_steps}] running loss = {total_loss / (i + 1):.4f}")

    val_loss = total_loss / args.eval_steps
    val_ppl = math.exp(min(val_loss, 20.0))
    print(f"\n  ── Validation Results ──")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  PPL:  {val_ppl:.1f}")

    # ── Sample completions ──
    if args.samples > 0:
        prompts = [
            "Once upon a time, there was a",
            "The little dog ran to the",
            "Lily wanted to play with her",
            "One day, a big bear walked into the",
            "The sun was shining and the birds were",
        ]
        print(f"\n  ── Sample Completions (temperature={args.temperature}) ──")
        for i in range(min(args.samples, len(prompts))):
            prompt = prompts[i]
            ids = tokenizer.encode_text(prompt)
            input_t = torch.tensor([ids], device=device).long()

            with torch.no_grad():
                output = model.generate(
                    input_t,
                    max_length=args.max_tokens,
                    tokenizer=tokenizer,
                    temperature=args.temperature,
                    repetition_penalty=1.3,
                )
            generated = tokenizer.decode_text(output[0].cpu().tolist())
            print(f"\n  [{i + 1}] Prompt: \"{prompt}\"")
            print(f"      Output: \"{generated}\"")

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
