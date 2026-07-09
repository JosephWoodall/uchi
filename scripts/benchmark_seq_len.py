#!/usr/bin/env python3
"""benchmark_seq_len.py -- real-hardware FLOPs/memory benchmark for 0.5.0
Item 2's max_seq_len decision.

Current default max_seq_len is 256, chosen for a 12GB VRAM budget, not an
architectural limit -- FLUX's SSM backbone gives O(1) inference memory
regardless of context length. This benchmark answers the actual question
Item 2's todo bullet asks: at what max_seq_len does the attention layers'
cost (quadratic in sequence length) stop being subdominant to the
checkpointed SSM layers' cost (near-linear)?

Real, not theoretical: instantiates the ACTUAL `AttentionBlock` and
`TSSMBlock` classes from `uchi/flux/`, at the real checkpoint's d_model
(768, confirmed by loading `flux_best.pt`), runs real forward+backward
passes under `torch.utils.checkpoint` (the same gradient-checkpointing
path `HybridTSSM.forward` uses in training, not an idealized FLOPs
formula) at each candidate seq_len, and measures actual wall-clock time
and peak CUDA memory. Scaled by the real layer counts (12 layers total,
`i % 4 == 3` attention placement -> 3 attention / 9 SSM at n_layers=12)
to report whole-model attention-vs-SSM cost, not just per-layer.

Usage:
    .venv/bin/python -m scripts.benchmark_seq_len
    .venv/bin/python -m scripts.benchmark_seq_len --seq-lens 256,512,1024,2048
"""
from __future__ import annotations

import argparse
import time

import torch

from uchi.flux.attention_block import AttentionBlock
from uchi.flux.tssm_block import TSSMBlock

D_MODEL = 768   # matches flux_best.pt, confirmed by loading the checkpoint
D_STATE = 32
N_HEADS = 8
N_LAYERS_TOTAL = 12                                    # matches flux_best.pt
N_ATTN_LAYERS = sum(1 for i in range(N_LAYERS_TOTAL) if i % 4 == 3)
N_SSM_LAYERS = N_LAYERS_TOTAL - N_ATTN_LAYERS


def _bench_layer(layer: torch.nn.Module, seq_len: int, batch: int, device: str,
                  checkpointed: bool, warmup: int = 2, iters: int = 5) -> tuple[float, float]:
    """Real forward+backward wall-clock (s) and peak memory (MB) for one
    layer at one seq_len, averaged over `iters` runs after `warmup`.
    """
    layer = layer.to(device)
    x = torch.randn(batch, seq_len, D_MODEL, device=device, requires_grad=True)

    def step():
        layer.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        if checkpointed:
            out, _ = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        else:
            out, _ = layer(x)
        loss = out.sum()
        loss.backward()

    for _ in range(warmup):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters

    peak_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if device == "cuda" else 0.0
    return dt, peak_mb


def run(seq_lens: list[int], batch: int, device: str, fuse_ssm: bool) -> None:
    print(f"\nDevice: {device}  |  d_model={D_MODEL}  d_state={D_STATE}  "
          f"n_layers={N_LAYERS_TOTAL} ({N_ATTN_LAYERS} attention / {N_SSM_LAYERS} SSM)  "
          f"|  UCHI_FUSE_SSM_SCAN={'1' if fuse_ssm else '0'}\n")
    header = f"{'seq_len':>8}  {'attn_ms/layer':>14}  {'ssm_ms/layer':>13}  " \
             f"{'attn_total_ms':>14}  {'ssm_total_ms':>13}  {'attn_share':>10}  {'attn_MB':>9}  {'ssm_MB':>9}"
    print(header)
    print("-" * len(header))

    results = []
    for seq_len in seq_lens:
        attn = AttentionBlock(D_MODEL, n_heads=N_HEADS)
        ssm = TSSMBlock(D_MODEL, D_STATE)

        try:
            attn_dt, attn_mb = _bench_layer(attn, seq_len, batch, device, checkpointed=True)
            ssm_dt, ssm_mb = _bench_layer(ssm, seq_len, batch, device, checkpointed=True)
        except torch.OutOfMemoryError:
            print(f"{seq_len:>8}  OOM at batch={batch} on this device -- skipping, "
                  f"real training would need a smaller batch or gradient accumulation here")
            del attn, ssm
            if device == "cuda":
                torch.cuda.empty_cache()
            continue

        attn_total = attn_dt * N_ATTN_LAYERS
        ssm_total = ssm_dt * N_SSM_LAYERS
        share = attn_total / (attn_total + ssm_total)

        print(f"{seq_len:>8}  {attn_dt*1000:>14.2f}  {ssm_dt*1000:>13.2f}  "
              f"{attn_total*1000:>14.2f}  {ssm_total*1000:>13.2f}  {share*100:>9.1f}%  "
              f"{attn_mb:>9.1f}  {ssm_mb:>9.1f}")
        results.append((seq_len, share))

        del attn, ssm
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\nRecommendation: pick the largest seq_len where attn_share stays")
    print("below ~50% of whole-model per-step cost (i.e. attention hasn't")
    print("overtaken the SSM scan as the dominant cost yet).")
    under_50 = [s for s, share in results if share < 0.5]
    if under_50:
        print(f"  -> largest candidate under 50% attention share: {max(under_50)}")
    else:
        print("  -> attention share already >=50% at every candidate tested; "
              "consider smaller seq_lens or reducing attention layer density.")


def main() -> int:
    parser = argparse.ArgumentParser(description="0.5.0 Item 2: max_seq_len FLOPs/memory benchmark")
    parser.add_argument("--seq-lens", default="256,512,1024,2048")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--fuse-ssm", action="store_true",
        help="Set UCHI_FUSE_SSM_SCAN=1 (torch.compile-fused scan, ~136s one-time "
             "compile cost per shape, documented 3.08x forward speedup) -- what real "
             "training entrypoints can opt into, unlike the unfused default.",
    )
    args = parser.parse_args()

    if args.fuse_ssm:
        import os
        os.environ["UCHI_FUSE_SSM_SCAN"] = "1"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    run(seq_lens, args.batch, device, fuse_ssm=args.fuse_ssm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
