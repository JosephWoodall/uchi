#!/bin/bash
# train_all.sh — Full FLUX pipeline, corrected & resumable.
#
#   0. Pre-tokenize corpus  -> uchi/flux/data/{train,val}.bin   (GPU-bound fast path)
#   1. Pre-training (116M)   -> checkpoints/ckpt_best.pt
#   2. SFT (SQuAD+Dolly+Code)-> checkpoints/sft_best.pt
#   3. CoT distill (GSM8K)   -> checkpoints/cot_best.pt
#   4. Ternary QAT (1.58-bit)-> checkpoints/qat_best.pt -> flux_best.pt
#
# Each phase is skipped only if its output exists AND is newer than its input
# checkpoint (via `-nt`) — proof it was actually built FROM that input, not a
# stale leftover from an earlier/abandoned run with the same filename. (A plain
# existence check bit us once: a stale sft_best.pt/cot_best.pt from a run that
# predated this pipeline caused SFT+CoT to be silently skipped, and QAT quantized
# untrained noise for 3 GPU-hours. Provenance, not existence.)
# Delete a checkpoint to force a rerun. torch.compile is DISABLED everywhere:
# it cannot compile the custom SSM scan.

set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# Force unbuffered stdout for every phase. Python block-buffers stdout when it's
# not a TTY (e.g. redirected to a log file) — a run once sat 3h47m into SFT with
# zero log output because of this. This is the fix, applied once for all phases.
export PYTHONUNBUFFERED=1

CKPT=uchi/flux/checkpoints
DATA=uchi/flux/data
COMMON="--no-compile"
PY="${PY:-.venv/bin/python}"

echo "============================================================"
echo "    FLUX FULL PIPELINE (corrected)"
echo "============================================================"

# ── Step 0: pre-tokenize (once) ──
if [ ! -f "$DATA/train.bin" ] || [ ! -f "$DATA/val.bin" ]; then
    echo "Step 0/4: Pre-tokenizing corpus -> $DATA/{train,val}.bin"
    $PY scripts/pretokenize.py --train-tokens 80000000 --val-tokens 1000000
else
    echo "Step 0/4: corpus bins present — skipping tokenization."
fi

# ── Step 1: pre-training ──
if [ ! -f "$CKPT/ckpt_best.pt" ]; then
    echo "Step 1/4: Phase 1 pre-training (116M)"
    $PY -m uchi.flux.train_v2 $COMMON \
        --max_steps 2000 --seq-len 512 --micro-batch 6 --grad-accum 11 \
        --d_model 768 --n_layers 12 --d_state 64 \
        --data-bin "$DATA/train.bin" --val-bin "$DATA/val.bin"
else
    echo "Step 1/4: ckpt_best.pt present — skipping pre-training."
fi
[ -f "$CKPT/ckpt_best.pt" ] || { echo "ERROR: Phase 1 produced no ckpt_best.pt"; exit 1; }

# ── Step 2: SFT ──
if [ ! -f "$CKPT/sft_best.pt" ] || [ ! "$CKPT/sft_best.pt" -nt "$CKPT/ckpt_best.pt" ]; then
    echo "Step 2/4: Phase 2 SFT (instruction following + grounded QA)"
    $PY -m uchi.flux.sft_train $COMMON \
        --base "$CKPT/ckpt_best.pt" --micro-batch 2 --grad-accum 32 --epochs 1
else
    echo "Step 2/4: sft_best.pt present and newer than its base — skipping SFT."
fi
[ -f "$CKPT/sft_best.pt" ] || { echo "ERROR: Phase 2 produced no sft_best.pt"; exit 1; }

# ── Step 3: CoT distillation ──
if [ ! -f "$CKPT/cot_best.pt" ] || [ ! "$CKPT/cot_best.pt" -nt "$CKPT/sft_best.pt" ]; then
    echo "Step 3/4: Phase 3 CoT distillation (real GSM8K teacher traces)"
    $PY -m uchi.flux.cot_distill $COMMON \
        --base "$CKPT/sft_best.pt" --micro-batch 2 --grad-accum 32
else
    echo "Step 3/4: cot_best.pt present and newer than its base — skipping CoT."
fi
[ -f "$CKPT/cot_best.pt" ] || { echo "ERROR: Phase 3 produced no cot_best.pt"; exit 1; }

# ── Step 4: ternary QAT (mixed CoT + general-text recovery — see qat_train.py
# module docstring: recovering on general text ALONE erases the reasoning
# format CoT just installed, so this mixes in real CoT-formatted batches too) ──
if [ ! -f "$CKPT/qat_best.pt" ] || [ ! "$CKPT/qat_best.pt" -nt "$CKPT/cot_best.pt" ]; then
    echo "Step 4/4: Phase 4 ternary QAT (1.58-bit, CoT-preserving)"
    $PY -m uchi.flux.qat_train $COMMON \
        --base "$CKPT/cot_best.pt" --micro-batch 2 --grad-accum 32 --seq-len 384 \
        --data-bin "$DATA/train.bin" --val-bin "$DATA/val.bin"
else
    echo "Step 4/4: qat_best.pt present and newer than its base — skipping QAT."
fi

# ── Canonical final checkpoint the Proposer loads by default ──
FINAL="$CKPT/qat_best.pt"
[ -f "$FINAL" ] || FINAL="$CKPT/cot_best.pt"
cp -f "$FINAL" "$CKPT/flux_best.pt"

echo "============================================================"
echo "    PIPELINE FINISHED — canonical model: $CKPT/flux_best.pt (from $FINAL)"
echo "============================================================"
