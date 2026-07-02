#!/bin/bash
# train_all.sh — Full FLUX pipeline, corrected & resumable.
#
#   0. Pre-tokenize corpus  -> uchi/flux/data/{train,val}.bin   (GPU-bound fast path)
#   1. Pre-training (116M)   -> checkpoints/ckpt_best.pt
#   2. SFT (SQuAD+Dolly+Code)-> checkpoints/sft_best.pt
#   3. CoT distill (GSM8K)   -> checkpoints/cot_best.pt
#   4. Ternary QAT (1.58-bit)-> checkpoints/qat_best.pt -> flux_best.pt
#
# Each phase is skipped if its output already exists, so a completed phase (e.g.
# the live Phase-1 run) is not redone. Delete a checkpoint to force a rerun.
# torch.compile is DISABLED everywhere: it cannot compile the custom SSM scan.

set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

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
if [ ! -f "$CKPT/sft_best.pt" ]; then
    echo "Step 2/4: Phase 2 SFT (instruction following + grounded QA)"
    $PY -m uchi.flux.sft_train $COMMON \
        --base "$CKPT/ckpt_best.pt" --micro-batch 2 --grad-accum 32 --epochs 1
else
    echo "Step 2/4: sft_best.pt present — skipping SFT."
fi
[ -f "$CKPT/sft_best.pt" ] || { echo "ERROR: Phase 2 produced no sft_best.pt"; exit 1; }

# ── Step 3: CoT distillation ──
if [ ! -f "$CKPT/cot_best.pt" ]; then
    echo "Step 3/4: Phase 3 CoT distillation (real GSM8K teacher traces)"
    $PY -m uchi.flux.cot_distill $COMMON \
        --base "$CKPT/sft_best.pt" --micro-batch 2 --grad-accum 32
else
    echo "Step 3/4: cot_best.pt present — skipping CoT."
fi
[ -f "$CKPT/cot_best.pt" ] || { echo "ERROR: Phase 3 produced no cot_best.pt"; exit 1; }

# ── Step 4: ternary QAT ──
if [ ! -f "$CKPT/qat_best.pt" ]; then
    echo "Step 4/4: Phase 4 ternary QAT (1.58-bit)"
    $PY -m uchi.flux.qat_train $COMMON \
        --base "$CKPT/cot_best.pt" --steps 1500 --micro-batch 6 --grad-accum 11 --seq-len 256 \
        --data-bin "$DATA/train.bin" --val-bin "$DATA/val.bin"
else
    echo "Step 4/4: qat_best.pt present — skipping QAT."
fi

# ── Canonical final checkpoint the Proposer loads by default ──
FINAL="$CKPT/qat_best.pt"
[ -f "$FINAL" ] || FINAL="$CKPT/cot_best.pt"
cp -f "$FINAL" "$CKPT/flux_best.pt"

echo "============================================================"
echo "    PIPELINE FINISHED — canonical model: $CKPT/flux_best.pt (from $FINAL)"
echo "============================================================"
