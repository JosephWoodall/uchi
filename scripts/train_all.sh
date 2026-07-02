#!/bin/bash
# train_all.sh — Master offline script for full FLUX training run.
#
# Runs:
#   1. Pre-training (OpenWebText, 133M params, 10k steps)
#   2. SFT (SQuAD + Dolly + OpenHermes)
#   3. CoT Distillation
#   4. Ternary QAT (1.58-bit)
#
# Gracefully stops if any phase fails.

set -e

# Enable PyTorch memory expansion for large models to reduce fragmentation OOMs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================================"
echo "    FLUX FULL PIPELINE — OFFLINE RUN"
echo "============================================================"
echo "Step 1/4: Phase 1 Pre-training (133M params)"
python -m uchi.flux.train_v2 --max_steps 10000 --micro-batch 2 --grad-accum 32 --d_model 768 --n_layers 12 --d_state 64
echo "✔ Phase 1 Complete."
echo ""

# The best checkpoint from Phase 1 is pretrain_best.pt
BASE_CKPT="uchi/flux/checkpoints/pretrain_best.pt"

if [ ! -f "$BASE_CKPT" ]; then
    echo "ERROR: Phase 1 did not produce $BASE_CKPT"
    exit 1
fi

echo "Step 2/4: Phase 2 SFT (Instruction Following & Reasoning)"
python -m uchi.flux.sft_train --base "$BASE_CKPT" --micro-batch 2 --grad-accum 32 --epochs 1
echo "✔ Phase 2 Complete."
echo ""

# SFT produces sft_best.pt
SFT_CKPT="uchi/flux/checkpoints/sft_best.pt"

if [ ! -f "$SFT_CKPT" ]; then
    echo "ERROR: Phase 2 did not produce $SFT_CKPT"
    exit 1
fi

echo "Step 3/4: Phase 3 Chain-of-Thought Distillation"
python -m uchi.flux.cot_distill --base "$SFT_CKPT" --micro-batch 2 --grad-accum 32
echo "✔ Phase 3 Complete."
echo ""

# CoT produces cot_best.pt
COT_CKPT="uchi/flux/checkpoints/cot_best.pt"

if [ ! -f "$COT_CKPT" ]; then
    echo "ERROR: Phase 3 did not produce $COT_CKPT"
    exit 1
fi

echo "Step 4/4: Phase 4 Ternary Quantization-Aware Training"
python -m uchi.flux.qat_train --base "$COT_CKPT" --micro-batch 2 --grad-accum 32
echo "✔ Phase 4 Complete."
echo ""

echo "============================================================"
echo "    FLUX TRAINING PIPELINE FINISHED SUCCESSFULLY"
echo "    Final model: uchi/flux/checkpoints/qat_best.pt"
echo "============================================================"
