#!/bin/bash
set -e

echo "========================================"
echo " UCHI FLUX TRAINING PIPELINE "
echo "========================================"

echo "Phase 1: Proof of Concept (TinyStories, BF16, Base Model)"
python3 uchi/flux/train_v2.py

echo "Phase 2: Supervised Fine-Tuning (Instruction Following)"
python3 uchi/flux/sft_train.py

echo "Phase 3: Chain-of-Thought Distillation (Reasoning)"
python3 uchi/flux/cot_distill.py

echo "Phase 4: Quantization-Aware Training (Ternary 1.58-bit)"
python3 uchi/flux/qat_train.py

echo "========================================"
echo " Training Complete. Best model saved to uchi/flux/checkpoints/"
echo "========================================"
