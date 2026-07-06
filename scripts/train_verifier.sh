#!/bin/bash
# train_verifier.sh — Verifier (entailment classifier) training pipeline.
#
#   1. Train EntailmentClassifier from scratch on MNLI+SNLI
#      -> checkpoints/verifier/verifier_best.pt
#
# This is a SEPARATE model from the FLUX proposer -- its own embedding
# table, its own checkpoint directory, never touches uchi/flux/checkpoints/
# *.pt (the proposer's checkpoints). Safe to run any time the GPU is free;
# do not launch this while a proposer training phase (train_all.sh) is
# using the GPU -- check `nvidia-smi` first. See tasks/0.4.0 Itemized
# Deliverables.md Item 17 for the full design (why an entailment classifier
# and not a second proposer, why it's additive-only, held-out validation
# required before it ever vetoes anything live).

set -e
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

CKPT=uchi/flux/checkpoints/verifier
PY="${PY:-.venv/bin/python}"

echo "============================================================"
echo "    VERIFIER TRAINING PIPELINE (entailment classifier)"
echo "============================================================"

if [ ! -f "$CKPT/verifier_best.pt" ]; then
    echo "Step 1/1: Training EntailmentClassifier on MNLI+SNLI"
    $PY -m uchi.flux.verifier_train \
        --max-examples 200000 --epochs 2 \
        --micro-batch 8 --grad-accum 8 --seq-len 192 \
        --d-model 256 --n-layers 8 --d-state 32 \
        --checkpoint-dir "$CKPT"
else
    echo "Step 1/1: verifier_best.pt present — skipping (delete it to force a rerun)."
fi

[ -f "$CKPT/verifier_best.pt" ] || { echo "ERROR: training produced no verifier_best.pt"; exit 1; }

echo "============================================================"
echo "    VERIFIER TRAINING FINISHED: $CKPT/verifier_best.pt"
echo "    NOT wired into the live oracle yet -- held-out validation"
echo "    against an adversarial set is required first (Item 17,"
echo "    Expected Benchmarks #7). Load explicitly via:"
echo "      EntailmentChecker.load('$CKPT/verifier_best.pt')"
echo "============================================================"
