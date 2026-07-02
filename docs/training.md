# Training FLUX — Pipeline & Artifacts

FLUX is the **Proposer** in the FLUX + Uchi architecture: a small, from-scratch
SSM/attention hybrid language model (~116M parameters) that Uchi's verifier
gates. This page documents how it is trained and, importantly, **what file you
end up with**.

> FLUX is a genuinely small model. Training produces a coherent proposer that
> feeds the verifier — not a frontier LLM. MMLU / SWE-bench / ARC-Challenge are
> tracked as a dashboard, not conquered; at this scale they stay near their
> baselines. The value is a grounded, honest system, not a leaderboard number.

## The one-liner

```bash
bash scripts/train_all.sh
```

This runs the full pipeline and, at the end, writes the **final artifact**:

```
uchi/flux/checkpoints/flux_best.pt
```

`flux_best.pt` is what `Uchi()` loads to drive the proposer. Nothing else is
required — construct `Uchi()` and it picks up the trained weights automatically.

## The four phases

Each phase reads the previous phase's best checkpoint and writes its own. Phases
are **skipped if their output already exists**, so an interrupted run resumes.

| # | Phase | Script | Data | Output |
|---|-------|--------|------|--------|
| 0 | Tokenize | `scripts/pretokenize.py` | FineWeb-Edu → memmap | `uchi/flux/data/{train,val}.bin` |
| 1 | Pre-training | `uchi/flux/train_v2.py` | `train.bin` | `checkpoints/ckpt_best.pt` |
| 2 | SFT | `uchi/flux/sft_train.py` | SQuAD + Dolly + CodeAlpaca | `checkpoints/sft_best.pt` |
| 3 | CoT distillation | `uchi/flux/cot_distill.py` | GSM8K teacher traces | `checkpoints/cot_best.pt` |
| 4 | Ternary QAT | `uchi/flux/qat_train.py` | `train.bin` | `checkpoints/qat_best.pt` |

**Final step:** `train_all.sh` copies the last successful phase
(`qat_best.pt`, or `cot_best.pt` if QAT is skipped) to `flux_best.pt` — the
canonical model the Proposer loads.

### What each phase does

1. **Pre-training** learns language from FineWeb-Edu (educational-filtered web).
   This is the bulk of training and where coherence comes from.
2. **SFT** teaches instruction-following and grounded answering-from-context
   (the shape `FluxProposer` uses at inference).
3. **CoT distillation** imitates *real* GSM8K worked-solution reasoning traces
   (static teacher distillation — a small model can't bootstrap reasoning from
   self-play).
4. **Ternary QAT** (optional) quantizes weights to 1.58-bit for efficiency.
   Skippable; on a small model it can trade quality for size, so `cot_best.pt`
   is a valid full-precision final model.

## Performance notes

- **GPU-bound throughput** requires the pre-tokenized `.bin` (step 0). Training
  directly from a streaming/tokenizing loop starves the GPU (~10× slower).
- **`torch.compile` is disabled** (`--no-compile`) — it cannot compile the
  custom SSM scan. All phases run eager.
- Default config: `d_model=768, n_layers=12, d_state=64`, seq 512, bf16, gradient
  checkpointing on (fits a 12 GB GPU at micro-batch 6).

## Loading the result

```python
from uchi import Uchi
u = Uchi()          # boots FLUX from uchi/flux/checkpoints/ (qat_best → cot_best → sft_best)
print(u.ask("..."))  # FLUX proposes, Uchi verifies
```

If **no** checkpoint is present, `Uchi()` still works: the proposer degrades to
`None` and the verifier falls back to grounded extraction / honest abstention.
FLUX makes the system *more capable*; the verifier keeps it *honest either way*.
