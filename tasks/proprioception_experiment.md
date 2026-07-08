# FLUX Proprioception — experimental idea, being integrated

**Status: pilot validated with corrected methodology (see "Corrected
methodology" and "Second validation pass" sections below) — Approach 1
now shows a clean, calibrated separation and is being wired into the
main pipeline. Approach 2 remains an unvalidated, lower-priority
alternative. Written up in depth so this can be picked back up without
re-deriving it.**

## Corrected methodology (important — supersedes the first pilot's exact numbers)

The first pilot in this file used a "Context:\n{ctx}\n\nQuestion: {q}\n
Answer:" template to build the reference set and to run the decisive
correct-vs-wrong-claim test. That does not match either of the two real
prompt shapes in this codebase: `FluxProposer._ANSWER` (`"Context:\n{ctx}
\n\nUsing ONLY the context above..."`) or, more importantly, the actual
evidence-grounded generation path `GenerateAndGround._candidates()`
uses (`propose(..., think=True)`), which sends FLUX **the raw question
alone, no context wrapper at all** — confirmed directly in
`uchi/proposer.py`'s `propose()`: grounding against evidence happens
downstream via the oracle, never by putting evidence text in FLUX's
prompt. This is deliberate, already-documented behavior (an earlier fix
in this project's history found wrapping the CoT path in the SFT-style
template caused degenerate output), not a bug.

Consequence: proprioception is a **pre-generation, question-only gate**
— "before FLUX even attempts an answer, is this question's shape/topic
familiar at all?" — not a check over (evidence, claim) pairs, which was
the first pilot's mistaken framing. This makes the "not a replacement
for the verifier" argument even more direct than before: proprioception
never even sees a generated claim, so it structurally cannot check
whether one is factually correct. `uchi/proprioception.py` and
`scripts/fit_proprioception.py` implement the corrected version.

## Second validation pass — real questions, real training sources, calibrated threshold

Reference set: 480 real questions sampled fairly across FLUX's actual
CoT training sources (GSM8K, OpenOrca, Magicoder, CommitPackFT), in the
exact raw-question shape `think=True` actually uses. 120 held out for
threshold calibration (95th percentile of their distances — not a
reused default from a different model, which was the first pilot's
other flaw).

| Prompt | Distance | Flagged unfamiliar? |
|---|---|---|
| "What does the following code change do..." (in-dist.) | 60.19 | No |
| "You are tasked with implementing a function..." (in-dist.) | 50.32 | No |
| "You are tasked with creating a function..." (in-dist.) | 58.95 | No |
| "You are tasked with implementing a parser..." (in-dist.) | 51.65 | No |
| Chinese quantum-physics text (genuinely OOD) | 227.96 | Yes |
| Gibberish string (genuinely OOD) | 214.57 | Yes |
| Symbol string (genuinely OOD) | 201.10 | Yes |

Calibrated threshold: 123.30, sitting cleanly between the in-distribution
cluster (~50–60) and the OOD cluster (~201–228) — roughly 4x separation,
markedly cleaner than the first pilot's 2.6x on an undersized, wrongly-
shaped reference set. **This is a real, validated result, not a promising
lead anymore** — worth actually integrating, with the same additive-only
discipline as everything else (a "familiar" verdict is silence, not an
acceptance; only "unfamiliar" does anything, and what it should do is
still an open design choice — see below).

## The core idea

FLUX (the proposer) currently has no internal sense of whether it's being
asked about something familiar or something structurally unlike its
training data. It will generate with the same apparent fluency whether
it's on solid ground or completely out of its depth. "Proprioception" here
means giving the proposer some analogue of that sense — named after the
biological term for a body's sense of its own position/state without
external input, since the goal is an *internal* signal, not an externally
supplied check.

## This is NOT a replacement for the verifier — read this before anything else

This point is important enough to repeat here even though it's already
been said in conversation, because the whole reason this file exists is so
someone (possibly future-you, without the conversation in front of them)
doesn't pick this up and conclude the verifier can be dropped.

The verifier answers: **"does this specific generated claim contradict
this specific retrieved evidence?"** Proprioception answers a different
question: **"is this general topic/prompt shape familiar to FLUX?"** These
are orthogonal axes of risk.

Concrete illustration — the motivating case from Item 17's original design
(the "$330m vs $500m" building-height mixup): that claim is about an
entirely ordinary, completely in-distribution kind of question (a
building's height). Proprioception, even if it works perfectly, would
correctly conclude "yes, this is exactly the kind of question FLUX is
trained on" — and it would be right about that — and it still would not
catch the error, because the error isn't a familiarity problem. It's a
grounding problem: does *this specific number* match *this specific
evidence*. Only the verifier checks that. No amount of "I recognize this
kind of question" can substitute for "I checked this specific fact against
its specific source." The same way a subject-matter expert who's
completely comfortable with a topic can still misstate one specific date
or swap two specific names — deep familiarity doesn't prevent a wrong
specific claim within it.

**If this idea works, it is a signal to layer on top of the existing
cascade — additional caution about *when* to trust generation at all,
or how many self-consistency votes to spend — not a substitute for
verifying *what* was generated.**

## Two candidate mechanisms

### Approach 1: Mahalanobis OOD detector on FLUX's own hidden states

Structurally the same technique already built and used for the verifier's
`OODDetector` (`uchi/flux/verifier_model.py`) — Mahalanobis distance over a
pooled representation, fit on a reference set, checked against a threshold
at inference time. Applied here to FLUX's own hidden state instead of the
entailment classifier's.

**The real, unverified assumption**: the verifier's version works (to the
extent it's been checked at all) because it sits on a *classification*
objective with a pooling step designed to produce a clean summary
representation for exactly this purpose. FLUX is trained on next-token
prediction — nothing guarantees its hidden states organize into a space
where "distance from training data" is meaningful. This might just work,
or it might not; that's the entire reason to test it rather than assume it.

**Design decisions that matter, not just implementation details:**
- *Which representation to pool*: the prompt/context's hidden state,
  before generation starts — not something mid-generation. Keeps this
  structurally analogous to the verifier's "check once, before trusting"
  shape rather than a much harder per-token version.
- *What counts as "in-distribution"*: FLUX trained across four very
  different phases (raw web text, SFT chat format, CoT reasoning traces,
  QAT mix). The reference set has to be real (context, question) pairs in
  the actual inference-time prompt shape
  (`GenerateAndGround`'s `_ANSWER` template), not raw pretraining text —
  fitting on the wrong phase's distribution would make everything from
  later phases look falsely OOD.

**Verification step, before trusting any result**: construct genuinely
held-out in-distribution prompts (same shape/topic as the reference set,
just not literally in it) and genuinely out-of-distribution prompts (a
different language, gibberish, a bizarre symbol sequence) and check
whether the fitted detector's distances actually separate the two groups.
If they don't, that's a real, valid outcome — it would mean next-token-
prediction hidden states aren't structured for this, the same way
`AnomalyDetector` turned out not to discriminate for numeric plausibility
earlier this session. Don't force it if this comes back negative.

### Approach 2: self-consistency strength via semantic clustering

Real research basis: semantic entropy (Kuhn et al. 2023) and SelfCheckGPT
(Manakul et al. 2023) — sample a model multiple times and check whether
the answers actually agree *in meaning*. Low agreement across resamples is
used as an uncertainty signal in that literature, validated on large,
capable LLMs.

Mechanically, this reuses infrastructure that already exists: dynamic-N
self-consistency voting already generates multiple candidates; it just
currently picks the plurality winner by exact string match
(`Counter(valid_candidates)`) without using the *strength* of that
consensus as a signal, and without any notion of semantic equivalence
("the tower is 330 meters tall" and "it stands 330m tall" currently count
as two different answers).

**The real, unverified assumption here is different from Approach 1, and
it's small-model-specific**: for a large, capable model, low output
diversity across resamples correlates with genuine confidence. For a
small, capacity-limited model like FLUX, low diversity might just reflect
a narrow output distribution in general — the model producing similar
text because it doesn't have the range to produce anything else, not
because it actually understands the topic. Consensus and confidence could
come apart in exactly the cases this is meant to catch. This has to be
checked at FLUX's actual scale, not assumed to transfer from large-LLM
research.

**Verification step, before trusting any result**: run this against
questions known to be well-covered (expect high consensus) and questions
known to be genuinely hard or OOD (expect low consensus), and confirm the
metric actually tracks that distinction for FLUX specifically.

## Experiment run

A first, honest, small-scale test of both — not a production benchmark,
run on CPU specifically so it wouldn't compete with verifier training that
was using the GPU at the time. Script:
`scratchpad/proprioception_experiment.py` (this session's scratchpad;
copy elsewhere if it needs to persist past session cleanup).

## Results (real, both approaches, first pass)

**One real bug caught mid-experiment, worth recording so it isn't
repeated**: the first run of Approach 2 embedded FLUX's full
`generate()` output, which returns prompt+continuation concatenated, not
just the new tokens. Since all 5 resamples of a given prompt share the
identical (long) prompt prefix, the shared prefix dominated the
skip-gram mean-pooled embedding and made every sample look artificially
similar regardless of what was actually generated — every single test
case came back with consensus=1.00, including the gibberish prompts,
which was the tell that something was wrong rather than a real finding.
Fixed by slicing the output to only the tokens generated beyond the
prompt length before embedding. Results below are from the corrected run.

### Approach 1 — Mahalanobis on FLUX's own hidden states

| Prompt | Distance | Flagged OOD (threshold=3.0) |
|---|---|---|
| "When did the Titanic sink?" (held-out in-dist.) | 260.84 | yes |
| "What do vaccines do?" (held-out in-dist.) | 295.48 | yes |
| "What is the largest hot desert?" (held-out in-dist.) | 329.75 | yes |
| "什么是量子纠缠?" (Chinese, genuinely OOD) | 949.36 | yes |
| "vex quorlth?" (gibberish, genuinely OOD) | 758.67 | yes |
| symbol string (genuinely OOD) | 633.23 | yes |

**Everything got flagged** at the default threshold (3.0, inherited
as-is from the verifier's `OODDetector` config) — that threshold is not
calibrated for FLUX's hidden-state space and needs its own fit, not
reuse. But look at the actual distances, not the flag: held-out
in-distribution prompts cluster at ~260–330, genuinely OOD prompts
cluster at ~630–950 — roughly **2.6x separation on average**, in exactly
the right direction. That's a real, promising signal.

**Caveat that has to be resolved before trusting this at all**: the
reference set here was only 30 examples, fitting a full covariance
matrix in a 768-dimensional space. That's severely underdetermined —
you generally want on the order of *the dimensionality* in samples,
ideally several times that, to get a reliable covariance estimate. With
only 30 points, the fitted covariance is likely dominated by the
regularization term (`OODDetector`'s `+ eye(d) * 1e-3`) rather than real
structure. The direction of the signal is encouraging; the magnitude and
reliability aren't trustworthy yet. Next step, if this gets picked up:
refit on hundreds-to-low-thousands of real (context, question) pairs
sampled from the actual SFT/CoT training data, then re-run this same
separation test before touching the threshold.

### Approach 2 — self-consistency strength via semantic clustering (corrected)

| Prompt | Consensus (largest cluster / 5 samples) |
|---|---|
| Eiffel Tower (easy) | 0.40 |
| Water boiling point (easy) | 0.60 |
| Gibberish (hard) | 0.20 |
| Symbol string (hard) | 0.40 |

Average "easy" = 0.50, average "hard" = 0.30 — directionally correct
(gibberish produced the single lowest consensus of all four, and the
generated samples for it were genuinely scattered with no thematic
overlap at all), but noisy at this scale: the symbol-string "hard" case
tied the Eiffel Tower "easy" case at 0.40, so there's real overlap, not a
clean separation. With only 5 samples per prompt and 2 prompts per
category, this is barely a pilot, not a validated result.

**Worth being honest about why the noise might be inherent, not just
small-sample-size**: reading the actual generated text (see the raw log),
none of it is very fluent or reliably on-topic even for the "easy"
prompts — this is a 64M-parameter, still-early-capability model. The
signal this technique is trying to measure (does the model converge on
similar answers *because it understands the topic*) may simply be
harder to detect while the model's overall coherence is this low, since
general incoherence adds noise regardless of true topic familiarity.
This might get cleaner as FLUX's capability improves through later
training, or it might not — that's untested.

### Where this leaves it

Approach 1 shows the stronger, cleaner-looking signal of the two, but
both need meaningfully more data before either could be trusted for
anything: Approach 1 needs a properly-sized reference set (hundreds+,
not 30) and a recalibrated threshold; Approach 2 needs more samples per
prompt and more prompts per category before the noise level is
distinguishable from a real ceiling on what this technique can measure
at FLUX's current scale.

## Decisive follow-up test: does this replace the verifier? (No — empirically, not just by argument)

Ran one more targeted test on Approach 1's fitted detector, using the
*exact* motivating case from Item 17's original design: same evidence,
same question, one correct claim and two wrong claims about the same
familiar topic (a building's height).

| Claim | Distance |
|---|---|
| "The Eiffel Tower is 330 meters tall." (correct) | 279.65 |
| "The Eiffel Tower is 25 meters tall." (wrong) | 279.22 |
| "The Eiffel Tower is 3300 meters tall." (wrong) | 286.45 |

**All three differ by under 3%** — statistically indistinguishable given
the noise already established in this same detector. Proprioception
shows *zero* discriminative power between a correct and an incorrect
claim about a topic FLUX is completely familiar with, because all three
claims are equally familiar in shape and domain — the thing that differs
between them (whether "25," "330," or "3300" is the right number) is
exactly the axis proprioception was never designed to see. This confirms
directly, not just by argument, that proprioception and the verifier
answer different questions and one cannot stand in for the other. **The
verifier remains required regardless of how well proprioception performs
once properly validated.**

## Open questions for whoever picks this up next

1. **Partially answered by the pilot above**: Approach 1's distances did
   separate in-distribution from OOD prompts (~2.6x), directionally
   correct — but the reference set (30 examples for a 768-dim covariance
   fit) is too small to trust yet. Next step is a proper-sized refit, not
   a fresh question of "does this work at all."
2. **Partially answered**: Approach 2's consensus strength was
   directionally correct on average but noisy at n=5 samples / 2 prompts
   per category, with real overlap between categories. Open question is
   whether more samples resolve the noise, or whether FLUX's current
   capability ceiling is the actual limit — untested either way.
3. If either works reliably at proper scale: what's the actual *action*
   on a positive signal —
   outright abstention, a hedge added to the answer, or feeding into
   `TaskConfigCache`'s dynamic-N decision to spend more self-consistency
   votes on prompts flagged as unfamiliar? This is a real design choice
   with a coverage-vs-caution tradeoff, mirroring the one already made
   for the verifier's own OOD gate.
4. Whichever direction this goes, re-confirm before building further: this
   augments the cascade, it does not replace the verifier.
