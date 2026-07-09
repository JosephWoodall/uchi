# This Repo's North Star

**One sentence:** Uchi separates the capacity to generate plausible text (a
small, from-scratch neural proposer) from the authority to assert it as true
(an independent, auditable, strictly additive verification cascade) — bound
together by a fully deterministic recall/credibility mechanism (ODUSP) and a
factual-grounding requirement enforced by retrieval, so the system can
synthesize novel answers without ever asserting a claim it cannot trace to
evidence.

## The Core Intuition

Three components, cleanly separated by construction, not by convention:

1. **ODUSP — the deterministic floor.** A credibility-weighted context trie
   (CTW-style multi-order blending, MWU credibility updates). Exact/near
   recall when it has seen something like this before, with calibrated
   confidence that degrades automatically for patterns that stop predicting
   reliably. Reused this session for a second purpose beyond recall —
   recommending how much self-consistency compute a question's structural
   shape warrants — without changing the trie mechanism itself.

2. **FLUX — the proposer, never trusted alone.** A from-scratch SSM/attention
   hybrid (`HybridTSSM`), trained via standard next-token cross-entropy
   across four phases (pretrain → SFT → CoT → QAT). It generates candidate
   answers. It is never the thing that decides whether an answer is true.

3. **The verification cascade — the authority, layered additively.** A
   deterministic word-overlap gate is the floor; an entailment classifier
   and a numeric-plausibility check sit on top as strictly additive vetoes —
   either can turn a pass into a reject, neither can turn a rejection into an
   acceptance. This is the one non-negotiable invariant defended against
   every proposed shortcut this project has considered: shared embeddings,
   joint training, RL-tuning the proposer against the verifier's reward, and
   a shared continuous latent space were all explored and rejected
   specifically because each one would let the generator and its judge share
   a failure mode.

If nothing retrieved grounds an answer, Uchi abstains. That is the leash:
synthesis is trustworthy only because it's checked by something structurally
incapable of agreeing with the proposer for the proposer's own reasons.

## What Changed From the Prior North Star

The previous version of this document (dated June 30) described a "256D
vector-symbolic manifold" performing HRR/circular-convolution binding of
query⊗evidence, and SSM "policy/value heads." None of that exists anywhere
in the current codebase — verified by grepping `uchi/` for HRR, circular
convolution, policy/value heads, and VSA: zero matches. It described a
design direction that was apparently superseded before this document was
updated to match. The actual architecture that's been built, defended, and
adversarially stress-tested since is the proposer/oracle/ODUSP separation
above — simpler than what the old document claimed, and considerably more
tested (the "500m vs 330m" false-accept case that motivated the entailment
classifier, the shared-latent-space/Langevin-dynamics debate, the
reward-hacking analysis for a designed-but-unbuilt MCTS verifier cascade).

## State-of-the-Art Grounding

- **CTW** — Willems, Shtarkov, Tjalkens 1995. ODUSP's recall mechanism,
  unchanged.
- **MWU** — Arora, Hazan, Kale 2012. ODUSP's credibility-update rule — a
  regret-bounded depth-selection scheme, not a heuristic.
- **NLI-based factual consistency checking** — Honovich et al. 2022
  ("TRUE"), and the broader retrieval-augmented-generation-with-verification
  line (Lewis et al. 2020). The entailment classifier's actual job: catching
  semantic contradictions no word-overlap check can see.
- **Mahalanobis OOD detection** — Lee et al. 2018. Gates the entailment
  veto: an out-of-distribution input's classifier judgment is treated as "no
  opinion," never trusted outright.
- **Conservative/additive ensembling** — "any layer can reject, none can
  override a rejection" is a precision-first ensemble design, not a novel
  mechanism, chosen because a hallucination that slips through costs more
  than an unnecessary abstention.

## Why This Beats Alternatives

| Alternative | Why rejected |
|---|---|
| Frontier pretrained LLM | Explicit, standing constraint: no LLMs. Also opaque and unauditable. |
| End-to-end neural verifier, no deterministic floor | A network can be confidently wrong; nothing bounds it. First idea considered this project's verifier-upgrade work, rejected first. |
| Joint/shared-weight proposer+verifier training, or a shared latent space | Reintroduces correlated failure modes — explored at length (frozen shared embeddings, Langevin dynamics over a joint space) and rejected each time for the same underlying reason. |
| RL-tuning the proposer against the verifier's reward | Unbounded iterations to find and permanently bake in a verifier blind spot — the textbook reward-hacking failure mode. |

## Drift Check

Every change must answer: **does this preserve the separation between what
generates and what has authority to assert truth?** A change that lets the
verifier's agreement resurrect a deterministic rejection, or lets the
proposer and verifier share weights, gradients, or a latent space, violates
the North Star regardless of how it's framed. A change that adds a new,
independent signal — another veto layer, a faster recall path, a cheaper
search over already-verified candidates — aligns with it.
