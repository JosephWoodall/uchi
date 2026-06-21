left off here: 
❯ what would be the general solution?

● The product/mixture choice is the wrong abstraction. Both are
  fixed strategies; what you actually want is a single strategy
  that self-selects between them based on how confident the trees
  are right now.

  The core observation:

  - Product-of-experts works when trees have strong individual
  opinions and agree — it sharpens the signal
  - Mixture works when trees are uncertain — it pools weak signals
  without punishing disagreement
  - The trees themselves tell you which situation you're in: their
  mean confidence

  General solution: adaptive voting

  At each prediction step, compute the mean confidence of the
  active trees. Use that as a continuous blend weight between
  product and mixture:

  α  =  mean confidence of active trees

  final distribution = α · product_distribution + (1-α) ·
  mixture_distribution

  When trees are individually confident (α → 1): vote like product
  — agreement required to win.
  When trees are individually uncertain (α → 0): vote like mixture
  — pool everything.

  No hyperparameter. No per-dataset tuning. The data tells you
  which regime you're in.

  Why this solves the three failure cases:

  ┌────────────────────┬──────────┬──────────────────────────┐
  │        Case        │  Tree    │       What α does        │
  │                    │  confs   │                          │
  ├────────────────────┼──────────┼──────────────────────────┤
  │ Airline —          │          │ α → product, sharpens    │
  │ structured,        │ High     │ the signal               │
  │ predictable        │          │                          │
  ├────────────────────┼──────────┼──────────────────────────┤
  │ Text — high vocab, │          │ α → mixture, includes    │
  │  sparse long-k     │ Low      │ uncertain trees without  │
  │ trees              │          │ punishing them           │
  ├────────────────────┼──────────┼──────────────────────────┤
  │ PRNG — spurious    │          │ α blends, neither fully  │
  │ agreement on noise │ Moderate │ committing nor fully     │
  │                    │          │ diffusing                │
  └────────────────────┴──────────┴──────────────────────────┘

  The PRNG case is the subtlest: the trees do memorize noise
  patterns and sometimes spuriously agree. Adaptive voting doesn't
  fully solve that — it softens it, because moderate confidence
  produces a partial product that only partially amplifies the
  spurious agreement. The real fix for PRNG is that it's
  fundamentally unpredictable and any above-random performance is
  overfitting. That's a data property, not a voting problem.

  Want me to implement voting='adaptive' as the new default and
  re-run?

✻ Cogitated for 59s

❯ lets do it; then please test
  ⎿  You've hit your session limit · resets 3:20am
     (America/New_York)

✻ Churned for 0s
