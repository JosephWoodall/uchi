"""
generative.py
=============
Generative capabilities for the Universal Sequence Predictor.

The predictor already stores the full conditional distribution P(next | context).
Generation is sampling from that distribution instead of taking argmax.

Sampling controls
-----------------
temperature : float  (default 1.0)
    < 1.0 → sharper / more deterministic
    > 1.0 → flatter / more creative / more random
    The distribution is raised to (1/T) then renormalised.

top_k : int | None
    Keep only the k most probable tokens before sampling.

top_p : float | None  (nucleus sampling)
    Keep the smallest set of tokens whose cumulative probability ≥ top_p,
    then sample within that nucleus.  Balances diversity and coherence
    better than top_k on large vocabularies.

Classes
-------
  SequenceGenerator   — auto-regressive text/symbol generation
  TabularGenerator    — synthetic tabular row generation (joint distribution)
  TimeSeriesGenerator — sampled multivariate time series generation
"""

from __future__ import annotations

import math
import random
from typing import Any

from .predictor  import UniversalPredictor
from .discretize import FeatureDiscretizer, LabelEncoder, _to_rows
from .tabular    import _make_predictor, _apply_order, _build_orders, _LABEL_NS
from .long_term_store import LongTermStore
from .online_tokenizer import OnlineTokenizer
from .omni_tokenizer import OmniTokenizer

try:
    from sklearn.base import BaseEstimator
    _SKLEARN = True
except ImportError:
    class BaseEstimator: pass
    _SKLEARN = False

# Lazy imports for optional components
def _get_online_tokenizer():
    from .online_tokenizer import OnlineTokenizer
    return OnlineTokenizer

def _get_dual_predictor():
    from .dual_predictor import DualPredictor
    return DualPredictor

def _get_long_term_store():
    from .long_term_store import LongTermStore
    return LongTermStore


# ══════════════════════════════════════════════════════════════════════════════
# Shared sampling primitive
# ══════════════════════════════════════════════════════════════════════════════

def _sample_dist(
    dist:        dict,
    temperature: float,
    top_k:       int   | None,
    top_p:       float | None,
    rng:         random.Random,
) -> Any:
    """
    Sample one token from a probability distribution.

    Order of operations: temperature → top_k → top_p → normalise → sample.
    """
    if not dist:
        return None

    tokens = list(dist.keys())
    probs  = list(dist.values())

    if temperature == 0.0:
        # Deterministic argmax
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        return tokens[best_idx]

    # Temperature: reshape p_i ← p_i^(1/T)
    if temperature != 1.0 and temperature > 0:
        inv_t = 1.0 / temperature
        probs = [p ** inv_t for p in probs]

    # Top-k: zero out all but the k highest
    if top_k is not None and top_k < len(probs):
        threshold = sorted(probs, reverse=True)[top_k - 1]
        probs = [p if p >= threshold else 0.0 for p in probs]

    # Nucleus (top-p): keep smallest prefix of sorted tokens summing to ≥ top_p
    if top_p is not None:
        order     = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        cumsum    = 0.0
        keep: set = set()
        for i in order:
            keep.add(i)
            cumsum += probs[i]
            if cumsum >= top_p:
                break
        probs = [p if i in keep else 0.0 for i, p in enumerate(probs)]

    total = sum(probs)
    if total < 1e-12:
        probs = [1.0 / len(probs)] * len(probs)
    else:
        probs = [p / total for p in probs]

    # Inverse CDF sample
    r      = rng.random()
    cumsum = 0.0
    for token, p in zip(tokens, probs):
        cumsum += p
        if r <= cumsum:
            return token
    return tokens[-1]


def _generate_from_predictor(
    p:           UniversalPredictor,
    n_tokens:    int,
    seed:        list | None,
    temperature: float,
    top_k:       int   | None,
    top_p:       float | None,
    rng:         random.Random,
    stop_tokens: set | None = None,
    tokenizer=None,
    long_term_store=None,
) -> list:
    """
    Auto-regressively sample n_tokens from predictor p.
    History is temporarily extended then restored.

    Optional: tokenizer applies merge rules to seed tokens.
    Optional: long_term_store provides three-layer fallback blending.
    """
    # To prevent context contamination from training data, we clear history
    # before observing the new seed tokens.
    saved = p.history[:]
    p.history.clear()

    if seed:
        seed_tokens = tokenizer.tokenize(seed) if tokenizer else seed
        for tok in seed_tokens:
            p.observe(tok)

    generated = []
    for _ in range(n_tokens):
        p.predict()
        
        # Halt on zero context depth to prevent "Frankenstein" word stitching
        # If depth is <= 0, the trie has lost causal pathing and is just spitting out
        # the most frequent unigram (or nothing).
        if p._last_prediction_depth <= 1:
            break
            
        dist = dict(p._last_distribution)

        # Three-layer fallback (Problem 8): blend with long-term store
        if long_term_store is not None and hasattr(p, 'history') and p.history:
            ctx = tuple(p.history[-p.k:]) if len(p.history) >= p.k else tuple(p.history)
            dist = long_term_store.blend(dist, ctx, p._vocab)

        # Repetition Penalty: only penalize recently *generated* tokens
        # to prevent loops, but never penalize seed tokens (which the trie
        # needs to reproduce in its response).
        repetition_penalty = 1.5
        for tok in set(generated[-20:]):
            if tok in dist:
                dist[tok] /= repetition_penalty

        token = _sample_dist(dist, temperature, top_k, top_p, rng)
        if token is None:
            break
        generated.append(token)
        if stop_tokens and token in stop_tokens:
            break
        p.observe(token)

    p.history = saved

    # Detokenize if we used a tokenizer
    if tokenizer is not None:
        generated = tokenizer.detokenize(generated)

    return generated


def _train_autoregressive(
    p: UniversalPredictor,
    tokens: list,
    tokenizer=None,
    long_term_store=None,
    use_skip_grams=False,
    unlearn=False,
) -> None:
    """
    Train p on every consecutive token pair within a single sequence.
    Each token is predicted from all preceding tokens in that sequence.
    This learns the joint distribution P(t_0) P(t_1|t_0) … P(t_n|t_0..t_{n-1}).

    Optional: tokenizer applies online merge rules before training.
    Optional: long_term_store receives replay after training completes.
    """
    if tokenizer is not None:
        tokens = tokenizer.tokenize(tokens)

    p.history.clear()
    correct = 0
    total = 0
    import random
    for i, token in enumerate(tokens):
        if not unlearn:
            p.predict()
            pred = p._last_prediction
        else:
            pred = None
        
        # Simulated Skip-Gram attention: randomly drop one context token during training
        if use_skip_grams and i > 2 and random.random() < 0.2:
            saved = p.history[:]
            idx_to_drop = random.randint(0, len(p.history) - 1)
            p.history.pop(idx_to_drop)
            if unlearn:
                p.unlearn(token)
                p.observe(token)
            else:
                p.observe(token)
                p.feedback(token)
            p.history = saved
            
        if unlearn:
            p.unlearn(token)
            p.observe(token)
        else:
            p.observe(token)
            p.feedback(token)
        total += 1
        if pred == token:
            correct += 1

    # Update tokenizer merge scores with running accuracy
    if tokenizer is not None and total > 0 and not unlearn:
        tokenizer.update(tokens, correct / total)

    # Replay high-confidence patterns into long-term store
    if long_term_store is not None and total > 0 and not unlearn:
        long_term_store.replay(p, tokens)

    p.history.clear()


# ══════════════════════════════════════════════════════════════════════════════
# SequenceGenerator
# ══════════════════════════════════════════════════════════════════════════════

class SequenceGenerator(BaseEstimator):
    """
    Auto-regressive sequence generator for any token type.

    Works with text (characters or words), DNA, symbol streams, event logs —
    anything the UniversalPredictor can model.

    Parameters
    ----------
    context_length : int
        Trie depth k — number of preceding tokens used as context.
    temperature : float
        Sampling temperature (default 1.0 = unmodified distribution).
    top_k : int | None
        If set, sample only from the top-k most probable tokens.
    top_p : float | None
        Nucleus sampling threshold (e.g. 0.9 = 90% probability mass).
    learning_rate, cred_max, lambda_power : float
    random_seed : int

    API
    ---
    gen.fit(sequences)                — train on one sequence or a list of sequences
    gen.partial_fit(sequences)        — online update
    gen.generate(n, seed, **kwargs)   — sample n tokens given optional seed
    gen.generate_text(n, seed, sep)   — convenience wrapper: joins tokens with sep
    gen.sample_next(**kwargs)         — sample one next token from current state
    gen.score(sequence)               — average bits-per-token (lower = better)
    """

    def __init__(
        self,
        context_length: int   = 6,
        min_context_length: int = 1,
        temperature:    float = 1.0,
        top_k:          int   | None = None,
        top_p:          float | None = None,
        learning_rate:  float = 0.08,
        cred_max:       float = 6.05,
        lambda_power:   float = 0.65,
        random_seed:    int   = 42,
        use_online_tokenizer: bool = False,
        tokenizer_max_merges: int  = 64,
        use_dual_predictor: bool = False,
        long_term_store: Any  = None,
        use_similarity_fallback: bool = False,
        use_positional_weights: bool = False,
        use_semantic_hashing: bool = False,
        use_skip_grams: bool = False,
    ):
        self.context_length = context_length
        self.min_context_length = min_context_length
        self.temperature    = temperature
        self.top_k          = top_k
        self.top_p          = top_p
        self.learning_rate  = learning_rate
        self.cred_max       = cred_max
        self.lambda_power   = lambda_power
        self.random_seed    = random_seed
        self.use_online_tokenizer = use_online_tokenizer
        self.tokenizer_max_merges = tokenizer_max_merges
        self.use_dual_predictor = use_dual_predictor
        self.long_term_store = long_term_store
        self.use_similarity_fallback = use_similarity_fallback
        self.use_positional_weights = use_positional_weights
        self.use_semantic_hashing = use_semantic_hashing
        self.use_skip_grams = use_skip_grams

    # ── public API ────────────────────────────────────────────────────────────

    def unlearn(self, sequences) -> 'SequenceGenerator':
        self._check_fitted()
        tok = getattr(self, '_tokenizer', None)
        lts = self.long_term_store
        if isinstance(sequences, str):
            _train_autoregressive(self._pred, list(sequences), tokenizer=tok, long_term_store=lts, use_skip_grams=self.use_skip_grams, unlearn=True)
        elif sequences and not isinstance(sequences[0], (list, tuple)):
            _train_autoregressive(self._pred, list(sequences), tokenizer=tok, long_term_store=lts, use_skip_grams=self.use_skip_grams, unlearn=True)
        else:
            for seq in sequences:
                _train_autoregressive(self._pred, list(seq), tokenizer=tok, long_term_store=lts, use_skip_grams=self.use_skip_grams, unlearn=True)
        return self

    def fit(self, sequences, y=None) -> 'SequenceGenerator':
        """
        Train on a sequence or list of sequences.
        sequences: str | list | list-of-lists
            A single string is treated as a character sequence.
        """
        if self.use_dual_predictor:
            DualPredictor = _get_dual_predictor()
            self._pred = DualPredictor(
                self.context_length,
                learning_rate=self.learning_rate,
            )
        else:
            self._pred = _make_predictor(
                self.context_length, self.learning_rate, self.cred_max, self.lambda_power,
                use_similarity_fallback=self.use_similarity_fallback,
                use_positional_weights=self.use_positional_weights,
                min_context_length=self.min_context_length,
            )
        self._rng = random.Random(self.random_seed)
        self._tokenizer = None
        if self.use_online_tokenizer:
            OT = _get_online_tokenizer()
            self._tokenizer = OT(max_merges=self.tokenizer_max_merges)
        self.omni_tokenizer = OmniTokenizer() if self.use_semantic_hashing else None
        self.is_fitted_ = True
        self._train_sequences(sequences)
        return self

    def partial_fit(self, sequences, y=None) -> 'SequenceGenerator':
        if not hasattr(self, '_pred'):
            return self.fit(sequences)
        self._train_sequences(sequences)
        return self

    def generate(
        self,
        n_tokens:    int,
        seed:        list | str | None = None,
        temperature: float | None      = None,
        top_k:       int   | None      = None,
        top_p:       float | None      = None,
        stop_tokens: list  | None      = None,
        use_mcts:    bool              = False,
        bias_context: str | None       = None,
        mcts_sims:   int               = 3,
        mcts_depth:  int               = 3,
    ) -> list:
        """
        Sample n_tokens auto-regressively.

        Parameters
        ----------
        seed : list | str | None
            Starting context.  A string is treated as a list of characters.
        temperature, top_k, top_p : override instance defaults for this call.
        stop_tokens : list | None
            Generation halts early if any of these tokens is sampled.

        Returns list of tokens (same type as training tokens).
        """
        self._check_fitted()
        seed_list = list(seed) if isinstance(seed, str) else (seed or [])
        
        if use_mcts:
            return _mcts_generate_from_predictor(
                self._pred, n_tokens, seed_list,
                n_sims=mcts_sims,
                top_k_candidates=top_k if top_k is not None else 3,
                lookahead_depth=mcts_depth,
                stop_tokens=set(stop_tokens) if stop_tokens else None,
                tokenizer=getattr(self, '_tokenizer', None),
                long_term_store=self.long_term_store,
                bias_context=bias_context,
            )
            
        return _generate_from_predictor(
            self._pred, n_tokens, seed_list,
            temperature if temperature is not None else self.temperature,
            top_k       if top_k       is not None else self.top_k,
            top_p       if top_p       is not None else self.top_p,
            self._rng,
            set(stop_tokens) if stop_tokens else None,
            tokenizer=getattr(self, '_tokenizer', None),
            long_term_store=self.long_term_store,
        )

    def generate_text(
        self,
        n_tokens:    int,
        seed:        str | None   = None,
        sep:         str          = '',
        **kwargs,
    ) -> str:
        """Convenience wrapper: generate and join tokens as a string."""
        tokens = self.generate(n_tokens, seed=list(seed) if seed else None, **kwargs)
        return sep.join(str(t) for t in tokens)

    def sample_next(self, temperature=None, top_k=None, top_p=None) -> Any:
        """Sample one next token given the current internal history."""
        self._check_fitted()
        self._pred.predict()
        return _sample_dist(
            dict(self._pred._last_distribution),
            temperature if temperature is not None else self.temperature,
            top_k       if top_k       is not None else self.top_k,
            top_p       if top_p       is not None else self.top_p,
            self._rng,
        )

    def observe(self, token) -> 'SequenceGenerator':
        """Advance internal state by one token (does not update trie)."""
        self._check_fitted()
        self._pred.observe(token)
        return self

    def score(self, sequence) -> float:
        """Average bits-per-token on a held-out sequence (lower = better)."""
        self._check_fitted()
        tokens = list(sequence)
        if not tokens:
            return float('inf')

        if self.omni_tokenizer:
            tokens = [self.omni_tokenizer.tokenize(t) for t in tokens]

        saved  = self._pred.history[:]
        total  = 0.0
        for token in tokens:
            self._pred.predict()
            prob   = max(self._pred._last_distribution.get(token, 1e-12), 1e-12)
            total += -math.log2(prob)
            self._pred.observe(token)
        self._pred.history = saved
        return total / len(tokens)

    def peek_distribution(self, seed_tokens: list) -> dict:
        """
        Return the trie's next-token probability distribution given seed_tokens
        without modifying any predictor state.  Safe to call at any time.
        """
        self._check_fitted()
        saved = self._pred.history[:]
        k = getattr(self._pred, 'k', len(seed_tokens))
        self._pred.history = list(seed_tokens)[-k:]
        self._pred.predict()
        dist = dict(self._pred._last_distribution)
        self._pred.history = saved
        return dist

    @property
    def vocab_(self) -> set:
        """Set of all tokens seen during training."""
        return set(self._pred._vocab) if hasattr(self, '_pred') else set()

    # ── internal ──────────────────────────────────────────────────────────────

    def _train_sequences(self, sequences) -> None:
        tok = getattr(self, '_tokenizer', None)
        lts = self.long_term_store
        if isinstance(sequences, str):
            _train_autoregressive(self._pred, list(sequences), tokenizer=tok, long_term_store=lts, use_skip_grams=self.use_skip_grams)
        elif sequences and not isinstance(sequences[0], (list, tuple)):
            # Flat list — treat as one sequence
            _train_autoregressive(self._pred, list(sequences), tokenizer=tok, long_term_store=lts, use_skip_grams=self.use_skip_grams)
        else:
            for seq in sequences:
                _train_autoregressive(self._pred, list(seq), tokenizer=tok, long_term_store=lts, use_skip_grams=self.use_skip_grams)

    def _check_fitted(self):
        if not hasattr(self, '_pred'):
            raise RuntimeError("Call fit() first.")


# ══════════════════════════════════════════════════════════════════════════════
# TabularGenerator
# ══════════════════════════════════════════════════════════════════════════════

class TabularGenerator(BaseEstimator):
    """
    Synthetic tabular row generator via joint distribution modeling.

    Unlike TabularPredictor (which learns P(label | features)), this learns the
    full joint P(f_0, f_1, ..., f_{n-1}, label) by treating every feature token
    AND the label token as one auto-regressive sequence.

    This allows:
      • Unconditional generation — sample complete rows from the joint
      • Class-conditional generation — fix the label, sample features
      • Feature-conditional generation — fix some features, sample the rest

    Feature order matters here (earlier features are conditioned on fewer
    preceding tokens).  MI-ascending order puts the most predictive feature
    last so it can condition on the most context.

    Parameters
    ----------
    n_bins : int
        Quantile bins for continuous features.
    context_length : int | None
        Trie depth.  None = n_features + 1 (full row context).
    n_orderings : int
        Ensemble size for diversity.
    n_epochs : int
    temperature : float
    top_k, top_p : sampling controls
    learning_rate, cred_max, lambda_power : float
    """

    def __init__(
        self,
        n_bins:         int        = 10,
        context_length: int | None = None,
        n_orderings:    int        = 3,
        n_epochs:       int        = 1,
        temperature:    float      = 1.0,
        top_k:          int | None = None,
        top_p:          float | None = None,
        learning_rate:  float      = 0.08,
        cred_max:       float      = 6.05,
        lambda_power:   float      = 0.65,
        random_seed:    int        = 42,
        long_term_store: Any       = None,
        use_similarity_fallback: bool = False,
        use_positional_weights: bool = False,
    ):
        self.n_bins         = n_bins
        self.context_length = context_length
        self.n_orderings    = n_orderings
        self.n_epochs       = n_epochs
        self.temperature    = temperature
        self.top_k          = top_k
        self.top_p          = top_p
        self.learning_rate  = learning_rate
        self.cred_max       = cred_max
        self.lambda_power   = lambda_power
        self.random_seed    = random_seed
        self.long_term_store = long_term_store
        self.use_similarity_fallback = use_similarity_fallback
        self.use_positional_weights = use_positional_weights

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, X, y) -> 'TabularGenerator':
        """
        Fit on a labelled dataset.  Learns the full joint distribution.
        X : feature matrix  (numpy, pandas, or list-of-lists)
        y : class labels
        """
        self._disc  = FeatureDiscretizer(n_bins=self.n_bins)
        self._lenc  = LabelEncoder()
        self._rng   = random.Random(self.random_seed)
        self._preds = []
        self._orders = []

        rows   = self._disc.fit_transform(X)
        labels = list(y)
        self._lenc.fit(labels)
        y_enc  = [self._lenc.encode(l) for l in labels]

        n_feat = self._disc.n_features
        k      = (n_feat + 1) if self.context_length is None else self.context_length
        self._orders = _build_orders(rows, y_enc, n_feat, self.n_orderings, self._rng)
        self._preds  = [
            _make_predictor(k, self.learning_rate, self.cred_max, self.lambda_power)
            for _ in self._orders
        ]

        # A second predictor trained label-first enables class-conditional generation.
        # P(f0,...,fn-1 | label) is modelled correctly only when label precedes features.
        # MI-descending order: most discriminative features immediately follow the label,
        # so even shallow-context fallbacks (label + 1 feature) capture class structure.
        self._cond_pred = _make_predictor(k, self.learning_rate, self.cred_max, self.lambda_power)
        self._cond_order = list(reversed(self._orders[0]))  # MI-descending for conditional

        for _ in range(self.n_epochs):
            pairs = list(zip(rows, labels))
            self._rng.shuffle(pairs)
            for tok_row, label in pairs:
                lt = (_LABEL_NS, self._lenc.encode(label))
                for p, order in zip(self._preds, self._orders):
                    full_seq = _apply_order(tok_row, order) + [lt]
                    _train_autoregressive(p, full_seq)
                # label-first sequence for conditional predictor
                cond_seq = [lt] + _apply_order(tok_row, self._cond_order)
                _train_autoregressive(self._cond_pred, cond_seq)

        self.is_fitted_ = True
        return self

    def partial_fit(self, X, y) -> 'TabularGenerator':
        if not hasattr(self, '_disc'):
            return self.fit(X, y)
        rows   = self._disc.transform(X)
        labels = list(y)
        self._lenc.partial_fit(labels)
        for tok_row, label in zip(rows, labels):
            lt = (_LABEL_NS, self._lenc.encode(label))
            for p, order in zip(self._preds, self._orders):
                full_seq = _apply_order(tok_row, order) + [lt]
                _train_autoregressive(p, full_seq)
            cond_seq = [lt] + _apply_order(tok_row, self._cond_order)
            _train_autoregressive(self._cond_pred, cond_seq)
        return self

    def sample(
        self,
        n_rows:           int             = 1,
        given_label:      Any             = None,
        given_features:   dict | None     = None,
        temperature:      float | None    = None,
        top_k:            int   | None    = None,
        top_p:            float | None    = None,
    ) -> list:
        """
        Generate n_rows synthetic rows.

        Parameters
        ----------
        given_label : any class label
            If set, condition on this label (label-first generation).
        given_features : {feature_index: value} | None
            Fix specific feature values; sample the rest.
        temperature, top_k, top_p : override instance sampling defaults.

        Returns
        -------
        list of dicts: [{'features': [...], 'label': ...}, ...]
        """
        self._check_fitted()
        T   = temperature if temperature is not None else self.temperature
        K   = top_k       if top_k       is not None else self.top_k
        P   = top_p       if top_p       is not None else self.top_p
        out = []
        for _ in range(n_rows):
            out.append(self._sample_row(given_label, given_features, T, K, P))
        return out

    def sample_dataframe(self, n_rows: int = 1, **kwargs):
        """Like sample() but returns a pandas DataFrame. Requires pandas."""
        import pandas as pd
        rows = self.sample(n_rows, **kwargs)
        X_cols = {f'feature_{i}': [r['features'][i] for r in rows]
                  for i in range(self._disc.n_features)}
        X_cols['label'] = [r['label'] for r in rows]
        return pd.DataFrame(X_cols)

    # ── internal ──────────────────────────────────────────────────────────────

    def _label_token(self, label) -> tuple:
        return (_LABEL_NS, self._lenc.encode(label))

    def _sample_row(self, given_label, given_features, T, K, P) -> dict:
        n_feat  = self._disc.n_features
        classes = self._lenc.classes_

        # Average distributions across all ordering predictors
        # for each position in the feature sequence
        def avg_dist_at_context(context_tokens):
            combined: dict = {}
            for p, order in zip(self._preds, self._orders):
                saved = p.history[:]
                p.history = list(context_tokens)
                p.predict()
                for tok, prob in p._last_distribution.items():
                    combined[tok] = combined.get(tok, 0.0) + prob
                p.history = saved
            total = sum(combined.values())
            if total < 1e-12:
                return combined
            return {t: v / total for t, v in combined.items()}

        if given_label is not None:
            # Class-conditional generation using the label-first predictor.
            # _cond_pred was trained on [label, f0, f1, ..., fn-1] so
            # P(f_i | label, f_0..f_{i-1}) is correctly modelled here.
            lt  = self._label_token(given_label)
            ctx = [lt]
            feature_values = [None] * n_feat

            def cond_dist_at(context_tokens):
                saved = self._cond_pred.history[:]
                self._cond_pred.history = list(context_tokens)
                self._cond_pred.predict()
                d = dict(self._cond_pred._last_distribution)
                self._cond_pred.history = saved
                return d

            for col_idx in self._cond_order:
                dist      = cond_dist_at(ctx)
                feat_dist = {t: v for t, v in dist.items()
                             if isinstance(t, tuple) and len(t) == 2
                             and isinstance(t[0], int) and t[0] == col_idx}
                token = _sample_dist(feat_dist, T, K, P, self._rng) if feat_dist else (col_idx, 0)
                ctx.append(token)
                feature_values[col_idx] = self._decode_feature_token(token)

            return {'features': feature_values, 'label': given_label}

        else:
            # Unconditional: sample features in MI order, then label
            # Average across orderings for each step
            order   = self._orders[0]
            context = []
            feature_values = [None] * n_feat

            for col_idx in order:
                dist = avg_dist_at_context(context)
                feat_dist = {t: v for t, v in dist.items()
                             if isinstance(t, tuple) and len(t) == 2
                             and isinstance(t[0], int) and t[0] == col_idx}
                if not feat_dist:
                    token = (col_idx, 0)
                else:
                    token = _sample_dist(feat_dist, T, K, P, self._rng)
                context.append(token)
                feature_values[col_idx] = self._decode_feature_token(token)

            # Sample label given all features
            dist      = avg_dist_at_context(context)
            lbl_dist  = {t: v for t, v in dist.items()
                         if isinstance(t, tuple) and t[0] == _LABEL_NS}
            if lbl_dist:
                lbl_token = _sample_dist(lbl_dist, T, K, P, self._rng)
                label     = self._lenc.decode(lbl_token[1]) if lbl_token else classes[0]
            else:
                label = classes[0]

            return {'features': feature_values, 'label': label}

    def _decode_feature_token(self, token):
        """Convert (col_idx, bin_or_code) back to an approximate feature value."""
        if token is None:
            return None
        col_idx, bin_val = token
        if col_idx < len(self._disc._types):
            if self._disc._types[col_idx] == 'numeric':
                return self._disc.bin_center(col_idx, bin_val)
            else:
                # Categorical: reverse the int code
                cat_map = self._disc._cat_maps.get(col_idx, {})
                rev     = {v: k for k, v in cat_map.items()}
                return rev.get(bin_val, bin_val)
        return bin_val

    def _check_fitted(self):
        if not hasattr(self, '_disc'):
            raise RuntimeError("Call fit() first.")


# ══════════════════════════════════════════════════════════════════════════════
# TimeSeriesGenerator
# ══════════════════════════════════════════════════════════════════════════════

class TimeSeriesGenerator(BaseEstimator):
    """
    Sampled multivariate time series generator.

    Extends the predictor to draw new sequences from the learned distribution
    instead of returning the argmax/mean (which forecast() does).

    Can be used for data augmentation, simulation, or scenario generation.

    Parameters
    ----------
    n_bins : int
    context_length : int
    temperature : float
    top_k, top_p : sampling controls (useful for avoiding repetitive sequences)
    learning_rate, cred_max, lambda_power : float

    API
    ---
    gen.fit(X)                       — fit discretizer and build trie
    gen.generate(n_steps, seed, ...) — sample a new sequence of n_steps
    gen.augment(X, n_copies, ...)    — generate n_copies similar sequences
    gen.score(X)                     — average bits-per-step
    """

    def __init__(
        self,
        n_bins:         int   = 8,
        context_length: int   = 5,
        temperature:    float = 1.0,
        top_k:          int   | None = None,
        top_p:          float | None = None,
        learning_rate:  float = 0.08,
        cred_max:       float = 6.05,
        lambda_power:   float = 0.65,
        random_seed:    int   = 42,
        use_dual_predictor: bool = False,
        long_term_store: Any  = None,
        use_similarity_fallback: bool = False,
        use_positional_weights: bool = False,
    ):
        self.n_bins         = n_bins
        self.context_length = context_length
        self.temperature    = temperature
        self.top_k          = top_k
        self.top_p          = top_p
        self.learning_rate  = learning_rate
        self.cred_max       = cred_max
        self.lambda_power   = lambda_power
        self.random_seed    = random_seed
        self.use_dual_predictor = use_dual_predictor
        self.long_term_store = long_term_store
        self.use_similarity_fallback = use_similarity_fallback
        self.use_positional_weights = use_positional_weights

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, X, y=None) -> 'TimeSeriesGenerator':
        from .timeseries import _compound_token, _make_predictor as _ts_make

        rows = _to_rows(X)
        if not rows:
            return self
        if not isinstance(rows[0], (list, tuple)):
            rows = [[v] for v in rows]

        self._n_dims = len(rows[0])
        self._disc   = FeatureDiscretizer(n_bins=self.n_bins)
        self._disc.fit(rows)
        self._pred   = _ts_make(
            self.context_length, self.learning_rate, self.cred_max, self.lambda_power,
        )
        self._rng    = random.Random(self.random_seed)
        self._compound_token = _compound_token

        for row in rows:
            token = _compound_token(self._disc._encode_row(row))
            self._pred.predict()
            self._pred.observe(token)
            self._pred.feedback(token)

        self.is_fitted_ = True
        return self

    def generate(
        self,
        n_steps:     int,
        seed:        list | None   = None,
        temperature: float | None  = None,
        top_k:       int   | None  = None,
        top_p:       float | None  = None,
    ) -> list:
        """
        Sample a new time series of n_steps steps.

        seed: list of float vectors to prime the context.
        Returns list of float vectors (one per step).
        """
        self._check_fitted()
        T = temperature if temperature is not None else self.temperature
        K = top_k       if top_k       is not None else self.top_k
        P = top_p       if top_p       is not None else self.top_p

        saved = self._pred.history[:]

        if seed:
            for x in seed:
                row = [x] if isinstance(x, (int, float)) else list(x)
                self._pred.observe(self._compound_token(self._disc._encode_row(row)))

        results = []
        for _ in range(n_steps):
            self._pred.predict()
            dist  = dict(self._pred._last_distribution)
            token = _sample_dist(dist, T, K, P, self._rng)
            if token is None:
                break
            results.append(self._decode_token(token))
            self._pred.observe(token)

        self._pred.history = saved
        return results

    def augment(
        self,
        X,
        n_copies:    int   = 1,
        temperature: float = 1.1,
        **kwargs,
    ) -> list:
        """
        Generate n_copies perturbed versions of X by seeding with X then sampling.

        temperature > 1.0 adds variety; temperature < 1.0 stays close to X.
        Returns list of generated sequences.
        """
        self._check_fitted()
        rows = _to_rows(X)
        if not rows:
            return []
        if not isinstance(rows[0], (list, tuple)):
            rows = [[v] for v in rows]

        generated = []
        for _ in range(n_copies):
            aug = self.generate(
                n_steps=len(rows),
                seed=rows,
                temperature=temperature,
                **kwargs,
            )
            generated.append(aug)
        return generated

    def score(self, X, y=None) -> float:
        """Average bits-per-step (lower = better). Trie not updated."""
        self._check_fitted()
        rows = _to_rows(X)
        if not rows:
            return float('inf')
        if not isinstance(rows[0], (list, tuple)):
            rows = [[v] for v in rows]

        saved = self._pred.history[:]
        total = 0.0
        for row in rows:
            token = self._compound_token(self._disc._encode_row(row))
            self._pred.predict()
            prob   = max(self._pred._last_distribution.get(token, 1e-12), 1e-12)
            total += -math.log2(prob)
            self._pred.observe(token)
        self._pred.history = saved
        return total / len(rows)

    # ── internal ──────────────────────────────────────────────────────────────

    def _decode_token(self, token) -> list:
        mid      = self.n_bins // 2
        fallback = [self._disc.bin_center(d, mid) for d in range(self._n_dims)]
        if not isinstance(token, tuple) or len(token) != self._n_dims:
            return fallback
        return [self._disc.bin_center(d, b) if isinstance(b, int) else 0.0
                for d, b in enumerate(token)]

    def _check_fitted(self):
        if not hasattr(self, '_pred'):
            raise RuntimeError("Call fit() first.")
def _mcts_generate_from_predictor(
    p,
    n_tokens: int,
    seed: list | None,
    n_sims: int = 3,
    top_k_candidates: int = 3,
    lookahead_depth: int = 3,
    stop_tokens: set | None = None,
    tokenizer=None,
    long_term_store=None,
    bias_context: str | None = None,
) -> list:
    """MCTS-guided token selection for Uchi."""
    import math
    import random
    
    saved = p.history[:]
    p.history.clear()
    
    if seed:
        seed_tokens = tokenizer.tokenize(seed) if tokenizer else seed
        for tok in seed_tokens:
            p.observe(tok)
            
    generated = []
    
    for step in range(n_tokens):
        p.predict()
        
        if p._last_prediction_depth <= 1:
            break
            
        dist = dict(p._last_distribution)
        
        # Apply simple repetition penalty to dist
        for tok in set(generated[-20:]):
            if tok in dist:
                dist[tok] /= 1.5
                
        # Get top-K candidates from dist
        if not dist:
            break
            
        sorted_candidates = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:top_k_candidates]
        
        best_tok = sorted_candidates[0][0]
        best_score = float('-inf')
        
        # Base state
        base_history = p.history[:]
        
        for cand_tok, prior in sorted_candidates:
            val_sum = 0.0
            
            for sim in range(n_sims):
                p.history = base_history[:]
                p.observe(cand_tok)
                sim_val = 0.0
                
                # --- Neuro-Symbolic Integration ---
                from uchi.neuro_symbolic import get_ssm
                ssm = get_ssm()
                current_state = ssm.get_state(p.history)
                # ----------------------------------
                
                for d in range(lookahead_depth):
                    p.predict()
                    
                    if p._last_prediction_depth <= 1:
                        sim_val -= 5.0
                        break
                        
                    # Calculate SSM value for current state
                    v = ssm.value(current_state).squeeze(-1)
                    sim_val += v.item()
                    
                    sim_dist = dict(p._last_distribution)
                    if not sim_dist:
                        break
                        
                    # Greedy choice for simulation
                    next_tok = max(sim_dist.items(), key=lambda x: x[1])[0]
                    if stop_tokens and next_tok in stop_tokens:
                        sim_val += 5.0 # Reward reaching a natural stop token
                        break
                        
                    # Semantic Context Masking Bonus
                    # Check both the full synset ID ("energy.n.01") and the word root
                    # ("energy") so plain-text web content and lemmatized memory both match.
                    _tok_str = str(next_tok).lower()
                    _tok_word = _tok_str.split(".")[0]
                    if bias_context and (_tok_str in bias_context.lower() or _tok_word in bias_context.lower()):
                        sim_val += 1.0
                        
                    p.observe(next_tok)
                    current_state, v = ssm.predict_next(current_state, next_tok)
                    sim_val += v.item() * 0.5
                    
                val_sum += sim_val
                
            avg_val = val_sum / max(1, n_sims)
            
            # Combine prior log-prob with value
            lp = math.log(max(1e-9, prior))
            score = 0.4 * lp + 0.6 * avg_val
            
            if score > best_score:
                best_score = score
                best_tok = cand_tok
                
        # Restore base history and observe the chosen token
        p.history = base_history[:]
        generated.append(best_tok)
        if stop_tokens and best_tok in stop_tokens:
            break
        p.observe(best_tok)
        
    p.history = saved
    
    if tokenizer is not None:
        generated = tokenizer.detokenize(generated)
        
    return generated
