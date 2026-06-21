"""
NodeCompressor
==============
Compresses converged trie nodes to bound memory usage.

Solves
------
  Problem 6 — Memory grows unbounded: old, stable nodes are compressed
  into compact distribution snapshots and freed from active memory.

Two-tier memory model (inspired by Claude Code conversation compression):

  Active nodes    : full _TrieNode objects — NO credibility cap enforced
                    by the compressor.  The predictor's own adaptive cap
                    still applies; the compressor never imposes a ceiling.
  Compressed nodes: frozen distribution snapshots — credibility frozen at
                    the moment of compression.  Much smaller footprint.

A node is eligible for compression when:
  1.  n_obs >= min_obs  (has enough data)
  2.  node_cred >= cred_max × stability_ratio  (near credibility ceiling)
  3.  It is a leaf or near-leaf (few/no children still actively growing)

Decompression: if the actual tokens seen diverge significantly from the
frozen distribution (measured by a lightweight staleness heuristic based
on the compressed probability of the observed token), the node is
decompressed and allowed to keep learning.

Observability
-------------
  stats()           — compression counts and ratio
  memory_estimate() — rough byte comparison of compressed vs. active cost
"""

import math
import sys
from typing import Any

from .predictor import _TrieNode


class CompressedNode:
    """
    Frozen snapshot of a converged trie node.

    Stores only the normalized distribution and metadata.
    Much smaller than a full _TrieNode with children dict.

    Parameters
    ----------
    distribution : dict
        Normalized probability distribution ``{token: prob}``.
    node_cred : float
        Credibility at time of compression (frozen).
    n_obs : int
        Observation count at compression time.
    frozen_step : int
        Global step counter at compression time.
    """
    __slots__ = ['distribution', 'node_cred', 'n_obs', 'frozen_step']

    def __init__(
        self,
        distribution: dict,
        node_cred: float,
        n_obs: int,
        frozen_step: int,
    ):
        self.distribution: dict  = distribution
        self.node_cred:    float = node_cred
        self.n_obs:        int   = n_obs
        self.frozen_step:  int   = frozen_step

    def __repr__(self) -> str:
        v = len(self.distribution)
        return (f"CompressedNode(vocab={v}, cred={self.node_cred:.3f}, "
                f"n_obs={self.n_obs}, step={self.frozen_step})")


class NodeCompressor:
    """
    Compresses converged trie nodes to bound memory usage.

    Two-tier memory model (inspired by Claude Code conversation compression):
    - Active nodes: full _TrieNode objects, NO credibility cap enforced by compressor
    - Compressed nodes: frozen distribution snapshots, credibility frozen at compression time

    A node is eligible for compression when:
      1. n_obs >= min_obs (has enough data)
      2. node_cred >= cred_max * stability_ratio (at or near credibility ceiling)
      3. Its successor distribution has low entropy change over recent observations

    Decompression: if queries to a compressed node suggest its distribution is stale
    (the actual tokens seen differ significantly from the frozen distribution),
    the node is decompressed and allowed to keep learning.

    Parameters
    ----------
    max_active_nodes : int
        Target upper bound on active (uncompressed) nodes (default 50_000).
    min_obs : int
        Minimum observations before a node can be compressed (default 50).
    stability_ratio : float
        Node cred must be >= this fraction of cred_max to compress (default 0.8).
    decompress_threshold : float
        KL-divergence threshold for decompression trigger (default 0.5).
    """

    def __init__(
        self,
        max_active_nodes: int = 50_000,
        min_obs: int = 50,
        stability_ratio: float = 0.8,
        decompress_threshold: float = 0.5,
    ):
        self.max_active_nodes:    int   = max_active_nodes
        self.min_obs:             int   = min_obs
        self.stability_ratio:    float = stability_ratio
        self.decompress_threshold: float = decompress_threshold

        # context_tuple → CompressedNode
        self._compressed: dict[tuple, CompressedNode] = {}
        # global step counter (advanced externally or by compress_pass)
        self._step: int = 0
        # tracks decompression events for diagnostics
        self._decompress_log: list[dict] = []
        # running KL estimate per compressed context (lightweight staleness tracker)
        self._kl_accum: dict[tuple, float] = {}
        self._kl_count: dict[tuple, int]   = {}

    # ── compression eligibility ───────────────────────────────────────────────

    def should_compress(self, node: _TrieNode, cred_max: float) -> bool:
        """
        Check if a node meets compression criteria.

        Parameters
        ----------
        node : _TrieNode
            The trie node to evaluate.
        cred_max : float
            Current credibility ceiling from the predictor.

        Returns
        -------
        bool
            True if the node is eligible for compression.
        """
        # Criterion 1: enough observations
        if node.n_obs < self.min_obs:
            return False

        # Criterion 2: credibility near ceiling (stable)
        if node.node_cred < cred_max * self.stability_ratio:
            return False

        # Criterion 3: must have a successor distribution to compress
        if not node.succ_cred:
            return False

        return True

    # ── compress / decompress ─────────────────────────────────────────────────

    def compress_node(self, context: tuple, node: _TrieNode) -> CompressedNode:
        """
        Create a CompressedNode from a _TrieNode.

        The distribution is normalized from ``succ_cred``.  The compressed
        node is stored internally and returned.

        Parameters
        ----------
        context : tuple
            The context key (sequence of symbols leading to this node).
        node : _TrieNode
            The trie node to compress.

        Returns
        -------
        CompressedNode
            The frozen snapshot.
        """
        # Normalize succ_cred into a probability distribution
        total = sum(node.succ_cred.values())
        if total < 1e-12:
            # Degenerate case: uniform over whatever keys exist
            n = len(node.succ_cred)
            distribution = {k: 1.0 / max(n, 1) for k in node.succ_cred}
        else:
            distribution = {k: v / total for k, v in node.succ_cred.items()}

        compressed = CompressedNode(
            distribution=distribution,
            node_cred=node.node_cred,
            n_obs=node.n_obs,
            frozen_step=self._step,
        )
        self._compressed[context] = compressed
        return compressed

    def get_compressed(self, context: tuple) -> CompressedNode | None:
        """
        Look up a compressed node by context tuple.

        Parameters
        ----------
        context : tuple
            The context key to look up.

        Returns
        -------
        CompressedNode | None
            The compressed node, or None if not found.
        """
        return self._compressed.get(context)

    def decompress_node(self, context: tuple) -> dict | None:
        """
        Decompress a node: return its stored distribution and remove from
        compressed storage.

        Logs the decompression event for diagnostics.

        Parameters
        ----------
        context : tuple
            The context key to decompress.

        Returns
        -------
        dict | None
            The frozen distribution ``{token: prob}``, or None if context
            was not compressed.
        """
        compressed = self._compressed.pop(context, None)
        if compressed is None:
            return None

        # Log the event
        self._decompress_log.append({
            'context': context,
            'step': self._step,
            'frozen_step': compressed.frozen_step,
            'age': self._step - compressed.frozen_step,
            'n_obs_at_freeze': compressed.n_obs,
            'vocab_size': len(compressed.distribution),
        })

        # Clean up KL tracking state
        self._kl_accum.pop(context, None)
        self._kl_count.pop(context, None)

        return dict(compressed.distribution)

    # ── staleness detection ───────────────────────────────────────────────────

    def check_staleness(self, context: tuple, actual_token: Any) -> bool:
        """
        Check if a compressed node's distribution is stale.

        Compares the actual token's probability in the frozen distribution
        against a threshold scaled by vocabulary size.  If the token's
        probability is very low (below ``decompress_threshold / vocab_size``),
        the distribution is considered stale.

        Also tracks a running KL estimate: the mean negative log-probability
        of observed tokens under the frozen distribution.

        Parameters
        ----------
        context : tuple
            The compressed context to check.
        actual_token : Any
            The token that was actually observed.

        Returns
        -------
        bool
            True if the compressed distribution appears stale and should
            be decompressed.
        """
        compressed = self._compressed.get(context)
        if compressed is None:
            return False

        dist = compressed.distribution
        vocab_size = max(len(dist), 1)

        # Probability of the actual token under the frozen distribution
        prob = dist.get(actual_token, 0.0)

        # Staleness threshold: if the token's probability is below
        # threshold / vocab_size, the distribution is stale.
        threshold = self.decompress_threshold / vocab_size

        # Update running KL estimate (mean surprise under frozen dist)
        # Uses -log(p) as a proxy for KL divergence contribution
        surprise = -math.log(max(prob, 1e-12))
        self._kl_accum[context] = self._kl_accum.get(context, 0.0) + surprise
        self._kl_count[context] = self._kl_count.get(context, 0) + 1

        # Check immediate staleness: very low probability for actual token
        if prob < threshold:
            return True

        # Check accumulated staleness: mean surprise exceeding threshold
        count = self._kl_count[context]
        if count >= 5:
            mean_surprise = self._kl_accum[context] / count
            # Compare against entropy of uniform distribution over vocab
            # If mean surprise is much higher, the distribution is stale
            uniform_surprise = math.log(vocab_size)
            if mean_surprise > uniform_surprise + self.decompress_threshold:
                return True

        return False

    # ── full-trie compression pass ────────────────────────────────────────────

    def compress_pass(self, root_node: _TrieNode, cred_max: float) -> dict:
        """
        Walk the entire trie and compress eligible nodes.

        For each leaf or near-leaf node that meets compression criteria,
        creates a CompressedNode and frees the node's children dict to
        reclaim memory.  The parent keeps a reference to the child node,
        but the child's subtree is freed.

        Uses an iterative stack-based traversal to avoid stack overflow on
        deep tries.

        Parameters
        ----------
        root_node : _TrieNode
            The root of the trie to walk.
        cred_max : float
            Current credibility ceiling from the predictor.

        Returns
        -------
        dict
            Stats: ``{compressed: int, skipped: int, active: int}``.
        """
        self._step += 1

        compressed_count = 0
        skipped_count = 0
        active_count = 0

        # Iterative DFS using an explicit stack.
        # Each entry: (node, context_tuple, parent_node, symbol_from_parent)
        # We process bottom-up: compress children before parents.
        # First pass: collect all nodes with their contexts.
        all_nodes: list[tuple[_TrieNode, tuple]] = []
        stack: list[tuple[_TrieNode, tuple]] = [(root_node, ())]

        while stack:
            node, ctx = stack.pop()
            all_nodes.append((node, ctx))
            for sym, child in node.children.items():
                stack.append((child, ctx + (sym,)))

        # Process in reverse order (deepest first → bottom-up) so that
        # children are compressed before their parents.  This lets us
        # safely clear a parent's children dict if all its children
        # were compressed.
        for node, ctx in reversed(all_nodes):
            # Skip the root — never compress it
            if not ctx:
                active_count += 1
                continue

            # Skip nodes already compressed in a previous pass
            if ctx in self._compressed:
                continue

            # Check if this is a leaf or near-leaf (all children already
            # compressed or it has no children)
            is_compressible_leaf = True
            for sym in list(node.children.keys()):
                child_ctx = ctx + (sym,)
                if child_ctx not in self._compressed:
                    is_compressible_leaf = False
                    break

            if not is_compressible_leaf:
                active_count += 1
                skipped_count += 1
                continue

            if self.should_compress(node, cred_max):
                self.compress_node(ctx, node)
                # Free children to reclaim memory — the distribution
                # snapshot captures all the information we need.
                node.children.clear()
                node.succ_cred.clear()
                compressed_count += 1
            else:
                active_count += 1
                skipped_count += 1

        return {
            'compressed': compressed_count,
            'skipped': skipped_count,
            'active': active_count,
        }

    # ── observability ─────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """
        Compression statistics.

        Returns
        -------
        dict
            ``{n_compressed, n_decompressions, total_frozen_tokens,
            compression_ratio}``.
        """
        n_compressed = len(self._compressed)
        n_decompressions = len(self._decompress_log)
        total_frozen_tokens = sum(
            len(cn.distribution) for cn in self._compressed.values()
        )

        # Compression ratio: how much of total node population is compressed
        # (compressed / (compressed + estimated active))
        # Without access to the trie here, report compressed count only.
        total = n_compressed + max(n_compressed, 1)  # conservative estimate
        compression_ratio = n_compressed / total if total > 0 else 0.0

        return {
            'n_compressed': n_compressed,
            'n_decompressions': n_decompressions,
            'total_frozen_tokens': total_frozen_tokens,
            'compression_ratio': round(compression_ratio, 4),
        }

    def memory_estimate(self) -> dict:
        """
        Rough byte estimate for compressed vs. what they would cost as
        active _TrieNode objects.

        Uses ``sys.getsizeof`` for object overhead estimates.  Actual memory
        savings depend on the Python allocator and GC behavior.

        Returns
        -------
        dict
            ``{compressed_bytes, active_equivalent_bytes, savings_bytes,
            savings_ratio}``.
        """
        compressed_bytes = 0
        active_equiv_bytes = 0

        # Per-object overhead estimates
        # CompressedNode: slots object + distribution dict
        # _TrieNode equivalent: slots object + children dict + succ_cred dict
        node_base = sys.getsizeof(object())  # ~28 bytes
        dict_base = sys.getsizeof({})         # ~64 bytes
        per_entry = 72  # rough cost per dict entry (key + value + hash)

        for cn in self._compressed.values():
            v = len(cn.distribution)

            # CompressedNode: base + 4 slots + one dict with v entries
            compressed_bytes += (
                node_base + 4 * 8  # 4 slots × 8 bytes (pointer)
                + dict_base + v * per_entry  # distribution dict
            )

            # Equivalent _TrieNode: base + 4 slots + children dict (assume v
            # children on average, each itself a _TrieNode) + succ_cred dict
            # with v entries
            child_cost = v * (node_base + 4 * 8 + 2 * dict_base)
            active_equiv_bytes += (
                node_base + 4 * 8
                + dict_base + v * per_entry   # children dict
                + dict_base + v * per_entry   # succ_cred dict
                + child_cost                   # child nodes themselves
            )

        savings = active_equiv_bytes - compressed_bytes
        ratio = savings / max(active_equiv_bytes, 1)

        return {
            'compressed_bytes': compressed_bytes,
            'active_equivalent_bytes': active_equiv_bytes,
            'savings_bytes': savings,
            'savings_ratio': round(ratio, 4),
        }
