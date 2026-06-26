import math
from typing import Any

def norm_pdf(x: float, mu: float, sigma: float) -> float:
    if sigma < 1e-6:
        sigma = 1e-6
    return (1.0 / (math.sqrt(2 * math.pi) * sigma)) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)

def norm_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma < 1e-6:
        return 1.0 if x >= mu else 0.0
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))

class HoeffdingNode:
    """
    A node in the Hoeffding Online Decision Tree.
    Tracks Gaussian statistics for Information Gain and Naive Bayes prediction.
    """
    def __init__(self, n_features: int, is_leaf: bool = True):
        self.is_leaf = is_leaf
        self.n_features = n_features
        
        # If leaf, track Gaussian statistics
        self.n_obs = 0
        self.class_counts = {}
        # feature_idx -> class_label -> {'sum': sum_x, 'sq_sum': sum_x2}
        self.feature_stats = {i: {} for i in range(n_features)}
        
        # If internal node, track split condition
        self.split_feature = None
        self.split_threshold = None
        self.left = None
        self.right = None

    def observe(self, row: list, label: Any):
        if not self.is_leaf:
            val = float(row[self.split_feature])
            if val <= self.split_threshold:
                self.left.observe(row, label)
            else:
                self.right.observe(row, label)
            return

        self.n_obs += 1
        self.class_counts[label] = self.class_counts.get(label, 0) + 1
        
        for f_idx, f_val in enumerate(row):
            v = float(f_val)
            if label not in self.feature_stats[f_idx]:
                self.feature_stats[f_idx][label] = {'sum': 0.0, 'sq_sum': 0.0}
            self.feature_stats[f_idx][label]['sum'] += v
            self.feature_stats[f_idx][label]['sq_sum'] += v * v

    def predict(self, row: list) -> dict:
        if not self.is_leaf:
            val = float(row[self.split_feature])
            if val <= self.split_threshold:
                return self.left.predict(row)
            else:
                return self.right.predict(row)
                
        if not self.class_counts:
            return {}
            
        # Gaussian Naive Bayes Prediction
        log_probs = {}
        for c, count in self.class_counts.items():
            # Log Prior P(c)
            log_p = math.log(count / self.n_obs)
            
            # Log Likelihoods
            for f_idx, f_val in enumerate(row):
                v = float(f_val)
                stats = self.feature_stats[f_idx].get(c, {'sum': 0.0, 'sq_sum': 0.0})
                n_c = self.class_counts.get(c, 0)
                
                if n_c > 1:
                    mu = stats['sum'] / n_c
                    var = (stats['sq_sum'] - (stats['sum'] ** 2) / n_c) / (n_c - 1)
                    sigma = math.sqrt(max(var, 1e-6))
                else:
                    mu = v
                    sigma = 1e-6
                    
                pdf = norm_pdf(v, mu, sigma)
                log_p += math.log(max(pdf, 1e-12))
            
            log_probs[c] = log_p
            
        # Convert log probs to normalized probabilities
        max_lp = max(log_probs.values())
        probs = {c: math.exp(lp - max_lp) for c, lp in log_probs.items()}
        total = sum(probs.values())
        return {c: p / total for c, p in probs.items()}

class HoeffdingPredictor:
    """
    Online Decision Tree using Hoeffding Bounds.
    Uses Gaussian Naive Bayes at the leaves and Continuous Threshold Splitting.
    """
    def __init__(
        self,
        n_features: int,
        delta: float = 1e-1,
        grace_period: int = 10,
        tie_threshold: float = 0.05
    ):
        self.n_features = n_features
        self.delta = delta
        self.grace_period = grace_period
        self.tie_threshold = tie_threshold
        self.root = HoeffdingNode(n_features)
        
    def _entropy(self, counts: dict) -> float:
        total = sum(counts.values())
        if total == 0:
            return 0.0
        ent = 0.0
        for c in counts.values():
            p = c / total
            if p > 0:
                ent -= p * math.log2(p)
        return ent

    def _evaluate_threshold(self, leaf: HoeffdingNode, f_idx: int, threshold: float) -> float:
        left_counts = {}
        right_counts = {}
        
        for c, count in leaf.class_counts.items():
            stats = leaf.feature_stats[f_idx].get(c, {'sum': 0.0, 'sq_sum': 0.0})
            if count > 1:
                mu = stats['sum'] / count
                var = (stats['sq_sum'] - (stats['sum'] ** 2) / count) / (count - 1)
                sigma = math.sqrt(max(var, 1e-6))
            else:
                mu = stats['sum'] / count if count > 0 else 0
                sigma = 1e-6
                
            p_left = norm_cdf(threshold, mu, sigma)
            left_counts[c] = count * p_left
            right_counts[c] = count * (1.0 - p_left)
            
        n_left = sum(left_counts.values())
        n_right = sum(right_counts.values())
        total = n_left + n_right
        
        if total == 0:
            return 0.0
            
        e_left = self._entropy(left_counts)
        e_right = self._entropy(right_counts)
        
        return (n_left / total) * e_left + (n_right / total) * e_right

    def _info_gain(self, leaf: HoeffdingNode, f_idx: int) -> tuple[float, float]:
        base_entropy = self._entropy(leaf.class_counts)
        
        # Collect candidate thresholds (class means for this feature)
        candidates = []
        for c, count in leaf.class_counts.items():
            stats = leaf.feature_stats[f_idx].get(c)
            if stats and count > 0:
                candidates.append(stats['sum'] / count)
                
        if not candidates:
            return 0.0, 0.0
            
        best_gain = -1.0
        best_thresh = 0.0
        
        for thresh in set(candidates):
            exp_entropy = self._evaluate_threshold(leaf, f_idx, thresh)
            gain = base_entropy - exp_entropy
            if gain > best_gain:
                best_gain = gain
                best_thresh = thresh
                
        return best_gain, best_thresh

    def _attempt_split(self, leaf: HoeffdingNode) -> None:
        if leaf.n_obs < self.grace_period:
            return
            
        # Calculate Information Gain for all features
        gains = []
        for i in range(self.n_features):
            gain, thresh = self._info_gain(leaf, i)
            gains.append((gain, thresh, i))
            
        gains.sort(reverse=True, key=lambda x: x[0])
        best_gain, best_thresh, best_idx = gains[0]
        second_gain = gains[1][0] if len(gains) > 1 else 0.0
        
        # Hoeffding Bound epsilon
        R = math.log2(len(leaf.class_counts)) if len(leaf.class_counts) > 0 else 1.0
        epsilon = math.sqrt((R**2 * math.log(1 / self.delta)) / (2 * leaf.n_obs))
        
        if (best_gain - second_gain > epsilon) or (epsilon < self.tie_threshold and best_gain > 0):
            # Split!
            leaf.is_leaf = False
            leaf.split_feature = best_idx
            leaf.split_threshold = best_thresh
            
            leaf.left = HoeffdingNode(self.n_features)
            leaf.right = HoeffdingNode(self.n_features)
            
            # Free memory
            leaf.feature_stats = None
            leaf.class_counts = None

    def _traverse_and_split(self, node: HoeffdingNode):
        if node.is_leaf:
            if node.n_obs % self.grace_period == 0:
                self._attempt_split(node)
        else:
            self._traverse_and_split(node.left)
            self._traverse_and_split(node.right)

    def partial_fit(self, row: list, label: Any) -> None:
        self.root.observe(row, label)
        self._traverse_and_split(self.root)
        
    def predict_proba(self, row: list) -> dict:
        return self.root.predict(row)
