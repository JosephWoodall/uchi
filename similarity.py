import math
from typing import Callable, Sequence


def gaussian(sigma: float = 1.0) -> Callable[[Sequence[float], Sequence[float]], float]:
    """Kernel similarity for numeric sequences. sigma controls how far 'similar' reaches."""
    def sim(a: Sequence[float], b: Sequence[float]) -> float:
        sq_dist = sum((x - y) ** 2 for x, y in zip(a, b))
        return math.exp(-sq_dist / (2.0 * sigma ** 2))
    return sim


def hamming(a: Sequence, b: Sequence) -> float:
    """Positional match rate — for text, DNA, or ordered categorical sequences."""
    if not a:
        return 0.0
    return sum(x == y for x, y in zip(a, b)) / len(a)


def jaccard(a: Sequence, b: Sequence) -> float:
    """Set-overlap similarity — for unordered event bags."""
    sa, sb = set(a), set(b)
    union = len(sa | sb)
    return len(sa & sb) / union if union > 0 else 0.0
