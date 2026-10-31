"""Word error rate. Pure Python, no torch — tests import this directly."""
from __future__ import annotations


def word_error_rate(ref: str, hyp: str) -> float:
    """Word error rate = edit distance / #ref words. 0.0 = perfect."""
    r = ref.upper().split()
    h = hyp.upper().split()
    if not r:
        return 1.0 if h else 0.0
    d = list(range(len(h) + 1))
    for i, x in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1] / len(r)
