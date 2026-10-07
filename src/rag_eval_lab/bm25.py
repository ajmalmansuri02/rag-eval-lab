"""A small, readable Okapi BM25 implementation.

score(q, d) = sum over query terms t of
    idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))

with idf(t) = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5)), which is always positive.

``k1`` controls term-frequency saturation and ``b`` controls length normalisation.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i if in into is it its of on or so "
    "than that the their then there these this to was what when where which who why will "
    "with you your".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, keep identifiers like ``dbx_pat_`` or ``8470`` intact, drop stopwords."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(d) for d in documents]
        self.doc_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.n_docs = len(documents)
        self.avgdl = (sum(self.doc_lens) / self.n_docs) if self.n_docs else 0.0
        df: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            df.update(set(tokens))
        self.idf = {
            term: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5)) for term, n in df.items()
        }

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = [0.0] * self.n_docs
        if not self.avgdl:
            return out
        for i, freqs in enumerate(self.doc_freqs):
            norm = self.k1 * (1 - self.b + self.b * self.doc_lens[i] / self.avgdl)
            total = 0.0
            for term in terms:
                tf = freqs.get(term)
                if tf:
                    total += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
            out[i] = total
        return out

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        """Indices and scores of the ``k`` best documents (zero-score documents excluded)."""
        scored = [(i, s) for i, s in enumerate(self.scores(query)) if s > 0]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]
