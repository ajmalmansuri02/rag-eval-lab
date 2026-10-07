"""Retrieval metrics, computed at the *document* level.

The retriever returns ranked chunks. Several chunks can come from the same document, so we
first collapse the chunk ranking into a ranking of unique document IDs (keeping the first
occurrence), then compare it with the golden ``doc_ids``:

- **recall@k**: fraction of the relevant documents that appear in the top-k chunks.
- **hit@k** (hit rate when averaged): 1 if at least one relevant document appears, else 0.
- **reciprocal rank**: 1 / (position of the first relevant document in the unique-document
  ranking), or 0 if none was retrieved. Averaged over questions this is MRR.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence


def unique_in_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def recall_at_k(retrieved_doc_ids: Sequence[str], relevant: Iterable[str]) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        raise ValueError("recall is undefined without relevant documents")
    return len(relevant_set & set(retrieved_doc_ids)) / len(relevant_set)


def hit_at_k(retrieved_doc_ids: Sequence[str], relevant: Iterable[str]) -> float:
    return 1.0 if set(relevant) & set(retrieved_doc_ids) else 0.0


def reciprocal_rank(retrieved_doc_ids: Sequence[str], relevant: Iterable[str]) -> float:
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(unique_in_order(retrieved_doc_ids), start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def mean(values: Iterable[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.fmean(clean) if clean else None


def percentile(values: Iterable[float], pct: float) -> float | None:
    """Nearest-rank percentile (``pct`` in 0-100)."""
    data = sorted(values)
    if not data:
        return None
    rank = max(1, min(len(data), math.ceil(pct / 100 * len(data))))
    return data[rank - 1]
