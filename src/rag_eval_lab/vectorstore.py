"""A minimal in-memory vector store backed by a numpy matrix.

For a corpus of a few hundred chunks, brute-force cosine similarity (one matrix-vector product)
is exact and takes microseconds, so an approximate-nearest-neighbour index would only add
complexity. The store can be saved to / loaded from a single ``.npz`` file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rag_eval_lab.chunking import Chunk
from rag_eval_lab.embeddings import l2_normalize


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        vectors = l2_normalize(vectors)
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if self._vectors is not None and vectors.shape[1] != self._vectors.shape[1]:
            raise ValueError("vector dimension mismatch")
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        """Return the ``k`` most similar chunks by cosine similarity, best first."""
        if self._vectors is None or k <= 0:
            return []
        query = l2_normalize(query_vector)[0]
        scores = self._vectors @ query
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(self._chunks[i], float(scores[i])) for i in top]

    def save(self, path: str | Path) -> None:
        if self._vectors is None:
            raise ValueError("cannot save an empty store")
        meta = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "text": c.text,
                "position": c.position,
                "metadata": c.metadata,
            }
            for c in self._chunks
        ]
        np.savez_compressed(path, vectors=self._vectors, meta=np.array(json.dumps(meta)))

    @classmethod
    def load(cls, path: str | Path) -> InMemoryVectorStore:
        store = cls()
        with np.load(path) as data:
            meta = json.loads(str(data["meta"]))
            store._vectors = data["vectors"]
        store._chunks = [Chunk(**m) for m in meta]
        return store
