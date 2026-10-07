from pathlib import Path

import numpy as np
import pytest

from rag_eval_lab.chunking import Chunk
from rag_eval_lab.vectorstore import InMemoryVectorStore


def _chunks(n: int) -> list[Chunk]:
    return [Chunk(f"d{i}#0", f"d{i}", f"text {i}", 0) for i in range(n)]


def test_search_returns_cosine_ordered_results():
    store = InMemoryVectorStore()
    store.add(_chunks(3), np.array([[1, 0], [0.7, 0.7], [0, 1]], dtype=np.float32))
    results = store.search(np.array([1.0, 0.1]), k=2)
    assert [c.doc_id for c, _ in results] == ["d0", "d1"]
    assert results[0][1] > results[1][1]
    assert results[0][1] == pytest.approx(1 / np.sqrt(1.01), rel=1e-5)


def test_k_larger_than_store_and_empty_store():
    store = InMemoryVectorStore()
    assert store.search(np.array([1.0, 0.0]), 3) == []
    store.add(_chunks(2), np.eye(2))
    assert len(store.search(np.array([1.0, 0.0]), 10)) == 2


def test_add_validates_shapes():
    store = InMemoryVectorStore()
    with pytest.raises(ValueError):
        store.add(_chunks(2), np.eye(3))
    store.add(_chunks(2), np.eye(2))
    with pytest.raises(ValueError, match="dimension"):
        store.add(_chunks(1), np.ones((1, 3)))


def test_save_and_load_roundtrip(tmp_path: Path):
    store = InMemoryVectorStore()
    store.add(_chunks(3), np.random.default_rng(0).normal(size=(3, 4)))
    path = tmp_path / "store.npz"
    store.save(path)
    loaded = InMemoryVectorStore.load(path)
    query = np.array([0.1, 0.2, 0.3, 0.4])
    assert [c.chunk_id for c, _ in loaded.search(query, 3)] == [
        c.chunk_id for c, _ in store.search(query, 3)
    ]
    assert len(loaded) == 3
