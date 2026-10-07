import json
from pathlib import Path

import httpx
import numpy as np

from rag_eval_lab.embeddings import (
    CachedEmbedder,
    EmbeddingConfig,
    HashingEmbedder,
    OllamaEmbedder,
    build_embedder,
)


def test_hashing_embedder_is_deterministic_and_normalised():
    emb = HashingEmbedder(dim=64)
    a = emb.embed(["the quick brown fox", "lazy dog"])
    b = HashingEmbedder(dim=64).embed(["the quick brown fox", "lazy dog"])
    assert a.shape == (2, 64)
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, rtol=1e-5)


def test_hashing_embedder_similarity_reflects_word_overlap():
    emb = HashingEmbedder()
    q, close, far = emb.embed(["backup retention policy", "retention policy for backup", "pizza"])
    assert q @ close > q @ far


class CountingEmbedder:
    name = "counting"

    def __init__(self):
        self.calls: list[tuple[list[str], str]] = []
        self.inner = HashingEmbedder(dim=16)

    def embed(self, texts, kind="document"):
        self.calls.append((list(texts), kind))
        return self.inner.embed(texts, kind)


def test_cached_embedder_caches_documents_not_queries(tmp_path: Path):
    inner = CountingEmbedder()
    cached = CachedEmbedder(inner, tmp_path)
    cached.embed(["a", "b"], "document")
    cached.embed(["b", "c"], "document")
    assert inner.calls[1] == (["c"], "document")
    cached.embed(["q"], "query")
    cached.embed(["q"], "query")
    assert sum(1 for _, kind in inner.calls if kind == "query") == 2

    cached.save()
    inner2 = CountingEmbedder()
    reloaded = CachedEmbedder(inner2, tmp_path)
    out = reloaded.embed(["a", "b", "c"], "document")
    assert inner2.calls == []
    assert out.shape == (3, 16)


def test_ollama_embedder_uses_prefixes_and_batches():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0] for _ in body["input"]]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    emb = OllamaEmbedder(
        "nomic-embed-text",
        base_url="http://ollama.test",
        query_prefix="search_query: ",
        document_prefix="search_document: ",
        batch_size=2,
        client=client,
    )
    out = emb.embed(["a", "b", "c"], "document")
    assert out.shape == (3, 2)
    assert [r["input"] for r in requests] == [
        ["search_document: a", "search_document: b"],
        ["search_document: c"],
    ]
    emb.embed(["q"], "query")
    assert requests[-1]["input"] == ["search_query: q"]
    assert requests[-1]["model"] == "nomic-embed-text"


def test_build_embedder_hashing():
    assert isinstance(build_embedder(EmbeddingConfig(provider="hashing")), HashingEmbedder)
