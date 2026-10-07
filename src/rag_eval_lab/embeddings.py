"""Embedding providers.

All embedders implement :class:`Embedder`: ``embed(texts, kind)`` returns an ``(n, dim)`` float32
array of L2-normalised vectors. ``kind`` is ``"query"`` or ``"document"`` because some models
(nomic-embed-text, for example) are trained with task prefixes and retrieve noticeably better
when queries and documents are prefixed differently.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import httpx
import numpy as np

from rag_eval_lab.env import env
from rag_eval_lab.http import RateLimiter, post_json

Kind = Literal["query", "document"]


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str], kind: Kind = "document") -> np.ndarray: ...


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


@dataclass
class EmbeddingConfig:
    provider: str = "ollama"  # ollama | gemini | hashing
    model: str = "nomic-embed-text"
    query_prefix: str = ""
    document_prefix: str = ""
    batch_size: int = 32

    @property
    def label(self) -> str:
        return self.model if self.provider != "hashing" else "hashing"


_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """A deterministic, dependency-free embedder based on the hashing trick.

    Each word and word-bigram is hashed into one of ``dim`` buckets with a +/-1 sign. This is
    essentially a sparse bag-of-words projected to a dense vector: it captures lexical overlap
    but no meaning (``"cost"`` and ``"price"`` are unrelated). It powers the offline mode and the
    unit tests, and doubles as a useful "how much do real embeddings actually add?" baseline.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0

    def embed(self, texts: Sequence[str], kind: Kind = "document") -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = _TOKEN_RE.findall(text.lower())
            features = tokens + [f"{a}_{b}" for a, b in itertools.pairwise(tokens)]
            for feature in features:
                idx, sign = self._bucket(feature)
                out[row, idx] += sign
        return l2_normalize(out)


class OllamaEmbedder:
    """Embeddings from a local Ollama server (``POST /api/embed``)."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        query_prefix: str = "",
        document_prefix: str = "",
        batch_size: int = 32,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.name = f"ollama:{model}"
        self.base_url = (base_url or env("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.prefixes = {"query": query_prefix, "document": document_prefix}
        self.batch_size = batch_size
        self._client = client or httpx.Client(timeout=timeout)

    def embed(self, texts: Sequence[str], kind: Kind = "document") -> np.ndarray:
        prefix = self.prefixes[kind]
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [prefix + t for t in texts[start : start + self.batch_size]]
            data = post_json(
                self._client, f"{self.base_url}/api/embed", {"model": self.model, "input": batch}
            )
            vectors.extend(data["embeddings"])
        return l2_normalize(np.array(vectors, dtype=np.float32))


class GeminiEmbedder:
    """Embeddings from the Gemini API (free tier). Requires ``GEMINI_API_KEY``."""

    BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        model: str | None = None,
        batch_size: int = 50,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = env("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set; see .env.example")
        model = model or env("GEMINI_EMBED_MODEL")
        if not model:
            raise ValueError("Set an embedding model in the config or GEMINI_EMBED_MODEL")
        self.model = model.removeprefix("models/")
        self.name = f"gemini:{self.model}"
        self.batch_size = batch_size
        self._client = client or httpx.Client(timeout=timeout)
        self._limiter = RateLimiter(float(env("GEMINI_REQUESTS_PER_MINUTE", "10") or 10))

    def embed(self, texts: Sequence[str], kind: Kind = "document") -> np.ndarray:
        task = "RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT"
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            payload = {
                "requests": [
                    {
                        "model": f"models/{self.model}",
                        "content": {"parts": [{"text": t}]},
                        "taskType": task,
                    }
                    for t in batch
                ]
            }
            data = post_json(
                self._client,
                f"{self.BASE}/models/{self.model}:batchEmbedContents",
                payload,
                headers={"x-goog-api-key": self.api_key},
                limiter=self._limiter,
            )
            vectors.extend(item["values"] for item in data["embeddings"])
        return l2_normalize(np.array(vectors, dtype=np.float32))


class CachedEmbedder:
    """Caches *document* embeddings by content hash, optionally persisted to ``.npz``.

    Experiments that share an embedding model but differ in retrieval settings re-embed the same
    chunks; caching makes a sweep of many configurations cost little more than one.
    """

    def __init__(self, inner: Embedder, cache_dir: str | Path | None = None) -> None:
        self.inner = inner
        self.name = inner.name
        self._memory: dict[str, np.ndarray] = {}
        self._path: Path | None = None
        if cache_dir is not None:
            slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", inner.name)
            self._path = Path(cache_dir) / f"{slug}.npz"
            self._load()

    @staticmethod
    def _key(text: str, kind: Kind) -> str:
        return hashlib.sha1(f"{kind}\x00{text}".encode()).hexdigest()

    def _load(self) -> None:
        if self._path and self._path.is_file():
            with np.load(self._path) as data:
                for key in data.files:
                    self._memory[key] = data[key]

    def save(self) -> None:
        if self._path and self._memory:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self._path, **self._memory)

    def embed(self, texts: Sequence[str], kind: Kind = "document") -> np.ndarray:
        if kind == "query":
            # Queries are not cached so that measured retrieval latency stays honest.
            return self.inner.embed(texts, kind)
        keys = [self._key(t, kind) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self._memory]
        if missing:
            fresh = self.inner.embed([texts[i] for i in missing], kind)
            for i, vector in zip(missing, fresh, strict=True):
                self._memory[keys[i]] = vector
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([self._memory[k] for k in keys])


def build_embedder(config: EmbeddingConfig) -> Embedder:
    provider = config.provider.lower()
    if provider == "hashing":
        return HashingEmbedder()
    if provider == "ollama":
        return OllamaEmbedder(
            config.model,
            query_prefix=config.query_prefix,
            document_prefix=config.document_prefix,
            batch_size=config.batch_size,
        )
    if provider == "gemini":
        return GeminiEmbedder(config.model or None)
    raise ValueError(f"Unknown embedding provider: {config.provider!r}")
