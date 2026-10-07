"""Retrievers: dense (embeddings), sparse (BM25), hybrid (reciprocal rank fusion), reranking."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from rag_eval_lab.bm25 import BM25
from rag_eval_lab.chunking import Chunk
from rag_eval_lab.embeddings import Embedder
from rag_eval_lab.llm import LLM
from rag_eval_lab.vectorstore import InMemoryVectorStore

Method = Literal["dense", "bm25", "hybrid"]


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[ScoredChunk]: ...


class DenseRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder) -> None:
        self.embedder = embedder
        self.store = InMemoryVectorStore()
        if chunks:
            self.store.add(chunks, embedder.embed([c.text for c in chunks], "document"))

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        vector = self.embedder.embed([query], "query")
        return [ScoredChunk(c, s) for c, s in self.store.search(vector, k)]


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.index = BM25([c.text for c in chunks], k1=k1, b=b)

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        return [ScoredChunk(self.chunks[i], s) for i, s in self.index.top_k(query, k)]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]], k: int = 60, weights: Sequence[float] | None = None
) -> list[ScoredChunk]:
    """Fuse ranked lists: score(d) = sum_i w_i / (k + rank_i(d)), ranks starting at 1.

    RRF only uses ranks, so it can combine retrievers whose raw scores live on different scales
    (cosine similarity vs. BM25). The constant ``k`` (60 in the original paper) dampens the
    advantage of the very top positions.
    """
    weights = list(weights) if weights is not None else [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match rankings")
    fused: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            cid = item.chunk.chunk_id
            by_id[cid] = item.chunk
            fused[cid] = fused.get(cid, 0.0) + weight / (k + rank)
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return [ScoredChunk(by_id[cid], score) for cid, score in ordered]


class HybridRetriever:
    """Runs dense and BM25 retrieval, each to ``candidates`` depth, then fuses with RRF."""

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: BM25Retriever,
        rrf_k: int = 60,
        candidates: int = 20,
        weights: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k
        self.candidates = candidates
        self.weights = weights

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        depth = max(k, self.candidates)
        fused = reciprocal_rank_fusion(
            [self.dense.retrieve(query, depth), self.sparse.retrieve(query, depth)],
            k=self.rrf_k,
            weights=self.weights,
        )
        return fused[:k]


RERANK_PROMPT = """You are a search relevance rater. Rate how useful the passage is for answering \
the query, on a scale from 0 (irrelevant) to 10 (directly contains the answer).
Respond with JSON only: {{"score": <integer 0-10>}}

### Query
{query}

### Passage
{passage}
"""


def parse_relevance(raw: str) -> float:
    """Extract a 0-10 relevance score from an LLM reply; unparseable replies score 0."""
    try:
        value = json.loads(raw).get("score")
        return max(0.0, min(10.0, float(value)))
    except (ValueError, AttributeError, TypeError):
        match = re.search(r"\d+(?:\.\d+)?", raw)
        return max(0.0, min(10.0, float(match.group()))) if match else 0.0


class LLMReranker:
    """Pointwise LLM reranker: asks the LLM to score each candidate, then sorts by that score.

    Cross-encoder rerankers are the usual production choice; an LLM rater keeps the lab
    dependency-free and runs on the same local Ollama. It costs one LLM call per candidate, so
    keep ``candidates`` small. Ties keep the first-stage order.
    """

    def __init__(self, base: Retriever, llm: LLM, candidates: int = 10) -> None:
        self.base = base
        self.llm = llm
        self.candidates = candidates

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        first_stage = self.base.retrieve(query, max(k, self.candidates))
        rescored = []
        for order, item in enumerate(first_stage):
            raw = self.llm.generate(
                RERANK_PROMPT.format(query=query, passage=item.chunk.text), json_mode=True
            )
            rescored.append((parse_relevance(raw), -order, item.chunk))
        rescored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [ScoredChunk(chunk, score) for score, _, chunk in rescored[:k]]
