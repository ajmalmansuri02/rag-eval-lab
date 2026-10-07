"""Runs every experiment in a :class:`RunConfig` against the golden set."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from rag_eval_lab.chunking import Chunk, ChunkingConfig, chunk_corpus
from rag_eval_lab.config import ExperimentConfig, RunConfig
from rag_eval_lab.corpus import Document, GoldenItem, load_corpus, load_golden_set
from rag_eval_lab.embeddings import CachedEmbedder, Embedder, EmbeddingConfig, build_embedder
from rag_eval_lab.generation import format_context, generate_answer
from rag_eval_lab.judge import judge_correctness, judge_faithfulness
from rag_eval_lab.llm import LLM, LLMConfig, build_llm
from rag_eval_lab.metrics import (
    hit_at_k,
    mean,
    percentile,
    recall_at_k,
    reciprocal_rank,
    unique_in_order,
)
from rag_eval_lab.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    LLMReranker,
    Retriever,
)


@dataclass
class QuestionResult:
    experiment: str
    question_id: str
    question: str
    tags: list[str]
    expected_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    retrieved_chunk_ids: list[str]
    recall: float | None
    reciprocal_rank: float | None
    hit: float | None
    retrieval_ms: float
    answer: str | None = None
    generation_ms: float | None = None
    faithfulness: float | None = None
    faithfulness_reason: str | None = None
    correctness: float | None = None
    correctness_reason: str | None = None


@dataclass
class ExperimentSummary:
    experiment: str
    chunking: str
    embedding: str
    retrieval: str
    rerank: str
    num_chunks: int
    index_seconds: float
    num_questions: int
    recall_at_k: float | None
    mrr: float | None
    hit_rate: float | None
    faithfulness: float | None
    correctness: float | None
    retrieval_ms_mean: float | None
    retrieval_ms_p95: float | None
    generation_ms_mean: float | None


@dataclass
class RunResult:
    config: RunConfig
    summaries: list[ExperimentSummary] = field(default_factory=list)
    questions: list[QuestionResult] = field(default_factory=list)


LLMFactory = Callable[[LLMConfig], LLM]
EmbedderFactory = Callable[[EmbeddingConfig], Embedder]


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


class Runner:
    """Builds indexes, runs retrieval/generation/judging and aggregates metrics.

    Factories are injectable so tests can swap in fake embedders and LLMs.
    """

    def __init__(
        self,
        config: RunConfig,
        *,
        embedder_factory: EmbedderFactory = build_embedder,
        llm_factory: LLMFactory = build_llm,
        log: Callable[[str], None] = _log,
    ) -> None:
        self.config = config
        self.embedder_factory = embedder_factory
        self.llm_factory = llm_factory
        self.log = log
        self._embedders: dict[tuple, CachedEmbedder] = {}
        self._chunks: dict[ChunkingConfig, list[Chunk]] = {}
        self._dense: dict[tuple, tuple[DenseRetriever, float]] = {}
        self._generator: LLM | None = None
        self._judge: LLM | None = None

    # -- lazily constructed components -------------------------------------------------

    def _embedder(self, cfg: EmbeddingConfig) -> CachedEmbedder:
        key = (cfg.provider, cfg.model, cfg.query_prefix, cfg.document_prefix)
        if key not in self._embedders:
            self._embedders[key] = CachedEmbedder(self.embedder_factory(cfg), self.config.cache_dir)
        return self._embedders[key]

    @property
    def generator(self) -> LLM:
        if self._generator is None:
            self._generator = self.llm_factory(self.config.generator)
        return self._generator

    @property
    def judge(self) -> LLM:
        if self._judge is None:
            self._judge = self.llm_factory(self.config.judge_llm)
        return self._judge

    def _get_chunks(self, docs: list[Document], cfg: ChunkingConfig) -> list[Chunk]:
        if cfg not in self._chunks:
            self._chunks[cfg] = chunk_corpus(docs, cfg)
        return self._chunks[cfg]

    def _dense_retriever(
        self, chunks: list[Chunk], exp: ExperimentConfig
    ) -> tuple[DenseRetriever, float]:
        e = exp.embedding
        key = (exp.chunking, e.provider, e.model, e.query_prefix, e.document_prefix)
        if key not in self._dense:
            start = time.perf_counter()
            retriever = DenseRetriever(chunks, self._embedder(e))
            self._dense[key] = (retriever, time.perf_counter() - start)
        return self._dense[key]

    def build_retriever(
        self, docs: list[Document], exp: ExperimentConfig
    ) -> tuple[Retriever, int, float]:
        """Return (retriever, number of chunks, seconds spent indexing)."""
        chunks = self._get_chunks(docs, exp.chunking)
        r = exp.retrieval
        index_seconds = 0.0
        retriever: Retriever
        if r.method == "bm25":
            start = time.perf_counter()
            retriever = BM25Retriever(chunks, k1=r.bm25_k1, b=r.bm25_b)
            index_seconds = time.perf_counter() - start
        else:
            dense, index_seconds = self._dense_retriever(chunks, exp)
            if r.method == "dense":
                retriever = dense
            else:
                start = time.perf_counter()
                sparse = BM25Retriever(chunks, k1=r.bm25_k1, b=r.bm25_b)
                index_seconds += time.perf_counter() - start
                retriever = HybridRetriever(dense, sparse, rrf_k=r.rrf_k, candidates=r.candidates)
        if exp.rerank.enabled:
            retriever = LLMReranker(retriever, self.generator, candidates=exp.rerank.candidates)
        return retriever, len(chunks), index_seconds

    # -- evaluation ----------------------------------------------------------------------

    def run(self) -> RunResult:
        cfg = self.config
        docs = load_corpus(cfg.corpus_dir)
        golden = load_golden_set(cfg.golden_set, {d.doc_id for d in docs})
        if cfg.limit:
            golden = golden[: int(cfg.limit)]
        result = RunResult(config=cfg)
        self.log(
            f"Loaded {len(docs)} documents and {len(golden)} questions; "
            f"running {len(cfg.experiments)} experiment(s)."
        )
        try:
            for i, exp in enumerate(cfg.experiments, start=1):
                self.log(f"[{i}/{len(cfg.experiments)}] {exp.name}")
                summary, rows = self.run_experiment(docs, golden, exp)
                result.summaries.append(summary)
                result.questions.extend(rows)
        finally:
            for embedder in self._embedders.values():
                embedder.save()
        return result

    def run_experiment(
        self, docs: list[Document], golden: list[GoldenItem], exp: ExperimentConfig
    ) -> tuple[ExperimentSummary, list[QuestionResult]]:
        cfg = self.config
        retriever, num_chunks, index_seconds = self.build_retriever(docs, exp)
        rows: list[QuestionResult] = []
        for item in golden:
            start = time.perf_counter()
            hits = retriever.retrieve(item.question, cfg.top_k)
            retrieval_ms = (time.perf_counter() - start) * 1000
            doc_ranking = unique_in_order(h.chunk.doc_id for h in hits)
            row = QuestionResult(
                experiment=exp.name,
                question_id=item.id,
                question=item.question,
                tags=list(item.tags),
                expected_doc_ids=list(item.doc_ids),
                retrieved_doc_ids=doc_ranking,
                retrieved_chunk_ids=[h.chunk.chunk_id for h in hits],
                recall=recall_at_k(doc_ranking, item.doc_ids) if item.answerable else None,
                reciprocal_rank=reciprocal_rank(doc_ranking, item.doc_ids)
                if item.answerable
                else None,
                hit=hit_at_k(doc_ranking, item.doc_ids) if item.answerable else None,
                retrieval_ms=retrieval_ms,
            )
            if cfg.generate:
                start = time.perf_counter()
                row.answer = generate_answer(self.generator, item.question, hits)
                row.generation_ms = (time.perf_counter() - start) * 1000
                if cfg.judge:
                    faith = judge_faithfulness(
                        self.judge, item.question, format_context(hits), row.answer
                    )
                    correct = judge_correctness(
                        self.judge, item.question, item.expected_answer, row.answer
                    )
                    row.faithfulness, row.faithfulness_reason = faith.normalized, faith.reason
                    row.correctness, row.correctness_reason = correct.normalized, correct.reason
            rows.append(row)

        summary = ExperimentSummary(
            experiment=exp.name,
            chunking=exp.chunking.label,
            embedding=exp.embedding_label,
            retrieval=exp.retrieval.method,
            rerank=f"llm@{exp.rerank.candidates}" if exp.rerank.enabled else "none",
            num_chunks=num_chunks,
            index_seconds=index_seconds,
            num_questions=len(rows),
            recall_at_k=mean(r.recall for r in rows),
            mrr=mean(r.reciprocal_rank for r in rows),
            hit_rate=mean(r.hit for r in rows),
            faithfulness=mean(r.faithfulness for r in rows),
            correctness=mean(r.correctness for r in rows),
            retrieval_ms_mean=mean(r.retrieval_ms for r in rows),
            retrieval_ms_p95=percentile([r.retrieval_ms for r in rows], 95),
            generation_ms_mean=mean(r.generation_ms for r in rows),
        )
        return summary, rows
