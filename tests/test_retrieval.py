import json

import pytest

from rag_eval_lab.embeddings import HashingEmbedder
from rag_eval_lab.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    LLMReranker,
    ScoredChunk,
    parse_relevance,
    reciprocal_rank_fusion,
)


def _ranking(chunks, ids):
    by_id = {c.chunk_id: c for c in chunks}
    return [ScoredChunk(by_id[i], 1.0) for i in ids]


def test_rrf_rewards_agreement(small_chunks):
    a = _ranking(small_chunks, ["pets#0", "food#0", "space#0"])
    b = _ranking(small_chunks, ["food#0", "space#0", "pets#0"])
    fused = reciprocal_rank_fusion([a, b], k=60)
    # food is 2nd + 1st, pets is 1st + 3rd: food wins
    assert [s.chunk.chunk_id for s in fused][:2] == ["food#0", "pets#0"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_includes_items_from_any_list_and_supports_weights(small_chunks):
    a = _ranking(small_chunks, ["pets#0"])
    b = _ranking(small_chunks, ["space#0"])
    fused = reciprocal_rank_fusion([a, b], weights=[1.0, 2.0])
    assert [s.chunk.chunk_id for s in fused] == ["space#0", "pets#0"]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([a, b], weights=[1.0])


def test_dense_bm25_and_hybrid_find_the_right_doc(small_chunks):
    dense = DenseRetriever(small_chunks, HashingEmbedder())
    sparse = BM25Retriever(small_chunks)
    hybrid = HybridRetriever(dense, sparse, candidates=3)
    for retriever in (dense, sparse, hybrid):
        top = retriever.retrieve("which planet is red? mars", 2)
        assert top[0].chunk.doc_id == "space"
        assert len(top) <= 2


def test_parse_relevance():
    assert parse_relevance('{"score": 7}') == 7
    assert parse_relevance("I'd say 12 out of 10") == 10
    assert parse_relevance("no idea") == 0
    assert parse_relevance('{"score": "bad"}') == 0


def test_llm_reranker_reorders_by_llm_score(small_chunks, fake_llm_cls):
    def responder(prompt: str, json_mode: bool) -> str:
        assert json_mode
        return json.dumps({"score": 9 if "Pizza" in prompt else 1})

    llm = fake_llm_cls(responder)
    base = BM25Retriever(small_chunks)
    # Every doc matches "is", so make a query that hits several docs lexically
    reranker = LLMReranker(
        HybridRetriever(DenseRetriever(small_chunks, HashingEmbedder()), base, candidates=3),
        llm,
        candidates=3,
    )
    top = reranker.retrieve("cats dogs pizza mars", 2)
    assert top[0].chunk.doc_id == "food"
    assert len(top) == 2
    assert len(llm.calls) == 3


def test_reranker_ties_keep_first_stage_order(small_chunks, fake_llm_cls):
    base = DenseRetriever(small_chunks, HashingEmbedder())
    first = [s.chunk.chunk_id for s in base.retrieve("cats pizza mars", 3)]
    reranker = LLMReranker(base, fake_llm_cls('{"score": 5}'), candidates=3)
    assert [s.chunk.chunk_id for s in reranker.retrieve("cats pizza mars", 3)] == first
