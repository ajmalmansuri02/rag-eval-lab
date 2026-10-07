import math

from rag_eval_lab.bm25 import BM25, tokenize


def test_tokenize_keeps_identifiers_and_drops_stopwords():
    tokens = tokenize("What is the dbx_pat_ prefix on port 8470? Use driftd.toml")
    assert "dbx_pat" in tokens
    assert "8470" in tokens
    assert "driftd.toml" in tokens
    assert "the" not in tokens and "what" not in tokens


def test_bm25_prefers_matching_document():
    bm25 = BM25(["cats sleep a lot", "dogs love walks", "pizza oven hot"])
    top = bm25.top_k("where do dogs walk", 3)
    assert top[0][0] == 1
    # documents without any query term are excluded
    assert all(score > 0 for _, score in top)
    assert len(top) == 1


def test_idf_rewards_rare_terms():
    bm25 = BM25(["common rare", "common", "common", "common"])
    assert bm25.idf["rare"] > bm25.idf["common"] > 0


def test_length_normalisation():
    short = "alpha beta"
    long = "alpha " + " ".join(f"filler{i}" for i in range(50))
    scores = BM25([short, long]).scores("alpha")
    assert scores[0] > scores[1]


def test_known_score_value():
    bm25 = BM25(["a1 b1", "c1 d1"], k1=1.5, b=0.75)
    idf = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    expected = idf * 1 * 2.5 / (1 + 1.5)
    assert math.isclose(bm25.scores("a1")[0], expected)


def test_empty_index():
    assert BM25([]).scores("anything") == []
