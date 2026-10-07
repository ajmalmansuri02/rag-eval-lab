from pathlib import Path

import pytest

from rag_eval_lab.corpus import load_corpus, load_golden_set
from tests.conftest import CORPUS_DIR, GOLDEN_PATH


def test_bundled_corpus_loads_and_skips_readme():
    docs = load_corpus(CORPUS_DIR)
    ids = {d.doc_id for d in docs}
    assert 12 <= len(docs) <= 20
    assert "README" not in ids and "readme" not in ids
    assert all(d.title and d.text for d in docs)


def test_bundled_golden_set_references_real_docs():
    docs = load_corpus(CORPUS_DIR)
    items = load_golden_set(GOLDEN_PATH, {d.doc_id for d in docs})
    assert len(items) >= 30
    assert len({i.id for i in items}) == len(items)
    assert any(not i.answerable for i in items), "expected some unanswerable questions"
    assert any(len(i.doc_ids) > 1 for i in items), "expected some multi-doc questions"


def test_golden_set_rejects_unknown_doc_ids(tmp_path: Path):
    path = tmp_path / "qa.yaml"
    path.write_text("- {id: a, question: q, expected_answer: x, doc_ids: [nope]}\n")
    with pytest.raises(ValueError, match="unknown docs"):
        load_golden_set(path, {"real"})


def test_golden_set_rejects_duplicate_ids(tmp_path: Path):
    path = tmp_path / "qa.yaml"
    path.write_text(
        "- {id: a, question: q, expected_answer: x}\n- {id: a, question: q2, expected_answer: y}\n"
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_golden_set(path)


def test_missing_corpus_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "missing")
