import csv
from pathlib import Path

import pytest

from rag_eval_lab.config import load_config
from rag_eval_lab.embeddings import HashingEmbedder
from rag_eval_lab.report import markdown_table, write_report
from rag_eval_lab.runner import Runner
from tests.conftest import REPO_ROOT


@pytest.fixture
def offline_config(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    config = load_config(REPO_ROOT / "configs" / "offline.yaml")
    config.limit = 8
    return config


def test_offline_run_end_to_end(offline_config, tmp_path: Path):
    result = Runner(offline_config, log=lambda _: None).run()
    assert len(result.summaries) == len(offline_config.experiments)
    assert len(result.questions) == 8 * len(offline_config.experiments)
    for s in result.summaries:
        assert 0.0 <= s.recall_at_k <= 1.0
        assert 0.0 <= s.mrr <= s.hit_rate <= 1.0
        assert s.correctness is not None and s.faithfulness is not None
    bm25 = next(s for s in result.summaries if s.retrieval == "bm25")
    assert bm25.embedding == "-"

    out = write_report(result, tmp_path)
    for name in ("summary.md", "summary.csv", "per_question.csv", "run.json"):
        assert (out / name).is_file()
    rows = list(csv.DictReader((out / "summary.csv").open()))
    assert [r["experiment"] for r in rows] == [e.name for e in offline_config.experiments]
    assert "Offline mode" in (out / "summary.md").read_text()


def test_unanswerable_questions_are_excluded_from_retrieval_metrics(monkeypatch, fake_llm_cls):
    monkeypatch.chdir(REPO_ROOT)
    config = load_config(REPO_ROOT / "configs" / "offline.yaml")
    config.experiments = config.experiments[:1]
    llm = fake_llm_cls('{"score": 5, "reason": "fake"}')
    result = Runner(
        config,
        embedder_factory=lambda cfg: HashingEmbedder(),
        llm_factory=lambda cfg: llm,
        log=lambda _: None,
    ).run()
    unanswerable = [q for q in result.questions if not q.expected_doc_ids]
    assert unanswerable
    assert all(q.recall is None and q.hit is None for q in unanswerable)
    assert all(q.correctness == 1.0 for q in result.questions)
    assert result.summaries[0].correctness == 1.0


def test_retrieval_only_skips_llm(offline_config):
    offline_config.generate = False
    offline_config.judge = False
    offline_config.experiments = [e for e in offline_config.experiments if not e.rerank.enabled]

    def no_llm(cfg):
        raise AssertionError("LLM should not be built")

    result = Runner(offline_config, llm_factory=no_llm, log=lambda _: None).run()
    assert all(q.answer is None for q in result.questions)
    assert all(s.correctness is None for s in result.summaries)


def test_dense_index_is_shared_between_experiments(offline_config):
    built = []

    def factory(cfg):
        built.append(cfg)
        return HashingEmbedder()

    Runner(offline_config, embedder_factory=factory, log=lambda _: None).run()
    assert len(built) == 1


def test_markdown_table_bolds_best(offline_config):
    offline_config.generate = False
    offline_config.experiments = [e for e in offline_config.experiments if not e.rerank.enabled]
    result = Runner(offline_config, log=lambda _: None).run()
    table = markdown_table(result.summaries)
    lines = table.splitlines()
    assert lines[0].startswith("| Experiment")
    assert len(lines) == 2 + len(result.summaries)
    assert "**" in table
