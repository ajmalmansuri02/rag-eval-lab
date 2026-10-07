from pathlib import Path

import pytest

from rag_eval_lab.config import deep_merge, load_config
from tests.conftest import REPO_ROOT

CONFIGS = sorted((REPO_ROOT / "configs").glob("*.yaml"))


def test_deep_merge_does_not_mutate():
    base = {"a": {"x": 1, "y": 2}, "b": 1}
    out = deep_merge(base, {"a": {"y": 3}})
    assert out == {"a": {"x": 1, "y": 3}, "b": 1}
    assert base["a"]["y"] == 2


@pytest.mark.parametrize("path", CONFIGS, ids=[p.name for p in CONFIGS])
def test_bundled_configs_parse(path: Path, monkeypatch):
    monkeypatch.delenv("RAG_LAB_PROVIDER", raising=False)
    config = load_config(path)
    assert config.experiments
    assert len({e.name for e in config.experiments}) == len(config.experiments)


def test_defaults_merge_into_experiments(tmp_path: Path):
    path = tmp_path / "c.yaml"
    path.write_text(
        """
name: t
defaults:
  chunking: {strategy: fixed, chunk_size: 50, overlap: 5}
  retrieval: {method: dense}
experiments:
  - name: a
  - name: b
    chunking: {chunk_size: 80}
    retrieval: {method: hybrid}
"""
    )
    config = load_config(path, provider="ollama")
    a, b = config.experiments
    assert a.chunking.chunk_size == 50 and a.chunking.strategy == "fixed"
    assert b.chunking.chunk_size == 80 and b.chunking.strategy == "fixed"
    assert b.retrieval.method == "hybrid"


def test_provider_override_offline():
    config = load_config(REPO_ROOT / "configs" / "quick.yaml", provider="offline")
    assert config.generator.provider == "offline"
    assert all(e.embedding.provider == "hashing" for e in config.experiments)


def test_env_provider_override(monkeypatch):
    monkeypatch.setenv("RAG_LAB_PROVIDER", "offline")
    config = load_config(REPO_ROOT / "configs" / "quick.yaml")
    assert config.judge_llm.provider == "offline"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("experiments: []", "no experiments"),
        ("experiments: [{name: a}, {name: a}]", "unique"),
        ("experiments: [{name: a, typo: 1}]", "unknown keys"),
        ("experiments: [{name: a, retrieval: {method: magic}}]", "retrieval method"),
    ],
)
def test_invalid_configs(tmp_path: Path, body: str, message: str):
    path = tmp_path / "bad.yaml"
    path.write_text(body)
    with pytest.raises(ValueError, match=message):
        load_config(path, provider="ollama")
