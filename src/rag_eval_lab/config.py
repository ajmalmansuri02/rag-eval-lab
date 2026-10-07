"""YAML configuration: one file describes a run made of several experiments.

Every experiment is ``defaults`` deep-merged with the experiment's own overrides, so a sweep
only has to spell out what changes. See ``configs/`` for examples.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rag_eval_lab.chunking import ChunkingConfig
from rag_eval_lab.embeddings import EmbeddingConfig
from rag_eval_lab.env import env
from rag_eval_lab.llm import LLMConfig

PROVIDERS = ("ollama", "gemini", "offline")


@dataclass(frozen=True)
class RetrievalConfig:
    method: str = "dense"  # dense | bm25 | hybrid
    rrf_k: int = 60
    candidates: int = 20  # depth of each ranked list fused by hybrid retrieval
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    def __post_init__(self) -> None:
        if self.method not in ("dense", "bm25", "hybrid"):
            raise ValueError(f"Unknown retrieval method: {self.method!r}")


@dataclass(frozen=True)
class RerankConfig:
    enabled: bool = False
    candidates: int = 10


@dataclass
class ExperimentConfig:
    name: str
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)

    @property
    def uses_embeddings(self) -> bool:
        return self.retrieval.method in ("dense", "hybrid")

    @property
    def embedding_label(self) -> str:
        return self.embedding.label if self.uses_embeddings else "-"


@dataclass
class RunConfig:
    name: str
    experiments: list[ExperimentConfig]
    corpus_dir: Path = Path("data/corpus")
    golden_set: Path = Path("data/golden/qa.yaml")
    output_dir: Path = Path("results")
    cache_dir: Path | None = Path(".cache/embeddings")
    top_k: int = 5
    limit: int | None = None
    generate: bool = True
    judge: bool = True
    generator: LLMConfig = field(default_factory=LLMConfig)
    judge_llm: LLMConfig = field(default_factory=lambda: LLMConfig(model="qwen2.5:7b"))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("corpus_dir", "golden_set", "output_dir", "cache_dir"):
            data[key] = str(data[key]) if data[key] is not None else None
        return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _experiment_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    if "name" not in data:
        raise ValueError("every experiment needs a name")
    known = {"name", "chunking", "embedding", "retrieval", "rerank"}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"experiment {data['name']!r} has unknown keys: {sorted(unknown)}")
    return ExperimentConfig(
        name=str(data["name"]),
        chunking=ChunkingConfig(**(data.get("chunking") or {})),
        embedding=EmbeddingConfig(**(data.get("embedding") or {})),
        retrieval=RetrievalConfig(**(data.get("retrieval") or {})),
        rerank=RerankConfig(**(data.get("rerank") or {})),
    )


def apply_provider_override(config: RunConfig, provider: str) -> RunConfig:
    """Point every component at one provider (handy for ``--provider offline`` smoke runs).

    For ``gemini`` the model names come from ``GEMINI_MODEL`` / ``GEMINI_EMBED_MODEL``.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"provider must be one of {PROVIDERS}, got {provider!r}")
    if provider == "ollama":
        return config
    llm_model = "" if provider == "gemini" else "offline"
    config.generator = LLMConfig(provider=provider, model=llm_model)
    config.judge_llm = LLMConfig(provider=provider, model=llm_model)
    for exp in config.experiments:
        if provider == "offline":
            exp.embedding = EmbeddingConfig(provider="hashing", model="hashing")
        else:
            exp.embedding = EmbeddingConfig(provider="gemini", model="")
    return config


def load_config(path: str | Path, provider: str | None = None) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    defaults = raw.get("defaults") or {}
    experiments_raw = raw.get("experiments") or []
    if not experiments_raw:
        raise ValueError(f"{path}: no experiments defined")
    experiments = [_experiment_from_dict(deep_merge(defaults, e)) for e in experiments_raw]
    names = [e.name for e in experiments]
    if len(names) != len(set(names)):
        raise ValueError(f"{path}: experiment names must be unique")

    cache_dir = raw.get("cache_dir", ".cache/embeddings")
    config = RunConfig(
        name=str(raw.get("name") or Path(path).stem),
        experiments=experiments,
        corpus_dir=Path(raw.get("corpus_dir", "data/corpus")),
        golden_set=Path(raw.get("golden_set", "data/golden/qa.yaml")),
        output_dir=Path(raw.get("output_dir", "results")),
        cache_dir=Path(cache_dir) if cache_dir else None,
        top_k=int(raw.get("top_k", 5)),
        limit=raw.get("limit"),
        generate=bool(raw.get("generate", True)),
        judge=bool(raw.get("judge", True)),
        generator=LLMConfig(**(raw.get("generator") or {})),
        judge_llm=LLMConfig(**(raw.get("judge_llm") or {"model": "qwen2.5:7b"})),
    )
    if config.judge and not config.generate:
        config.judge = False
    provider = provider or env("RAG_LAB_PROVIDER")
    if provider:
        config = apply_provider_override(config, provider)
    return config
