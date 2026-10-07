"""Shared fixtures. Nothing here needs Ollama, Gemini or the network."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from rag_eval_lab.chunking import Chunk
from rag_eval_lab.corpus import Document

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "data" / "corpus"
GOLDEN_PATH = REPO_ROOT / "data" / "golden" / "qa.yaml"


class FakeLLM:
    """Scripted LLM: ``responder(prompt, json_mode)`` decides the reply; calls are recorded."""

    name = "fake"

    def __init__(self, responder: Callable[[str, bool], str] | str = "ok") -> None:
        self.responder = responder
        self.calls: list[tuple[str, bool]] = []

    def generate(self, prompt: str, *, json_mode: bool = False) -> str:
        self.calls.append((prompt, json_mode))
        if callable(self.responder):
            return self.responder(prompt, json_mode)
        return self.responder


@pytest.fixture
def fake_llm_cls() -> type[FakeLLM]:
    return FakeLLM


@pytest.fixture
def small_docs() -> list[Document]:
    return [
        Document("pets", "Pets", "# Pets\n\nCats sleep a lot. Dogs love long walks in the park."),
        Document("food", "Food", "# Food\n\nPizza is baked in a hot oven. Sushi uses raw fish."),
        Document("space", "Space", "# Space\n\nMars is the red planet. Jupiter is a gas giant."),
    ]


@pytest.fixture
def small_chunks(small_docs: list[Document]) -> list[Chunk]:
    return [Chunk(f"{d.doc_id}#0", d.doc_id, d.text, 0) for d in small_docs]
