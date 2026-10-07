"""Loading the document corpus and the golden question set."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

SKIPPED_FILES = {"readme.md"}


@dataclass(frozen=True)
class Document:
    """One source document. ``doc_id`` is the file stem, e.g. ``pricing-plans``."""

    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class GoldenItem:
    """One evaluation question with its reference answer and supporting documents."""

    id: str
    question: str
    expected_answer: str
    doc_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def answerable(self) -> bool:
        return bool(self.doc_ids)


def _title_from_markdown(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def load_corpus(corpus_dir: str | Path) -> list[Document]:
    """Load every ``*.md`` file in ``corpus_dir`` (except README) as a :class:`Document`."""
    path = Path(corpus_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {path}")
    docs: list[Document] = []
    for file in sorted(path.glob("*.md")):
        if file.name.lower() in SKIPPED_FILES:
            continue
        text = file.read_text(encoding="utf-8")
        docs.append(
            Document(doc_id=file.stem, title=_title_from_markdown(text, file.stem), text=text)
        )
    if not docs:
        raise ValueError(f"No markdown documents found in {path}")
    return docs


def load_golden_set(path: str | Path, known_doc_ids: set[str] | None = None) -> list[GoldenItem]:
    """Load the golden set YAML. If ``known_doc_ids`` is given, validate references against it."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Golden set {path} must be a YAML list of items")
    items: list[GoldenItem] = []
    seen: set[str] = set()
    for entry in raw:
        item = GoldenItem(
            id=str(entry["id"]),
            question=str(entry["question"]).strip(),
            expected_answer=str(entry["expected_answer"]).strip(),
            doc_ids=tuple(entry.get("doc_ids") or ()),
            tags=tuple(entry.get("tags") or ()),
        )
        if item.id in seen:
            raise ValueError(f"Duplicate golden item id: {item.id}")
        seen.add(item.id)
        if known_doc_ids is not None:
            unknown = set(item.doc_ids) - known_doc_ids
            if unknown:
                raise ValueError(
                    f"Golden item {item.id} references unknown docs: {sorted(unknown)}"
                )
        items.append(item)
    return items
