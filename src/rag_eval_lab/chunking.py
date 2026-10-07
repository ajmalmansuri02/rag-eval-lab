"""Chunking strategies.

Sizes are measured in *words* (whitespace-separated tokens). Real tokenizers produce roughly
1.3 tokens per English word, so ``chunk_size: 200`` is about 260 model tokens. Words keep the lab
dependency-free and make the numbers easy to reason about.

Two strategies:

``fixed``
    A sliding window of ``chunk_size`` words with ``overlap`` words shared between neighbours.
    Ignores document structure entirely, so chunks can start mid-sentence and mix sections.

``recursive``
    Heading-aware. The document is first split into sections by markdown headings. Sections that
    are too large are split recursively on paragraph breaks, then line breaks, then sentence
    ends, then words, and the pieces are greedily merged back up to ``chunk_size``. Each chunk is
    prefixed with its heading path ("Doc title > Section") so it is self-describing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from rag_eval_lab.corpus import Document

Strategy = Literal["fixed", "recursive"]

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    position: int
    metadata: dict[str, str] = field(default_factory=dict, hash=False, compare=False)


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: Strategy = "recursive"
    chunk_size: int = 200
    overlap: int = 20
    include_headings: bool = True

    def __post_init__(self) -> None:
        if self.strategy not in ("fixed", "recursive"):
            raise ValueError(f"Unknown chunking strategy: {self.strategy!r}")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.overlap < self.chunk_size:
            raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    @property
    def label(self) -> str:
        return f"{self.strategy}-{self.chunk_size}/{self.overlap}"


def _word_count(text: str) -> int:
    return len(text.split())


def fixed_size_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split ``text`` into ``chunk_size``-word windows; neighbours share ``overlap`` words."""
    words = text.split()
    if not words:
        return []
    step = chunk_size - overlap
    pieces: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        pieces.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return pieces


def split_markdown_sections(text: str) -> list[tuple[list[str], str]]:
    """Split markdown into ``(heading_path, body)`` sections.

    Headings inside fenced code blocks are ignored. The heading path tracks nesting, e.g.
    ``["Driftbox Overview", "Network ports"]``.
    """
    sections: list[tuple[list[str], str]] = []
    path: list[tuple[int, str]] = []
    buffer: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append(([title for _, title in path], body))
        buffer.clear()

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        match = None if in_fence else _HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            path = [(lvl, title) for lvl, title in path if lvl < level]
            path.append((level, match.group(2).strip()))
        else:
            buffer.append(line)
    flush()
    return sections


def _split_recursive(text: str, chunk_size: int, separators: tuple[str, ...]) -> list[str]:
    """Break ``text`` into pieces of at most ``chunk_size`` words, coarse separators first."""
    if _word_count(text) <= chunk_size:
        return [text]
    if not separators:
        return fixed_size_chunks(text, chunk_size, 0)
    sep, rest = separators[0], separators[1:]
    parts = [p for p in text.split(sep) if p.strip()]
    if len(parts) <= 1:
        return _split_recursive(text, chunk_size, rest)
    pieces: list[str] = []
    for i, part in enumerate(parts):
        # Keep the sentence terminator with the sentence.
        if sep == ". " and i < len(parts) - 1:
            part = part + "."
        pieces.extend(_split_recursive(part, chunk_size, rest))
    return pieces


def _merge_pieces(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily pack small pieces into chunks of up to ``chunk_size`` words.

    When a chunk is closed, its trailing ``overlap`` words are carried into the next chunk.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for piece in pieces:
        n = _word_count(piece)
        if current and current_words + n > chunk_size:
            chunks.append("\n".join(current))
            tail = " ".join(" ".join(current).split()[-overlap:]) if overlap else ""
            current = [tail] if tail and _word_count(tail) + n <= chunk_size else []
            current_words = _word_count(" ".join(current))
        current.append(piece)
        current_words += n
    if current:
        chunks.append("\n".join(current))
    return chunks


def recursive_chunks(
    text: str, chunk_size: int, overlap: int, include_headings: bool = True
) -> list[tuple[str, str]]:
    """Heading-aware recursive chunking. Returns ``(heading_path, chunk_text)`` pairs."""
    out: list[tuple[str, str]] = []
    for headings, body in split_markdown_sections(text):
        heading_path = " > ".join(headings)
        prefix = f"{heading_path}\n" if include_headings and heading_path else ""
        budget = max(chunk_size - _word_count(prefix), max(1, chunk_size // 2))
        pieces = _split_recursive(body, budget, _SEPARATORS)
        for chunk in _merge_pieces(pieces, budget, min(overlap, budget - 1)):
            out.append((heading_path, prefix + chunk))
    return out


def chunk_document(doc: Document, config: ChunkingConfig) -> list[Chunk]:
    if config.strategy == "fixed":
        texts = [("", t) for t in fixed_size_chunks(doc.text, config.chunk_size, config.overlap)]
    else:
        texts = recursive_chunks(
            doc.text, config.chunk_size, config.overlap, config.include_headings
        )
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}#{i}",
            doc_id=doc.doc_id,
            text=text,
            position=i,
            metadata={"heading": heading} if heading else {},
        )
        for i, (heading, text) in enumerate(texts)
    ]


def chunk_corpus(docs: list[Document], config: ChunkingConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, config))
    return chunks
