import pytest

from rag_eval_lab.chunking import (
    ChunkingConfig,
    chunk_corpus,
    chunk_document,
    fixed_size_chunks,
    recursive_chunks,
    split_markdown_sections,
)
from rag_eval_lab.corpus import Document, load_corpus
from tests.conftest import CORPUS_DIR

WORDS = " ".join(f"w{i}" for i in range(25))


def test_fixed_size_windows_and_overlap():
    chunks = fixed_size_chunks(WORDS, chunk_size=10, overlap=3)
    assert [len(c.split()) for c in chunks] == [10, 10, 10, 4]
    first, second = chunks[0].split(), chunks[1].split()
    assert first[-3:] == second[:3]
    # every word is covered
    assert set(WORDS.split()) == {w for c in chunks for w in c.split()}


def test_fixed_size_short_and_empty_text():
    assert fixed_size_chunks("one two", 10, 2) == ["one two"]
    assert fixed_size_chunks("   ", 10, 2) == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"chunk_size": 0}, "positive"),
        ({"chunk_size": 10, "overlap": 10}, "overlap"),
        ({"strategy": "semantic"}, "Unknown"),
    ],
)
def test_config_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ChunkingConfig(**kwargs)


def test_split_markdown_sections_tracks_heading_path_and_ignores_code_fences():
    text = "# Title\nintro\n## A\nalpha\n```\n# not a heading\n```\n### A1\ndeep\n## B\nbeta\n"
    sections = split_markdown_sections(text)
    paths = [tuple(p) for p, _ in sections]
    assert paths == [("Title",), ("Title", "A"), ("Title", "A", "A1"), ("Title", "B")]
    assert "# not a heading" in sections[1][1]


def test_recursive_chunks_respect_size_and_prefix_headings():
    body = "\n\n".join(f"Paragraph {i} " + "word " * 30 for i in range(6))
    text = f"# Doc\n\n## Section\n\n{body}"
    chunks = recursive_chunks(text, chunk_size=80, overlap=10)
    assert len(chunks) > 1
    for heading, chunk in chunks:
        assert heading == "Doc > Section"
        assert chunk.startswith("Doc > Section\n")
        assert len(chunk.split()) <= 80


def test_recursive_chunks_never_mix_sections():
    text = "# T\n## One\napple apple apple\n## Two\nbanana banana banana\n"
    chunks = recursive_chunks(text, chunk_size=200, overlap=0)
    assert len(chunks) == 2
    assert "banana" not in chunks[0][1] and "apple" not in chunks[1][1]


def test_recursive_handles_single_giant_word_run():
    text = "# T\n" + "x " * 500
    chunks = recursive_chunks(text, chunk_size=50, overlap=5)
    assert all(len(c.split()) <= 50 for _, c in chunks)


def test_recursive_without_headings():
    chunks = recursive_chunks("# T\n## S\nhello world", 50, 0, include_headings=False)
    assert chunks == [("T > S", "hello world")]


def test_chunk_ids_are_unique_and_positional():
    doc = Document("d", "D", "# D\n" + "\n\n".join("para " * 40 for _ in range(4)))
    chunks = chunk_document(doc, ChunkingConfig(chunk_size=50, overlap=5))
    assert [c.chunk_id for c in chunks] == [f"d#{i}" for i in range(len(chunks))]
    assert all(c.doc_id == "d" for c in chunks)


@pytest.mark.parametrize("strategy", ["fixed", "recursive"])
def test_whole_corpus_chunks(strategy):
    docs = load_corpus(CORPUS_DIR)
    chunks = chunk_corpus(docs, ChunkingConfig(strategy=strategy, chunk_size=100, overlap=10))
    assert {c.doc_id for c in chunks} == {d.doc_id for d in docs}
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert max(len(c.text.split()) for c in chunks) <= 100
