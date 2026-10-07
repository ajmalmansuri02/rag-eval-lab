"""Answer generation: build a grounded prompt from retrieved chunks and call the LLM."""

from __future__ import annotations

from collections.abc import Sequence

from rag_eval_lab.llm import LLM
from rag_eval_lab.retrieval import ScoredChunk

IDK = "I don't know based on the provided documents."

ANSWER_PROMPT = """You answer questions about the Driftbox product using ONLY the context below.
Rules:
- If the context does not contain the answer, reply exactly: "{idk}"
- Be concise: one to three sentences.
- Do not use outside knowledge.

### Context
{context}

### Question
{question}

### Answer
"""


def format_context(chunks: Sequence[ScoredChunk]) -> str:
    blocks = [
        f"[{i}] source: {item.chunk.doc_id}\n{item.chunk.text}" for i, item in enumerate(chunks, 1)
    ]
    return "\n\n".join(blocks)


def build_answer_prompt(question: str, chunks: Sequence[ScoredChunk]) -> str:
    return ANSWER_PROMPT.format(idk=IDK, context=format_context(chunks), question=question)


def generate_answer(llm: LLM, question: str, chunks: Sequence[ScoredChunk]) -> str:
    return llm.generate(build_answer_prompt(question, chunks)).strip()
