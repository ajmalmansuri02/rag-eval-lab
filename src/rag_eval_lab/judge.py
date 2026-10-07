"""LLM-as-judge for answer quality.

Two separate judgments, each on a 1-5 rubric and normalised to 0-1 via (score - 1) / 4:

- **Faithfulness**: is every claim in the answer supported by the retrieved context?
  Measures hallucination. It does not care whether the answer is right.
- **Correctness**: does the answer convey the same facts as the reference answer?
  Measures end-to-end usefulness. It does not care where the facts came from.

Keeping them separate lets you tell "retrieval failed, model guessed right" apart from
"retrieval worked, model ignored it". Judges are noisy: use a larger model than the generator
when you can, run with temperature 0, and spot-check the reasons in ``per_question.csv``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from rag_eval_lab.llm import LLM

FAITHFULNESS_RUBRIC = """\
5 - Every claim in the answer is directly supported by the context, OR the answer correctly \
says the context does not contain the information.
4 - All key claims are supported; a minor detail is not explicitly in the context.
3 - Some claims are supported, but at least one meaningful claim is not.
2 - Most claims are not supported by the context.
1 - The answer contradicts the context or is fabricated."""

CORRECTNESS_RUBRIC = """\
5 - Fully correct: conveys the same key facts as the reference answer (wording may differ).
4 - Correct on the main point but misses a secondary detail from the reference.
3 - Partially correct: gets something right but misses or garbles a key fact.
2 - Mostly incorrect, with only a small overlap with the reference.
1 - Incorrect, or says "I don't know" when the reference contains an answer.
Special case: if the reference says the documentation does not contain the answer, score 5 \
when the candidate declines to answer and 1 when it invents an answer."""

FAITHFULNESS_PROMPT = """You are a strict evaluator checking a RAG system for hallucination.
Judge ONLY whether the candidate answer is supported by the retrieved context. Ignore whether \
it is correct in the real world.

Rubric:
{rubric}

Respond with JSON only: {{"score": <integer 1-5>, "reason": "<one short sentence>"}}

### Question
{question}

### Retrieved context
{context}

### Candidate answer
{answer}
"""

CORRECTNESS_PROMPT = """You are a strict evaluator comparing a candidate answer to a reference \
answer.

Rubric:
{rubric}

Respond with JSON only: {{"score": <integer 1-5>, "reason": "<one short sentence>"}}

### Question
{question}

### Reference answer
{reference}

### Candidate answer
{answer}
"""


@dataclass(frozen=True)
class Judgment:
    score: int | None  # raw 1-5, None if the judge reply could not be parsed
    reason: str

    @property
    def normalized(self) -> float | None:
        return None if self.score is None else (self.score - 1) / 4


def parse_judgment(raw: str) -> Judgment:
    """Parse ``{"score": n, "reason": "..."}`` robustly (models sometimes wrap JSON in prose)."""
    candidates = [raw]
    match = re.search(r"\{.*\}", raw, re.S)
    if match:
        candidates.append(match.group())
    for text in candidates:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "score" in data:
            try:
                score = round(float(data["score"]))
            except (TypeError, ValueError):
                break
            if 1 <= score <= 5:
                return Judgment(score, str(data.get("reason", "")).strip())
    match = re.search(r"score\D{0,5}([1-5])\b", raw, re.I)
    if match:
        return Judgment(int(match.group(1)), "parsed from free text")
    return Judgment(None, f"unparseable judge output: {raw[:120]!r}")


def judge_faithfulness(llm: LLM, question: str, context: str, answer: str) -> Judgment:
    prompt = FAITHFULNESS_PROMPT.format(
        rubric=FAITHFULNESS_RUBRIC, question=question, context=context, answer=answer
    )
    return parse_judgment(llm.generate(prompt, json_mode=True))


def judge_correctness(llm: LLM, question: str, reference: str, answer: str) -> Judgment:
    prompt = CORRECTNESS_PROMPT.format(
        rubric=CORRECTNESS_RUBRIC, question=question, reference=reference, answer=answer
    )
    return parse_judgment(llm.generate(prompt, json_mode=True))
