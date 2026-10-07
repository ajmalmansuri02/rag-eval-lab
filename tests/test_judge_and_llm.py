import json

import httpx
import pytest

from rag_eval_lab.generation import IDK, build_answer_prompt, generate_answer
from rag_eval_lab.judge import (
    Judgment,
    judge_correctness,
    judge_faithfulness,
    parse_judgment,
)
from rag_eval_lab.llm import HeuristicLLM, LLMConfig, OllamaLLM, build_llm
from rag_eval_lab.retrieval import ScoredChunk


@pytest.mark.parametrize(
    ("raw", "score"),
    [
        ('{"score": 4, "reason": "fine"}', 4),
        ('Sure! Here you go: {"score": 5, "reason": "x"} hope that helps', 5),
        ('{"score": "3"}', 3),
        ("Score: 2. The answer is mostly wrong.", 2),
        ('{"score": 9}', None),
        ("I cannot judge this", None),
    ],
)
def test_parse_judgment(raw, score):
    assert parse_judgment(raw).score == score


def test_normalisation():
    assert Judgment(1, "").normalized == 0.0
    assert Judgment(5, "").normalized == 1.0
    assert Judgment(3, "").normalized == 0.5
    assert Judgment(None, "").normalized is None


def test_judges_send_rubric_and_inputs(fake_llm_cls):
    llm = fake_llm_cls('{"score": 5, "reason": "ok"}')
    j1 = judge_faithfulness(llm, "Q?", "CTX", "ANS")
    j2 = judge_correctness(llm, "Q?", "REF", "ANS")
    assert j1.score == j2.score == 5
    faith_prompt, json_mode = llm.calls[0]
    assert json_mode
    assert "CTX" in faith_prompt and "ANS" in faith_prompt and "Rubric" in faith_prompt
    assert "REF" in llm.calls[1][0]


def test_answer_prompt_contains_numbered_sources(small_chunks, fake_llm_cls):
    chunks = [ScoredChunk(c, 1.0) for c in small_chunks[:2]]
    prompt = build_answer_prompt("What do cats do?", chunks)
    assert "[1] source: pets" in prompt and "[2] source: food" in prompt
    assert IDK in prompt
    llm = fake_llm_cls("  They sleep.  ")
    assert generate_answer(llm, "What do cats do?", chunks) == "They sleep."


def test_heuristic_llm_handles_each_prompt_type(small_chunks):
    llm = HeuristicLLM()
    chunks = [ScoredChunk(c, 1.0) for c in small_chunks]
    answer = llm.generate(build_answer_prompt("Which planet is red?", chunks))
    assert "Mars" in answer
    unknown = llm.generate(build_answer_prompt("zebra quantum?", chunks))
    assert unknown == IDK
    judged = judge_correctness(llm, "Which planet?", "Mars is the red planet.", answer)
    assert judged.score is not None and judged.score >= 4
    faith = judge_faithfulness(llm, "q", "Mars is the red planet.", "Bananas are yellow.")
    assert faith.score == 1


def test_ollama_llm_payload():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"response": ' {"score": 3} '})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = OllamaLLM("qwen2.5:3b", base_url="http://x", client=client)
    assert llm.generate("hi", json_mode=True) == '{"score": 3}'
    assert seen["format"] == "json" and seen["stream"] is False
    assert seen["options"]["temperature"] == 0.0


def test_build_llm_offline_and_unknown():
    assert isinstance(build_llm(LLMConfig(provider="offline")), HeuristicLLM)
    with pytest.raises(ValueError):
        build_llm(LLMConfig(provider="nope"))


def test_gemini_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        build_llm(LLMConfig(provider="gemini", model="m"))
