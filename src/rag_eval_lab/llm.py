"""Text-generation providers used for answering, reranking and judging."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

import httpx

from rag_eval_lab.env import env
from rag_eval_lab.http import RateLimiter, post_json


class LLM(Protocol):
    name: str

    def generate(self, prompt: str, *, json_mode: bool = False) -> str: ...


@dataclass
class LLMConfig:
    provider: str = "ollama"  # ollama | gemini | offline
    model: str = "qwen2.5:3b"
    temperature: float = 0.0
    num_ctx: int = 8192

    @property
    def label(self) -> str:
        return self.model if self.provider != "offline" else "offline-heuristic"


class OllamaLLM:
    """Generation via a local Ollama server (``POST /api/generate``)."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        temperature: float = 0.0,
        num_ctx: int = 8192,
        timeout: float = 300.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.name = f"ollama:{model}"
        self.base_url = (base_url or env("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.options = {"temperature": temperature, "num_ctx": num_ctx, "seed": 7}
        self._client = client or httpx.Client(timeout=timeout)

    def generate(self, prompt: str, *, json_mode: bool = False) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": self.options,
        }
        if json_mode:
            payload["format"] = "json"
        data = post_json(self._client, f"{self.base_url}/api/generate", payload)
        return str(data.get("response", "")).strip()


class GeminiLLM:
    """Generation via the Gemini API free tier. Requires ``GEMINI_API_KEY``."""

    BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = env("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set; see .env.example")
        model = model or env("GEMINI_MODEL")
        if not model:
            raise ValueError("Set a generation model in the config or GEMINI_MODEL")
        self.model = model.removeprefix("models/")
        self.name = f"gemini:{self.model}"
        self.temperature = temperature
        self._client = client or httpx.Client(timeout=timeout)
        self._limiter = RateLimiter(float(env("GEMINI_REQUESTS_PER_MINUTE", "10") or 10))

    def generate(self, prompt: str, *, json_mode: bool = False) -> str:
        config: dict[str, object] = {"temperature": self.temperature}
        if json_mode:
            config["responseMimeType"] = "application/json"
        data = post_json(
            self._client,
            f"{self.BASE}/models/{self.model}:generateContent",
            {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": config,
            },
            headers={"x-goog-api-key": self.api_key},
            limiter=self._limiter,
        )
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts).strip()


_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by can do does for from how i if in is it its my of on or the to what "
    "when where which who why will with you your this that".split()
)


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOP}


def _section(prompt: str, name: str) -> str:
    """Extract the body of a ``### name`` section from one of the lab's prompts."""
    match = re.search(rf"^### {re.escape(name)}\s*\n(.*?)(?=^### |\Z)", prompt, re.S | re.M)
    return match.group(1).strip() if match else ""


class HeuristicLLM:
    """A deterministic stand-in for an LLM, used by the offline mode.

    It is **not** a language model. It recognises the lab's own prompt templates (by their
    ``### Section`` headers) and answers with simple word-overlap heuristics:

    - answer prompts: returns the context sentence that overlaps most with the question;
    - relevance prompts (reranking): scores word overlap between query and passage;
    - judge prompts: scores word overlap between answer and reference/context.

    This keeps the whole pipeline runnable in CI and without a GPU, so you can see the moving
    parts. Its scores say nothing about real answer quality.
    """

    name = "offline-heuristic"

    def generate(self, prompt: str, *, json_mode: bool = False) -> str:
        if "### Passage" in prompt:
            return self._relevance(prompt)
        if "### Reference answer" in prompt:
            return self._overlap_judge(
                _section(prompt, "Candidate answer"), _section(prompt, "Reference answer"), f1=True
            )
        if "### Retrieved context" in prompt and "### Candidate answer" in prompt:
            return self._overlap_judge(
                _section(prompt, "Candidate answer"), _section(prompt, "Retrieved context")
            )
        if "### Context" in prompt:
            return self._answer(prompt)
        return "I don't know."

    @staticmethod
    def _answer(prompt: str) -> str:
        question = _content_words(_section(prompt, "Question"))
        context = re.sub(r"^\[\d+\].*$", "", _section(prompt, "Context"), flags=re.M)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", context) if s.strip()]
        best, best_score = "", 0
        for sentence in sentences:
            score = len(question & _content_words(sentence))
            if score > best_score:
                best, best_score = sentence, score
        return best if best_score >= 1 else "I don't know based on the provided documents."

    @staticmethod
    def _relevance(prompt: str) -> str:
        query = _content_words(_section(prompt, "Query"))
        passage = _content_words(_section(prompt, "Passage"))
        score = round(10 * len(query & passage) / max(len(query), 1))
        return json.dumps({"score": min(score, 10)})

    @staticmethod
    def _overlap_judge(candidate: str, reference: str, f1: bool = False) -> str:
        """Precision of answer words against the reference (or F1 when ``f1``), mapped to 1-5."""
        cand, ref = _content_words(candidate), _content_words(reference)
        if not cand or not ref:
            return json.dumps({"score": 1, "reason": "empty answer"})
        common = len(cand & ref)
        precision, recall = common / len(cand), common / len(ref)
        if f1:
            ratio = 2 * precision * recall / (precision + recall) if common else 0.0
        else:
            ratio = precision
        score = 1 + round(4 * ratio)
        return json.dumps({"score": score, "reason": f"word overlap {ratio:.2f} (heuristic)"})


def build_llm(config: LLMConfig) -> LLM:
    provider = config.provider.lower()
    if provider == "offline":
        return HeuristicLLM()
    if provider == "ollama":
        return OllamaLLM(config.model, temperature=config.temperature, num_ctx=config.num_ctx)
    if provider == "gemini":
        return GeminiLLM(config.model or None, temperature=config.temperature)
    raise ValueError(f"Unknown LLM provider: {config.provider!r}")
