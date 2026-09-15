"""LLMProvider abstraction (D4): the default implementation speaks the
OpenAI-compatible chat completions API over plain httpx, so Ollama, vLLM,
llama.cpp server and LM Studio are all interchangeable via base_url/model.
No provider SDK is imported.

Stage 17.3: two calls — hypotheses (LLM #1) and, conditionally, the final
wording (LLM #2). Both return the parsed response plus token/latency usage.
"""

import json
import time
from typing import Any, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from ..charts.rules import Recommendation
from ..profiling.models import DatasetProfile
from .coverage import ValidatedHypothesis
from .prompts import build_final_messages, build_hypothesis_messages
from .schemas import FinalResponse, HypothesisResponse, LLMUsage

TEMPERATURE = 0.1
MAX_TOKENS = 2000

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category  # surfaced in the fallback message (critique #8)


class LLMProvider(Protocol):
    def generate_hypotheses(
        self, profile: DatasetProfile, rule_candidates: list[Recommendation]
    ) -> tuple[HypothesisResponse, LLMUsage]: ...

    def finalize_insights(
        self, profile: DatasetProfile, validated: list[ValidatedHypothesis]
    ) -> tuple[FinalResponse, LLMUsage]: ...


class DisabledProvider:
    def generate_hypotheses(
        self, profile: DatasetProfile, rule_candidates: list[Recommendation]
    ) -> tuple[HypothesisResponse, LLMUsage]:
        return HypothesisResponse(), LLMUsage()

    def finalize_insights(
        self, profile: DatasetProfile, validated: list[ValidatedHypothesis]
    ) -> tuple[FinalResponse, LLMUsage]:
        return FinalResponse(), LLMUsage()


class OpenAICompatProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_seconds: float = 30,
        include_sample_rows: bool = True,
        transport: httpx.BaseTransport | None = None,  # tests inject MockTransport
    ) -> None:
        self._model = model
        self._include_sample_rows = include_sample_rows
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout_seconds, transport=transport
        )

    def generate_hypotheses(
        self, profile: DatasetProfile, rule_candidates: list[Recommendation]
    ) -> tuple[HypothesisResponse, LLMUsage]:
        messages = build_hypothesis_messages(profile, rule_candidates, self._include_sample_rows)
        return self._complete(messages, HypothesisResponse)

    def finalize_insights(
        self, profile: DatasetProfile, validated: list[ValidatedHypothesis]
    ) -> tuple[FinalResponse, LLMUsage]:
        messages = build_final_messages(profile, validated)
        return self._complete(messages, FinalResponse)

    # the single-call name is kept as a thin alias for older callers
    def recommend_charts(
        self, profile: DatasetProfile, rule_candidates: list[Recommendation]
    ) -> HypothesisResponse:
        return self.generate_hypotheses(profile, rule_candidates)[0]

    def _complete(self, messages: list[dict[str, str]], model_cls: type[T]) -> tuple[T, LLMUsage]:
        if not self._model:
            raise LLMError("misconfigured: model not set", "llm_model is empty")
        last: Exception | None = None
        for _ in range(2):  # one retry for connection/HTTP/parse failures
            try:
                return self._call(messages, model_cls)
            except httpx.TimeoutException as exc:
                raise LLMError("timeout", str(exc)) from exc  # timeouts never retry (critique #4)
            except (httpx.HTTPError, ValueError, ValidationError) as exc:
                last = exc
        raise LLMError("request failed after retry", str(last)) from last

    def _call(self, messages: list[dict[str, str]], model_cls: type[T]) -> tuple[T, LLMUsage]:
        started = time.perf_counter()
        response = self._client.post(
            "/chat/completions",
            json={
                "model": self._model,
                "messages": messages,
                "response_format": {"type": "json_object"},
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
            },
        )
        response.raise_for_status()
        body = response.json()
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ValueError("malformed chat completion envelope") from None
        # json_object mode is best-effort server-side; Pydantic is the authority (D4)
        extracted = _extract_json(content)
        try:
            payload = json.loads(extracted)
        except ValueError:
            # e.g. fenced JSON followed by trailing prose: retry on the
            # widest brace slice before giving up
            start, end = extracted.find("{"), extracted.rfind("}")
            if start == -1 or end <= start:
                raise
            payload = json.loads(extracted[start : end + 1])
        usage = _usage(body, messages, content, (time.perf_counter() - started) * 1000)
        return model_cls.model_validate(payload), usage


def _usage(body: Any, messages: list[dict[str, str]], content: str, latency_ms: float) -> LLMUsage:
    raw = body.get("usage") if isinstance(body, dict) else None
    prompt_tokens = completion_tokens = None
    if isinstance(raw, dict):
        prompt_tokens = raw.get("prompt_tokens") if isinstance(raw.get("prompt_tokens"), int) else None
        completion_tokens = (
            raw.get("completion_tokens") if isinstance(raw.get("completion_tokens"), int) else None
        )
    return LLMUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_chars=sum(len(m.get("content", "")) for m in messages),
        completion_chars=len(content or ""),
        latency_ms=round(latency_ms, 1),
    )


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        text = text[first_newline + 1 :] if first_newline != -1 else ""
        text = text.rstrip()
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    if not text.startswith("{"):  # tolerate prose around the JSON object
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    return text
