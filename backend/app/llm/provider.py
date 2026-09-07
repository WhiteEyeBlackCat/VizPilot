"""LLMProvider abstraction (D4): the default implementation speaks the
OpenAI-compatible chat completions API over plain httpx, so Ollama, vLLM,
llama.cpp server and LM Studio are all interchangeable via base_url/model.
No provider SDK is imported.
"""

import json
from typing import Protocol

import httpx
from pydantic import ValidationError

from ..charts.rules import Recommendation
from ..profiling.models import DatasetProfile
from .prompts import build_messages
from .schemas import LLMResponse

TEMPERATURE = 0.1
MAX_TOKENS = 2000


class LLMError(Exception):
    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category  # surfaced in the fallback message (critique #8)


class LLMProvider(Protocol):
    def recommend_charts(
        self, profile: DatasetProfile, rule_candidates: list[Recommendation]
    ) -> LLMResponse: ...


class DisabledProvider:
    def recommend_charts(
        self, profile: DatasetProfile, rule_candidates: list[Recommendation]
    ) -> LLMResponse:
        return LLMResponse()


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

    def recommend_charts(
        self, profile: DatasetProfile, rule_candidates: list[Recommendation]
    ) -> LLMResponse:
        if not self._model:
            raise LLMError("misconfigured: model not set", "llm_model is empty")
        messages = build_messages(profile, rule_candidates, self._include_sample_rows)
        last: Exception | None = None
        for _ in range(2):  # one retry for connection/HTTP/parse failures
            try:
                return self._call(messages)
            except httpx.TimeoutException as exc:
                raise LLMError("timeout", str(exc)) from exc  # timeouts never retry (critique #4)
            except (httpx.HTTPError, ValueError, ValidationError) as exc:
                last = exc
        raise LLMError("request failed after retry", str(last)) from last

    def _call(self, messages: list[dict[str, str]]) -> LLMResponse:
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
        try:
            content = response.json()["choices"][0]["message"]["content"]
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
        return LLMResponse.model_validate(payload)


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
