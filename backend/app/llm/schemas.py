"""LLM response contracts for the two-stage workflow (stage 17.3).

LLM #1 (hypothesis generator) returns hypotheses — structured claims with the
probe the backend should run when the evidence tables do not already answer
them. LLM #2 (final analyst) returns the wording for hypotheses the backend
has validated. Both are parsed per item and tolerantly (stage 5 principle): a
malformed item is dropped, never the whole response. Loose types on purpose —
every field is re-checked against the profile in the service (the LLM never
decides what is true).
"""

from typing import Any

from pydantic import BaseModel, ConfigDict


class LLMChartSuggestion(BaseModel):
    """One chart item, parsed per-item and tolerantly (stage5 blocking #1):
    every field has a default so a sparse item still parses; types are loose
    strings — validate_spec is the real gate."""

    title: str = ""
    type: str = ""
    x: str | None = None
    y: str | None = None
    group_by: str | None = None
    aggregation: str | None = None
    reason: str = ""
    priority: int = 50


class Hypothesis(BaseModel):
    """One LLM #1 item. `statement` is the claim in words; `test_needed` is
    the probe type the backend should run (None when the LLM believes the
    evidence already covers it — the service checks either way); `columns`
    are the probe roles; `chart` the supporting chart suggestion."""

    model_config = ConfigDict(extra="allow")

    statement: str = ""
    variables: list[Any] = []
    evidence_available: bool | None = None
    test_needed: str | None = None
    columns: Any = None  # dict[role, column]; validated in the service
    chart: Any = None  # LLMChartSuggestion-shaped dict
    reason: str = ""
    importance: int = 3  # 1..5, 5 = most important


class HypothesisResponse(BaseModel):
    # Any on purpose: one malformed item must not sink the response — the
    # service normalises item by item
    hypotheses: Any = []
    # optional bare chart suggestions (re-ranking rule candidates, extra
    # charts); no claim attached, so they only go through the chart merge
    charts: list[Any] = []


class FinalInsight(BaseModel):
    """One LLM #2 item: the wording for a validated hypothesis. Numbers and
    claims are checked against the validated evidence in the service."""

    model_config = ConfigDict(extra="allow")

    hypothesis_id: Any = None
    text: str = ""
    why_it_matters: str = ""
    priority: int = 3
    chart_id: Any = None


class FinalResponse(BaseModel):
    insights: Any = []
    message: str | None = None


class LLMUsage(BaseModel):
    """Token accounting for one call (prompt/completion from the server's
    `usage` block when present, otherwise character counts)."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    prompt_chars: int = 0
    completion_chars: int = 0
    latency_ms: float = 0.0
