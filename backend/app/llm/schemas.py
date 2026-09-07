from typing import Any

from pydantic import BaseModel


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


class LLMInsight(BaseModel):
    """One insight paired with its supporting chart (stage8 contract).
    chart is an LLMChartSuggestion-shaped dict, validated per item in the
    service; None means the LLM attached no chart (kept as unverified)."""

    text: str = ""
    chart: Any = None


class LLMResponse(BaseModel):
    # Any, not list[str]: a malformed insights value must not invalidate the
    # whole response — the service normalizes it item-by-item
    insights: Any = []
    # raw items on purpose: one malformed chart (wrong shape, not even a
    # dict) must not sink the whole response — the service validates each
    # item individually
    charts: list[Any] = []
