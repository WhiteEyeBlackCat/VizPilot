"""Merges rule-based recommendations with LLM suggestions (D8).

With the LLM disabled or failing, the output is exactly the Stage 3
rule-engine result — the system never depends on the LLM (SPEC §13).
"""

import logging
import threading
from typing import Any

from pydantic import ValidationError

from ..charts.rules import MAX_CHARTS, Recommendation, recommend_charts
from ..charts.spec import ChartSpec, validate_spec
from ..profiling.models import DatasetProfile
from .provider import LLMError, LLMProvider
from .schemas import LLMChartSuggestion, LLMResponse

logger = logging.getLogger(__name__)

LLM_NEW_CHART_SCORE = 0.75
MAX_INSIGHTS = 5
MAX_INSIGHT_CHARS = 300
FALLBACK_MESSAGE = "AI suggestions unavailable ({reason}); showing rule-based recommendations."

SpecKey = tuple[str | None, ...]


def _key(spec: ChartSpec) -> SpecKey:
    return (spec.type, spec.x, spec.y, spec.group_by)


class RecommendationService:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider
        # merged results per (dataset_id, llm_used); profiles are immutable so
        # entries never expire (D5). Per-key locks prevent duplicate LLM calls
        # for the same key without serializing unrelated requests.
        self._cache: dict[tuple[str, bool], dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._key_locks: dict[tuple[str, bool], threading.Lock] = {}

    def get(self, profile: DatasetProfile, use_llm: bool) -> dict[str, Any]:
        cache_key = (profile.dataset_id, use_llm)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        # per-key lock: an in-flight LLM build must not block the llm=false
        # escape hatch or other datasets
        with self._lock:
            key_lock = self._key_locks.setdefault(cache_key, threading.Lock())
        with key_lock:
            if cache_key not in self._cache:
                self._cache[cache_key] = self._build(profile, use_llm)
            return self._cache[cache_key]

    def _build(self, profile: DatasetProfile, use_llm: bool) -> dict[str, Any]:
        rules = recommend_charts(profile)
        if not use_llm:
            return _shape(rules, [], None)

        try:
            response = self._provider.recommend_charts(profile, rules)
        except LLMError as exc:
            logger.warning("LLM call failed (%s): %s", exc.category, exc)
            return _shape(rules, [], FALLBACK_MESSAGE.format(reason=exc.category))

        merged, kept_llm_items = _merge(profile, rules, response)
        message = None
        if response.charts and kept_llm_items == 0:
            message = FALLBACK_MESSAGE.format(reason="all suggestions were invalid")
        insights = _clean_insights(response.insights)
        return _shape(merged, insights, message)


def _clean_insights(raw: Any) -> list[str]:
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    kept = [i for i in raw if isinstance(i, str) and i.strip()]
    if len(kept) < len(raw):
        logger.warning("dropped %d malformed insight(s)", len(raw) - len(kept))
    return [i[:MAX_INSIGHT_CHARS] for i in kept[:MAX_INSIGHTS]]


def _shape(recs: list[Recommendation], insights: list[str], message: str | None) -> dict[str, Any]:
    charts = [rec.model_dump() for rec in recs]
    if message is None and not charts:
        message = "No charts could be recommended for this dataset."
    return {"charts": charts, "insights": insights, "message": message}


def _parse_suggestions(response: LLMResponse) -> list[LLMChartSuggestion]:
    suggestions = []
    for item in response.charts:  # per-item tolerance (blocking #1)
        if not isinstance(item, dict):
            logger.warning("dropping non-object LLM chart item %r", item)
            continue
        try:
            suggestions.append(LLMChartSuggestion.model_validate(item))
        except ValidationError as exc:
            logger.warning("dropping malformed LLM chart item %r: %s", item, exc)
    return suggestions


def _merge(
    profile: DatasetProfile, rules: list[Recommendation], response: LLMResponse
) -> tuple[list[Recommendation], int]:
    """Returns the merged, re-ranked list plus how many LLM items survived."""
    by_key = {_key(rec.spec): rec for rec in rules}
    llm_priority: dict[SpecKey, int] = {}
    added: list[Recommendation] = []
    kept = 0

    for suggestion in _parse_suggestions(response):
        if not suggestion.title:
            logger.warning("dropping LLM chart without a title: %r", suggestion)
            continue
        try:
            spec = ChartSpec(
                title=suggestion.title,
                type=suggestion.type,  # type: ignore[arg-type]  # Literal rejects bad values
                x=suggestion.x,
                y=suggestion.y,
                group_by=suggestion.group_by,
                aggregation=suggestion.aggregation,  # type: ignore[arg-type]
                reason=suggestion.reason,
            )
        except ValidationError as exc:
            logger.warning("dropping LLM chart '%s': %s", suggestion.title, exc)
            continue
        errors = validate_spec(spec, profile)
        if errors:
            logger.warning("dropping LLM chart '%s': %s", suggestion.title, "; ".join(errors))
            continue

        key = _key(spec)
        if key in by_key:
            # duplicate of a rules chart: keep the rules spec, adopt the LLM
            # reason and let the LLM priority pull it forward
            if suggestion.reason:
                by_key[key].spec.reason = suggestion.reason
        elif key not in llm_priority:
            added.append(Recommendation(spec=spec, score=LLM_NEW_CHART_SCORE, source="llm"))
        else:
            continue  # LLM repeated itself
        llm_priority[key] = min(suggestion.priority, llm_priority.get(key, suggestion.priority))
        kept += 1

    def sort_key(rec: Recommendation) -> tuple:
        # blocking #2: LLM-ranked charts first in LLM order, the rest by score
        key = _key(rec.spec)
        if key in llm_priority:
            return (0, llm_priority[key], -rec.score, rec.spec.title)
        return (1, -rec.score, rec.spec.type, rec.spec.x or "", rec.spec.y or "")

    merged = sorted(rules + added, key=sort_key)[:MAX_CHARTS]
    for i, rec in enumerate(merged):
        rec.spec.priority = i + 1
    return merged, kept
