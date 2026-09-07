"""Merges rule-based recommendations with LLM suggestions (D8, stage8).

Flow principle (confirmed with the user): statistics produce evidence, the
LLM produces hypotheses and narrative, statistics verify the hypotheses —
one pass, no loop. With the LLM disabled or failing, the output is exactly
the rule-engine result (SPEC §13).
"""

import logging
import threading
from typing import Any, Literal

from pydantic import ValidationError

from ..charts.rules import (
    Recommendation,
    Tier,
    VerificationLevel,
    apply_diversity_caps,
    assign_tiers,
    evaluate_llm_spec,
    recommend_charts,
)
from ..charts.spec import ChartSpec, validate_spec
from ..profiling.models import DatasetProfile
from .provider import LLMError, LLMProvider
from .schemas import LLMChartSuggestion, LLMInsight, LLMResponse

logger = logging.getLogger(__name__)

MAX_INSIGHTS = 5
MAX_INSIGHT_CHARS = 300
UNVERIFIED_NOTE = "hypothesis not verified against the data"
FALLBACK_MESSAGE = "AI suggestions unavailable ({reason}); showing rule-based recommendations."

SpecKey = tuple[str | None, ...]
Supported = Literal["strong", "weak", "unverified"]

_TIER_CAP: dict[VerificationLevel, Tier | None] = {
    "strong": None,
    "neutral": None,  # no testable hypothesis -> rules fixed score, no cap
    "weak": "secondary",
    "unverified": "exploratory",
}


def _key(spec: ChartSpec) -> SpecKey:
    return (spec.type, spec.x, spec.y, spec.group_by)


def _supported_label(level: VerificationLevel) -> Supported:
    # "neutral" charts (heatmap/histogram) carry nothing to verify; the badge
    # must not read "unverified" (critique #3) — "weak" is the honest middle
    return {"strong": "strong", "weak": "weak", "neutral": "weak", "unverified": "unverified"}[level]


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

        merged, insights, attempted, kept = _merge(profile, rules, response)
        message = None
        if attempted > 0 and kept == 0:
            message = FALLBACK_MESSAGE.format(reason="all suggestions were invalid")
        return _shape(merged, insights, message)


def _shape(
    recs: list[Recommendation], insights: list[dict[str, Any]], message: str | None
) -> dict[str, Any]:
    charts = [rec.model_dump() for rec in recs]
    if message is None and not charts:
        message = "No charts could be recommended for this dataset."
    return {"charts": charts, "insights": insights, "message": message}


def _parse_suggestion(item: Any) -> LLMChartSuggestion | None:
    if not isinstance(item, dict):
        logger.warning("dropping non-object LLM chart item %r", item)
        return None
    try:
        return LLMChartSuggestion.model_validate(item)
    except ValidationError as exc:
        logger.warning("dropping malformed LLM chart item %r: %s", item, exc)
        return None


def _parse_insights(raw: Any) -> list[tuple[str, Any]]:
    """Tolerant per-item parse -> (text, chart-dict | None). Legacy plain
    strings become chartless insights; non-object items are dropped."""
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        if raw:
            logger.warning("dropping malformed insights payload %r", raw)
        return []
    items: list[tuple[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            if item.strip():
                items.append((item, None))  # legacy format: no supporting chart
            continue
        if not isinstance(item, dict):
            logger.warning("dropping non-object insight item %r", item)
            continue
        try:
            insight = LLMInsight.model_validate(item)
        except ValidationError as exc:
            logger.warning("dropping malformed insight item %r: %s", item, exc)
            continue
        if insight.text.strip():
            items.append((insight.text, insight.chart))
    return items


class _Merger:
    def __init__(self, profile: DatasetProfile, rules: list[Recommendation]) -> None:
        self.profile = profile
        self.by_key = {_key(rec.spec): rec for rec in rules}
        self.added: dict[SpecKey, Recommendation] = {}
        self.llm_priority: dict[SpecKey, int] = {}
        self.attempted = 0
        self.kept = 0

    def integrate(self, suggestion: LLMChartSuggestion) -> tuple[SpecKey, VerificationLevel] | None:
        """Validates, evidence-scores and merges one LLM chart. Returns its
        dedup key and verification level, or None when dropped."""
        self.attempted += 1
        if not suggestion.title:
            logger.warning("dropping LLM chart without a title: %r", suggestion)
            return None
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
            return None
        errors = validate_spec(spec, self.profile)
        if errors:
            logger.warning("dropping LLM chart '%s': %s", suggestion.title, "; ".join(errors))
            return None

        score, level = evaluate_llm_spec(spec, self.profile)
        key = _key(spec)
        if key in self.by_key:
            # duplicate of a rules chart: keep the rules spec (and its
            # evidence score), adopt the LLM reason
            if suggestion.reason:
                self.by_key[key].spec.reason = suggestion.reason
        elif key not in self.added:
            if level == "unverified":
                note = f" ({UNVERIFIED_NOTE})"
                spec.reason = (spec.reason + note) if spec.reason else UNVERIFIED_NOTE
            self.added[key] = Recommendation(
                spec=spec, score=score, source="llm", tier_cap=_TIER_CAP[level]
            )
        self.llm_priority[key] = min(suggestion.priority, self.llm_priority.get(key, suggestion.priority))
        self.kept += 1
        return key, level


def _merge(
    profile: DatasetProfile, rules: list[Recommendation], response: LLMResponse
) -> tuple[list[Recommendation], list[dict[str, Any]], int, int]:
    """Returns (final list, insights payload, attempted, kept LLM items)."""
    merger = _Merger(profile, rules)

    # insight state machine (stage8 blocking #1)
    insight_rows: list[tuple[str, SpecKey | None, Supported]] = []
    for text, chart in _parse_insights(response.insights):
        if chart is None:
            insight_rows.append((text, None, "unverified"))
            continue
        suggestion = _parse_suggestion(chart)
        if suggestion is None:
            logger.warning("dropping insight with malformed supporting chart: %r", text)
            continue
        result = merger.integrate(suggestion)
        if result is None:
            logger.warning("dropping insight whose supporting chart was rejected: %r", text)
            continue
        key, level = result
        insight_rows.append((text, key, _supported_label(level)))

    for item in response.charts:  # bare chart suggestions, per-item tolerant
        suggestion = _parse_suggestion(item)
        if suggestion is not None:
            merger.integrate(suggestion)

    # merge -> re-apply diversity caps (critique #4) -> display order ->
    # score-based tiers with LLM caps (blocking #2)
    capped = apply_diversity_caps(rules + list(merger.added.values()))
    llm_priority = merger.llm_priority

    def display_key(rec: Recommendation) -> tuple:
        key = _key(rec.spec)
        if key in llm_priority:
            return (0, llm_priority[key], -rec.score, rec.spec.title)
        return (1, -rec.score, rec.spec.type, rec.spec.x or "", rec.spec.y or "")

    merged = sorted(capped, key=display_key)
    for i, rec in enumerate(merged):
        rec.spec.priority = i + 1
    assign_tiers(merged)

    final_priority = {_key(rec.spec): rec.spec.priority for rec in merged}
    insights = [
        {
            "text": text[:MAX_INSIGHT_CHARS],
            "supported": supported,
            # None when the chart fell to the caps/MAX_CHARTS cut — the
            # insight survives, it just has no chart to link to
            "chart_priority": final_priority.get(key) if key is not None else None,
        }
        # deliberate: charts from beyond-cap insights were already integrated —
        # they carry their own evidence score and diversity caps, so keeping
        # them is harmless while the insight text itself is dropped
        for text, key, supported in insight_rows[:MAX_INSIGHTS]
    ]
    return merged, insights, merger.attempted, merger.kept