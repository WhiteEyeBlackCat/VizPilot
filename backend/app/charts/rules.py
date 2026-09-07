"""Deterministic chart recommendations from a DatasetProfile.

Every candidate passes validate_spec before it can be returned. Scores are
internal ordering only; priority (1..N) is the cross-stage contract (D8).
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from ..profiling.models import ColumnProfile, DatasetProfile
from .spec import ChartSpec, TimeGranularity, validate_spec

MAX_CHARTS = 12
MAX_PER_TYPE = 3  # diversity cap (stage3 blocking #1)
MAX_MISSING_RATIO = 0.5
MAX_GRANULARITY_POINTS = 500
SCATTER_MIN_CORR = 0.3
SCATTER_MAX_PAIRS = 5
GROUP_CATEGORIES_RANGE = (2, 8)


class Recommendation(BaseModel):
    spec: ChartSpec
    score: float
    source: Literal["rules", "llm"] = "rules"


def choose_time_granularity(unique_count: int, span_days: float) -> TimeGranularity:
    """Finest bucket whose estimated point count is <= 500; month as fallback."""
    if span_days < 2:
        # intraday/short spans: no hour bucket exists, and "day" would collapse
        # everything into a single point; keep raw and let render-time sampling cap it
        return "raw"
    estimates = [
        ("raw", unique_count),
        ("day", span_days),
        ("week", span_days / 7),
        ("month", span_days / 30),
    ]
    for granularity, points in estimates:
        if points <= MAX_GRANULARITY_POINTS:
            return granularity  # type: ignore[return-value]
    return "month"


def recommend_charts(profile: DatasetProfile) -> list[Recommendation]:
    usable = [
        c
        for c in profile.columns
        if c.semantic_type not in ("id", "unknown", "text")
        and c.missing_ratio <= MAX_MISSING_RATIO
    ]
    datetime_cols = [c for c in usable if c.semantic_type == "datetime"]
    numeric_cols = [c for c in usable if c.semantic_type == "numeric"]
    cat_cols = [c for c in usable if c.semantic_type in ("categorical", "boolean")]
    group_by = _group_candidate(cat_cols)

    candidates: list[Recommendation] = []
    candidates += _line_charts(profile, datetime_cols, numeric_cols, group_by)
    candidates += _bar_charts(cat_cols, numeric_cols)
    candidates += _box_charts(cat_cols, numeric_cols)
    candidates += _scatter_charts(profile, numeric_cols, group_by)
    candidates += _histogram_charts(numeric_cols)
    candidates += _heatmap_chart(numeric_cols)

    valid = _dedup(r for r in candidates if not validate_spec(r.spec, profile))
    ranked = _rank(valid)
    for i, rec in enumerate(ranked):
        rec.spec.priority = i + 1
    return ranked


def _group_candidate(cat_cols: list[ColumnProfile]) -> str | None:
    lo, hi = GROUP_CATEGORIES_RANGE
    eligible = [c for c in cat_cols if c.n_categories is not None and lo <= c.n_categories <= hi]
    if not eligible:
        return None
    return min(eligible, key=lambda c: (c.n_categories, c.name)).name


def _line_charts(
    profile: DatasetProfile,
    datetime_cols: list[ColumnProfile],
    numeric_cols: list[ColumnProfile],
    group_by: str | None,
) -> list[Recommendation]:
    recs = []
    for dt in datetime_cols:
        granularity = choose_time_granularity(dt.unique_count, _span_days(dt))
        has_duplicates = dt.unique_count < profile.n_rows - dt.missing_count
        aggregation = "mean" if granularity != "raw" or has_duplicates else None
        score = 0.8 + (0.1 if dt.inferred_frequency in ("daily", "weekly", "monthly") else 0.0)
        for num in numeric_cols:
            suffix = f" by {group_by}" if group_by else ""
            recs.append(
                Recommendation(
                    spec=ChartSpec(
                        title=f"{num.name} over {dt.name}{suffix}",
                        type="line",
                        x=dt.name,
                        y=num.name,
                        group_by=group_by,  # grouped version replaces ungrouped
                        aggregation=aggregation,
                        time_granularity=granularity,
                        reason=f"Shows how {num.name} changes over time{suffix}.",
                    ),
                    score=score,
                )
            )
    return recs


def _span_days(dt: ColumnProfile) -> float:
    if not isinstance(dt.min, str) or not isinstance(dt.max, str):
        return 0.0
    return (datetime.fromisoformat(dt.max) - datetime.fromisoformat(dt.min)).total_seconds() / 86400


def _bar_score(cat: ColumnProfile) -> float:
    return 0.6 + (0.1 if 3 <= (cat.n_categories or 0) <= 12 else 0.0)


def _bar_charts(cat_cols: list[ColumnProfile], numeric_cols: list[ColumnProfile]) -> list[Recommendation]:
    recs = []
    bar_cats = [c for c in cat_cols if 2 <= (c.n_categories or 0) <= 20]
    for cat in bar_cats:
        for num in numeric_cols:
            recs.append(
                Recommendation(
                    spec=ChartSpec(
                        title=f"Mean {num.name} by {cat.name}",
                        type="bar",
                        x=cat.name,
                        y=num.name,
                        aggregation="mean",
                        reason=f"Compares average {num.name} across {cat.name} categories.",
                    ),
                    score=_bar_score(cat),
                )
            )
    if bar_cats:
        count_cat = max(bar_cats, key=lambda c: (c.n_categories, c.name))
        recs.append(
            Recommendation(
                spec=ChartSpec(
                    title=f"Record count by {count_cat.name}",
                    type="bar",
                    x=count_cat.name,
                    aggregation="count",
                    reason=f"Shows how many records fall in each {count_cat.name} category.",
                ),
                score=_bar_score(count_cat),
            )
        )
    return recs


def _box_charts(cat_cols: list[ColumnProfile], numeric_cols: list[ColumnProfile]) -> list[Recommendation]:
    return [
        Recommendation(
            spec=ChartSpec(
                title=f"{num.name} distribution by {cat.name}",
                type="box",
                x=cat.name,
                y=num.name,
                reason=f"Shows the distribution of {num.name} within each {cat.name} category.",
            ),
            score=0.55,
        )
        for cat in cat_cols
        if 2 <= (cat.n_categories or 0) <= 12
        for num in numeric_cols
        if num.std is not None and num.std > 0
    ]


def _scatter_charts(
    profile: DatasetProfile, numeric_cols: list[ColumnProfile], group_by: str | None
) -> list[Recommendation]:
    corr = profile.correlations
    if corr is None:
        return []
    usable = {c.name for c in numeric_cols}
    pairs = []
    for i, a in enumerate(corr.columns):
        for j in range(i + 1, len(corr.columns)):
            b = corr.columns[j]
            value = corr.matrix[i][j]
            if a in usable and b in usable and value is not None and abs(value) >= SCATTER_MIN_CORR:
                pairs.append((abs(value), a, b, value))
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
    recs = []
    for strength, a, b, value in pairs[:SCATTER_MAX_PAIRS]:
        suffix = f", colored by {group_by}" if group_by else ""
        recs.append(
            Recommendation(
                spec=ChartSpec(
                    title=f"{b} vs {a}",
                    type="scatter",
                    x=a,
                    y=b,
                    group_by=group_by,  # grouped version replaces ungrouped
                    reason=f"Reveals the relationship between {a} and {b} "
                    f"(correlation {value:.2f}){suffix}.",
                ),
                score=0.5 + 0.4 * strength,
            )
        )
    return recs


def _histogram_charts(numeric_cols: list[ColumnProfile]) -> list[Recommendation]:
    recs = []
    for num in numeric_cols:
        if num.std is None or num.std <= 0:
            continue
        skewed = num.skewness is not None and abs(num.skewness) > 1
        recs.append(
            Recommendation(
                spec=ChartSpec(
                    title=f"Distribution of {num.name}",
                    type="histogram",
                    x=num.name,
                    reason=f"Shows the distribution of {num.name}.",
                ),
                score=0.5 + (0.05 if skewed else 0.0),
            )
        )
    return recs


def _heatmap_chart(numeric_cols: list[ColumnProfile]) -> list[Recommendation]:
    if len(numeric_cols) < 3:
        return []
    return [
        Recommendation(
            spec=ChartSpec(
                title="Correlation heatmap",
                type="heatmap",
                reason="Shows pairwise correlations between all numeric columns.",
            ),
            score=0.65,
        )
    ]


def _dedup(recs) -> list[Recommendation]:
    seen: set[tuple] = set()
    result = []
    for rec in recs:
        key = (rec.spec.type, rec.spec.x, rec.spec.y, rec.spec.group_by)
        if key not in seen:
            seen.add(key)
            result.append(rec)
    return result


def _sort_key(rec: Recommendation) -> tuple:
    return (-rec.score, rec.spec.type, rec.spec.x or "", rec.spec.y or "", rec.spec.group_by or "")


def _rank(recs: list[Recommendation]) -> list[Recommendation]:
    per_type: dict[str, list[Recommendation]] = {}
    for rec in sorted(recs, key=_sort_key):
        per_type.setdefault(rec.spec.type, []).append(rec)
    kept = [rec for group in per_type.values() for rec in group[:MAX_PER_TYPE]]
    return sorted(kept, key=_sort_key)[:MAX_CHARTS]
