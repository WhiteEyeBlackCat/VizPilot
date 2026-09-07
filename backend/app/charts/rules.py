"""Deterministic chart recommendations, ranked by profile evidence (stage 7).

Scores are anchored to effect sizes (eta = sqrt(adjusted eta-squared), which
shares the |corr| scale) instead of cardinality guesses; lexicographic
tie-breaks remain only for determinism. Every candidate passes validate_spec;
priority (1..N) is the cross-stage contract (D8).
"""

import math
import statistics
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from ..profiling.models import ColumnProfile, Correlations, DatasetProfile, Evidence
from .spec import ChartSpec, TimeGranularity, validate_spec

MAX_CHARTS = 12
MAX_PER_TYPE = 3  # diversity cap (stage3 blocking #1)
MAX_PER_X = 2  # within a type, one x column may fill at most 2 slots
MAX_MISSING_RATIO = 0.5
MAX_GRANULARITY_POINTS = 500
SCATTER_MIN_CORR = 0.3
SCATTER_MAX_PAIRS = 5
GROUP_CATEGORIES_RANGE = (2, 8)
NONLINEAR_CORR_GAP = 0.15  # |spearman| - |pearson| above this -> non-linear note
LINE_GROUP_INTERACTION_MIN = 0.05
LINE_GROUP_MAIN_EFFECT_MIN = 0.1  # eta-squared(group, y): main-effect grouping still earns a split

Tier = Literal["top", "secondary", "exploratory"]

# verification level of an LLM-suggested chart against the evidence table
VerificationLevel = Literal["strong", "weak", "unverified", "neutral"]
TOP_SCORE_FLOOR = 0.68
UNVERIFIED_SCORE = 0.5  # combinations outside the evidence scan


class Recommendation(BaseModel):
    spec: ChartSpec
    score: float
    source: Literal["rules", "llm"] = "rules"
    tier: Tier = "exploratory"
    # LLM-verification ceiling ("secondary"/"exploratory"); internal only —
    # applied by assign_tiers and never serialized into the API response
    tier_cap: Tier | None = Field(default=None, exclude=True)


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


def slope_spread_threshold(n_min: int) -> float:
    """Dynamic false-positive guard (stage7 blocking #4): small groups produce
    noisy per-group correlations, so the required spread grows as ~1/sqrt(n)."""
    return max(0.3, 3.5 / math.sqrt(max(n_min - 3, 1)))


class _EvidenceIndex:
    """Fast lookups over profile.evidence for the chart generators."""

    def __init__(self, evidence: Evidence) -> None:
        self.eta2 = {(e.cat, e.num): e.eta_squared for e in evidence.cat_num}
        self.time_eta2 = {(t.datetime_col, t.num): t.eta_squared for t in evidence.time_effects}
        self.spearman: dict[tuple[str, str], float] = {}
        sp = evidence.num_num_spearman
        if sp is not None:
            for i, a in enumerate(sp.columns):
                for j, b in enumerate(sp.columns):
                    value = sp.matrix[i][j]
                    if i != j and value is not None:
                        self.spearman[(a, b)] = value
        # (datetime_col, group, num) -> best time-bucket interaction strength
        self.time_interaction: dict[tuple[str, str, str], float] = {}
        for e in evidence.interactions:
            for first, second in ((e.cat1, e.cat2), (e.cat2, e.cat1)):
                if "@" in first and "@" not in second:
                    key = (first.split("@", 1)[0], second, e.num)
                    self.time_interaction[key] = max(
                        self.time_interaction.get(key, 0.0), e.strength
                    )
        # (x, y, group) -> (spread, n_min), both axis orders
        self.slope: dict[tuple[str, str, str], tuple[float, int]] = {}
        for s in evidence.slope_heterogeneity:
            for key in ((s.x, s.y, s.group), (s.y, s.x, s.group)):
                self.slope[key] = (s.spread, s.n_min)

    def eta(self, cat: str, num: str) -> float:
        return math.sqrt(self.eta2.get((cat, num), 0.0))

    def time_eta(self, dt: str, num: str) -> float:
        return math.sqrt(self.time_eta2.get((dt, num), 0.0))


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
    lo, hi = GROUP_CATEGORIES_RANGE
    group_cols = [c.name for c in cat_cols if lo <= (c.n_categories or 0) <= hi]
    index = _EvidenceIndex(profile.evidence)

    candidates: list[Recommendation] = []
    candidates += _line_charts(profile, datetime_cols, numeric_cols, group_cols, index)
    candidates += _bar_charts(cat_cols, numeric_cols, index)
    candidates += _box_charts(cat_cols, numeric_cols, index)
    candidates += _scatter_charts(profile, numeric_cols, group_cols, index)
    candidates += _histogram_charts(numeric_cols)
    candidates += _heatmap_chart(numeric_cols)

    valid = _dedup(r for r in candidates if not validate_spec(r.spec, profile))
    ranked = apply_diversity_caps(valid)
    for i, rec in enumerate(ranked):
        rec.spec.priority = i + 1
    assign_tiers(ranked)
    return ranked


def assign_tiers(recs: list[Recommendation]) -> None:
    """Score-based tier boundaries, decoupled from display order (stage8
    blocking #2 — the merged list may show LLM-ranked charts first, so
    display position must never decide "top"). Calibrated on
    dataset/sales_basic.csv and the planted-signal eval:

    - top: scanning by score DESCENDING, the first 3 charts that clear the
      dataset's median by 0.05 AND an absolute floor of 0.68 — strictly above
      every fixed score (heatmap 0.65, count bar 0.6, histogram <=0.55), so
      only charts with actual effect-size evidence can be top. Charts with an
      LLM-verification cap never take a top slot.
    - secondary: at or above the median.
    - finally the LLM caps are applied (weak -> at most secondary,
      unverified -> exploratory).
    """
    if not recs:
        return
    median = statistics.median(rec.score for rec in recs)
    threshold = max(median + 0.05, TOP_SCORE_FLOOR)
    top_slots = 3
    for rec in sorted(recs, key=_sort_key):  # score-descending scan
        if top_slots > 0 and rec.tier_cap is None and rec.score >= threshold:
            rec.tier = "top"
            top_slots -= 1
        elif rec.score >= median:
            rec.tier = "secondary"
        else:
            rec.tier = "exploratory"
    for rec in recs:
        if rec.tier_cap == "exploratory":
            rec.tier = "exploratory"
        elif rec.tier_cap == "secondary" and rec.tier == "top":
            # defensive only: the scan above never hands top to a capped chart
            rec.tier = "secondary"


def _line_charts(
    profile: DatasetProfile,
    datetime_cols: list[ColumnProfile],
    numeric_cols: list[ColumnProfile],
    group_cols: list[str],
    index: _EvidenceIndex,
) -> list[Recommendation]:
    recs = []
    for dt in datetime_cols:
        granularity = choose_time_granularity(dt.unique_count, _span_days(dt))
        has_duplicates = dt.unique_count < profile.n_rows - dt.missing_count
        regularity = 0.05 if dt.inferred_frequency in ("daily", "weekly", "monthly") else 0.0
        for num in numeric_cols:
            group = _line_group(dt.name, num.name, group_cols, index)
            aggregation = "mean" if granularity != "raw" or has_duplicates or group else None
            score = 0.5 + 0.3 * index.time_eta(dt.name, num.name) + regularity
            suffix = f" by {group}" if group else ""
            recs.append(
                Recommendation(
                    spec=ChartSpec(
                        title=f"{num.name} over {dt.name}{suffix}",
                        type="line",
                        x=dt.name,
                        y=num.name,
                        group_by=group,
                        aggregation=aggregation,
                        time_granularity=granularity,
                        reason=f"Shows how {num.name} changes over time{suffix}.",
                    ),
                    score=score,
                )
            )
    return recs


def _line_group(dt: str, num: str, group_cols: list[str], index: _EvidenceIndex) -> str | None:
    """A grouped line must earn its split: either the group's trend differs
    over time (interaction) or the group shifts the level (main effect)."""
    best: tuple[float, str] | None = None
    for group in group_cols:
        interaction = index.time_interaction.get((dt, group, num), 0.0)
        main_effect = index.eta2.get((group, num), 0.0)
        if interaction >= LINE_GROUP_INTERACTION_MIN or main_effect >= LINE_GROUP_MAIN_EFFECT_MIN:
            strength = max(interaction, main_effect)
            if best is None or (strength, group) > (best[0], best[1]):
                best = (strength, group)
    return best[1] if best else None


def _span_days(dt: ColumnProfile) -> float:
    if not isinstance(dt.min, str) or not isinstance(dt.max, str):
        return 0.0
    return (datetime.fromisoformat(dt.max) - datetime.fromisoformat(dt.min)).total_seconds() / 86400


def _bar_charts(
    cat_cols: list[ColumnProfile], numeric_cols: list[ColumnProfile], index: _EvidenceIndex
) -> list[Recommendation]:
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
                    score=0.45 + 0.45 * index.eta(cat.name, num.name),
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
                score=0.6,
            )
        )
    return recs


def _box_charts(
    cat_cols: list[ColumnProfile], numeric_cols: list[ColumnProfile], index: _EvidenceIndex
) -> list[Recommendation]:
    return [
        Recommendation(
            spec=ChartSpec(
                title=f"{num.name} distribution by {cat.name}",
                type="box",
                x=cat.name,
                y=num.name,
                reason=f"Shows the distribution of {num.name} within each {cat.name} category.",
            ),
            score=0.4 + 0.45 * index.eta(cat.name, num.name),
        )
        for cat in cat_cols
        if 2 <= (cat.n_categories or 0) <= 12
        for num in numeric_cols
        if num.std is not None and num.std > 0
    ]


def _scatter_charts(
    profile: DatasetProfile,
    numeric_cols: list[ColumnProfile],
    group_cols: list[str],
    index: _EvidenceIndex,
) -> list[Recommendation]:
    corr = profile.correlations
    if corr is None:
        return []
    usable = {c.name for c in numeric_cols}
    pairs = []
    for i, a in enumerate(corr.columns):
        for j in range(i + 1, len(corr.columns)):
            b = corr.columns[j]
            pearson = corr.matrix[i][j]
            if a not in usable or b not in usable:
                continue
            spearman = index.spearman.get((a, b))
            strength = max(
                abs(pearson) if pearson is not None else 0.0,
                abs(spearman) if spearman is not None else 0.0,
            )
            if strength >= SCATTER_MIN_CORR:
                pairs.append((strength, a, b, pearson, spearman))
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))

    recs = []
    for strength, a, b, pearson, spearman in pairs[:SCATTER_MAX_PAIRS]:
        group = _scatter_group(a, b, group_cols, index)
        shown = pearson if pearson is not None else spearman
        reason = f"Reveals the relationship between {a} and {b} (correlation {shown:.2f})"
        if (
            pearson is not None
            and spearman is not None
            and abs(spearman) - abs(pearson) > NONLINEAR_CORR_GAP
        ):
            reason += (
                f"; the rank correlation ({spearman:.2f}) is notably stronger, "
                "suggesting a non-linear relationship"
            )
        if group:
            reason += f", colored by {group} (the relationship differs across its groups)"
        recs.append(
            Recommendation(
                spec=ChartSpec(
                    title=f"{b} vs {a}",
                    type="scatter",
                    x=a,
                    y=b,
                    group_by=group,
                    reason=reason + ".",
                ),
                score=0.5 + 0.4 * strength,
            )
        )
    return recs


def _scatter_group(a: str, b: str, group_cols: list[str], index: _EvidenceIndex) -> str | None:
    """Grouped scatter only when the per-group correlations demonstrably
    diverge (slope heterogeneity above the dynamic threshold)."""
    best: tuple[float, str] | None = None
    for group in group_cols:
        entry = index.slope.get((a, b, group))
        if entry is None:
            continue
        spread, n_min = entry
        if spread >= slope_spread_threshold(n_min):
            if best is None or (spread, group) > (best[0], best[1]):
                best = (spread, group)
    return best[1] if best else None


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


def evaluate_llm_spec(spec: ChartSpec, profile: DatasetProfile) -> tuple[float, VerificationLevel]:
    """Scores an LLM-suggested chart with the same evidence formulas as the
    rule engine and grades the hypothesis (stage8):

    - "strong": evidence-backed and score >= TOP_SCORE_FLOOR (may reach top)
    - "weak": evidence exists but is below the floor (capped at secondary)
    - "unverified": the combination is outside the evidence scan, e.g. a
      numeric-backed categorical as y (capped at exploratory)
    - "neutral": the type carries no testable hypothesis (heatmap, histogram,
      count bar) — rules' fixed scores, no cap, no unverified label
    """
    index = _EvidenceIndex(profile.evidence)
    columns = {c.name: c for c in profile.columns}

    if spec.type == "heatmap":
        return 0.65, "neutral"
    if spec.type == "histogram":
        col = columns.get(spec.x or "")
        skewed = col is not None and col.skewness is not None and abs(col.skewness) > 1
        return 0.5 + (0.05 if skewed else 0.0), "neutral"
    if spec.type in ("bar", "box") and spec.y is None:
        return 0.6, "neutral"  # count bar: no group-effect hypothesis

    if spec.type in ("bar", "box"):
        eta2 = index.eta2.get((spec.x or "", spec.y or ""))
        if eta2 is None:
            return UNVERIFIED_SCORE, "unverified"
        score = (0.45 if spec.type == "bar" else 0.4) + 0.45 * math.sqrt(eta2)
    elif spec.type == "scatter":
        pearson = _pearson_lookup(profile.correlations, spec.x or "", spec.y or "")
        spearman = index.spearman.get((spec.x or "", spec.y or ""))
        if pearson is None and spearman is None:
            return UNVERIFIED_SCORE, "unverified"
        strength = max(abs(pearson or 0.0), abs(spearman or 0.0))
        score = 0.5 + 0.4 * strength
    elif spec.type == "line":
        eta2 = index.time_eta2.get((spec.x or "", spec.y or ""))
        if eta2 is None:
            return UNVERIFIED_SCORE, "unverified"
        dt = columns.get(spec.x or "")
        regularity = (
            0.05 if dt is not None and dt.inferred_frequency in ("daily", "weekly", "monthly") else 0.0
        )
        score = 0.5 + 0.3 * math.sqrt(eta2) + regularity
    else:
        return UNVERIFIED_SCORE, "unverified"

    level: VerificationLevel = "strong" if score >= TOP_SCORE_FLOOR else "weak"
    if spec.group_by is not None and not _group_supported(spec, index):
        level = "weak"  # ungrouped evidence is fine, but the split is a guess
    return score, level


def _pearson_lookup(correlations: Correlations | None, a: str, b: str) -> float | None:
    if correlations is None or a not in correlations.columns or b not in correlations.columns:
        return None
    return correlations.matrix[correlations.columns.index(a)][correlations.columns.index(b)]


def _group_supported(spec: ChartSpec, index: _EvidenceIndex) -> bool:
    if spec.type == "scatter":
        entry = index.slope.get((spec.x or "", spec.y or "", spec.group_by or ""))
        return entry is not None and entry[0] >= slope_spread_threshold(entry[1])
    if spec.type == "line":
        interaction = index.time_interaction.get(
            (spec.x or "", spec.group_by or "", spec.y or ""), 0.0
        )
        main_effect = index.eta2.get((spec.group_by or "", spec.y or ""), 0.0)
        return (
            interaction >= LINE_GROUP_INTERACTION_MIN or main_effect >= LINE_GROUP_MAIN_EFFECT_MIN
        )
    return True  # bar/histogram grouping carries no scanned hypothesis


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


def apply_diversity_caps(recs: list[Recommendation]) -> list[Recommendation]:
    """Per-type/per-x slots picked in score order plus the overall MAX_CHARTS
    cut. Also called after the LLM merge (stage8 critique #4): re-suggesting a
    chart the caps removed must not bypass the diversity defenses."""
    per_type: dict[str, list[Recommendation]] = {}
    for rec in sorted(recs, key=_sort_key):
        per_type.setdefault(rec.spec.type, []).append(rec)
    kept = []
    for chart_type, group in per_type.items():
        # per-x cap for bar/box: one strong categorical must not monopolize a
        # type's slots (stage7 review: quantity took all three bar slots on
        # sales_basic). Other types put their variety on y, not x.
        if chart_type not in ("bar", "box"):
            kept += group[:MAX_PER_TYPE]
            continue
        picked: list[Recommendation] = []
        per_x: dict[str, int] = {}
        for rec in group:
            x = rec.spec.x or ""
            if per_x.get(x, 0) >= MAX_PER_X:
                continue
            per_x[x] = per_x.get(x, 0) + 1
            picked.append(rec)
            if len(picked) >= MAX_PER_TYPE:
                break
        kept += picked
    return sorted(kept, key=_sort_key)[:MAX_CHARTS]
