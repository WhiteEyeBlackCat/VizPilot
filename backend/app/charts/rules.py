"""Deterministic chart recommendations, ranked by profile evidence (stage 7).

Scores are anchored to effect sizes (eta = sqrt(adjusted eta-squared), which
shares the |corr| scale) instead of cardinality guesses; lexicographic
tie-breaks remain only for determinism. Every candidate passes validate_spec;
priority (1..N) is the cross-stage contract (D8).
"""

import math
import re
import statistics
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from ..profiling.models import ColumnProfile, Correlations, DatasetProfile, DerivedColumn, Evidence
from .confidence import (
    MAX_MISSING_RATIO,
    Confidence,
    TierCap,
    Warning,
    assess,
    has_suspected_sentinels,
    with_caution,
)
from .spec import ChartSpec, TimeGranularity, validate_spec

MAX_CHARTS = 12
MAX_PER_TYPE = 3  # diversity cap (stage3 blocking #1)
MAX_PER_X = 2  # within a type, one x column may fill at most 2 slots
MAX_PER_Y = 3  # stage 14: one column as y in at most 3 bar/box/line charts
# MAX_MISSING_RATIO lives in confidence.py (stage 9) and is re-exported here
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
# stage 13: a chart of a derived column against one of its components draws
# the definition (sales = price × quantity × (1 − discount)), not a finding —
# the score is discounted and the tier capped at exploratory
DERIVED_SCORE_FACTOR = 0.6


class Recommendation(BaseModel):
    spec: ChartSpec
    # FINAL score = base evidence score x confidence.overall (stage 9); the
    # base is recoverable as score / confidence.overall
    score: float
    source: Literal["rules", "llm"] = "rules"
    tier: Tier = "exploratory"
    confidence: Confidence | None = None  # stage 9, additive
    warnings: list[Warning] = []  # stage 9, additive
    # tier ceiling ("secondary"/"exploratory") from LLM verification and/or
    # the confidence layer; internal only — applied by assign_tiers and never
    # serialized into the API response
    tier_cap: Tier | None = Field(default=None, exclude=True)


_CAP_RANK: dict[Tier | None, int] = {None: 0, "secondary": 1, "exploratory": 2}


def stricter_cap(a: Tier | None, b: TierCap | Tier | None) -> Tier | None:
    """The tighter of two tier ceilings (None < secondary < exploratory)."""
    return a if _CAP_RANK[a] >= _CAP_RANK[b] else b


def apply_confidence(recs: list[Recommendation], profile: DatasetProfile) -> None:
    """Runs every candidate through the confidence chain (stage 9): discounts
    the score, attaches the structured confidence/warnings, tightens the tier
    ceiling and appends the caution suffix to the reason. Called once per
    candidate — by recommend_charts for rules charts and by the LLM merge for
    new LLM charts — so fixed-score and evidence-scored charts are treated alike."""
    for rec in recs:
        confidence, warnings, cap = assess(rec.spec, profile)
        rec.score = rec.score * confidence.overall
        rec.confidence = confidence
        rec.warnings = warnings
        rec.tier_cap = stricter_cap(rec.tier_cap, cap)
        rec.spec.reason = with_caution(rec.spec.reason, warnings)


def apply_derived_caps(recs: list[Recommendation], profile: DatasetProfile) -> None:
    """Stage 13: charts whose x/y pair is a derived column against one of its
    components are definitional. Score x DERIVED_SCORE_FACTOR, tier capped at
    exploratory, an info warning attached and the reason rewritten so the
    card says why. Runs after apply_confidence (the caution suffix is part
    of the reason it prefixes) for rules and LLM candidates alike. Near-copy
    pairs only get the warning and a note — no cap, no discount."""
    index = _EvidenceIndex(profile.evidence)
    for rec in recs:
        derived = index.derived_for(rec.spec)
        if derived is not None:
            rec.score = rec.score * DERIVED_SCORE_FACTOR
            rec.tier_cap = stricter_cap(rec.tier_cap, "exploratory")
            rec.warnings = rec.warnings + [derived_relationship_warning(derived)]
            rec.spec.reason = _definitional_reason(derived, rec.spec.reason)
            continue
        near = index.near_copy_for(rec.spec)
        if near is not None:
            rec.warnings = rec.warnings + [derived_relationship_warning(near)]
            rec.spec.reason = _definitional_reason(near, rec.spec.reason)


def definitional_reason(spec: ChartSpec, profile: DatasetProfile, reason: str | None) -> str | None:
    """The note apply_derived_caps prefixes, recomputed for a replacement
    reason (the LLM merge adopts the LLM's wording for a rules chart and must
    not lose the disclosure). None when the pair is not definitional."""
    index = _EvidenceIndex(profile.evidence)
    derived = index.derived_for(spec) or index.near_copy_for(spec)
    return None if derived is None else _definitional_reason(derived, reason)


def _definitional_reason(derived: DerivedColumn, reason: str | None) -> str:
    if derived.kind == "near_copy":
        note = (
            f"{derived.target} is nearly a transformed copy of {derived.components[0]} "
            f"(rank correlation {derived.match_ratio:.2f}); the relationship is likely definitional."
        )
    else:
        note = (
            f"{derived.target} is computed as {derived.formula}; "
            "this relationship is definitional, not a finding."
        )
    return f"{note} {reason}".strip() if reason else note


def derived_relationship_warning(derived: DerivedColumn) -> Warning:
    """Per-chart info warning (severity info: never part of the caution text)."""
    return Warning(
        code="derived_relationship",
        severity="info",
        message=(
            f"{derived.target} is nearly a transformed copy of {derived.components[0]}."
            if derived.kind == "near_copy"
            else f"{derived.target} is computed as {derived.formula}."
        ),
        meta={
            "target": derived.target,
            "components": list(derived.components),
            "formula": derived.formula,
            "kind": derived.kind,
            "match_ratio": derived.match_ratio,
        },
    )


def derived_column_warnings(profile: DatasetProfile) -> list[Warning]:
    """Dataset-level: one `derived_column` per detected identity (and one
    `near_duplicate_column` per suppressed duplicate, stage 14) so the
    response discloses what the ranking demoted or left out."""
    out = []
    for d in getattr(profile.evidence, "derived_columns", []):
        if d.kind == "near_copy":
            message = (
                f"{d.target} is a near-duplicate of {d.components[0]} "
                f"(rank correlation {d.match_ratio:.2f}); recommendations use {d.components[0]} only."
            )
            code = "near_duplicate_column"
        else:
            components = ", ".join(d.components)
            message = (
                f"{d.target} appears to be computed as {d.formula}; charts of {d.target} "
                f"against {components} show the formula, not a finding."
            )
            code = "derived_column"
        out.append(
            Warning(
                code=code,
                severity="info",
                message=message,
                meta={
                    "target": d.target,
                    "components": list(d.components),
                    "formula": d.formula,
                    "kind": d.kind,
                    "match_ratio": d.match_ratio,
                },
            )
        )
    return out


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
        # stage 13: {target, component} pairs that only draw a definition.
        # near_copy entries are disclosure-only (see DerivedColumn) and are
        # kept apart so they never cap or discount a chart.
        self.derived_pairs: dict[frozenset[str], DerivedColumn] = {}
        self.near_copy_pairs: dict[frozenset[str], DerivedColumn] = {}
        for d in getattr(evidence, "derived_columns", []):
            table = self.near_copy_pairs if d.kind == "near_copy" else self.derived_pairs
            for component in d.components:
                table.setdefault(frozenset((d.target, component)), d)
        # stage 14: near-duplicate columns -> their group's representative;
        # only the representative takes part in candidates (heatmap excepted)
        self.suppressed: dict[str, str] = {}
        self.group_rho: dict[str, float] = {}
        for g in getattr(evidence, "near_duplicate_groups", []):
            for dup in g.duplicates:
                self.suppressed[dup] = g.representative
                self.group_rho[dup] = g.rho.get(dup, 0.0)

    def canonical(self, name: str | None) -> str | None:
        return None if name is None else self.suppressed.get(name, name)

    def derived_for(self, spec: ChartSpec) -> DerivedColumn | None:
        """The identity a chart's x/y pair merely restates, if any."""
        if not spec.x or not spec.y:
            return None
        return self.derived_pairs.get(frozenset((spec.x, spec.y)))

    def near_copy_for(self, spec: ChartSpec) -> DerivedColumn | None:
        if not spec.x or not spec.y:
            return None
        return self.near_copy_pairs.get(frozenset((spec.x, spec.y)))

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
    index = _EvidenceIndex(profile.evidence)
    datetime_cols = [c for c in usable if c.semantic_type == "datetime"]
    all_numeric = [c for c in usable if c.semantic_type == "numeric"]
    # stage 14: a near-duplicate (atemp next to temp) never earns its own
    # charts — its representative already draws them. The heatmap keeps
    # every column so the duplication itself stays visible.
    numeric_cols = [c for c in all_numeric if c.name not in index.suppressed]
    cat_cols = [c for c in usable if c.semantic_type in ("categorical", "boolean")]
    lo, hi = GROUP_CATEGORIES_RANGE
    group_cols = [c.name for c in cat_cols if lo <= (c.n_categories or 0) <= hi]

    candidates: list[Recommendation] = []
    candidates += _line_charts(profile, datetime_cols, numeric_cols, group_cols, index)
    candidates += _bar_charts(cat_cols, numeric_cols, index)
    candidates += _box_charts(cat_cols, numeric_cols, index)
    candidates += _scatter_charts(profile, numeric_cols, group_cols, index)
    candidates += _histogram_charts(numeric_cols)
    candidates += _heatmap_chart(all_numeric)

    valid = _dedup(r for r in candidates if not validate_spec(r.spec, profile))
    valid, _ = dedup_equivalent(valid, profile)  # stage 14: no-op for rules, shared post-pass
    apply_confidence(valid, profile)  # before the caps: slots go to confident charts
    apply_derived_caps(valid, profile)  # stage 13: definitional pairs sink before the slot cut
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
    # stage 13: definitional pairs go last so they only take a slot when
    # nothing else clears the threshold (they are capped downstream anyway)
    pairs.sort(key=lambda p: (frozenset((p[1], p[2])) in index.derived_pairs, -p[0], p[1], p[2]))

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
        # the skew bonus rewards a genuinely skewed distribution; skew
        # manufactured by suspected sentinels is the artefact the robustness
        # factor penalises, so it must not also earn the bonus (stage 9 #6)
        skewed = num.skewness is not None and abs(num.skewness) > 1 and not has_suspected_sentinels(num)
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
        skewed = (
            col is not None
            and col.skewness is not None
            and abs(col.skewness) > 1
            and not has_suspected_sentinels(col)  # same rule as _histogram_charts
        )
        return 0.5 + (0.05 if skewed else 0.0), "neutral"
    if spec.type in ("bar", "box") and spec.y is None:
        return 0.6, "neutral"  # count bar: no group-effect hypothesis
    if index.derived_for(spec) is not None:
        # stage 13: a hypothesis about a definitional pair is not a finding
        # the data can verify — capped at exploratory, insight unverified
        return UNVERIFIED_SCORE, "unverified"

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


def canonicalize_spec(spec: ChartSpec, profile: DatasetProfile) -> tuple[ChartSpec | None, list[str]]:
    """Stage 14: maps near-duplicate columns in x / y / group_by to their
    representative (an LLM may still ask for atemp). Returns the rewritten
    spec and the substitutions made ("atemp -> temp"); None when the
    substitution collapses x and y onto the same column (the chart would
    only show the duplication)."""
    index = _EvidenceIndex(profile.evidence)
    if not index.suppressed:
        return spec, []
    changes: dict[str, str | None] = {}
    notes: list[str] = []
    for field in ("x", "y", "group_by"):
        value = getattr(spec, field)
        mapped = index.canonical(value)
        if mapped != value:
            changes[field] = mapped
            notes.append(f"{value} -> {mapped}")
    if not changes:
        return spec, []
    x = changes.get("x", spec.x)
    y = changes.get("y", spec.y)
    if x is not None and x == y:
        return None, notes
    title = spec.title
    for note in notes:
        old_name, new_name = note.split(" -> ")
        title = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(old_name)}(?![A-Za-z0-9_])", new_name, title)
    changes["title"] = title
    return spec.model_copy(update=changes), notes


def near_duplicate_substituted_warning(notes: list[str]) -> Warning:
    pairs = [n.split(" -> ") for n in notes]
    return Warning(
        code="near_duplicate_substituted",
        severity="info",
        message="; ".join(f"{a} replaced by {b} (near-duplicate)" for a, b in pairs) + ".",
        meta={"substitutions": [{"from": a, "to": b} for a, b in pairs]},
    )


def _canonical_key(spec: ChartSpec, index: _EvidenceIndex) -> tuple:
    return (
        spec.type,
        index.canonical(spec.x),
        index.canonical(spec.y),
        index.canonical(spec.group_by),
        spec.aggregation,
    )


def dedup_equivalent(
    recs: list[Recommendation], profile: DatasetProfile
) -> tuple[list[Recommendation], dict[tuple, tuple]]:
    """Stage 14: charts that are the same chart once near-duplicates are
    mapped to their representative keep only the strongest (rules before
    LLM on ties). Returns the survivors in input order plus a map from a
    dropped chart's dedup key to the survivor's, so an insight that pointed
    at the dropped chart can follow it."""
    index = _EvidenceIndex(profile.evidence)
    best: dict[tuple, Recommendation] = {}
    for rec in recs:
        key = _canonical_key(rec.spec, index)
        current = best.get(key)
        if current is None or (
            (-rec.score, rec.source != "rules", _sort_key(rec)) < (-current.score, current.source != "rules", _sort_key(current))
        ):
            best[key] = rec
    kept_ids = {id(rec) for rec in best.values()}
    redirect: dict[tuple, tuple] = {}
    for rec in recs:
        if id(rec) not in kept_ids:
            survivor = best[_canonical_key(rec.spec, index)]
            redirect[_dedup_key(rec.spec)] = _dedup_key(survivor.spec)
    return [rec for rec in recs if id(rec) in kept_ids], redirect


def _dedup_key(spec: ChartSpec) -> tuple:
    return (spec.type, spec.x, spec.y, spec.group_by)


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
    """Slots picked in score order under three caps, then the overall
    MAX_CHARTS cut. Also called after the LLM merge (stage8 critique #4):
    re-suggesting a chart the caps removed must not bypass the defenses.

    - per type: at most MAX_PER_TYPE charts of one type
    - per x (bar/box): one strong categorical must not monopolize a type's
      slots (stage7 review: quantity took all three bar slots on sales_basic)
    - per y (bar/box/line, stage 14): one measure must not fill the list
      from every angle (temp took 5 of 12 slots on the bike-sharing hours)
    """
    kept: list[Recommendation] = []
    per_type: dict[str, int] = {}
    per_x: dict[tuple[str, str], int] = {}
    per_y: dict[str, int] = {}
    for rec in sorted(recs, key=_sort_key):
        chart_type = rec.spec.type
        if per_type.get(chart_type, 0) >= MAX_PER_TYPE:
            continue
        x_key = (chart_type, rec.spec.x or "")
        if chart_type in ("bar", "box") and per_x.get(x_key, 0) >= MAX_PER_X:
            continue
        y = rec.spec.y
        counts_y = chart_type in ("bar", "box", "line") and y is not None
        if counts_y and per_y.get(y, 0) >= MAX_PER_Y:
            continue
        kept.append(rec)
        per_type[chart_type] = per_type.get(chart_type, 0) + 1
        if chart_type in ("bar", "box"):
            per_x[x_key] = per_x.get(x_key, 0) + 1
        if counts_y:
            per_y[y] = per_y.get(y, 0) + 1
    return sorted(kept, key=_sort_key)[:MAX_CHARTS]
