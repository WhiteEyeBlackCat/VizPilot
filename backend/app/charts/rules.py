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
from .intent import HEXBIN_MIN_N, ChartChoice, Intent, IntentContext, classify_intent, select_chart
from .roles import ColumnRole, roles_for
from .spec import CATEGORY_AXIS_TYPES, ChartSpec, ChartType, TimeGranularity, validate_spec

MAX_CHARTS = 12
MAX_PER_TYPE = 3  # diversity cap (stage3 blocking #1)
MAX_PER_X = 2  # within a type, one x column may fill at most 2 slots
MAX_PER_Y = 3  # stage 14: one column as y in at most 3 bar/box/line charts
# stage 19.1: diminishing returns instead of a bare raw-type cap. A slot's
# value is the chart's score discounted by what the list already shows: each
# kept chart of the same analytical question — (intent, x, group_by): "how
# does <measure> move over the hours", "how does <measure> trend over time"
# with the measure varying — discounts the next answer to that question
# (three trend lines over the same date column answer one question three
# times), and beyond TYPE_FREE_SLOTS charts drawn with the same raw type a
# milder discount applies across questions (a cyclic line and a trend line
# are different findings, so they are never eliminated for looking alike —
# only nudged). Scores and tiers are untouched: this only decides which
# candidates take the MAX_CHARTS slots.
QUESTION_DECAY = 0.8
TYPE_DECAY = 0.85
TYPE_FREE_SLOTS = 2
# MAX_MISSING_RATIO lives in confidence.py (stage 9) and is re-exported here
MAX_GRANULARITY_POINTS = 500
SCATTER_MIN_CORR = 0.3
SCATTER_MAX_PAIRS = 5
GROUP_CATEGORIES_RANGE = (2, 8)
NONLINEAR_CORR_GAP = 0.15  # |spearman| - |pearson| above this -> non-linear note
LINE_GROUP_INTERACTION_MIN = 0.05
LINE_GROUP_MAIN_EFFECT_MIN = 0.1  # eta-squared(group, y): main-effect grouping still earns a split
# stage 19.1: a finding is a distribution difference (raincloud / ridgeline /
# box) when evidence.dispersion carries a STRONG entry for the pair — the
# thresholds live in profiling/evidence.py (DISPERSION_*)
TWO_STATE_INTERACTION_MIN = 0.05  # interaction share for a category × two-state chart
# stage 20b composition: the same interaction evidence says the mix of parts is
# not the same in every bar — without it a stack only restates two bar charts.
# The floor is deliberately above the two-state one: a dumbbell asks a narrow
# question about two states, a stacked bar takes a slot to claim the whole mix
# shifts, so a share that barely clears the noise does not earn one.
COMPOSITION_INTERACTION_MIN = 0.08
COMPOSITION_PARTS_RANGE = (2, 8)
COMPOSITION_X_RANGE = (2, 20)
# a rate, a share or an average is already normalised: adding its values up
# produces no total, so it is never the measure of a composition
_NON_ADDITIVE_NAME_RE = re.compile(
    r"(^|_)(pct|percent|percentage|ratio|rate|share|avg|average|mean|median|per)($|_)", re.I
)
# stage 20b flow: a sankey claims "these rows moved from A to B". The only
# honest, statistic-free signal for that is the author's own naming — a
# declared from/to (source/target, prev/next …) pair over one stem. A pair of
# look-alike category columns is NOT treated as a flow: it is as likely to be
# a copy, whose sankey would be all self-loops.
FLOW_SCORE = 0.55  # descriptive, like the count bar; no effect size behind it
FLOW_NODES_RANGE = (2, 20)
_FLOW_TOKEN_PAIRS = (
    ("source", "target"),
    ("src", "dst"),
    ("src", "dest"),
    ("from", "to"),
    ("origin", "destination"),
    ("origin", "dest"),
    ("orig", "dest"),
    ("start", "end"),
    ("prev", "next"),
    ("previous", "next"),
    ("before", "after"),
    ("old", "new"),
    ("parent", "child"),
    ("sender", "receiver"),
)
# types sharing the bar's per-x cap and the per-y cap (stage 19 family)
_X_CAPPED_TYPES = frozenset({"bar", "lollipop", "box", "violin", "raincloud", "ridgeline", "dot_interval", "polar_line", "rose", "line", "stacked_bar"})
_Y_CAPPED_TYPES = _X_CAPPED_TYPES | frozenset({"dumbbell", "slopegraph"})

Tier = Literal["top", "secondary", "exploratory"]

# verification level of an LLM-suggested chart against the evidence table
VerificationLevel = Literal["strong", "weak", "unverified", "neutral"]
# stage 19: evidence family of each chart type (how evaluate_llm_spec scores it)
_FAMILY: dict[str, str] = {
    "lollipop": "bar",
    "polar_line": "bar",
    "rose": "bar",
    "stacked_bar": "bar",
    "dumbbell": "bar",
    "slopegraph": "bar",
    "violin": "box",
    "raincloud": "box",
    "ridgeline": "box",
    "dot_interval": "box",
    "hexbin": "scatter",
}
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
    # stage 19 (additive): why this chart family was chosen, and the other
    # eligible charts of the same finding (the UI shows the preferred one)
    intent: Intent | None = None
    alternatives: list[ChartSpec] = []
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
    elif derived.kind == "bucket":
        note = (
            f"{derived.target} is a banding of {derived.components[0]} (its groups are ordered, "
            "disjoint ranges of that column); this relationship is definitional, not a finding."
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
        # stage 19: (cat_a, cat_b, num) -> interaction share, both orders (no time markers)
        self.interaction: dict[tuple[str, str, str], float] = {}
        for e in evidence.interactions:
            if "@" in e.cat1 or "@" in e.cat2:
                continue
            for a, b in ((e.cat1, e.cat2), (e.cat2, e.cat1)):
                self.interaction[(a, b, e.num)] = max(self.interaction.get((a, b, e.num), 0.0), e.strength)
        # stage 19: (cat, num) -> smallest group and whether the group spreads differ
        self.n_min = {(e.cat, e.num): e.n_min for e in evidence.cat_num}
        # stage 19.1: the spread evidence is layer 1's own (Brown–Forsythe
        # eta-squared + robust spread ratio, evidence.dispersion), decoupled
        # from the eta-squared gate of the L2 group summaries: equal means
        # with different spreads count. Only a strong entry (every group
        # >= 30 rows) moves a finding to the distribution family — the same
        # floor the density charts need. Stage 19.2: an entry flagged
        # definitional (num derived from cat, or cat = bucket(num)) is kept
        # as evidence but never promotes a headline distribution chart —
        # "price spread differs across price_band" restates the banding.
        self.dispersion = {(d.cat, d.num): d for d in getattr(evidence, "dispersion", [])}
        self.spread_differs: dict[tuple[str, str], bool] = {
            key: True
            for key, d in self.dispersion.items()
            if d.strength == "strong" and not getattr(d, "definitional", False)
        }
        # stage 19: (x, y) -> binned eta-squared of the L2 non-linear scan (numeric cycles)
        self.binned_eta2 = {
            (n.x, n.y): n.binned_eta2 for n in getattr(getattr(evidence, "layer2", None), "nonlinear", []) or []
        }
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
    roles = roles_for(profile.columns)
    # a cyclic numeric (hour 0..23) is a category axis for the cycle charts
    # and stays a numeric for scatter / histogram / heatmap
    cyclic_numeric = [c for c in numeric_cols if roles[c.name].cyclic]
    measures = [c for c in numeric_cols if not roles[c.name].cyclic]

    candidates: list[Recommendation] = []
    candidates += _line_charts(profile, datetime_cols, measures, group_cols, index)
    candidates += _category_charts(profile, cat_cols, measures, index, roles)
    candidates += _cyclic_numeric_charts(profile, cyclic_numeric, measures, index, roles)
    candidates += _count_chart(profile, cat_cols, roles)
    candidates += _two_state_charts(profile, cat_cols, measures, index, roles)
    candidates += _matrix_charts(profile, cat_cols, measures, index, roles)
    candidates += _composition_charts(profile, cat_cols, measures, index, roles)
    candidates += _flow_charts(profile, cat_cols, roles)
    # a cycle (hour) is drawn once, by the cyclic line: never also as a scatter / hexbin
    candidates += _scatter_charts(profile, measures, group_cols, index)
    candidates += _histogram_charts(profile, numeric_cols)
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
                    intent="trend",
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


def _variant(spec: ChartSpec, chart_type: ChartType, choice: ChartChoice) -> ChartSpec:
    """The same finding drawn as another eligible chart: the fields each
    type accepts, the type-specific ones dropped, the order carried over.
    The spec's own type comes back unchanged (only the order is filled in)."""
    sort = choice.sort if chart_type in CATEGORY_AXIS_TYPES else None
    if chart_type == spec.type:
        return spec if spec.sort is not None or sort is None else spec.model_copy(update={"sort": sort})
    update: dict = {"type": chart_type, "sort": sort, "top_n": None, "percent": None}
    if chart_type in ("stacked_bar", "sankey"):
        # only sum / count compose (spec.py COMPOSITION_AGGREGATIONS)
        update.update(aggregation="count" if spec.y is None else (spec.aggregation if spec.aggregation == "sum" else "sum"))
        if chart_type == "stacked_bar":
            update.update(percent=spec.percent)
    elif chart_type in ("box", "violin", "raincloud", "ridgeline", "dot_interval"):
        update.update(aggregation=None, group_by=None)
    elif chart_type in ("bar", "lollipop", "polar_line", "rose"):
        update.update(aggregation=spec.aggregation or ("count" if spec.y is None else "mean"))
        if chart_type in ("polar_line", "rose"):
            update.update(group_by=None)
    elif chart_type == "line":
        # over a cycle / category the line needs an aggregate; a trend keeps its own
        update.update(aggregation=spec.aggregation or "mean")
        if choice.intent == "cyclic_pattern":
            update.update(group_by=None)
    elif chart_type in ("scatter", "hexbin"):
        update.update(aggregation=None)
    elif chart_type in ("dumbbell", "slopegraph"):
        update.update(aggregation=spec.aggregation or "mean")
    return spec.model_copy(update=update)


def _finding(
    profile: DatasetProfile,
    base: ChartSpec,
    score: float,
    x: ColumnRole | None,
    y: ColumnRole | None,
    group: ColumnRole | None,
    ctx: IntentContext,
    hint: ChartType | None = None,
) -> Recommendation | None:
    """One recommendation per analytical finding: the intent picks the chart
    family, the preferred eligible chart becomes the spec and the other
    eligible ones its alternatives (each validated on this profile)."""
    intent = classify_intent(x, y, group, ctx)
    if intent is None:
        return None
    choice = select_chart(intent, x, group, ctx, hint=hint)
    if choice is None:
        return None
    candidates = [choice.preferred, *choice.alternatives]
    specs = [_variant(base, t, choice) for t in candidates]
    valid = [sp for sp in specs if not validate_spec(sp, profile)]
    if not valid:
        return None
    return Recommendation(spec=valid[0], score=score, intent=intent, alternatives=valid[1:])


def _category_ctx(cat: ColumnProfile, num: ColumnProfile | None, index: _EvidenceIndex, profile: DatasetProfile, **extra) -> IntentContext:
    key = (cat.name, num.name) if num is not None else None
    return IntentContext(
        n_rows=profile.profiled_rows or profile.n_rows,
        n_groups=cat.n_categories or 0,
        n_min_group=index.n_min.get(key, 0) if key else 0,
        distribution_evidence=index.spread_differs.get(key, False) if key else False,
        **extra,
    )


def _category_charts(
    profile: DatasetProfile,
    cat_cols: list[ColumnProfile],
    numeric_cols: list[ColumnProfile],
    index: _EvidenceIndex,
    roles: dict[str, ColumnRole],
) -> list[Recommendation]:
    """Stage 19: one chart per (category, measure) finding. The intent
    (group comparison / distribution by group / cyclic pattern / ranking)
    follows the column roles and the evidence; the chart follows the intent."""
    recs = []
    for cat in [c for c in cat_cols if 2 <= (c.n_categories or 0) <= 20]:
        for num in numeric_cols:
            eta = index.eta(cat.name, num.name)
            ctx = _category_ctx(cat, num, index, profile)
            x_role, y_role = roles[cat.name], roles[num.name]
            intent = classify_intent(x_role, y_role, None, ctx)
            if intent == "distribution_by_group":
                title = f"{num.name} distribution by {cat.name}"
                reason = f"Shows how the distribution of {num.name} differs across {cat.name} categories, not just its average."
                score = 0.4 + 0.45 * eta
            elif intent == "cyclic_pattern":
                title = f"{num.name} across {cat.name}"
                reason = f"Shows how {num.name} moves over the {cat.name} cycle, in its natural order."
                score = 0.45 + 0.45 * eta
            elif intent == "ranking":
                title = f"{cat.name} ranked by mean {num.name}"
                reason = f"Ranks {cat.name} categories by average {num.name}."
                score = 0.45 + 0.45 * eta
            else:
                title = f"Mean {num.name} by {cat.name}"
                reason = f"Compares average {num.name} across {cat.name} categories."
                score = 0.45 + 0.45 * eta
            base = ChartSpec(title=title, type="bar", x=cat.name, y=num.name, aggregation="mean", reason=reason)
            rec = _finding(profile, base, score, x_role, y_role, None, ctx)
            if rec is not None:
                recs.append(rec)
    return recs


def _cyclic_numeric_charts(
    profile: DatasetProfile,
    cyclic_cols: list[ColumnProfile],
    numeric_cols: list[ColumnProfile],
    index: _EvidenceIndex,
    roles: dict[str, ColumnRole],
) -> list[Recommendation]:
    """A numeric cycle (hour 0..23) against a measure: strength from the L2
    non-linear scan (binned eta-squared) or the rank correlation; drawn as a
    line over the cycle in its natural order (polar / rose as alternatives)."""
    recs = []
    for cyc in cyclic_cols:
        role = roles[cyc.name]
        for num in numeric_cols:
            strength = math.sqrt(index.binned_eta2.get((cyc.name, num.name), 0.0))
            strength = max(strength, abs(index.spearman.get((cyc.name, num.name), 0.0)))
            ctx = IntentContext(
                n_rows=profile.profiled_rows or profile.n_rows,
                n_groups=min(cyc.unique_count, role.period or cyc.unique_count),
                n_min_group=(profile.profiled_rows or profile.n_rows) // max(cyc.unique_count, 1),
            )
            base = ChartSpec(
                title=f"{num.name} across {cyc.name}",
                type="line",
                x=cyc.name,
                y=num.name,
                aggregation="mean",
                reason=f"Shows how {num.name} moves over the {cyc.name} cycle (mean per {cyc.name}).",
            )
            rec = _finding(profile, base, 0.45 + 0.45 * strength, role, roles[num.name], None, ctx)
            if rec is not None:
                recs.append(rec)
    return recs


def _count_chart(profile: DatasetProfile, cat_cols: list[ColumnProfile], roles: dict[str, ColumnRole]) -> list[Recommendation]:
    """Record count over the widest low-cardinality category (fixed score):
    a ranking for a nominal column, the cycle order for a cyclic one."""
    bar_cats = [c for c in cat_cols if 2 <= (c.n_categories or 0) <= 20]
    if not bar_cats:
        return []
    count_cat = max(bar_cats, key=lambda c: (c.n_categories, c.name))
    role = roles[count_cat.name]
    intent: Intent = "cyclic_pattern" if role.cyclic else "ranking"
    ctx = IntentContext(n_rows=profile.n_rows, n_groups=count_cat.n_categories or 0, count_measure=True, magnitude_measure=True)
    choice = select_chart(intent, role, None, ctx, hint="bar")
    spec = ChartSpec(
        title=f"Record count by {count_cat.name}",
        type="bar",
        x=count_cat.name,
        aggregation="count",
        sort=choice.sort if choice else None,
        reason=f"Shows how many records fall in each {count_cat.name} category.",
    )
    alternatives = []
    if choice is not None:
        alternatives = [sp for sp in (_variant(spec, t, choice) for t in choice.alternatives) if not validate_spec(sp, profile)]
    return [Recommendation(spec=spec, score=0.6, intent=intent, alternatives=alternatives)]


def _two_state_charts(
    profile: DatasetProfile,
    cat_cols: list[ColumnProfile],
    numeric_cols: list[ColumnProfile],
    index: _EvidenceIndex,
    roles: dict[str, ColumnRole],
) -> list[Recommendation]:
    """Category × two-state (before/after, weekday/weekend, A/B) → measure,
    when the interaction evidence says the two states move the categories
    differently. Drawn as a dumbbell (slopegraph / grouped bar alternatives)."""
    recs = []
    states = [c for c in cat_cols if roles[c.name].two_state]
    for state in states:
        for cat in cat_cols:
            if cat.name == state.name or not 2 <= (cat.n_categories or 0) <= 12:
                continue
            for num in numeric_cols:
                strength = index.interaction.get((cat.name, state.name, num.name), 0.0)
                if strength < TWO_STATE_INTERACTION_MIN:
                    continue
                ctx = _category_ctx(cat, num, index, profile)
                base = ChartSpec(
                    title=f"{num.name} by {cat.name}: {state.name} compared",
                    type="dumbbell",
                    x=cat.name,
                    y=num.name,
                    group_by=state.name,
                    aggregation="mean",
                    reason=f"Compares mean {num.name} between the two {state.name} states within each {cat.name} (the gap differs by category).",
                )
                rec = _finding(profile, base, 0.45 + 0.45 * math.sqrt(strength), roles[cat.name], roles[num.name], roles[state.name], ctx)
                if rec is not None:
                    recs.append(rec)
    return recs


def _matrix_charts(
    profile: DatasetProfile,
    cat_cols: list[ColumnProfile],
    numeric_cols: list[ColumnProfile],
    index: _EvidenceIndex,
    roles: dict[str, ColumnRole],
) -> list[Recommendation]:
    """Two ordered dimensions (month × hour, level × weekday) → measure, when
    the interaction evidence is there: a matrix heatmap of the aggregate."""
    recs = []
    ordered = [c for c in cat_cols if roles[c.name].ordered and not roles[c.name].two_state and 2 <= (c.n_categories or 0) <= 20]
    for i, a in enumerate(ordered):
        for b in ordered[i + 1 :]:
            for num in numeric_cols:
                strength = index.interaction.get((a.name, b.name, num.name), 0.0)
                if strength < TWO_STATE_INTERACTION_MIN:
                    continue
                ctx = IntentContext(n_rows=profile.n_rows, n_groups=a.n_categories or 0, n_categories=b.n_categories or 0)
                base = ChartSpec(
                    title=f"Mean {num.name} by {a.name} × {b.name}",
                    type="heatmap",
                    x=a.name,
                    y=num.name,
                    group_by=b.name,
                    aggregation="mean",
                    reason=f"Shows how mean {num.name} depends on the combination of {a.name} and {b.name}.",
                )
                rec = _finding(profile, base, 0.45 + 0.45 * math.sqrt(strength), roles[a.name], roles[num.name], roles[b.name], ctx)
                if rec is not None:
                    recs.append(rec)
    return recs


# --- stage 20b: composition / flow -------------------------------------------


def _is_additive_measure(num: ColumnProfile) -> bool:
    """A measure whose parts add up to a whole: a genuine numeric quantity,
    never negative (a signed measure has no part-to-whole reading, and a
    rating or a code is not a quantity to sum), and not already a rate or an
    average — the sum of percentages or of per-unit values is not a total."""
    if num.semantic_type != "numeric" or _NON_ADDITIVE_NAME_RE.search(num.name):
        return False
    return isinstance(num.min, (int, float)) and not isinstance(num.min, bool) and num.min >= 0


def _composition_charts(
    profile: DatasetProfile,
    cat_cols: list[ColumnProfile],
    numeric_cols: list[ColumnProfile],
    index: _EvidenceIndex,
    roles: dict[str, ColumnRole],
) -> list[Recommendation]:
    """Category × parts → summed magnitude, when the interaction evidence says
    the mix of parts is not the same in every category. Drawn as a stacked bar
    (grouped bar as the alternative) — never a pie."""
    recs: list[Recommendation] = []
    x_lo, x_hi = COMPOSITION_X_RANGE
    p_lo, p_hi = COMPOSITION_PARTS_RANGE
    for cat in [c for c in cat_cols if x_lo <= (c.n_categories or 0) <= x_hi]:
        for part in cat_cols:
            if part.name == cat.name or not p_lo <= (part.n_categories or 0) <= p_hi:
                continue
            # a composition is directional: the bars are the whole, the stack
            # the split. Only the reading whose stack is the narrower
            # dimension is drawn (ties by name), so one pair never yields two
            # charts of the same numbers transposed.
            if (part.n_categories or 0, part.name) > (cat.n_categories or 0, cat.name):
                continue
            for num in numeric_cols:
                if not _is_additive_measure(num):
                    continue
                strength = index.interaction.get((cat.name, part.name, num.name), 0.0)
                if strength < COMPOSITION_INTERACTION_MIN:
                    continue
                ctx = _category_ctx(
                    cat,
                    num,
                    index,
                    profile,
                    n_categories=part.n_categories or 0,
                    part_to_whole=True,
                    magnitude_measure=True,
                )
                base = ChartSpec(
                    title=f"Total {num.name} by {cat.name}, split by {part.name}",
                    type="stacked_bar",
                    x=cat.name,
                    y=num.name,
                    group_by=part.name,
                    aggregation="sum",
                    reason=(
                        f"Shows how the total {num.name} of each {cat.name} is made up of its "
                        f"{part.name} parts (the mix differs by {cat.name})."
                    ),
                )
                rec = _finding(
                    profile, base, 0.45 + 0.45 * math.sqrt(strength), roles[cat.name], roles[num.name], roles[part.name], ctx
                )
                if rec is not None:
                    recs.append(rec)
    return recs


def _name_parts(name: str) -> list[tuple[str, str]]:
    """(token, stem) readings of a column name: the whole name, its first
    underscore-separated token and its last one ("from_city" -> ("from",
    "city"), "city_from" -> ("from", "city"))."""
    low = name.strip().lower()
    out = [(low, "")]
    if "_" in low:
        head, rest = low.split("_", 1)
        out.append((head, rest))
        stem, tail = low.rsplit("_", 1)
        out.append((tail, stem))
    return out


def flow_direction(a: str, b: str) -> tuple[str, str] | None:
    """(source, target) when the two column names declare a transition over
    one stem (from_city / to_city, status_prev / status_next, src / dst);
    None otherwise. Deterministic and independent of the argument order."""
    for first, second in ((a, b), (b, a)):
        for token_a, stem_a in _name_parts(first):
            for token_b, stem_b in _name_parts(second):
                if stem_a == stem_b and (token_a, token_b) in _FLOW_TOKEN_PAIRS:
                    return first, second
    return None


def _flow_charts(
    profile: DatasetProfile,
    cat_cols: list[ColumnProfile],
    roles: dict[str, ColumnRole],
) -> list[Recommendation]:
    """A declared from → to pair of category columns: how many rows move from
    each source to each target, drawn as a sankey. The weight is a row count:
    the flow itself is the finding, so there is no effect size to score and
    the chart carries the descriptive FLOW_SCORE."""
    recs: list[Recommendation] = []
    lo, hi = FLOW_NODES_RANGE
    usable = {c.name: c for c in cat_cols if lo <= (c.n_categories or 0) <= hi}
    names = sorted(usable)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            direction = flow_direction(a, b)
            if direction is None:
                continue
            src, dst = direction
            ctx = IntentContext(
                n_rows=profile.profiled_rows or profile.n_rows,
                n_groups=usable[src].n_categories or 0,
                n_categories=usable[dst].n_categories or 0,
                count_measure=True,
                magnitude_measure=True,
                flow_pair=True,
            )
            base = ChartSpec(
                title=f"Flow from {src} to {dst}",
                type="sankey",
                x=src,
                group_by=dst,
                aggregation="count",
                reason=f"Shows how many records move from each {src} to each {dst}.",
            )
            rec = _finding(profile, base, FLOW_SCORE, roles[src], None, roles[dst], ctx)
            if rec is not None:
                recs.append(rec)
    return recs


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
        n_rows = profile.profiled_rows or profile.n_rows
        spec = ChartSpec(title=f"{b} vs {a}", type="scatter", x=a, y=b, group_by=group, reason=reason + ".")
        alternatives: list[ChartSpec] = []
        if group is None and n_rows > HEXBIN_MIN_N:
            # stage 19: overplotting — the hexbin is the readable chart, the
            # (sampled) scatter stays available as an alternative
            alternatives = [spec]
            spec = spec.model_copy(update={"type": "hexbin"})
        elif group is None and n_rows >= 300:
            alternatives = [spec.model_copy(update={"type": "hexbin"})]
        recs.append(Recommendation(spec=spec, score=0.5 + 0.4 * strength, intent="numeric_relationship", alternatives=alternatives))
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


def _histogram_charts(profile: DatasetProfile, numeric_cols: list[ColumnProfile]) -> list[Recommendation]:
    recs = []
    n_rows = profile.profiled_rows or profile.n_rows
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
                intent="distribution",
                alternatives=[ChartSpec(title=f"Distribution of {num.name}", type="box", y=num.name, reason=f"Shows the distribution of {num.name}.")]
                + ([ChartSpec(title=f"Distribution of {num.name}", type="violin", y=num.name, reason=f"Shows the distribution of {num.name}.")] if n_rows >= 30 else []),
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


def select_for_spec(spec: ChartSpec, profile: DatasetProfile) -> tuple[ChartSpec, list[ChartSpec], Intent | None]:
    """Stage 19: an LLM- or probe-suggested chart goes through the same
    intent → selection step as the rule charts. The suggested type is only a
    hint: it stays when it is eligible for the finding, otherwise the
    preferred eligible chart replaces it (the hint never becomes an
    alternative when the guards reject it). Returns (spec, alternatives,
    intent); a spec the classifier has no intent for is returned as is."""
    roles = roles_for(profile.columns)
    x, y, group = roles.get(spec.x or ""), roles.get(spec.y or ""), roles.get(spec.group_by or "")
    if spec.type in ("heatmap", "histogram") and spec.x is None:
        return spec, [], ("distribution" if spec.type == "histogram" else None)
    index = _EvidenceIndex(profile.evidence)
    columns = {c.name: c for c in profile.columns}
    xcol = columns.get(spec.x or "")
    n_rows = profile.profiled_rows or profile.n_rows
    n_groups = 0
    if xcol is not None:
        n_groups = xcol.n_categories or (min(xcol.unique_count, x.period) if x is not None and x.period else xcol.unique_count)
    key = (spec.x or "", spec.y or "")
    ctx = IntentContext(
        n_rows=n_rows,
        n_groups=n_groups,
        n_min_group=index.n_min.get(key, n_rows // max(n_groups, 1) if n_groups else 0),
        n_categories=(columns.get(spec.group_by or "").n_categories or 0) if spec.group_by in columns else 0,
        distribution_evidence=index.spread_differs.get(key, False) or spec.type in ("box", "violin", "raincloud", "ridgeline"),
        count_measure=spec.y is None,
        magnitude_measure=spec.aggregation in ("sum", "count"),
        ranking_requested=spec.sort == "value",
        # stage 20b: a suggested stacked bar / sankey states the intent itself
        # (the guards in intent.py still decide whether it is eligible)
        part_to_whole=spec.type == "stacked_bar",
        flow_pair=spec.type == "sankey" or flow_direction(spec.x or "", spec.group_by or "") == (spec.x, spec.group_by),
    )
    if spec.type == "histogram":
        return spec, [], "distribution"
    intent = classify_intent(x, y, group, ctx)
    if intent is None:
        return spec, [], None
    choice = select_chart(intent, x, group, ctx, hint=spec.type)
    if choice is None:
        return spec, [], intent
    ordered = [choice.preferred, *choice.alternatives]
    specs = [sp for sp in (_variant(spec, t, choice) for t in ordered) if not validate_spec(sp, profile)]
    if not specs:
        return spec, [], intent
    return specs[0], specs[1:], intent


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

    if spec.type == "heatmap" and spec.x is not None:
        # stage 19 matrix pattern: scored on the interaction share of its two dimensions
        strength = index.interaction.get((spec.x, spec.group_by or "", spec.y or ""))
        if strength is None:
            return UNVERIFIED_SCORE, "unverified"
        score = 0.45 + 0.45 * math.sqrt(strength)
        return score, "strong" if score >= TOP_SCORE_FLOOR else "weak"
    if spec.type == "heatmap":
        return 0.65, "neutral"
    if spec.type == "sankey" or (spec.type == "stacked_bar" and spec.y is None):
        # stage 20b: counting how rows split across two category columns is a
        # description, not a hypothesis the evidence scan can grade
        return 0.6, "neutral"
    if spec.type == "stacked_bar":
        # the claim is "the mix of parts differs by category": the same
        # interaction share the matrix heatmap is scored on
        strength = index.interaction.get((spec.x or "", spec.group_by or "", spec.y or ""))
        if strength is None:
            return UNVERIFIED_SCORE, "unverified"
        score = 0.45 + 0.45 * math.sqrt(strength)
        return score, "strong" if score >= TOP_SCORE_FLOOR else "weak"
    # stage 19: the new types are scored as the family they belong to
    family = _FAMILY.get(spec.type, spec.type)
    if spec.type == "line" and (col := columns.get(spec.x or "")) is not None and col.semantic_type == "categorical":
        family = "bar"  # a line over an ordered category is a group comparison
    if spec.type in ("polar_line", "rose", "line") and (col := columns.get(spec.x or "")) is not None and col.semantic_type == "numeric":
        # a numeric cycle (hour): the non-linear scan or the rank correlation
        strength = math.sqrt(index.binned_eta2.get((spec.x or "", spec.y or ""), 0.0))
        strength = max(strength, abs(index.spearman.get((spec.x or "", spec.y or ""), 0.0)))
        if spec.type == "line" and strength == 0.0:
            family = "line"
        else:
            score = 0.45 + 0.45 * strength
            return score, "strong" if score >= TOP_SCORE_FLOOR else "weak"
    spec = spec.model_copy(update={"type": family})
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

    - per family: at most MAX_PER_TYPE charts of one intent (or type when
      the chart carries no intent)
    - per x (bar/box): one strong categorical must not monopolize a type's
      slots (stage7 review: quantity took all three bar slots on sales_basic)
    - per y (bar/box/line, stage 14): one measure must not fill the list
      from every angle (temp took 5 of 12 slots on the bike-sharing hours)
    - stage 19.1, diminishing returns: slots are filled one at a time by the
      highest *slot value* = score × QUESTION_DECAY^(kept answers to the same
      (intent, x, group_by) question) × TYPE_DECAY^(kept charts of the same
      raw type beyond TYPE_FREE_SLOTS), so a third trend line over the same
      date column competes against the best histogram at a discount instead
      of winning on raw score. Deterministic: ties fall back to the score
      order. Membership is decided here; the returned list is score-ordered
      as before and scores / tiers are unchanged.
    """
    kept: list[Recommendation] = []
    per_type: dict[str, int] = {}
    per_question: dict[tuple[str, str, str], int] = {}
    per_raw_type: dict[str, int] = {}
    per_x: dict[tuple[str, str], int] = {}
    per_y: dict[str, int] = {}
    pool = sorted(recs, key=_sort_key)

    def _question(rec: Recommendation) -> tuple[str, str, str]:
        return (rec.intent or rec.spec.type, rec.spec.x or "", rec.spec.group_by or "")

    def _admissible(rec: Recommendation) -> bool:
        chart_type = rec.spec.type
        # stage 19: the type cap counts chart families (the intent) — a cyclic
        # line and a trend line are different findings, two bars of the same
        # intent are not more diverse for being bars
        if per_type.get(rec.intent or chart_type, 0) >= MAX_PER_TYPE:
            return False
        # trend lines all share the dataset's datetime column: the per-x cap
        # applies to lines over a cycle only
        if _x_capped(rec) and per_x.get((chart_type, rec.spec.x or ""), 0) >= MAX_PER_X:
            return False
        if _y_capped(rec) and per_y.get(rec.spec.y or "", 0) >= MAX_PER_Y:
            return False
        return True

    def _slot_value(rec: Recommendation) -> float:
        answers = per_question.get(_question(rec), 0)
        raw = max(0, per_raw_type.get(rec.spec.type, 0) - TYPE_FREE_SLOTS)
        return rec.score * QUESTION_DECAY**answers * TYPE_DECAY**raw

    while pool and len(kept) < MAX_CHARTS:
        best: Recommendation | None = None
        best_value = -1.0
        for rec in pool:
            if not _admissible(rec):
                continue
            value = _slot_value(rec)
            if value > best_value + 1e-12:  # strict: the first (score-ordered) candidate wins ties
                best, best_value = rec, value
        if best is None:
            break
        pool.remove(best)
        kept.append(best)
        family = best.intent or best.spec.type
        per_type[family] = per_type.get(family, 0) + 1
        per_question[_question(best)] = per_question.get(_question(best), 0) + 1
        per_raw_type[best.spec.type] = per_raw_type.get(best.spec.type, 0) + 1
        if _x_capped(best):
            key = (best.spec.type, best.spec.x or "")
            per_x[key] = per_x.get(key, 0) + 1
        if _y_capped(best):
            per_y[best.spec.y or ""] = per_y.get(best.spec.y or "", 0) + 1
    return sorted(kept, key=_sort_key)


def _x_capped(rec: Recommendation) -> bool:
    return rec.spec.type in _X_CAPPED_TYPES and (rec.spec.type != "line" or rec.intent == "cyclic_pattern")


def _y_capped(rec: Recommendation) -> bool:
    return rec.spec.type in _Y_CAPPED_TYPES and rec.spec.y is not None
