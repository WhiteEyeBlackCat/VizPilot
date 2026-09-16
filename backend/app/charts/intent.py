"""Visualization intent and deterministic chart selection (stage 19).

    evidence + semantic roles → VisualizationIntent → eligible chart family
    → preferred chart (+ alternatives) → ChartSpec

Nothing here reads the data, calls the LLM or knows ECharts: it is a pure
function of column roles, a few evidence facts and sample sizes, so the same
inputs always give the same charts. The LLM proposes *what* is worth
showing (columns + claim); this module decides *how*.
"""

from dataclasses import dataclass, field
from typing import Literal

from .roles import ColumnRole
from .spec import ChartType

Intent = Literal[
    "trend",
    "distribution",
    "distribution_by_group",
    "group_comparison",
    "ranking",
    "numeric_relationship",
    "cyclic_pattern",
    "matrix_pattern",
    "two_state_comparison",
    "composition",  # contract only: no generator / renderer in stage 19
    "flow",  # contract only
]

# --- guards (one place, named) -----------------------------------------------

DENSITY_MIN_N = 30  # per group: below this a density (violin / raincloud / ridgeline) over-smooths
RAINCLOUD_MAX_GROUPS = 6
RIDGELINE_GROUPS = (3, 12)  # ordered groups only
DOT_INTERVAL_MIN_N = 5  # per group: a mean ± interval needs a few rows
LOLLIPOP_MIN_CATEGORIES = 9  # above a bar's comfortable width
RANKING_MIN_CATEGORIES = 9  # nominal groups this many or more read as a ranking
HEXBIN_MIN_N = 5000  # overplotting: hexbin beats scatter
POLAR_PERIOD_MAX = 24
POLAR_MIN_CATEGORIES = 4
SLOPEGRAPH_MAX_CATEGORIES = 8
DUMBBELL_MIN_CATEGORIES = 2


@dataclass(frozen=True)
class IntentContext:
    """The few facts selection needs besides the roles."""

    n_rows: int = 0
    n_groups: int = 0  # categories of x (or of the state column for two-state)
    n_min_group: int = 0  # rows in the smallest group
    n_categories: int = 0  # second dimension (two-state: categories per state; matrix: rows)
    distribution_evidence: bool = False  # group spreads / shapes differ, not just means
    nonlinear: bool = False
    count_measure: bool = False  # y is None: counting rows
    magnitude_measure: bool = False  # sum / count of a positive measure (rose eligible)
    ranking_requested: bool = False  # the caller explicitly wants a ranking


@dataclass(frozen=True)
class ChartChoice:
    intent: Intent
    preferred: ChartType
    alternatives: tuple[ChartType, ...] = ()
    # category order the chart must keep: "natural" (role order / value
    # ascending) or "value" (ranked) — None when the axis is not categorical
    sort: Literal["natural", "value"] | None = None
    notes: tuple[str, ...] = field(default=())


# --- intent ------------------------------------------------------------------


def classify_intent(
    x: ColumnRole | None,
    y: ColumnRole | None,
    group: ColumnRole | None,
    ctx: IntentContext,
) -> Intent | None:
    """Deterministic intent from roles + context; None when the combination
    has no chart (two numerics as x and group, text axes …)."""
    categorical = ("nominal", "ordinal", "cyclic", "boolean")
    if x is None and y is not None and y.role == "numeric" and group is None:
        return "distribution"
    if x is None or y is None:
        return None
    if y.role not in ("numeric",) and not (y.role == "ordinal" and y.order is None):
        # a measure must be numeric (integer-coded ratings count)
        return None
    if x.role == "datetime":
        return "trend"
    if x.role == "numeric":
        return "numeric_relationship"
    if x.role == "cyclic" and not ctx.count_measure:
        if group is not None and group.role in categorical and group.ordered:
            return "matrix_pattern"
        if ctx.distribution_evidence:
            return "distribution_by_group"  # "the spread differs by month": a ridge / raincloud
        return "cyclic_pattern"
    if x.role == "cyclic":
        return "cyclic_pattern"
    if x.role in categorical:
        if group is not None and group.two_state and ctx.n_groups >= DUMBBELL_MIN_CATEGORIES and not ctx.count_measure:
            return "two_state_comparison"
        if group is not None and group.role in categorical and group.ordered and x.ordered and not ctx.count_measure:
            return "matrix_pattern"
        if ctx.ranking_requested:
            return "ranking"
        if ctx.distribution_evidence and not ctx.count_measure:
            return "distribution_by_group"
        if x.role == "nominal" and ctx.n_groups >= RANKING_MIN_CATEGORIES:
            return "ranking"
        return "group_comparison"
    return None


# --- eligibility -------------------------------------------------------------


def eligible_charts(intent: Intent, x: ColumnRole | None, group: ColumnRole | None, ctx: IntentContext) -> list[ChartType]:
    """Every chart type that may show this intent under its guards, most
    preferred first. Simpler charts come first when they are as clear."""
    out: list[ChartType] = []
    dense = ctx.n_min_group >= DENSITY_MIN_N and ctx.n_rows >= DENSITY_MIN_N
    if intent == "distribution":
        out = ["histogram", "box"]
        if ctx.n_rows >= DENSITY_MIN_N:
            out.append("violin")
    elif intent == "distribution_by_group":
        ordered = x is not None and x.ordered
        few = 2 <= ctx.n_groups <= RAINCLOUD_MAX_GROUPS
        if dense and few:
            out.append("raincloud")
        if dense and ordered and RIDGELINE_GROUPS[0] <= ctx.n_groups <= RIDGELINE_GROUPS[1]:
            out.append("ridgeline")
        out.append("box")
        if dense and ctx.n_groups <= RIDGELINE_GROUPS[1]:
            out.append("violin")
    elif intent == "group_comparison":
        if ctx.n_groups >= LOLLIPOP_MIN_CATEGORIES:
            out = ["lollipop", "bar"]
        else:
            out = ["bar"]
        if not ctx.count_measure and ctx.n_min_group >= DOT_INTERVAL_MIN_N:
            out.append("dot_interval")
        if not ctx.count_measure and ctx.n_groups <= RIDGELINE_GROUPS[1]:
            out.append("box")  # the spread behind the means, always readable
    elif intent == "ranking":
        out = ["bar", "lollipop"] if ctx.n_groups < LOLLIPOP_MIN_CATEGORIES else ["lollipop", "bar"]
    elif intent == "numeric_relationship":
        out = ["hexbin", "scatter"] if ctx.n_rows > HEXBIN_MIN_N else ["scatter"]
        if ctx.n_rows > HEXBIN_MIN_N:
            pass
        elif ctx.n_rows >= DENSITY_MIN_N * 10:
            out.append("hexbin")  # readable, but scatter is clearer at this size
    elif intent == "trend":
        out = ["line"]
    elif intent == "cyclic_pattern":
        out = ["line"]
        if x is not None and x.cyclic and x.period is not None:
            if POLAR_MIN_CATEGORIES <= ctx.n_groups and x.period <= POLAR_PERIOD_MAX:
                out.append("polar_line")
            # a rose encodes a magnitude (sum / count) as sector area; never a mean
            if ctx.magnitude_measure and POLAR_MIN_CATEGORIES <= ctx.n_groups and x.period <= POLAR_PERIOD_MAX:
                out.append("rose")
        if ctx.count_measure or ctx.magnitude_measure:
            out.append("bar")
    elif intent == "two_state_comparison":
        out = ["dumbbell"]
        if ctx.n_groups <= SLOPEGRAPH_MAX_CATEGORIES:
            out.append("slopegraph")
        out.append("bar")  # grouped bar: always readable
    elif intent == "matrix_pattern":
        out = ["heatmap"]
    # composition / flow: no renderer yet → nothing eligible
    return out


def select_chart(
    intent: Intent,
    x: ColumnRole | None,
    group: ColumnRole | None,
    ctx: IntentContext,
    hint: ChartType | None = None,
) -> ChartChoice | None:
    """Preferred chart + alternatives for an intent. `hint` (an LLM's or a
    caller's suggested type) only promotes a chart that is eligible anyway —
    never one the guards reject."""
    eligible = eligible_charts(intent, x, group, ctx)
    if not eligible:
        return None
    preferred = eligible[0]
    notes: list[str] = []
    if hint is not None and hint in eligible and hint != preferred:
        preferred = hint
        notes.append(f"chart type {hint} taken from the suggestion")
    elif hint is not None and hint not in eligible:
        notes.append(f"chart type {hint} is not eligible for {intent}; {preferred} used")
    alternatives = tuple(t for t in eligible if t != preferred)
    sort: Literal["natural", "value"] | None = None
    if intent == "ranking":
        sort = "value"
    elif x is not None and x.role in ("nominal", "ordinal", "cyclic", "boolean") and preferred in (
        "bar", "lollipop", "box", "violin", "raincloud", "ridgeline", "dot_interval", "line", "polar_line", "rose", "dumbbell", "slopegraph",
    ):
        sort = "natural"
    return ChartChoice(intent=intent, preferred=preferred, alternatives=alternatives, sort=sort, notes=tuple(notes))
