"""Unified data-quality / confidence layer (stage 9).

    final score = base evidence score x sample_size x missingness x robustness

Every recommendation — evidence-scored (bar/box/line/scatter) and fixed-score
(count bar, histogram, heatmap) alike, rules and LLM alike — passes through
the same chain, so a fixed-score chart can never become "best" on a 4-row
dataset merely because the evidence charts were penalised.

Semantic validity is the existing hard gate (id/text/unknown columns are
excluded by the rule engine and banned as axes by validate_spec) and has no
factor here. "Extremely small sample -> never top" is enforced through the
existing Recommendation.tier_cap mechanism, derived from the confidence value
itself — no dataset special case and no raw-row cutoff.

Scale convention: every count inside the profile is SAMPLE-level (profiled
rows); every number under `Confidence` is FULL-TABLE (n_rows), obtained by
scaling sample rates by n_rows.
"""

import math
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel

from ..profiling.models import ColumnProfile, DatasetProfile
from .spec import ChartSpec

# --- central constants (the only numbers in this layer) ------------------------

N_FULL = 30
# Rows at which an UNGROUPED statistic earns full confidence. Classic CLT rule
# of thumb; the standard error of r at n=30 for |r|~0.45 (the scatter top
# floor: 0.5 + 0.4*0.45 = 0.68) is ~0.16, i.e. the floor is resolved at ~2.8
# SE. At and above N_FULL the factor is exactly 1.0, so healthy datasets are
# numerically untouched.

N_GROUP_FULL = 20
# Rows PER GROUP for full confidence in a group mean/median (bar, box, grouped
# line/scatter). The SE of a group mean at n=20 is 0.22 sigma; a 0.5 sigma
# separation between groups (eta~0.5, the bar/box top floor) is >2 SE apart.
# At n=5 the SE is 0.45 sigma and that separation is indistinguishable from
# noise -> factor 0.5.

TOP_CONFIDENCE_FLOOR = 0.5
# sample_size confidence below this caps the tier at "secondary" (coordinator
# decision B.1). Equivalent to n < N_FULL/4 (~7 rows) or every group below
# N_GROUP_FULL/4 (5 rows). Numerically redundant today (max base score 0.90 x
# 0.5 < TOP_SCORE_FLOOR 0.68) but stated as an explicit, tested invariant so
# the guarantee survives future changes to the score formulas.

MAX_MISSING_RATIO = 0.5
# Columns with more raw nulls than this are left out of the rule engine
# entirely (pre-stage-9 rule, kept: decision B.2) and reported through the
# dataset-level `column_excluded` warning instead of vanishing silently.

MISSING_PENALTY_WEIGHT = 0.5
# missingness = 1 - weight x (share of rows the chart cannot use). Missingness
# is a BIAS risk (the observed rows may not represent the table), not a
# precision loss — precision is what sample_size measures — so the discount
# is linear and independent of n: 700/1000 and 7/10 share this factor and
# differ on sample_size. At the MAX_MISSING_RATIO ceiling the discount is 25%
# (decision B.3).

MISSING_WARN_RATIO = 0.2  # one row in five unusable -> "warning"
MISSING_SEVERE_RATIO = 0.4  # -> "severe"

SMALL_GROUP_N = N_GROUP_FULL // 4  # = 5
# A plotted group below this cannot be estimated at all (size_confidence(5) =
# 0.5, the same point where TOP_CONFIDENCE_FLOOR bites). The row-weighted
# score factor deliberately ignores a few such groups among large ones, so a
# separate guard is needed when they are the MAJORITY of what is drawn: two
# 2-row bars next to a 996-row bar are still two unreliable bars.
SMALL_GROUP_MAJORITY = 0.5  # share of plotted groups below SMALL_GROUP_N that caps the tier
DEGENERATE_GROUP_N = 2  # a 1-row group has no within-group variation -> "severe"
BAR_DEFAULT_TOP_N = 20
# Bars draw only the top_n categories (render.DEFAULT_TOP_N; importing render
# here would be circular). Which categories survive depends on the aggregate
# values, unknown before render — the largest-by-count groups are the proxy.

CAUTION_PREFIX = " Caution: "  # reason-string suffix marker (idempotent re-application)

SENTINEL_FACTOR = 0.6
# A suspected sentinel on a scale/mean-sensitive axis: a handful of -999 moves
# a mean of 25 by tens of units and collapses a histogram into one bin. The
# damage does not scale with the count, so this is a level, not a ratio.
# 0.6 x the maximum base score (0.90) = 0.54 < TOP_SCORE_FLOOR: such a chart
# cannot be top by arithmetic alone; the explicit tier cap makes it an invariant.
SENTINEL_FACTOR_ROBUST_DISPLAY = 0.85
# The same column on a robust display (box: median/IQR; median or count
# aggregations): the statistic is unaffected, but the sentinels still appear
# as extreme points and must be named. No tier cap.
EXTREME_WEIGHT = 0.2
EXTREME_REF_RATIO = 0.10
# Legitimate extreme values: factor = 1 - EXTREME_WEIGHT x min(1, ratio /
# EXTREME_REF_RATIO). The data is real and the chart is correct; only
# readability suffers (axis compression), so the discount is gentle and only
# saturates at 0.8 when >= 10% of the rows lie beyond the far-out fences.
# Calibration (coordinator decision, stage 5): the detector's far-out ratios
# on CLEAN skewed distributions are lognormal(7, 0.6) 1.2%, exp(U(0, 6)) 2.5%,
# lognormal(4, 1) 3.3%, Poisson / lognormal mixtures 5-6% — legitimate heavy
# tails, and the spec says extreme values are never treated as errors. The
# strong signal is a suspected sentinel, not an extreme; so a clean
# heavy-tailed column (factor >= 0.88 at 6%) must not push itself out of the
# top tier, while the extreme_value warning still discloses it. Never a tier
# cap, never applied to box charts (showing outliers is their purpose).
EXTREME_WARN_RATIO = 0.01  # below one row in a hundred the note is "info", above "warning"
HEATMAP_SENTINEL_CAP_SHARE = 0.5
# A heatmap is prorated: factor = 1 - (1 - SENTINEL_FACTOR) x k/m over its m
# numeric columns. One bad column must not sink a 10-column heatmap but must
# be named; the tier is capped only when at least half the columns are hit.

# aggregations whose value a single sentinel can move (raw/None counts as sensitive)
SENSITIVE_AGGREGATIONS = frozenset({"mean", "sum", "min", "max"})
ROBUST_AGGREGATIONS = frozenset({"median", "count"})

Severity = Literal["info", "warning", "severe"]
NSource = Literal["exact", "estimated"]
TierCap = Literal["secondary", "exploratory"]


class Warning(BaseModel):
    code: str  # small_sample | small_groups | missing_data | invalid_values | column_excluded
    severity: Severity
    message: str
    meta: dict[str, Any] = {}


class Confidence(BaseModel):
    sample_size: float
    missingness: float
    robustness: float = 1.0  # stage 5
    overall: float
    n_total: int  # full-table rows (profile.n_rows)
    n_effective: int  # rows the chart is actually built from, full-table scale
    missing_ratio: float  # 1 - n_effective / n_total
    min_group_n: int | None = None  # smallest plotted group (grouped charts only)
    # "exact" only when every count came from the profile's quality/evidence
    # fields; any fallback to the independence estimate marks "estimated"
    n_source: NSource


class Assessment(NamedTuple):
    confidence: Confidence
    warnings: list[Warning]
    tier_cap: TierCap | None


# --- the two confidence curves ---------------------------------------------------


def size_confidence(n: int, n_full: int) -> float:
    """Ratio of achieved precision to the precision at n_full, capped at 1:
    the standard error of a mean or correlation shrinks as 1/sqrt(n)."""
    if n <= 0:
        return 0.0
    return min(1.0, math.sqrt(n / n_full))


def group_confidence(group_counts: dict[str, int]) -> float:
    """Row-weighted mean of the per-group curve: the share of the plotted
    mass that is well estimated. One small tail category among large ones
    barely moves it; all-tiny groups sink it. Empty -> 1 (nothing to judge)."""
    total = sum(group_counts.values())
    if total <= 0:
        return 1.0
    return sum(n * size_confidence(n, N_GROUP_FULL) for n in group_counts.values()) / total


# --- effective row counts per chart type ------------------------------------------


class _Counts(NamedTuple):
    rate: float  # share of profiled rows the chart can use (sample-level)
    groups: dict[str, int] | None  # sample-level per-group counts, or None
    exact: bool


class _View:
    """Profile lookups with the sample->full-table scale factor."""

    def __init__(self, profile: DatasetProfile) -> None:
        self.profile = profile
        self.cols = {c.name: c for c in profile.columns}
        self.n_rows = profile.n_rows
        profiled = getattr(profile, "profiled_rows", 0) or 0
        # pre-v3 payloads carry no profiled_rows; they are unsampled in practice
        self.profiled_rows = profiled if profiled > 0 else profile.n_rows
        self.scale = self.n_rows / self.profiled_rows if self.profiled_rows else 1.0
        evidence = profile.evidence
        self.cat_num = {(e.cat, e.num): e for e in evidence.cat_num}
        self.time = {(t.datetime_col, t.num): t for t in evidence.time_effects}

    def col(self, name: str | None) -> ColumnProfile | None:
        return self.cols.get(name) if name is not None else None

    def rate(self, name: str | None) -> tuple[float, bool]:
        """(share of profiled rows with a usable value, exact?). Falls back to
        raw nulls when the quality block is absent (cast failures invisible)."""
        col = self.col(name)
        if col is None:
            return 1.0, True
        quality = getattr(col, "quality", None)
        if quality is not None and getattr(quality, "profiled_rows", 0) > 0:
            return float(quality.valid_ratio), True
        return 1.0 - col.missing_ratio, False

    def pair_count(self, a: str | None, b: str | None) -> int | None:
        corr = self.profile.correlations
        counts = getattr(corr, "pair_counts", None) if corr is not None else None
        if counts is None or a not in corr.columns or b not in corr.columns:
            return None
        return int(counts[corr.columns.index(a)][corr.columns.index(b)])

    def category_counts(self, name: str | None) -> tuple[dict[str, int], bool] | None:
        """Per-category rows of a categorical column from top_values; exact
        when every category is listed, otherwise the unlisted tail is spread
        evenly over synthetic '?' keys (never shown by name)."""
        col = self.col(name)
        if col is None or not col.top_values:
            return None
        counts = {str(t.value): int(t.count) for t in col.top_values}
        k = col.n_categories or len(counts)
        if k <= len(counts):
            return counts, True
        rate, _ = self.rate(name)
        valid = int(round(rate * self.profiled_rows))
        tail_k = k - len(counts)
        rest = max(valid - sum(counts.values()), 0)
        for i in range(tail_k):
            counts[f"?{i}"] = rest // tail_k
        return counts, False


def _scaled_groups(groups: dict[str, int] | None, factor: float) -> dict[str, int] | None:
    if groups is None:
        return None
    return {k: max(int(round(v * factor)), 0) for k, v in groups.items()}


def _with_extra_columns(view: _View, counts: _Counts, extra: list[str | None]) -> _Counts:
    """Multiply by the observed rate of columns the exact count did not cover
    (independence assumption -> estimated unless the column is complete)."""
    rate, groups, exact = counts
    for name in extra:
        if name is None:
            continue
        r, r_exact = view.rate(name)
        rate *= r
        groups = _scaled_groups(groups, r)
        exact = exact and r_exact and r >= 1.0
    return _Counts(rate, groups, exact)


def _single_column(view: _View, name: str | None) -> _Counts:
    rate, exact = view.rate(name)
    return _Counts(rate, None, exact)


def _cat_num_counts(view: _View, cat: str | None, num: str | None) -> _Counts:
    effect = view.cat_num.get((cat or "", num or ""))
    n_total = getattr(effect, "n_total", 0) if effect is not None else 0
    groups = getattr(effect, "group_counts", None) if effect is not None else None
    if n_total > 0 and groups:
        return _Counts(n_total / view.profiled_rows, dict(groups), True)
    # independence estimate: category counts x observed rate of num
    cat_rate, cat_exact = view.rate(cat)
    num_rate, _ = view.rate(num)
    cats = view.category_counts(cat)
    groups = _scaled_groups(cats[0], num_rate) if cats else None
    return _Counts(cat_rate * num_rate, groups, False)


def _pair_counts(view: _View, a: str | None, b: str | None) -> _Counts:
    n = view.pair_count(a, b)
    if n is not None:
        return _Counts(n / view.profiled_rows, None, True)
    ra, _ = view.rate(a)
    rb, _ = view.rate(b)
    return _Counts(ra * rb, None, False)


def _grouped(view: _View, counts: _Counts, group: str | None, x: str | None, num: str | None) -> _Counts:
    """Attach per-group counts of `group` to an (x, y) pair count.

    The overall rate gains the group column's own rate (rows with a null
    group are not drawn). The per-group split, when it comes from the
    evidence table, is already (group, y)-complete and must NOT be rescaled
    by the group's rate again (stage 2 verification D1); it only ignores
    x-missingness, so it is scaled by x's rate and stays exact only when x is
    complete."""
    if group is None:
        return counts
    g_rate, g_exact = view.rate(group)
    rate = counts.rate * g_rate
    exact = counts.exact and g_exact and g_rate >= 1.0
    effect = view.cat_num.get((group, num or ""))
    groups = getattr(effect, "group_counts", None) if effect is not None else None
    if groups:
        x_rate, x_exact = view.rate(x)
        groups = _scaled_groups(dict(groups), x_rate)
        exact = exact and x_exact and x_rate >= 1.0
    else:
        cats = view.category_counts(group)
        groups = _scaled_groups(cats[0], counts.rate) if cats else None
        exact = False
    return _Counts(rate, groups, exact)


def _heatmap_counts(view: _View) -> _Counts:
    corr = view.profile.correlations
    counts = getattr(corr, "pair_counts", None) if corr is not None else None
    if counts is not None and len(corr.columns) >= 2:
        n = min(counts[i][j] for i in range(len(counts)) for j in range(len(counts)) if i != j)
        return _Counts(n / view.profiled_rows, None, True)
    numeric = [c.name for c in view.profile.columns if c.semantic_type == "numeric"]
    rates = [view.rate(name)[0] for name in numeric] or [1.0]
    return _Counts(min(rates), None, False)


def effective_counts(spec: ChartSpec, view: _View) -> _Counts:
    """Sample-level usable-row rate and per-group counts for a spec.

    histogram: valid rows of x (+group)        bar count: valid rows of x, groups = categories
    bar agg / box: pairwise (x, y) + groups     scatter: pair_counts[x][y] (+group)
    line: TimeEffect / pair_counts (+group)     heatmap: min off-diagonal pair count
    """
    x, y, group = spec.x, spec.y, spec.group_by
    if spec.type == "histogram":
        return _with_extra_columns(view, _single_column(view, x), [group])
    if spec.type == "bar" and y is None:
        cats = view.category_counts(x)
        rate, exact = view.rate(x)
        counts = _Counts(rate, cats[0] if cats else None, exact and (cats[1] if cats else True))
        return _with_extra_columns(view, counts, [group])
    if spec.type in ("bar", "box"):
        return _with_extra_columns(view, _cat_num_counts(view, x, y), [group])
    if spec.type == "scatter":
        return _grouped(view, _pair_counts(view, x, y), group, x, y)
    if spec.type == "line":
        effect = view.time.get((x or "", y or ""))
        n_total = getattr(effect, "n_total", 0) if effect is not None else 0
        if n_total > 0:
            counts = _Counts(n_total / view.profiled_rows, None, True)
        else:
            counts = _pair_counts(view, x, y)  # numeric x, or outside the time scan
        return _grouped(view, counts, group, x, y)
    return _heatmap_counts(view)


def plotted_groups(spec: ChartSpec, groups: dict[str, int]) -> dict[str, int]:
    """Groups that actually appear on the chart: bars are truncated to top_n
    (largest-by-count proxy, see BAR_DEFAULT_TOP_N); every other type draws
    all groups."""
    if spec.type != "bar":
        return groups
    top_n = spec.top_n or BAR_DEFAULT_TOP_N
    if len(groups) <= top_n:
        return groups
    ranked = sorted(groups.items(), key=lambda kv: (-kv[1], kv[0]))
    return dict(ranked[:top_n])


def small_group_majority(groups: dict[str, int]) -> tuple[bool, bool]:
    """(majority of groups below SMALL_GROUP_N, majority below DEGENERATE_GROUP_N)."""
    if not groups:
        return False, False
    total = len(groups)
    tiny = sum(1 for n in groups.values() if n < SMALL_GROUP_N)
    degenerate = sum(1 for n in groups.values() if n < DEGENERATE_GROUP_N)
    return tiny / total > SMALL_GROUP_MAJORITY, degenerate / total > SMALL_GROUP_MAJORITY


# --- assessment -----------------------------------------------------------------


def assess(spec: ChartSpec, profile: DatasetProfile) -> Assessment:
    view = _View(profile)
    rate, groups, exact = effective_counts(spec, view)
    rate = min(max(rate, 0.0), 1.0)
    n_total = view.n_rows
    n_effective = int(round(rate * n_total))
    full_groups = _scaled_groups(groups, view.scale)
    n_source: NSource = "exact" if exact else "estimated"

    c_overall = size_confidence(n_effective, N_FULL)
    drawn = plotted_groups(spec, full_groups) if full_groups else None
    c_group = group_confidence(drawn) if drawn else 1.0
    sample_size = min(c_overall, c_group)
    majority_tiny, majority_degenerate = small_group_majority(drawn or {})

    missing_ratio = max(0.0, 1.0 - rate)
    missingness = 1.0 - MISSING_PENALTY_WEIGHT * missing_ratio
    robustness, robustness_warnings, robustness_cap = _robustness(spec, view)

    warnings = _sample_warnings(
        spec, view, n_effective, drawn, sample_size, c_overall, majority_tiny, majority_degenerate, n_source
    )
    warnings += _missing_warnings(spec, view, n_effective, n_total, missing_ratio, n_source)
    warnings += robustness_warnings
    # the guards cap the tier without touching the score factors
    capped = sample_size < TOP_CONFIDENCE_FLOOR or majority_tiny or robustness_cap
    tier_cap: TierCap | None = "secondary" if capped else None

    confidence = Confidence(
        sample_size=sample_size,
        missingness=missingness,
        robustness=robustness,
        overall=sample_size * missingness * robustness,
        n_total=n_total,
        n_effective=n_effective,
        missing_ratio=missing_ratio,
        min_group_n=min(full_groups.values()) if full_groups else None,
        n_source=n_source,
    )
    return Assessment(confidence, warnings, tier_cap)


def _group_label(spec: ChartSpec) -> str:
    if spec.group_by is not None:
        return spec.group_by
    return spec.x or ""


def _sample_warnings(
    spec: ChartSpec,
    view: _View,
    n_effective: int,
    groups: dict[str, int] | None,
    sample_size: float,
    c_overall: float,
    majority_tiny: bool,
    majority_degenerate: bool,
    n_source: NSource,
) -> list[Warning]:
    warnings: list[Warning] = []
    severe = sample_size < TOP_CONFIDENCE_FLOOR
    label = _group_label(spec)
    named = {k: v for k, v in (groups or {}).items() if not k.startswith("?")}
    if c_overall >= 1.0 and majority_tiny and groups:
        # J1 guard: most of what is drawn rests on fewer than SMALL_GROUP_N rows
        tiny = sum(1 for n in groups.values() if n < SMALL_GROUP_N)
        warnings.append(
            Warning(
                code="small_groups",
                severity="severe" if majority_degenerate else "warning",
                message=(
                    f"{tiny} of {len(groups)} {label} groups have fewer than {SMALL_GROUP_N} rows; "
                    "most per-group values are unreliable."
                ),
                meta={
                    "group_sizes": {k: v for k, v in named.items() if v < SMALL_GROUP_N} or None,
                    "small_groups": tiny,
                    "plotted_groups": len(groups),
                    "min_group_n": min(groups.values()),
                    "n_source": n_source,
                },
            )
        )
    elif c_overall < 1.0:
        detail = ""
        if groups:
            lo, hi = min(groups.values()), max(groups.values())
            per = f"{lo}" if lo == hi else f"{lo} to {hi}"
            detail = f" ({per} per {label} group)"
        warnings.append(
            Warning(
                code="small_sample",
                severity="severe" if severe else "info",
                message=f"Only {n_effective} rows{detail}; differences may be noise.",
                meta={
                    "n_effective": n_effective,
                    "n_total": view.n_rows,
                    "group_sizes": named or None,
                    "n_source": n_source,
                },
            )
        )
    elif groups and min(groups.values()) < N_GROUP_FULL:
        small = {k: v for k, v in named.items() if v < N_GROUP_FULL}
        unlisted = sum(1 for k, v in groups.items() if k.startswith("?") and v < N_GROUP_FULL)
        parts = [f"{k} ({v} rows)" for k, v in sorted(small.items(), key=lambda kv: (kv[1], kv[0]))]
        if unlisted:
            parts.append(f"{unlisted} unlisted {label} categor{'y' if unlisted == 1 else 'ies'}")
        warnings.append(
            Warning(
                code="small_groups",
                severity="severe" if severe else "warning",
                message=(
                    f"{', '.join(parts)} fall below {N_GROUP_FULL} rows; "
                    "their values may be noise."
                ),
                meta={"group_sizes": small or None, "min_group_n": min(groups.values()), "n_source": n_source},
            )
        )
    return warnings


def _missing_warnings(
    spec: ChartSpec,
    view: _View,
    n_effective: int,
    n_total: int,
    missing_ratio: float,
    n_source: NSource,
) -> list[Warning]:
    warnings: list[Warning] = []
    axes = [name for name in (spec.x, spec.y, spec.group_by) if name is not None]
    if spec.type == "heatmap":
        axes = [c.name for c in view.profile.columns if c.semantic_type == "numeric"]
    incomplete = {name: 1.0 - view.rate(name)[0] for name in axes if view.rate(name)[0] < 1.0}
    if missing_ratio >= MISSING_WARN_RATIO:
        names = " or ".join(sorted(incomplete, key=lambda n: -incomplete[n])) or "the plotted columns"
        warnings.append(
            Warning(
                code="missing_data",
                severity="severe" if missing_ratio >= MISSING_SEVERE_RATIO else "warning",
                message=(
                    f"{missing_ratio:.0%} of rows lack {names}; "
                    f"the chart uses {n_effective} of {n_total} rows."
                ),
                meta={
                    "n_effective": n_effective,
                    "n_total": n_total,
                    "missing_ratio": missing_ratio,
                    "columns": {k: round(v, 4) for k, v in incomplete.items()},
                    "n_source": n_source,
                },
            )
        )
    for name in axes:
        col = view.col(name)
        quality = getattr(col, "quality", None) if col is not None else None
        invalid = int(getattr(quality, "invalid_count", 0) or 0) if quality is not None else 0
        if invalid > 0:
            count = int(round(invalid * view.scale))
            kind = "dates" if col.semantic_type == "datetime" else "numbers"
            warnings.append(
                Warning(
                    code="invalid_values",
                    severity="warning",
                    message=f"{count} values in {name} could not be read as {kind} and are excluded.",
                    meta={"column": name, "invalid_count": count, "n_source": n_source},
                )
            )
    return warnings


# --- robustness (stage 5): suspected sentinels and extreme values ----------------


def has_suspected_sentinels(col: ColumnProfile | None) -> bool:
    """Single definition used by ranking (skew bonus), render (clipped
    histogram) and this layer; getattr-defensive for pre-stage-5 profiles."""
    quality = getattr(col, "quality", None) if col is not None else None
    return bool(getattr(quality, "suspected_sentinels", None))


def _format_value(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def sentinel_summary(col: ColumnProfile, scale: float = 1.0) -> str:
    """'9999 x11, -999 x8' — distinct values with counts, largest first."""
    candidates = sorted(col.quality.suspected_sentinels, key=lambda s: (-s.count, s.value))
    return ", ".join(f"{_format_value(s.value)} x{int(round(s.count * scale))}" for s in candidates)


def column_robustness(col: ColumnProfile | None, sensitive: bool) -> tuple[float, str | None]:
    """(factor, kind) for one numeric axis: 'sentinel', 'extreme' or None."""
    quality = getattr(col, "quality", None) if col is not None else None
    if quality is None:
        return 1.0, None
    if has_suspected_sentinels(col):
        return (SENTINEL_FACTOR if sensitive else SENTINEL_FACTOR_ROBUST_DISPLAY), "sentinel"
    ratio = float(getattr(quality, "extreme_value_ratio", 0.0) or 0.0)
    if ratio > 0 and sensitive:
        return 1.0 - EXTREME_WEIGHT * min(1.0, ratio / EXTREME_REF_RATIO), "extreme"
    return 1.0, None


def _robust_axes(spec: ChartSpec, view: _View) -> list[tuple[str, bool]]:
    """Numeric axes of a spec as (column, sensitive?). Box is a robust display
    of y; bar/line depend on the aggregation; scatter/histogram axes are
    always sensitive (their range is what is drawn)."""
    if spec.type == "histogram":
        return [(spec.x, True)] if spec.x else []
    if spec.type == "scatter":
        return [(name, True) for name in (spec.x, spec.y) if name]
    if spec.type == "box":
        return [(spec.y, False)] if spec.y else []
    if spec.type in ("bar", "line"):
        if spec.y is None:
            return []
        sensitive = spec.aggregation not in ROBUST_AGGREGATIONS  # None (raw) is sensitive
        return [(spec.y, sensitive)]
    return []


def _robustness(spec: ChartSpec, view: _View) -> tuple[float, list[Warning], bool]:
    """Factor, warnings and whether the tier must be capped."""
    warnings: list[Warning] = []
    if spec.type == "heatmap":
        return _heatmap_robustness(view)
    factor, cap = 1.0, False
    for name, sensitive in _robust_axes(spec, view):
        col = view.col(name)
        col_factor, kind = column_robustness(col, sensitive)
        factor = min(factor, col_factor)
        if kind == "sentinel":
            cap = cap or sensitive
            summary = sentinel_summary(col, view.scale)
            rows = int(round(int(getattr(col.quality, "sentinel_row_count", 0) or 0) * view.scale))
            if sensitive:
                message = (
                    f"{name} contains suspected sentinel values ({summary}) that distort "
                    "its mean and range; treat the extremes with care."
                )
            else:
                message = (
                    f"{name} contains suspected sentinel values ({summary}); this display "
                    "is robust to them, but they appear as extreme points."
                )
            warnings.append(
                Warning(
                    code="suspected_sentinel",
                    severity="severe" if sensitive else "warning",
                    message=message,
                    meta={
                        "column": name,
                        "values": [s.model_dump() for s in col.quality.suspected_sentinels],
                        "sentinel_rows": rows,
                        "sensitive": sensitive,
                    },
                )
            )
        elif kind == "extreme":
            ratio = float(col.quality.extreme_value_ratio)
            count = int(round(int(col.quality.extreme_value_count) * view.scale))
            warnings.append(
                Warning(
                    code="extreme_value",
                    severity="warning" if ratio >= EXTREME_WARN_RATIO else "info",
                    message=(
                        f"{name} has {count} extreme values ({ratio:.1%}) far outside its "
                        "typical range; the axis may be compressed."
                    ),
                    meta={"column": name, "extreme_count": count, "extreme_ratio": round(ratio, 4)},
                )
            )
    return factor, warnings, cap


def _heatmap_robustness(view: _View) -> tuple[float, list[Warning], bool]:
    corr = view.profile.correlations
    names = list(corr.columns) if corr is not None else [
        c.name for c in view.profile.columns if c.semantic_type == "numeric"
    ]
    if not names:
        return 1.0, [], False
    hit = [name for name in names if has_suspected_sentinels(view.col(name))]
    if not hit:
        return 1.0, [], False
    share = len(hit) / len(names)
    factor = 1.0 - (1.0 - SENTINEL_FACTOR) * share
    cap = share >= HEATMAP_SENTINEL_CAP_SHARE
    warning = Warning(
        code="suspected_sentinel",
        severity="severe" if cap else "warning",
        message=(
            f"{len(hit)} of {len(names)} numeric columns ({', '.join(hit)}) contain suspected "
            "sentinel values; their correlations may be distorted."
        ),
        meta={"columns": hit, "share": round(share, 4)},
    )
    return factor, [warning], cap


# --- helpers used by the rule engine and the LLM merge ---------------------------


def with_caution(reason: str, warnings: list[Warning]) -> str:
    """Appends ' Caution: <messages>' for warnings of severity >= warning.
    Idempotent: an earlier caution suffix is replaced, never duplicated."""
    base = reason.split(CAUTION_PREFIX, 1)[0] if CAUTION_PREFIX in reason else reason
    messages = [w.message for w in warnings if w.severity != "info"]
    if not messages:
        return base
    return (base.rstrip() + CAUTION_PREFIX + " ".join(messages)).strip()


def excluded_column_warnings(profile: DatasetProfile) -> list[Warning]:
    """Dataset-level: one `column_excluded` per column the rule engine drops
    for exceeding MAX_MISSING_RATIO (decision B.2). id/text/unknown columns
    are typing exclusions, not quality ones, and are not listed."""
    return [
        Warning(
            code="column_excluded",
            severity="warning",
            message=(
                f"{c.name} was left out of the recommendations: "
                f"{c.missing_ratio:.0%} of its values are missing."
            ),
            meta={"column": c.name, "missing_ratio": round(c.missing_ratio, 4)},
        )
        for c in profile.columns
        if c.semantic_type not in ("id", "unknown", "text") and c.missing_ratio > MAX_MISSING_RATIO
    ]
