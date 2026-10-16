"""Effect-size evidence for recommendation ranking (stage 7).

Everything is computed on the profiler's seeded <=100k sample and follows the
NaN/inf -> null convention. Scan sizes are bounded by the caps below, so the
whole table stays in the millisecond-to-sub-second range.
"""

import math
import re
from datetime import timedelta
from itertools import combinations
from typing import NamedTuple

import polars as pl

from .models import (
    CatNumEffect,
    ColumnProfile,
    CorrelationTest,
    Correlations,
    DerivedColumn,
    DerivedKind,
    DispersionEffect,
    Evidence,
    FdrSummary,
    InteractionEffect,
    NearDuplicateGroup,
    SlopeHet,
    TimeBucket,
    TimeEffect,
)
from .pairwise import pairwise_pearson

MAX_CAT_CATEGORIES = 20
MAX_CAT_COLUMNS = 15  # above this, keep the lowest-cardinality ones (critique #9)
MIN_CELL_COUNT = 5  # interaction cells with fewer rows are dropped (critique #6)
MIN_VALID_CELLS = 4
MIN_SLOPE_GROUP_ROWS = 30  # per-group floor for group correlations (critique #4)
SLOPE_GROUP_RANGE = (2, 8)
TOP_CATS_FOR_INTERACTIONS = 3
TOP_NUMS_FOR_INTERACTIONS = 5
TOP_PAIRS_FOR_SLOPES = 5
TOP_CATS_FOR_SLOPES = 3
MIN_ABS_CORR = 0.3

# --- derived columns (stage 13) ---
DERIVED_MAX_ROWS = 5000  # identity checks run on a seeded subsample of this size
DERIVED_SAMPLE_SEED = 13
DERIVED_MAX_COLUMNS = 30  # candidate columns considered (matches MAX_CORRELATION_COLUMNS)
DERIVED_FULL_POOL_COLUMNS = 12  # up to this many candidates every pair/triple is tested
DERIVED_POOL = 8  # above that, each target is tested against its 8 most correlated peers
DERIVED_MIN_ROWS = 30
DERIVED_MATCH_RATIO = 0.99
DERIVED_MIN_TARGET_UNIQUE = 10  # near-constant targets match anything within tolerance
# a target is only searched when some peer correlates with it at least this
# much (Pearson or Spearman): every identity form co-moves with at least one
# of its components (a − b: 0.71 for independent equal-variance inputs), and
# the guard keeps wide uncorrelated tables (63 sensors) out of the search
DERIVED_MIN_PEER_CORR = 0.3
DERIVED_MAX_DECIMALS = 6
# a transformed duplicate (unit conversion, scaling) ranks identically:
# |rho| = 1.000 up to rounding. 0.995 leaves genuinely noisy relationships
# alone: y = 2x + N(0, 1) with x ~ U(0, 10) measures r = 0.985 and is a
# finding, not a copy (calibration point: test_confidence healthy dataset)
NEAR_COPY_MIN_RHO = 0.995
# stage 14: a slightly weaker rank correlation still marks a duplicate when
# the column NAMES corroborate it (temp / atemp: rho 0.989; price / price_usd).
# Correlation alone cannot separate that case from y = 2x + N(0, 1) at
# r = 0.985, so the name signal is required below NEAR_COPY_MIN_RHO.
NEAR_DUP_NAMED_MIN_RHO = 0.98
_NUMERIC_DTYPE_RE = re.compile(r"^(U?Int|Float)\d+$")
_NAME_SPLIT_RE = re.compile(r"[^a-z]+")
_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
_NAME_MIN_TOKEN = 3  # shared token length that counts as related
_NAME_MIN_SUBSTRING = 4  # one normalised name inside the other

# --- dispersion (stage 19.1) ---
# A spread difference is evidence on its own, whatever the means do. The
# Brown–Forsythe form (deviations from the group MEDIAN, robust to skew and
# outliers) is scored as an adjusted eta-squared so it is df-corrected like
# the mean-difference evidence; the robust spread ratio (IQR, else MAD, else
# std) must agree so a single heavy tail in one group cannot flag a pair.
DISPERSION_MIN_ETA = 0.02  # adjusted eta-squared of |x − group median| over the compared groups
DISPERSION_MIN_RATIO = 1.5  # widest / narrowest per-group robust spread
# Stage 19.2 sample-size semantics: a group is SUPPORTED with >= STRONG_MIN_N
# rows and takes part in a strong comparison; a group below WEAK_MIN_N never
# takes part at all. With >= 2 supported groups the comparison runs over
# exactly those (smaller groups are excluded and listed, they no longer
# cancel the evidence of the well-supported groups); otherwise every group
# >= WEAK_MIN_N is compared and the entry can only be weak. 30 is the same
# per-group floor the density charts (raincloud / ridgeline) need.
DISPERSION_STRONG_MIN_N = 30
DISPERSION_WEAK_MIN_N = 10
# Stage 19.2 calibration: the effect-size gates alone (eta >= 0.02 ~ F 2.2
# at n=60, p ~ 0.14) let ~8-12 % of equal-spread pairs through at 30 rows
# per group. A strong entry now also needs the Brown–Forsythe F test
# (deviations from the group median, Brown & Forsythe 1974) to reject at the
# conventional alpha = 0.05 — the null strong rate is bounded by alpha by
# construction and the ratio gate keeps it lower (measured ~1-3 % at N=30,
# tests/test_dispersion_calibration.py), while sd 1 vs 2 at 100 rows per
# group is still found in > 90 % of replicates.
DISPERSION_ALPHA = 0.05
# A weak entry (LLM-only, labelled weak) needs the same test at a looser
# alpha so an equal-spread pair of 10-29-row groups is not listed 10-15 % of
# the time on the effect gates alone (checkpoint B, concern 1).
DISPERSION_WEAK_ALPHA = 0.10
# Stage 19.2 bucket detection (`band = bucket(num)`, see _is_bucket): the
# per-group ranges must be disjoint in one consistent order, every group
# must hold >= DISPERSION_WEAK_MIN_N rows, every boundary gap must be small
# against the neighbours' mean spacing — gap <= max(BUCKET_GAP_FACTOR ×
# width / n of either neighbour, BUCKET_RESOLUTION_SLACK × the column's
# smallest step: an integer partition has a gap of exactly one step) — and
# at least one side of the boundary must be dense (the BUCKET_EDGE_SHARE
# slice of the group's width adjacent to the boundary holds >=
# BUCKET_EDGE_MIN_ROWS rows and >= BUCKET_EDGE_MIN_SHARE of the group).
# Independent clusters have sparse tails on both sides of a gap and wide
# gaps against their spacing, so they are not flagged; a partition of a
# continuous column has a sharp, dense edge on at least one side of every
# cut. Calibration (seeded scratch simulation, 150-200 replicates; the
# regression tests in test_dispersion.py cover the clear cases): true
# partitions of uniform / lognormal / integer columns are flagged in 72-100 %
# of replicates at 25-33 rows per bin and >= 85 % at 100 per bin; independent
# Gaussian, exponential and integer clusters at any separation are flagged
# in <= 2 % of replicates at 30 rows per group and 0 % at >= 100 rows (only
# chance disjointness with dense tails passes). A gap factor of 10 would
# raise the cluster rate to ~7 %, an edge share of 0.12 would halve recall.
BUCKET_GAP_FACTOR = 5.0
BUCKET_RESOLUTION_SLACK = 1.5
BUCKET_EDGE_SHARE = 0.1
BUCKET_EDGE_MIN_SHARE = 0.08
BUCKET_EDGE_MIN_ROWS = 2
# stage 21: false-discovery control. Every L1 test with a p-value joins one
# Benjamini–Hochberg family per dataset at FDR_Q; the rules only let a chart
# whose supporting test is a discovery into the top tier. Measured on 1 200
# seeded noise datasets (audit phase B): n = 30 × 20 columns produced a top
# chart in 84 % of them by multiplicity alone, 6 % with BH.
FDR_Q = 0.05
# Bartlett effective n for time-ordered correlations is floored here so the
# t-test keeps at least one degree of freedom
MIN_EFFECTIVE_N = 3.0

_BUCKET_TRUNC: dict[TimeBucket, str] = {"day": "1d", "month": "1mo", "year": "1y"}
_COARSER: dict[TimeBucket, TimeBucket | None] = {"day": "month", "month": "year", "year": None}


def compute_evidence(
    df: pl.DataFrame, columns: list[ColumnProfile], correlations: Correlations | None
) -> Evidence:
    cats = _eligible_cats(columns)
    nums = [c for c in columns if c.semantic_type == "numeric"]
    dts = [c for c in columns if c.semantic_type == "datetime"]

    cat_num = _cat_num_effects(df, cats, nums)
    best_eta = _best_eta_by_column(cat_num)
    spearman = _spearman_matrix(df, correlations)
    derived = derived_columns(df, columns, correlations, spearman)
    near_dups = near_duplicate_groups(spearman, derived, columns)
    # stage 19.2: the dispersion pass also detects `cat = bucket(num)`
    # bandings (a DerivedColumn of kind bucket, appended to the identities so
    # the derived caps, layer 2 and the probes treat them like any identity)
    # and flags entries that restate an identity / near copy / banding
    dispersion, buckets = _dispersion_effects(df, cats, nums, derived)
    evidence = Evidence(
        cat_num=cat_num,
        time_effects=_time_effects(df, dts, nums),
        num_num_spearman=spearman,
        interactions=_interactions(df, cats, nums, dts, correlations, best_eta),
        slope_heterogeneity=_slope_heterogeneity(df, correlations, cats, best_eta),
        derived_columns=derived + buckets,
        near_duplicate_groups=near_dups,
        dispersion=dispersion,
        correlation_tests=_correlation_tests(df, correlations, dts, spearman),
    )
    apply_fdr(evidence)
    return evidence


class EtaResult(NamedTuple):
    eta_squared: float
    n_groups: int
    n_total: int
    n_min: int
    group_counts: dict[str, int]  # str(group value) -> valid rows; only non-empty groups
    p_value: float = 1.0  # stage 21: one-way ANOVA F-test, df (k - 1, n - k)


def adjusted_eta_squared(df: pl.DataFrame, cat: str, num: str) -> tuple[float, int] | None:
    """1 - MS_within/MS_total, clipped at 0. Raw eta-squared is inflated on
    high-cardinality small samples (blocking #1), the df-corrected form is not.
    Returns None when undefined (constant num, <2 groups, no within-group df).
    Single-pair convenience wrapper over adjusted_eta_squared_many."""
    result = adjusted_eta_squared_many(df, cat, [num])[num]
    return None if result is None else (result.eta_squared, result.n_groups)


def adjusted_eta_squared_many(
    df: pl.DataFrame, cat: str, nums: list[str]
) -> dict[str, EtaResult | None]:
    """adjusted_eta_squared for every num against one cat in two polars
    executions instead of one per pair (stage 9 #1: 180 pairs cost 3.9s).

    Row set per num is unchanged: cat non-null, num non-null and finite —
    non-finite floats are masked to null so mean/sum/count skip them exactly
    like the former drop_nulls + is_finite filter. SS_total stays two-pass
    (grand mean first, then squared deviations) so the arithmetic matches the
    single-pair form to floating tolerance; maintain_order keeps the group
    summation order deterministic."""
    if not nums:
        return {}
    exprs = []
    for num in nums:
        col = pl.col(num)
        if df.schema[num].is_float():
            col = pl.when(col.is_finite()).then(col).otherwise(None)
        exprs.append(col.alias(num))
    data = df.filter(pl.col(cat).is_not_null()).select(pl.col(cat), *exprs)
    grand = data.select(pl.col(num).mean().alias(num) for num in nums).row(0, named=True)
    aggs = []
    for num in nums:
        mean = grand[num]
        aggs.append(pl.col(num).count().alias(f"n:{num}"))
        aggs.append(((pl.col(num) - pl.col(num).mean()) ** 2).sum().alias(f"ssw:{num}"))
        aggs.append(((pl.col(num) - pl.lit(mean)) ** 2).sum().alias(f"sst:{num}"))
    groups = data.group_by(cat, maintain_order=True).agg(aggs)
    keys = [str(key) for key in groups[cat].to_list()]  # same labelling as render/SlopeHet

    results: dict[str, EtaResult | None] = {}
    for num in nums:
        counts = groups[f"n:{num}"]
        non_empty = counts.filter(counts > 0)
        k = len(non_empty)
        n = int(counts.sum())
        if grand[num] is None or k < 2 or n - k < 2:
            results[num] = None
            continue
        ss_total = groups[f"sst:{num}"].sum()
        if ss_total is None or ss_total <= 0:
            results[num] = None
            continue
        ss_within = groups[f"ssw:{num}"].sum()
        adjusted = 1 - (ss_within / (n - k)) / (ss_total / (n - 1))
        group_counts = {key: int(c) for key, c in zip(keys, counts.to_list()) if c > 0}
        results[num] = EtaResult(
            max(float(adjusted), 0.0), k, n, int(non_empty.min()), group_counts,
            _anova_p(float(ss_total), float(ss_within), k, n),
        )
    return results


def _anova_p(ss_total: float, ss_within: float, k: int, n: int) -> float:
    """One-way ANOVA F-test p-value from the two sums of squares."""
    ss_between = max(ss_total - ss_within, 0.0)
    if ss_within <= 0:
        return 0.0 if ss_between > 0 else 1.0
    f = (ss_between / (k - 1)) / (ss_within / (n - k))
    return f_survival(f, k - 1, n - k)


def _eligible_cats(columns: list[ColumnProfile]) -> list[ColumnProfile]:
    cats = [
        c
        for c in columns
        if c.semantic_type in ("categorical", "boolean")
        and 2 <= (c.n_categories or 0) <= MAX_CAT_CATEGORIES
    ]
    if len(cats) > MAX_CAT_COLUMNS:
        cats = sorted(cats, key=lambda c: (c.n_categories, c.name))[:MAX_CAT_COLUMNS]
    return cats


def _cat_num_effects(
    df: pl.DataFrame, cats: list[ColumnProfile], nums: list[ColumnProfile]
) -> list[CatNumEffect]:
    effects = []
    num_names = [num.name for num in nums]
    for cat in cats:
        results = adjusted_eta_squared_many(df, cat.name, num_names)
        for num in num_names:
            result = results[num]
            if result is not None:
                effects.append(
                    CatNumEffect(
                        cat=cat.name,
                        num=num,
                        eta_squared=result.eta_squared,
                        n_groups=result.n_groups,
                        n_total=result.n_total,
                        n_min=result.n_min,
                        group_counts=result.group_counts,
                        p_value=result.p_value,
                    )
                )
    return effects


def _best_eta_by_column(effects: list[CatNumEffect]) -> dict[str, float]:
    best: dict[str, float] = {}
    for e in effects:
        best[e.cat] = max(best.get(e.cat, 0.0), e.eta_squared)
        best[e.num] = max(best.get(e.num, 0.0), e.eta_squared)
    return best


# --- dispersion (stage 19.1 / 19.2) -------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes
    3rd ed. §6.4, modified Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b) for a, b > 0 and 0 <= x <= 1 (Numerical Recipes betai)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def f_survival(f: float, d1: int, d2: int) -> float:
    """P(F(d1, d2) > f): the F-distribution survival function as the
    regularised incomplete beta I_x(d2/2, d1/2) with x = d2 / (d2 + d1·f)
    (Abramowitz & Stegun 26.6.2). Pure Python and deterministic (no scipy);
    accurate to ~1e-12 for the degrees of freedom seen here — checked
    against the tabulated 5 % critical values in test_dispersion_calibration."""
    if not math.isfinite(f):
        return 0.0
    if f <= 0.0:
        return 1.0
    x = d2 / (d2 + d1 * f)
    return min(1.0, max(0.0, regularized_incomplete_beta(d2 / 2.0, d1 / 2.0, x)))


class _GroupStat(NamedTuple):
    label: str
    n: int
    std: float | None
    iqr: float | None
    mad: float | None
    lo: float | None
    hi: float | None
    dev_sum: float | None
    dev_ssw: float | None
    top_edge: int  # rows in the BUCKET_EDGE_SHARE slice below the group's max
    bottom_edge: int  # rows in the slice above the group's min


class _BrownForsythe(NamedTuple):
    eta_squared: float  # adjusted, clipped at 0
    f: float
    p_value: float


def _brown_forsythe(groups: list[_GroupStat]) -> _BrownForsythe | None:
    """One-way ANOVA of |x − group median| from the per-group aggregates of
    the compared groups: SS_between from the group means, SS_within summed.
    Returns None when undefined (< 2 groups, no within df, no variance)."""
    k = len(groups)
    n = sum(g.n for g in groups)
    if k < 2 or n - k < 2:
        return None
    if any(g.dev_sum is None or g.dev_ssw is None for g in groups):
        return None
    grand = sum(g.dev_sum for g in groups) / n  # type: ignore[misc]
    ss_between = sum(g.n * (g.dev_sum / g.n - grand) ** 2 for g in groups)  # type: ignore[operator]
    ss_within = sum(g.dev_ssw for g in groups)  # type: ignore[misc]
    ss_total = ss_between + ss_within
    if ss_total <= 0 or not math.isfinite(ss_total):
        return None
    eta = max(1.0 - (ss_within / (n - k)) / (ss_total / (n - 1)), 0.0)
    if ss_within <= 0:
        return _BrownForsythe(float(eta), math.inf, 0.0)
    f = (ss_between / (k - 1)) / (ss_within / (n - k))
    return _BrownForsythe(float(eta), float(f), f_survival(f, k - 1, n - k))


def _is_bucket(groups: list[_GroupStat], resolution: float | None) -> bool:
    """Is the categorical a banding of the numeric (`band = bucket(num)`)?
    Rule (stage 19.2, conservative — a false flag demotes a real finding):
    >= 2 non-empty groups, every group >= DISPERSION_WEAK_MIN_N rows, the
    per-group [min, max] ranges pairwise disjoint in one consistent order,
    every boundary gap <= max(BUCKET_GAP_FACTOR × mean spacing of either
    neighbour, BUCKET_RESOLUTION_SLACK × column resolution) and dense on at
    least one side (see the BUCKET_* constants). Independent groups whose ranges are merely
    separated fail the gap and edge-density conditions."""
    if len(groups) < 2 or any(g.n < DISPERSION_WEAK_MIN_N for g in groups):
        return False
    if any(g.lo is None or g.hi is None for g in groups):
        return False
    ordered = sorted(groups, key=lambda g: (g.lo, g.hi, g.label))  # type: ignore[arg-type]
    floor = resolution if resolution is not None and resolution > 0 else 0.0
    for below, above in zip(ordered, ordered[1:]):
        gap = above.lo - below.hi  # type: ignore[operator]
        if gap <= 0:
            return False  # overlapping (or touching) ranges
        spacing = max((below.hi - below.lo) / below.n, (above.hi - above.lo) / above.n)  # type: ignore[operator]
        if gap > max(BUCKET_GAP_FACTOR * spacing, BUCKET_RESOLUTION_SLACK * floor):
            return False
        dense_below = below.top_edge >= max(BUCKET_EDGE_MIN_ROWS, BUCKET_EDGE_MIN_SHARE * below.n)
        dense_above = above.bottom_edge >= max(BUCKET_EDGE_MIN_ROWS, BUCKET_EDGE_MIN_SHARE * above.n)
        if not (dense_below or dense_above):
            return False
    return True


def _finite_col(name: str, df: pl.DataFrame) -> pl.Expr:
    col = pl.col(name)
    if df.schema[name].is_float():
        col = pl.when(col.is_finite()).then(col).otherwise(None)
    return col.alias(name)


def _dispersion_entry(
    cat: str,
    num: str,
    stats: list[_GroupStat],
    compared: list[_GroupStat],
    may_be_strong: bool,
    empty: dict[str, int],
    definition: DerivedColumn | None,
) -> DispersionEffect | None:
    """One comparison over `compared` (a subset of `stats`): None when a gate
    fails, else the entry with every other group of `stats` (and the groups
    without any value, `empty`) listed as excluded."""
    bf = _brown_forsythe(compared)
    if bf is None or bf.eta_squared < DISPERSION_MIN_ETA:
        return None
    strong = may_be_strong and bf.p_value <= DISPERSION_ALPHA
    if not strong and bf.p_value > DISPERSION_WEAK_ALPHA:
        return None
    ratios: dict[str, tuple[float, str, str] | None] = {}
    for attr in ("std", "iqr", "mad"):
        values = [(getattr(g, attr), g.label) for g in compared]
        if any(v is None or not math.isfinite(v) or v <= 0 for v, _ in values):
            ratios[attr] = None  # a constant group: no ratio
            continue
        widest, narrowest = max(values), min(values)  # ties broken by the group label
        ratios[attr] = (widest[0] / narrowest[0], widest[1], narrowest[1])
    robust = ratios["iqr"] or ratios["mad"] or ratios["std"]
    if robust is None or robust[0] < DISPERSION_MIN_RATIO:
        return None
    compared_labels = {g.label for g in compared}
    excluded = {g.label: g.n for g in stats if g.label not in compared_labels}
    excluded.update(empty)
    return DispersionEffect(
        cat=cat,
        num=num,
        n_total=sum(g.n for g in compared),
        n_groups=len(compared),
        n_min=min(g.n for g in compared),
        bf_eta_squared=round(bf.eta_squared, 6),
        bf_f=round(bf.f, 6) if math.isfinite(bf.f) else None,
        bf_p_value=round(bf.p_value, 6),
        std_ratio=round(ratios["std"][0], 6) if ratios["std"] else None,
        iqr_ratio=round(ratios["iqr"][0], 6) if ratios["iqr"] else None,
        mad_ratio=round(ratios["mad"][0], 6) if ratios["mad"] else None,
        spread_ratio=round(robust[0], 6),
        widest_group=robust[1],
        narrowest_group=robust[2],
        strength="strong" if strong else "weak",
        compared_groups=[g.label for g in compared],
        excluded_groups=dict(sorted(excluded.items())),
        definitional=definition is not None,
        definitional_kind=None if definition is None else definition.kind,
        definition=None if definition is None else f"{definition.target} = {definition.formula}",
    )


def _dispersion_effects(
    df: pl.DataFrame, cats: list[ColumnProfile], nums: list[ColumnProfile], derived: list[DerivedColumn]
) -> tuple[list[DispersionEffect], list[DerivedColumn]]:
    """Spread differences of every num across every eligible cat, decoupled
    from the mean difference: the Brown–Forsythe deviations |num − group
    median| as an adjusted eta-squared (equal spreads → 0) plus the F test
    p-value, and the widest / narrowest per-group std, IQR and MAD. Row set
    per num = cat non-null and num finite, like the layer-1 eta-squared; one
    group_by per cat with all nums batched, the per-group aggregates are
    combined in Python (<= 20 groups). Only pairs that clear the gates are
    listed (see the constants above); the comparison runs over the
    supported groups first and falls back to every group >= WEAK_MIN_N at
    most weak (DispersionEffect docstring).

    Also returns the `cat = bucket(num)` bandings found on the way (every
    pair, listed or not) as DerivedColumn(kind="bucket"); listed entries
    that restate an identity, near copy or banding carry definitional=True."""
    if not cats or not nums:
        return [], []
    names = [n.name for n in nums]
    identities: dict[frozenset[str], DerivedColumn] = {}
    copies: dict[str, set[str]] = {}  # near-copy adjacency (duplicate <-> representative)
    for d in derived:
        for component in d.components:
            identities.setdefault(frozenset((d.target, component)), d)
        if d.kind == "near_copy":
            copies.setdefault(d.target, set()).add(d.components[0])
            copies.setdefault(d.components[0], set()).add(d.target)

    def definition_of(cat: str, num: str) -> DerivedColumn | None:
        """The identity a (cat, num) pair restates: directly, or through a
        near copy — when num is (transitively) a near copy of a column that
        is definitionally related to cat, the pair is definitional too
        (checkpoint B follow-up 2: quantity -> total_k with total_k ≈ total
        = unit_price × quantity)."""
        direct = identities.get(frozenset((cat, num)))
        if direct is not None:
            return direct
        seen, queue = {num}, sorted(copies.get(num, ()))
        while queue:
            twin = queue.pop(0)
            if twin in seen:
                continue
            seen.add(twin)
            base = identities.get(frozenset((cat, twin)))
            if base is not None and base.kind != "near_copy":
                return DerivedColumn(
                    target=num,
                    components=[twin],
                    formula=f"near copy of {twin}; {base.target} = {base.formula}",
                    kind="near_copy",
                    match_ratio=base.match_ratio,
                    n=base.n,
                )
            queue.extend(sorted(copies.get(twin, ()) - seen))
        return None
    # smallest positive step of each num (its resolution: 1 for counts, 0.01
    # for prices) — the floor for the bucket gap tolerance
    steps = df.select(*(_finite_col(n, df) for n in names))
    res_exprs = []
    for n in names:
        diff = pl.col(n).drop_nulls().unique().sort().diff()
        res_exprs.append(diff.filter(diff > 0).min().alias(n))
    resolution = steps.select(res_exprs).row(0, named=True) if len(steps) else {n: None for n in names}
    effects: list[DispersionEffect] = []
    buckets: list[DerivedColumn] = []
    for cat in cats:
        key = cat.name
        data = df.filter(pl.col(key).is_not_null()).select(pl.col(key), *(_finite_col(n, df) for n in names))
        both = data.with_columns(
            [(pl.col(n) - pl.col(n).median().over(key)).abs().alias(f"dev:{n}") for n in names]
        )
        aggs: list[pl.Expr] = []
        for n in names:
            col = pl.col(n)
            dev = pl.col(f"dev:{n}")
            width = col.max() - col.min()
            aggs += [
                col.count().alias(f"n:{n}"),
                col.std().alias(f"std:{n}"),
                (
                    col.quantile(0.75, interpolation="linear") - col.quantile(0.25, interpolation="linear")
                ).alias(f"iqr:{n}"),
                dev.median().alias(f"mad:{n}"),
                col.min().alias(f"lo:{n}"),
                col.max().alias(f"hi:{n}"),
                dev.sum().alias(f"dsum:{n}"),
                ((dev - dev.mean()) ** 2).sum().alias(f"dssw:{n}"),
                (col >= col.max() - BUCKET_EDGE_SHARE * width).sum().alias(f"top:{n}"),
                (col <= col.min() + BUCKET_EDGE_SHARE * width).sum().alias(f"bottom:{n}"),
            ]
        groups = both.group_by(key, maintain_order=True).agg(aggs)
        rows = sorted(groups.iter_rows(named=True), key=lambda r: str(r[key]))
        for n in names:
            stats = [
                _GroupStat(
                    str(r[key]), int(r[f"n:{n}"]), r[f"std:{n}"], r[f"iqr:{n}"], r[f"mad:{n}"],
                    r[f"lo:{n}"], r[f"hi:{n}"], r[f"dsum:{n}"], r[f"dssw:{n}"],
                    int(r[f"top:{n}"] or 0), int(r[f"bottom:{n}"] or 0),
                )
                for r in rows
                if r[f"n:{n}"]
            ]
            # groups of cat with no value of num at all: excluded, traced with 0 rows
            empty = {str(r[key]): 0 for r in rows if not r[f"n:{n}"]}
            definition = definition_of(key, n)
            if definition is None and _is_bucket(stats, resolution[n]):
                definition = DerivedColumn(
                    target=key,
                    components=[n],
                    formula=f"bucket({n})",
                    kind="bucket",
                    match_ratio=1.0,
                    n=sum(s.n for s in stats),
                )
                buckets.append(definition)
            # the supported comparison first (>= 2 groups of >= STRONG_MIN_N
            # rows, may be strong); when it lists nothing, the wider
            # comparison over every group >= WEAK_MIN_N (at most weak) —
            # follow-up 1: a supported subset must not hide evidence that
            # the smaller groups carry
            supported = [g for g in stats if g.n >= DISPERSION_STRONG_MIN_N]
            wider = [g for g in stats if g.n >= DISPERSION_WEAK_MIN_N]
            attempts: list[tuple[list[_GroupStat], bool]] = []
            if len(supported) >= 2:
                attempts.append((supported, True))
            if len(wider) >= 2 and len(wider) > len(supported):
                attempts.append((wider, False))
            for compared, may_be_strong in attempts:
                entry = _dispersion_entry(key, n, stats, compared, may_be_strong, empty, definition)
                if entry is not None:
                    effects.append(entry)
                    break
    return effects, buckets


# --- time effects -----------------------------------------------------------


def span_bucket(span_days: float) -> TimeBucket:
    """Span-adaptive bucket (blocking #5): short series must not sink because
    the bucket is too coarse for their span."""
    if span_days <= 90:
        return "day"
    if span_days <= 3 * 365:
        return "month"
    return "year"


def _datetime_span_days(s: pl.Series) -> float | None:
    non_null = s.drop_nulls()
    if len(non_null) < 2:
        return None
    return (non_null.max() - non_null.min()) / timedelta(days=1)


def _time_effects(
    df: pl.DataFrame, dts: list[ColumnProfile], nums: list[ColumnProfile]
) -> list[TimeEffect]:
    effects = []
    for dt in dts:
        span = _datetime_span_days(df[dt.name])
        if span is None:
            continue
        # one batched pass per bucket; a num whose finer bucket has no
        # within-group df (e.g. one observation per day) is retried at the
        # coarser bucket, as before — only the nums still pending are retried
        found: dict[str, TimeEffect] = {}
        pending = [num.name for num in nums]
        bucket: TimeBucket | None = span_bucket(span)
        while bucket is not None and pending:
            bucketed = df.select(
                pl.col(dt.name).dt.truncate(_BUCKET_TRUNC[bucket]).alias("_bucket"),
                *[pl.col(num) for num in pending],
            )
            results = adjusted_eta_squared_many(bucketed, "_bucket", pending)
            still_pending = []
            for num in pending:
                result = results[num]
                if result is None:
                    still_pending.append(num)
                    continue
                found[num] = TimeEffect(
                    datetime_col=dt.name,
                    num=num,
                    eta_squared=result.eta_squared,
                    bucket=bucket,
                    n_total=result.n_total,
                    n_min=result.n_min,
                    p_value=result.p_value,
                )
            pending = still_pending
            bucket = _COARSER[bucket]
        effects += [found[num.name] for num in nums if num.name in found]  # stable num order
    return effects


# --- Spearman ---------------------------------------------------------------


def _spearman_matrix(df: pl.DataFrame, correlations: Correlations | None) -> Correlations | None:
    """Rank each column once, then Pearson on the ranks pairwise. Per-pair
    pl.corr(method="spearman") re-ranks 100k rows twice per pair (measured
    5.2s at 100k x 30 columns), so the rank-once approximation is used —
    exact without nulls, and ranks shift only marginally when pairwise-null
    rows are dropped. The pairwise Pearson itself is the single-execution
    helper shared with the profiler (stage 9 #1)."""
    if correlations is None:
        return None
    cols = correlations.columns
    ranked = df.select(
        pl.when(pl.col(c).cast(pl.Float64).is_finite())
        .then(pl.col(c).cast(pl.Float64))
        .otherwise(None)
        .rank(method="average")
        .alias(c)
        for c in cols
    )
    matrix, counts = pairwise_pearson(ranked, cols)
    return Correlations(
        columns=cols, matrix=matrix, truncated=correlations.truncated, pair_counts=counts
    )


# --- interactions -----------------------------------------------------------


def _interactions(
    df: pl.DataFrame,
    cats: list[ColumnProfile],
    nums: list[ColumnProfile],
    dts: list[ColumnProfile],
    correlations: Correlations | None,
    best_eta: dict[str, float],
) -> list[InteractionEffect]:
    top_cats = sorted((c.name for c in cats), key=lambda c: (-best_eta.get(c, 0.0), c))[
        :TOP_CATS_FOR_INTERACTIONS
    ]
    num_rank = dict(best_eta)
    if correlations is not None:
        for i, a in enumerate(correlations.columns):
            for j in range(i + 1, len(correlations.columns)):
                value = correlations.matrix[i][j]
                if value is not None:
                    b = correlations.columns[j]
                    num_rank[a] = max(num_rank.get(a, 0.0), abs(value))
                    num_rank[b] = max(num_rank.get(b, 0.0), abs(value))
    top_nums = sorted((c.name for c in nums), key=lambda c: (-num_rank.get(c, 0.0), c))[
        :TOP_NUMS_FOR_INTERACTIONS
    ]

    # factor columns: real categoricals, plus time buckets labeled "col@bucket"
    frame = df
    factors: list[tuple[str, str]] = [(name, name) for name in top_cats]  # (label, column)
    for dt in dts:
        span = _datetime_span_days(df[dt.name])
        if span is None:
            continue
        bucket = span_bucket(span)
        label = f"{dt.name}@{bucket}"
        frame = frame.with_columns(
            pl.col(dt.name).dt.truncate(_BUCKET_TRUNC[bucket]).alias(label)
        )
        factors.append((label, label))

    pairs = [(a, b) for (a, _), (b, _) in combinations(factors, 2) if not ("@" in a and "@" in b)]
    effects = []
    for label1, label2 in pairs:
        for num in top_nums:
            result = _interaction_test(frame, label1, label2, num)
            if result is not None:
                strength, n_total, p_value = result
                effects.append(
                    InteractionEffect(
                        cat1=label1, cat2=label2, num=num, strength=strength, n_total=n_total,
                        p_value=p_value,
                    )
                )
    return effects


def _interaction_strength(df: pl.DataFrame, f1: str, f2: str, num: str) -> float | None:
    result = _interaction_effect(df, f1, f2, num)
    return None if result is None else result[0]


def _interaction_effect(df: pl.DataFrame, f1: str, f2: str, num: str) -> tuple[float, int] | None:
    result = _interaction_test(df, f1, f2, num)
    return None if result is None else (result[0], result[1])


def _interaction_test(df: pl.DataFrame, f1: str, f2: str, num: str) -> tuple[float, int, float] | None:
    """Additive-prediction residual share of variance, with the number of
    rows in the kept cells.

    Marginal effects use UNWEIGHTED means of cell means: with raw weighted
    marginals, an unbalanced but purely additive design produces a large fake
    interaction (the eval pins this), so the unweighted-means ANOVA form is
    required. Not df-corrected — small cells bias it upward, which the
    MIN_CELL_COUNT floor bounds. Stage 21: the third value is the F-test
    p-value of SS_interaction / (cells − a − b + 1) against the within-cell
    residual / (N − cells), which does account for the cell count."""
    data = df.select(f1, f2, num).drop_nulls()
    if data.schema[num].is_float():
        data = data.filter(pl.col(num).is_finite())
    cells = (
        data.group_by(f1, f2, maintain_order=True)
        .agg(pl.len().alias("_n"), pl.col(num).mean().alias("_m"))
        .sort(f1, f2)
    )
    kept = cells.filter(pl.col("_n") >= MIN_CELL_COUNT)
    # unweighted marginal means are only additive-exact on a near-complete
    # grid; a sparse grid can fake interaction on purely additive data
    full_grid = cells[f1].n_unique() * cells[f2].n_unique()
    if full_grid and kept.height / full_grid < 0.7:
        return None
    if kept.height < MIN_VALID_CELLS:
        return None
    if kept[f1].n_unique() < 2 or kept[f2].n_unique() < 2:
        return None
    data = data.join(kept.select(f1, f2), on=[f1, f2], how="semi")
    ss_total = ((data[num] - data[num].mean()) ** 2).sum()
    if ss_total is None or ss_total <= 0:
        return None
    row_mean = {
        key: value
        for key, value in kept.group_by(f1, maintain_order=True)
        .agg(pl.col("_m").mean())
        .iter_rows()
    }
    col_mean = {
        key: value
        for key, value in kept.group_by(f2, maintain_order=True)
        .agg(pl.col("_m").mean())
        .iter_rows()
    }
    grand = kept["_m"].mean()
    ss_interaction = sum(
        w * (m - (row_mean[i] + col_mean[j] - grand)) ** 2
        for i, j, w, m in kept.select(f1, f2, "_n", "_m").iter_rows()
    )
    cell_mean = kept.select(f1, f2, "_m")
    ss_within = float(
        data.join(cell_mean, on=[f1, f2], how="left")
        .select(((pl.col(num) - pl.col("_m")) ** 2).sum())
        .item()
        or 0.0
    )
    a, b = kept[f1].n_unique(), kept[f2].n_unique()
    df_int = kept.height - a - b + 1
    df_within = data.height - kept.height
    # the test uses the least-squares additive fit, not the unweighted-means
    # one behind `strength`: on an unbalanced or incomplete grid the latter is
    # not the LS fit and inflates the interaction SS (independent review:
    # 400/400 additive seeds "significant" with two empty cells)
    ss_int_ls = _ls_interaction_ss(list(kept.select(f1, f2, "_n", "_m").iter_rows()))
    if df_int < 1 or df_within < 1:
        p_value = 1.0
    elif ss_within <= 0:
        p_value = 0.0 if ss_int_ls > 0 else 1.0
    else:
        p_value = f_survival((ss_int_ls / df_int) / (ss_within / df_within), df_int, df_within)
    return float(ss_interaction / ss_total), data.height, p_value


def _ls_interaction_ss(cells: list[tuple]) -> float:
    """Interaction sum of squares of a two-way layout = RSS(additive LS fit)
    − RSS(cell means), from the cell table (row, col, n, mean). The weighted
    additive fit is solved by Gauss–Seidel backfitting, which converges to
    the least-squares solution for a connected design of any balance."""
    rows = {r for r, _, _, _ in cells}
    cols = {c for _, c, _, _ in cells}
    total_n = sum(n for _, _, n, _ in cells)
    mu = sum(n * m for _, _, n, m in cells) / total_n
    row_eff = {r: 0.0 for r in rows}
    col_eff = {c: 0.0 for c in cols}
    row_w = {r: sum(n for rr, _, n, _ in cells if rr == r) for r in rows}
    col_w = {c: sum(n for _, cc, n, _ in cells if cc == c) for c in cols}
    previous = math.inf
    for _ in range(1000):
        for r in rows:
            row_eff[r] = sum(n * (m - mu - col_eff[c]) for rr, c, n, m in cells if rr == r) / row_w[r]
        for c in cols:
            col_eff[c] = sum(n * (m - mu - row_eff[r]) for r, cc, n, m in cells if cc == c) / col_w[c]
        rss = sum(n * (m - mu - row_eff[r] - col_eff[c]) ** 2 for r, c, n, m in cells)
        if previous - rss <= 1e-12 * max(rss, 1e-300):
            break
        previous = rss
    return max(rss, 0.0)


# --- correlation significance + false-discovery control (stage 21) -----------


def _lag1_autocorrelations(df: pl.DataFrame, cols: list[str], dts: list[ColumnProfile]) -> dict[str, float]:
    """Lag-1 autocorrelation of each numeric column with the rows ordered by
    the dataset's most complete datetime column; empty without one."""
    if not dts or not cols:
        return {}
    dt = max(dts, key=lambda c: (-(c.missing_count or 0), c.name)).name
    ordered = df.filter(pl.col(dt).is_not_null()).sort(dt, maintain_order=True).select(
        pl.when(pl.col(c).cast(pl.Float64).is_finite()).then(pl.col(c).cast(pl.Float64)).otherwise(None).alias(c)
        for c in cols
    )
    if ordered.height < 4:
        return {}
    row = ordered.select(pl.corr(pl.col(c), pl.col(c).shift(1)).alias(c) for c in cols).row(0, named=True)
    return {c: float(v) for c, v in row.items() if v is not None and math.isfinite(v)}


def _t_test_p(r: float, n_eff: float) -> float:
    if abs(r) >= 1:
        return 0.0
    return f_survival(r * r * (n_eff - 2) / (1 - r * r), 1, n_eff - 2)


def _correlation_tests(
    df: pl.DataFrame,
    correlations: Correlations | None,
    dts: list[ColumnProfile],
    spearman: Correlations | None = None,
) -> list[CorrelationTest]:
    """t-test of every Pearson pair in the correlation matrix. With a datetime
    column the sample is Bartlett's effective n (independent random walks
    correlate at |r| >= 0.3 in ~70 % of seeded pairs whatever n is; with n_eff
    the false-positive rate falls to 0.5–4 %, audit phase B §4)."""
    if correlations is None:
        return []
    cols = correlations.columns
    rho = _lag1_autocorrelations(df, cols, dts)
    tests = []
    for i, a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            r = correlations.matrix[i][j]
            n = correlations.pair_counts[i][j] if correlations.pair_counts else None
            if r is None or n is None or n < 4:
                continue
            ra, rb = rho.get(a), rho.get(b := cols[j])
            n_eff = float(n)
            if ra is not None and rb is not None and ra * rb > 0:
                n_eff = n * (1 - ra * rb) / (1 + ra * rb)
            n_eff = min(float(n), max(MIN_EFFECTIVE_N, n_eff))
            rho_s = spearman.matrix[i][j] if spearman is not None and spearman.columns == cols else None
            tests.append(
                CorrelationTest(
                    a=a, b=b, r=float(r), n=int(n), n_eff=round(n_eff, 3),
                    autocorr_a=None if ra is None else round(ra, 6),
                    autocorr_b=None if rb is None else round(rb, 6),
                    p_value=_t_test_p(r, n_eff),
                    rho=rho_s,
                    p_value_spearman=None if rho_s is None else _t_test_p(rho_s, n_eff),
                )
            )
    return tests


def benjamini_hochberg(p_values: list[float], q: float = FDR_Q) -> list[bool]:
    """Step-up BH: True for every p-value rejected at FDR level q."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    cutoff = 0
    for rank, i in enumerate(order, 1):
        if p_values[i] <= q * rank / m:
            cutoff = rank
    rejected = [False] * m
    for i in order[:cutoff]:
        rejected[i] = True
    return rejected


def apply_fdr(evidence: Evidence, q: float = FDR_Q) -> None:
    """One BH family per dataset over every L1 entry with a p-value; sets
    each entry's `discovery` and the summary in `evidence.fdr`. Interactions
    and dispersion entries are listed after screening, so their family is
    the listed subset (documented limitation)."""
    entries: list = []
    for e in evidence.cat_num:
        entries.append((e, e.p_value))
    for e in evidence.time_effects:
        entries.append((e, e.p_value))
    for e in evidence.interactions:
        entries.append((e, e.p_value))
    for e in evidence.correlation_tests:
        entries.append((e, e.p_value))
    spearman_entries = [(e, e.p_value_spearman) for e in evidence.correlation_tests]
    for e in evidence.dispersion:
        entries.append((e, e.bf_p_value))
    tested = [(e, p, "discovery") for e, p in entries if p is not None and math.isfinite(p)]
    tested += [(e, p, "discovery_spearman") for e, p in spearman_entries if p is not None and math.isfinite(p)]
    rejected = benjamini_hochberg([p for _, p, _ in tested], q) if tested else []
    for (e, _, field), flag in zip(tested, rejected):
        setattr(e, field, flag)
    evidence.fdr = FdrSummary(q=q, m=len(tested), discoveries=sum(rejected))


# --- slope heterogeneity ----------------------------------------------------


def _slope_heterogeneity(
    df: pl.DataFrame,
    correlations: Correlations | None,
    cats: list[ColumnProfile],
    best_eta: dict[str, float],
) -> list[SlopeHet]:
    if correlations is None:
        return []
    pairs = []
    for i, a in enumerate(correlations.columns):
        for j in range(i + 1, len(correlations.columns)):
            value = correlations.matrix[i][j]
            if value is not None and abs(value) >= MIN_ABS_CORR:
                pairs.append((abs(value), a, correlations.columns[j]))
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))

    lo, hi = SLOPE_GROUP_RANGE
    group_cols = [c.name for c in cats if lo <= (c.n_categories or 0) <= hi]
    group_cols = sorted(group_cols, key=lambda c: (-best_eta.get(c, 0.0), c))[:TOP_CATS_FOR_SLOPES]

    results = []
    for _, x, y in pairs[:TOP_PAIRS_FOR_SLOPES]:
        for group in group_cols:
            data = df.select(x, y, group).drop_nulls()
            for name in (x, y):
                if data.schema[name].is_float():
                    data = data.filter(pl.col(name).is_finite())
            corrs: dict[str, float | None] = {}
            valid_sizes = []
            parts = sorted(data.group_by(group), key=lambda kv: str(kv[0][0]))
            for key, part in parts:  # sorted so the stored dict order is stable
                if part.height < MIN_SLOPE_GROUP_ROWS:
                    continue
                value = _finite(part.select(pl.corr(x, y)).item())
                corrs[str(key[0])] = value
                if value is not None:
                    valid_sizes.append(part.height)
            valid = [v for v in corrs.values() if v is not None]
            if len(valid) < 2:
                continue
            results.append(
                SlopeHet(
                    x=x,
                    y=y,
                    group=group,
                    corrs=corrs,
                    spread=max(valid) - min(valid),
                    n_min=min(valid_sizes),
                )
            )
    return results


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


# --- derived columns (stage 13) ---------------------------------------------


class _Form(NamedTuple):
    kind: DerivedKind
    components: tuple[str, ...]
    formula: str
    expr: pl.Expr


def derived_candidates(columns: list[ColumnProfile]) -> list[str]:
    """Columns a definitional identity can involve: numeric measurements and
    numeric-backed categoricals that are quantities (quantity=1..10 is a
    categorical to the profiler but a factor in sales = price × quantity).
    Nominal codes, ids, booleans and constants are out; capped in column order."""
    names = []
    for c in columns:
        if c.semantic_type == "numeric":
            if c.std is None or c.std <= 0:
                continue
        elif c.semantic_type == "categorical":
            if c.nominal or not _NUMERIC_DTYPE_RE.match(c.original_dtype):
                continue
            if (c.n_categories or 0) < 2:
                continue
        else:
            continue
        names.append(c.name)
    return names[:DERIVED_MAX_COLUMNS]


def derived_columns(
    df: pl.DataFrame,
    columns: list[ColumnProfile],
    correlations: Correlations | None,
    spearman: Correlations | None,
) -> list[DerivedColumn]:
    """Row-wise identity search plus near-copy disclosure.

    Identities tested for every target t and distinct peers a, b, c:
    a×b, a+b, a−b, b−a, a/b, b/a, a×(1−c), a×b×c, a×b×(1−c) (the (1−c) forms
    only when c lies in [0, 1]). A row matches when |t − f| is within
    max(1e-6·|t|, half a unit of t's last observed decimal); a form holds
    when >= 99% of the >= 30 finite rows match. One result per target: the
    best match ratio, then the fewest components (a×b beats a×b×(1−c) when c
    is all zero), then formula text — all deterministic. No constant fitting,
    no more than three components: those are out of scope.

    Cost is bounded: a seeded subsample of DERIVED_MAX_ROWS rows, every
    form of one target evaluated in ONE polars select, and above
    DERIVED_FULL_POOL_COLUMNS candidates each target only meets its
    DERIVED_POOL most Pearson-correlated peers (a wide sensor table is not
    scanned combinatorially; a weakly correlated factor such as a discount
    can then be missed — documented limitation)."""
    names = derived_candidates(columns)
    if len(names) < 2:
        return _near_copies(spearman, [], columns)
    data = df.select(
        pl.when(pl.col(n).cast(pl.Float64).is_finite())
        .then(pl.col(n).cast(pl.Float64))
        .otherwise(None)
        .alias(n)
        for n in names
    )
    if data.height > DERIVED_MAX_ROWS:
        data = data.sample(DERIVED_MAX_ROWS, seed=DERIVED_SAMPLE_SEED)
    if data.height < DERIVED_MIN_ROWS:
        return _near_copies(spearman, [], columns)

    stats = _column_stats(data, names)
    unit_interval = {
        n for n in names if stats[n]["min"] is not None and stats[n]["min"] >= 0 and stats[n]["max"] <= 1
    }
    results: list[DerivedColumn] = []
    for target in names:
        if stats[target]["n_unique"] < DERIVED_MIN_TARGET_UNIQUE:
            continue
        if not _has_correlated_peer(target, correlations, spearman):
            continue
        pool = _peer_pool(target, names, correlations, data)
        forms = _forms(target, pool, unit_interval)
        if not forms:
            continue
        found = _best_form(data, target, stats[target]["decimals"], forms)
        if found is not None:
            results.append(found)
    results = _dedup_rearrangements(results, names)
    return results + _near_copies(spearman, results, columns)


def _dedup_rearrangements(results: list[DerivedColumn], names: list[str]) -> list[DerivedColumn]:
    """c = a + b also matches as a = c − b and b = c − a: one identity, three
    targets. Keep one entry per variable set — the best match, then the
    target that comes LAST in column order (a computed column is usually
    appended after its inputs: sales after price, quantity, discount)."""
    best: dict[frozenset[str], DerivedColumn] = {}
    for d in results:
        key = frozenset([d.target, *d.components])
        current = best.get(key)
        if current is None or (d.match_ratio, names.index(d.target)) > (
            current.match_ratio,
            names.index(current.target),
        ):
            best[key] = d
    kept = set(map(id, best.values()))
    return [d for d in results if id(d) in kept]


def _column_stats(data: pl.DataFrame, names: list[str]) -> dict[str, dict]:
    exprs = []
    for n in names:
        col = pl.col(n)
        exprs += [col.n_unique().alias(f"u:{n}"), col.min().alias(f"lo:{n}"), col.max().alias(f"hi:{n}")]
        for k in range(DERIVED_MAX_DECIMALS + 1):
            scaled = col * (10.0**k)
            exprs.append((scaled - scaled.round(0)).abs().max().alias(f"d{k}:{n}"))
    row = data.select(exprs).row(0, named=True)
    stats = {}
    for n in names:
        decimals = DERIVED_MAX_DECIMALS
        for k in range(DERIVED_MAX_DECIMALS + 1):
            gap = row[f"d{k}:{n}"]
            if gap is not None and gap < 1e-6:
                decimals = k
                break
        # nulls count as a distinct value in n_unique; they never match
        stats[n] = {"n_unique": int(row[f"u:{n}"]), "min": row[f"lo:{n}"], "max": row[f"hi:{n}"], "decimals": decimals}
    return stats


def _has_correlated_peer(
    target: str, correlations: Correlations | None, spearman: Correlations | None
) -> bool:
    """False only when the target IS in the correlation scan and nothing
    there co-moves with it; a target outside the scan (a numeric-backed
    categorical such as quantity) is always searched."""
    seen = False
    for table in (correlations, spearman):
        if table is None or target not in table.columns:
            continue
        seen = True
        row = table.matrix[table.columns.index(target)]
        for j, value in enumerate(row):
            if j != table.columns.index(target) and value is not None and abs(value) >= DERIVED_MIN_PEER_CORR:
                return True
    return not seen


def _peer_pool(
    target: str, names: list[str], correlations: Correlations | None, data: pl.DataFrame
) -> list[str]:
    """Every other candidate when the table is small; above
    DERIVED_FULL_POOL_COLUMNS the DERIVED_POOL peers with the strongest
    |Pearson| to the target. Strengths come from the profile's correlation
    table; candidates outside it (numeric-backed categoricals such as
    quantity, which the profiler never correlates) get theirs computed on
    the subsample in one select, so a factor is never ranked as 0 merely
    for being categorical (verifier finding, stage 13)."""
    peers = [n for n in names if n != target]
    if len(names) <= DERIVED_FULL_POOL_COLUMNS:
        return peers

    strength: dict[str, float] = {}
    if correlations is not None and target in correlations.columns:
        row = correlations.matrix[correlations.columns.index(target)]
        for peer in peers:
            if peer in correlations.columns:
                value = row[correlations.columns.index(peer)]
                strength[peer] = abs(value) if value is not None else 0.0
    missing = [p for p in peers if p not in strength]
    if missing:
        computed = data.select(pl.corr(pl.col(target), pl.col(p)).alias(p) for p in missing).row(0, named=True)
        for peer in missing:
            value = computed[peer]
            strength[peer] = abs(value) if value is not None and math.isfinite(value) else 0.0

    ranked = sorted(peers, key=lambda p: (-strength[p], names.index(p)))[:DERIVED_POOL]
    return [p for p in peers if p in ranked]  # keep column order for determinism


def _forms(target: str, pool: list[str], unit_interval: set[str]) -> list[_Form]:
    forms: list[_Form] = []
    col = pl.col
    for a, b in combinations(pool, 2):
        forms.append(_Form("product", (a, b), f"{a} × {b}", col(a) * col(b)))
        forms.append(_Form("sum", (a, b), f"{a} + {b}", col(a) + col(b)))
        forms.append(_Form("difference", (a, b), f"{a} − {b}", col(a) - col(b)))
        forms.append(_Form("difference", (b, a), f"{b} − {a}", col(b) - col(a)))
        forms.append(_Form("ratio", (a, b), f"{a} / {b}", col(a) / col(b)))
        forms.append(_Form("ratio", (b, a), f"{b} / {a}", col(b) / col(a)))
        if b in unit_interval:
            forms.append(_Form("product_discount", (a, b), f"{a} × (1 − {b})", col(a) * (1 - col(b))))
        if a in unit_interval:
            forms.append(_Form("product_discount", (b, a), f"{b} × (1 − {a})", col(b) * (1 - col(a))))
    if len(pool) + 1 <= DERIVED_FULL_POOL_COLUMNS or len(pool) <= DERIVED_POOL:
        for a, b, c in combinations(pool, 3):
            forms.append(_Form("product", (a, b, c), f"{a} × {b} × {c}", col(a) * col(b) * col(c)))
            for x, y, d in ((a, b, c), (a, c, b), (b, c, a)):
                if d in unit_interval:
                    forms.append(
                        _Form("product_discount", (x, y, d), f"{x} × {y} × (1 − {d})", col(x) * col(y) * (1 - col(d)))
                    )
    return forms


def _best_form(data: pl.DataFrame, target: str, decimals: int, forms: list[_Form]) -> DerivedColumn | None:
    """Two passes because polars cost here is per expression, not per row
    (~0.1 ms each): pass 1 counts matching rows for every form; only forms
    with at least DERIVED_MIN_ROWS matches (a handful on real data) get the
    exact valid-row count in pass 2."""
    t = pl.col(target)
    # half a unit of the last observed decimal (rounding) plus a relative
    # floating-point allowance — summed, so a value rounded exactly at the
    # half-unit boundary still matches
    tol = t.abs() * 1e-6 + 0.5 * 10.0 ** (-decimals)
    # a null anywhere (missing / non-finite input, division by zero) is
    # neither a match nor a valid row: the comparison is null and sum skips it
    matches = data.select(
        ((t - form.expr).abs() <= tol).sum().alias(f"m{i}") for i, form in enumerate(forms)
    ).row(0)
    survivors = [
        (i, int(m or 0)) for i, m in enumerate(matches) if int(m or 0) >= DERIVED_MIN_ROWS * DERIVED_MATCH_RATIO
    ]
    if not survivors:
        return None
    valids = data.select(
        (t - forms[i].expr).is_finite().sum().alias(f"v{i}") for i, _ in survivors
    ).row(0)
    best: tuple[tuple, DerivedColumn] | None = None
    for (i, matched), valid in zip(survivors, valids):
        valid = int(valid or 0)
        if valid < DERIVED_MIN_ROWS:
            continue
        ratio = min(matched / valid, 1.0)
        if ratio < DERIVED_MATCH_RATIO:
            continue
        form = forms[i]
        key = (-ratio, len(form.components), form.formula)
        if best is None or key < best[0]:
            best = (
                key,
                DerivedColumn(
                    target=target,
                    components=list(form.components),
                    formula=form.formula,
                    kind=form.kind,
                    match_ratio=ratio,
                    n=valid,
                ),
            )
    return None if best is None else best[1]


def _near_copies(
    spearman: Correlations | None, found: list[DerivedColumn], columns: list[ColumnProfile]
) -> list[DerivedColumn]:
    """One near_copy entry per suppressed duplicate (target = duplicate,
    component = its group's representative), derived from the stage 14
    groups so the disclosure and the suppression agree by construction."""
    out = []
    for group in near_duplicate_groups(spearman, found, columns):
        for dup in group.duplicates:
            rho = group.rho[dup]
            out.append(
                DerivedColumn(
                    target=dup,
                    components=[group.representative],
                    formula=f"≈ monotone transform of {group.representative} (rank correlation {rho:.3f})",
                    kind="near_copy",
                    match_ratio=rho,
                    n=group.n,
                )
            )
    return out


def _normalised_name(name: str) -> str:
    return _NAME_SPLIT_RE.sub("", _CAMEL_RE.sub("_", name).lower())


def _name_tokens(name: str) -> set[str]:
    return {t for t in _NAME_SPLIT_RE.split(_CAMEL_RE.sub("_", name).lower()) if len(t) >= _NAME_MIN_TOKEN}


def names_related(a: str, b: str) -> bool:
    """temp / atemp, price / price_usd, tempC / temp_f: one normalised name
    inside the other (>= 4 chars) or a shared alphabetic token (>= 3 chars).
    Digits and separators never count, so s01 / s02 are unrelated."""
    na, nb = _normalised_name(a), _normalised_name(b)
    if len(na) >= _NAME_MIN_SUBSTRING and len(nb) >= _NAME_MIN_SUBSTRING and (na in nb or nb in na):
        return True
    return bool(_name_tokens(a) & _name_tokens(b))


def near_duplicate_groups(
    spearman: Correlations | None, found: list[DerivedColumn], columns: list[ColumnProfile]
) -> list[NearDuplicateGroup]:
    """Connected components of near-duplicate pairs among the numeric columns
    (stage 14). A pair qualifies with |rho| >= NEAR_COPY_MIN_RHO, or with
    |rho| >= NEAR_DUP_NAMED_MIN_RHO when names_related; >= DERIVED_MIN_ROWS
    shared rows; pairs an identity already explains are skipped. The
    representative is the member with the fewest missing values, then the
    shorter name, then column order; rho maps each duplicate to |Spearman|
    with the representative (falling back to the pair that linked it)."""
    if spearman is None:
        return []
    by_name = {c.name: c for c in columns}
    eligible = {n for n in derived_candidates(columns)}
    explained = {frozenset((d.target, c)) for d in found if d.kind != "near_copy" for c in d.components}
    counts = spearman.pair_counts
    cols = spearman.columns
    order = {name: i for i, name in enumerate(cols)}
    parent: dict[str, str] = {}
    pair_rho: dict[frozenset[str], float] = {}
    pair_n: dict[frozenset[str], int] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    for i, a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            b = cols[j]
            if a not in eligible or b not in eligible:
                continue
            rho = spearman.matrix[i][j]
            n = counts[i][j] if counts is not None else DERIVED_MIN_ROWS
            if rho is None or n < DERIVED_MIN_ROWS:
                continue
            strength = abs(rho)
            if strength < NEAR_COPY_MIN_RHO and not (
                strength >= NEAR_DUP_NAMED_MIN_RHO and names_related(a, b)
            ):
                continue
            key = frozenset((a, b))
            if key in explained:
                continue
            pair_rho[key] = strength
            pair_n[key] = n
            parent.setdefault(a, a)
            parent.setdefault(b, b)
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

    members: dict[str, list[str]] = {}
    for name in parent:
        members.setdefault(find(name), []).append(name)

    groups = []
    for group in members.values():
        group.sort(key=lambda n: order[n])
        rep = min(group, key=lambda n: (by_name[n].missing_count, len(n), order[n]))
        dups = [n for n in group if n != rep]
        rho: dict[str, float] = {}
        n_min = None
        for dup in dups:
            direct = pair_rho.get(frozenset((dup, rep)))
            if direct is None:
                # linked through another member: report the strongest link
                direct = max(v for k, v in pair_rho.items() if dup in k)
            rho[dup] = direct
            n_pair = pair_n.get(frozenset((dup, rep)))
            if n_pair is not None:
                n_min = n_pair if n_min is None else min(n_min, n_pair)
        groups.append(
            NearDuplicateGroup(
                representative=rep,
                duplicates=dups,
                rho=rho,
                n=n_min if n_min is not None else min(pair_n[k] for k in pair_n if any(m in k for m in group)),
            )
        )
    groups.sort(key=lambda g: order[g.representative])
    return groups
