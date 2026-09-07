"""Effect-size evidence for recommendation ranking (stage 7).

Everything is computed on the profiler's seeded <=100k sample and follows the
NaN/inf -> null convention. Scan sizes are bounded by the caps below, so the
whole table stays in the millisecond-to-sub-second range.
"""

import math
from datetime import timedelta
from itertools import combinations

import polars as pl

from .models import (
    CatNumEffect,
    ColumnProfile,
    Correlations,
    Evidence,
    InteractionEffect,
    SlopeHet,
    TimeBucket,
    TimeEffect,
)

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
    return Evidence(
        cat_num=cat_num,
        time_effects=_time_effects(df, dts, nums),
        num_num_spearman=_spearman_matrix(df, correlations),
        interactions=_interactions(df, cats, nums, dts, correlations, best_eta),
        slope_heterogeneity=_slope_heterogeneity(df, correlations, cats, best_eta),
    )


def adjusted_eta_squared(df: pl.DataFrame, cat: str, num: str) -> tuple[float, int] | None:
    """1 - MS_within/MS_total, clipped at 0. Raw eta-squared is inflated on
    high-cardinality small samples (blocking #1), the df-corrected form is not.
    Returns None when undefined (constant num, <2 groups, no within-group df)."""
    data = df.select(cat, num).drop_nulls()
    if data.schema[num].is_float():
        data = data.filter(pl.col(num).is_finite())
    n = data.height
    k = data[cat].n_unique()
    if k < 2 or n - k < 2:
        return None
    grand = data[num].mean()
    ss_total = ((data[num] - grand) ** 2).sum()
    if ss_total is None or ss_total <= 0:
        return None
    # maintain_order keeps the float summation order deterministic across runs
    ss_within = (
        data.group_by(cat, maintain_order=True)
        .agg(((pl.col(num) - pl.col(num).mean()) ** 2).sum().alias("ss"))["ss"]
        .sum()
    )
    adjusted = 1 - (ss_within / (n - k)) / (ss_total / (n - 1))
    return max(float(adjusted), 0.0), k


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
    for cat in cats:
        for num in nums:
            result = adjusted_eta_squared(df, cat.name, num.name)
            if result is not None:
                effects.append(
                    CatNumEffect(cat=cat.name, num=num.name, eta_squared=result[0], n_groups=result[1])
                )
    return effects


def _best_eta_by_column(effects: list[CatNumEffect]) -> dict[str, float]:
    best: dict[str, float] = {}
    for e in effects:
        best[e.cat] = max(best.get(e.cat, 0.0), e.eta_squared)
        best[e.num] = max(best.get(e.num, 0.0), e.eta_squared)
    return best


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
        for num in nums:
            # coarsen when the finer bucket has no within-group df
            # (e.g. one observation per day)
            bucket: TimeBucket | None = span_bucket(span)
            while bucket is not None:
                bucketed = df.select(
                    pl.col(dt.name).dt.truncate(_BUCKET_TRUNC[bucket]).alias("_bucket"),
                    pl.col(num.name),
                )
                result = adjusted_eta_squared(bucketed, "_bucket", num.name)
                if result is not None:
                    effects.append(
                        TimeEffect(
                            datetime_col=dt.name, num=num.name, eta_squared=result[0], bucket=bucket
                        )
                    )
                    break
                bucket = _COARSER[bucket]
    return effects


# --- Spearman ---------------------------------------------------------------


def _spearman_matrix(df: pl.DataFrame, correlations: Correlations | None) -> Correlations | None:
    """Rank each column once, then Pearson on the ranks pairwise. Per-pair
    pl.corr(method="spearman") re-ranks 100k rows twice per pair (measured
    5.2s at 100k x 30 columns), so the rank-once approximation is used —
    exact without nulls, and ranks shift only marginally when pairwise-null
    rows are dropped."""
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
    n = len(cols)
    matrix: list[list[float | None]] = [[None] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            pair = ranked.select(cols[i], cols[j]).drop_nulls()
            value = pair.select(pl.corr(cols[i], cols[j])).item() if pair.height >= 2 else None
            matrix[i][j] = matrix[j][i] = _finite(value)
    return Correlations(columns=cols, matrix=matrix, truncated=correlations.truncated)


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
            strength = _interaction_strength(frame, label1, label2, num)
            if strength is not None:
                effects.append(
                    InteractionEffect(cat1=label1, cat2=label2, num=num, strength=strength)
                )
    return effects


def _interaction_strength(df: pl.DataFrame, f1: str, f2: str, num: str) -> float | None:
    """Additive-prediction residual share of variance.

    Marginal effects use UNWEIGHTED means of cell means: with raw weighted
    marginals, an unbalanced but purely additive design produces a large fake
    interaction (the eval pins this), so the unweighted-means ANOVA form is
    required. Not df-corrected — small cells bias it upward, which the
    MIN_CELL_COUNT floor bounds."""
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
    return float(ss_interaction / ss_total)


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
