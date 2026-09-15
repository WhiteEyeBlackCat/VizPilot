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
    Correlations,
    DerivedColumn,
    DerivedKind,
    Evidence,
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
    return Evidence(
        cat_num=cat_num,
        time_effects=_time_effects(df, dts, nums),
        num_num_spearman=spearman,
        interactions=_interactions(df, cats, nums, dts, correlations, best_eta),
        slope_heterogeneity=_slope_heterogeneity(df, correlations, cats, best_eta),
        derived_columns=derived,
        near_duplicate_groups=near_duplicate_groups(spearman, derived, columns),
    )


class EtaResult(NamedTuple):
    eta_squared: float
    n_groups: int
    n_total: int
    n_min: int
    group_counts: dict[str, int]  # str(group value) -> valid rows; only non-empty groups


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
        results[num] = EtaResult(max(float(adjusted), 0.0), k, n, int(non_empty.min()), group_counts)
    return results


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
                    )
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
            result = _interaction_effect(frame, label1, label2, num)
            if result is not None:
                strength, n_total = result
                effects.append(
                    InteractionEffect(
                        cat1=label1, cat2=label2, num=num, strength=strength, n_total=n_total
                    )
                )
    return effects


def _interaction_strength(df: pl.DataFrame, f1: str, f2: str, num: str) -> float | None:
    result = _interaction_effect(df, f1, f2, num)
    return None if result is None else result[0]


def _interaction_effect(df: pl.DataFrame, f1: str, f2: str, num: str) -> tuple[float, int] | None:
    """Additive-prediction residual share of variance, with the number of
    rows in the kept cells.

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
    return float(ss_interaction / ss_total), data.height


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
