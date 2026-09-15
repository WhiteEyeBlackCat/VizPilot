"""Evidence layer 2 (stage 17.1): bounded, deterministic structural summaries
for the LLM workflow.

Layer 1 (evidence.py) reduces every relationship to one effect size. That is
enough to RANK charts but not to REASON about a dataset: the LLM cannot tell
a U shape from a flat relationship, a bimodal column from a skewed one, or a
subgroup that behaves differently from the rest. Layer 2 adds explainable
summaries of exactly those structures, each with the row count it was
computed on, each capped so the cost stays bounded and the prompt small.

Rules of this module:
- Everything runs on the profiler's seeded sample (`df` is `casted`), in
  batched polars executions (one per categorical / datetime / x column).
- Nothing here is read by the rule engine; recommendations on the
  `llm=false` path are byte-identical with or without layer 2.
- Definitional structure is left out: near-duplicate columns (stage 14) are
  represented by their representative only, and derived-column pairs
  (stage 13) never appear as a group summary, conditional relationship,
  non-linear signal or anomaly — the LLM must not be handed the formula as
  a pattern to explain.
- Deterministic: stable sort keys everywhere, no RNG.
"""

import math
from typing import Any

import polars as pl

from .evidence import (
    MIN_SLOPE_GROUP_ROWS,
    _BUCKET_TRUNC,
    _COARSER,
    _datetime_span_days,
    _eligible_cats,
    _finite,
    span_bucket,
)
from .models import (
    ChangePoint,
    ColumnProfile,
    ConditionalRelationship,
    Correlations,
    DistributionSummary,
    Evidence,
    EvidenceLayer2,
    GroupCorrelation,
    GroupStat,
    GroupSummary,
    GroupTimePattern,
    NonlinearBin,
    NonlinearSignal,
    RobustRange,
    SubgroupAnomaly,
    TimeBucket,
    TimeSeriesPoint,
)

# --- caps and thresholds (the only numbers in this layer) --------------------

MIN_ROWS = 30  # below this no layer-2 entry is produced for a column / pair

# distributions
MAX_DISTRIBUTIONS = 20  # numeric columns, column order
DIST_BINS = 10
BIMODALITY_THRESHOLD = 0.555  # Sarle's bimodality coefficient of the uniform distribution
PEAK_MIN_SHARE = 0.10  # a histogram peak must hold at least this share of rows
VALLEY_MAX_RATIO = 0.6  # ...and the valley between two peaks at most this share of the lower peak
SKEW_SYMMETRIC = 0.5  # |skewness| below this is "symmetric"
# a column with few distinct values (a 24-level hour, a 4-level discount) is
# discrete, and its equal-width histogram is a comb whose teeth are not modes
MULTIMODAL_MIN_UNIQUE = 50

# group summaries
GROUP_SUMMARY_TOP = 8  # cat×num pairs by eta-squared
# a grouping is summarised only when it explains at least this much of the
# column (top pairs and the per-column fallback alike): below it the
# per-group means differ by noise only (weathersit x hr 0.002, 63 sensors
# 0.0003-0.005, sales_basic 0.0) and would just pad the prompt
GROUP_SUMMARY_MIN_ETA = 0.01
MAX_GROUP_SUMMARIES = 12  # ...plus one pair per otherwise uncovered numeric column
MAX_GROUPS_LISTED = 20

# conditional relationships / group time patterns
MAX_CONDITIONAL = 8
CONDITIONAL_TOP_GROUPS = 3  # grouping columns (2..8 categories) by best eta-squared
CONDITIONAL_TOP_NUMS = 8  # numeric columns: layer-1 slope pairs first, then by strongest correlation
MAX_GROUP_TIME_PATTERNS = 8
MAX_TIME_POINTS = 24  # buckets per group; coarsened first, then evenly thinned
MIN_TIME_POINTS = 4  # a group series shorter than this (and any group under MIN_ROWS) is noise
TIME_PATTERN_GROUP_RANGE = (2, 8)  # categories of the grouping column
TIME_PATTERN_TOP_NUMS = 4  # per datetime column, by time effect
TIME_PATTERN_TOP_CATS = 2  # per num, by eta-squared

# nonlinear signals
MAX_NONLINEAR = 10
NONLINEAR_BINS = 10
NONLINEAR_MIN_ROWS = 100  # 10 rows per equal-frequency bin
NONLINEAR_MIN_X_UNIQUE = 10  # fewer distinct x values is a categorical effect, not a curve
MAX_NONLINEAR_COLUMNS = 20
SHAPE_FLAT_ETA = 0.05  # binned eta-squared below this: no dependence worth a shape
SHAPE_LINEAR_GAP = 0.05  # nonlinear_gap below this: a straight line explains it
MONOTONE_TOLERANCE = 0.05  # a step against the trend below this share of the range is noise
# multi_peak: interior local maxima of the bin means (normalised to the
# series range), each at least CURVE_PEAK_MIN above the minimum, separated by
# a valley at most CURVE_VALLEY_MAX_RATIO of the lower peak
CURVE_PEAK_MIN = 0.10
CURVE_VALLEY_MAX_RATIO = 0.75

# change points
MAX_CHANGE_POINTS = 8
MIN_SEGMENT_BUCKETS = 5
MAX_CHANGE_BUCKETS = 400
CHANGE_FLAG_EFFECT = 1.5  # |after − before| in pooled SDs of the bucket means (signal vs bucket noise)
CHANGE_FLAG_DIFF_SD = 0.3  # ...AND in SDs of the column itself (a change that matters in data units)

# subgroup anomalies
MAX_ANOMALIES = 8
ANOMALY_MIN_GROUPS = 5
ANOMALY_MIN_ROWS = 30
ANOMALY_Z = 3.0  # robust z of the group mean among the group means
ANOMALY_MIN_DIFF_SD = 0.5  # ...AND at least this far from the others in pooled within-group SDs
MAD_SCALE = 1.4826

_BIN = "__l2_bin"
_FINER: dict[TimeBucket, TimeBucket | None] = {"year": "month", "month": "day", "day": None}


def compute_layer2(
    df: pl.DataFrame,
    columns: list[ColumnProfile],
    correlations: Correlations | None,
    layer1: Evidence,
) -> EvidenceLayer2:
    suppressed = {d for g in layer1.near_duplicate_groups for d in g.duplicates}
    nums = [
        c
        for c in columns
        if c.semantic_type == "numeric" and c.std is not None and c.std > 0 and c.name not in suppressed
    ]
    cats = _eligible_cats(columns)
    dts = [c for c in columns if c.semantic_type == "datetime"]
    by_name = {c.name: c for c in columns}
    definitional = _definitional_pairs(layer1)

    group_stats = _group_stats(df, cats, nums)  # cat -> num -> [GroupStat]
    return EvidenceLayer2(
        distributions=_distributions(df, nums),
        group_summaries=_group_summaries(layer1, nums, group_stats, definitional),
        conditional_relationships=_conditional_relationships(df, layer1, nums, cats, correlations, definitional),
        group_time_patterns=_group_time_patterns(df, layer1, by_name, dts, suppressed),
        nonlinear=_nonlinear(df, nums, correlations, definitional),
        change_points=_change_points(df, dts, nums),
        subgroup_anomalies=_subgroup_anomalies(layer1, group_stats, definitional),
    )


# --- helpers ------------------------------------------------------------------


def _finite_expr(name: str) -> pl.Expr:
    col = pl.col(name).cast(pl.Float64)
    return pl.when(col.is_finite()).then(col).otherwise(None).alias(name)


def _q(name: str, q: float, alias: str) -> pl.Expr:
    return pl.col(name).quantile(q, interpolation="linear").alias(alias)


def _definitional_pairs(layer1: Evidence) -> set[frozenset[str]]:
    pairs: set[frozenset[str]] = set()
    for d in layer1.derived_columns:
        for c in d.components:
            pairs.add(frozenset((d.target, c)))
    for g in layer1.near_duplicate_groups:
        members = [g.representative, *g.duplicates]
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                pairs.add(frozenset((a, b)))
    return pairs


def _median(sorted_values: list[float]) -> float:
    n = len(sorted_values)
    mid = n // 2
    if n % 2:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2


# --- distributions --------------------------------------------------------------


def _distributions(df: pl.DataFrame, nums: list[ColumnProfile]) -> list[DistributionSummary]:
    nums = nums[:MAX_DISTRIBUTIONS]
    if not nums:
        return []
    names = [c.name for c in nums]
    data = df.select(_finite_expr(n) for n in names)
    exprs: list[pl.Expr] = []
    for n in names:
        col = pl.col(n)
        exprs += [
            col.count().alias(f"n:{n}"),
            col.min().alias(f"min:{n}"),
            col.max().alias(f"max:{n}"),
            col.skew().alias(f"skew:{n}"),
            col.kurtosis().alias(f"kurt:{n}"),
        ]
        exprs += [_q(n, q, f"p{int(q * 100):02d}:{n}") for q in (0.05, 0.25, 0.5, 0.75, 0.95)]
    stats = data.select(exprs).row(0, named=True)

    windows: dict[str, tuple[float, float]] = {}
    for c in nums:
        n = stats[f"n:{c.name}"]
        if n < MIN_ROWS:
            continue
        rr = c.quality.robust_range if c.quality is not None else None
        lo, hi = (rr.lo, rr.hi) if rr is not None else (stats[f"min:{c.name}"], stats[f"max:{c.name}"])
        if lo is None or hi is None or not hi > lo:
            continue
        windows[c.name] = (float(lo), float(hi))
    if not windows:
        return []

    bin_exprs: list[pl.Expr] = []
    for name, (lo, hi) in windows.items():
        width = (hi - lo) / DIST_BINS
        col = pl.col(name)
        for i in range(DIST_BINS):
            left = lo + i * width
            right = hi if i == DIST_BINS - 1 else lo + (i + 1) * width
            upper = col <= right if i == DIST_BINS - 1 else col < right
            bin_exprs.append(((col >= left) & upper).sum().alias(f"b{i}:{name}"))
    counts = data.select(bin_exprs).row(0, named=True)

    results = []
    for c in nums:
        if c.name not in windows:
            continue
        name = c.name
        n = int(stats[f"n:{name}"])
        lo, hi = windows[name]
        raw = [int(counts[f"b{i}:{name}"]) for i in range(DIST_BINS)]
        inside = sum(raw)
        bins = [round(b / n, 6) for b in raw]
        skew = _finite(stats[f"skew:{name}"])
        kurt = _finite(stats[f"kurt:{name}"])
        bc = _bimodality_coefficient(skew, kurt, n)
        modes = _count_peaks(bins)
        rr = c.quality.robust_range if c.quality is not None else None
        results.append(
            DistributionSummary(
                column=name,
                n=n,
                p05=float(stats[f"p05:{name}"]),
                p25=float(stats[f"p25:{name}"]),
                p50=float(stats[f"p50:{name}"]),
                p75=float(stats[f"p75:{name}"]),
                p95=float(stats[f"p95:{name}"]),
                skewness=skew,
                robust_range=RobustRange(lo=rr.lo, hi=rr.hi) if rr is not None else None,
                bin_lo=lo,
                bin_hi=hi,
                bins=bins,
                outside_ratio=round((n - inside) / n, 6),
                shape=_shape_label(skew),
                bimodality_coefficient=bc,
                modes=modes,
                multimodal_signal=(
                    modes >= 2
                    and bc is not None
                    and bc > BIMODALITY_THRESHOLD
                    and c.unique_count >= MULTIMODAL_MIN_UNIQUE
                ),
            )
        )
    return results


def _shape_label(skew: float | None) -> str:
    if skew is None or abs(skew) < SKEW_SYMMETRIC:
        return "symmetric"
    return "right_skewed" if skew > 0 else "left_skewed"


def _bimodality_coefficient(skew: float | None, kurt: float | None, n: int) -> float | None:
    """Sarle's BC = (g² + 1) / (k + 3(n−1)²/((n−2)(n−3))), k = excess
    kurtosis. Uniform = 0.555; a well-separated two-component mixture is
    above it, a normal (0.33) or a skewed unimodal column below."""
    if skew is None or kurt is None or n < 4:
        return None
    denominator = kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    if denominator <= 0:
        return None
    return round((skew**2 + 1) / denominator, 6)


def _count_peaks(bins: list[float]) -> int:
    """Separated peaks in a coarse histogram: local maxima holding at least
    PEAK_MIN_SHARE, merged when the valley between two of them is not deep
    enough (above VALLEY_MAX_RATIO of the lower peak). A plateau counts once."""
    peaks = []
    for i, b in enumerate(bins):
        left = bins[i - 1] if i > 0 else -1.0
        right = bins[i + 1] if i < len(bins) - 1 else -1.0
        if b > left and b >= right and b >= PEAK_MIN_SHARE:
            peaks.append(i)
    if len(peaks) < 2:
        return len(peaks)
    merged = [peaks[0]]
    for p in peaks[1:]:
        last = merged[-1]
        valley = min(bins[last : p + 1])
        if valley <= VALLEY_MAX_RATIO * min(bins[last], bins[p]):
            merged.append(p)
        elif bins[p] > bins[last]:
            merged[-1] = p  # same hill, keep its higher point
    return len(merged)


# --- group summaries and anomalies ---------------------------------------------


def _group_stats(
    df: pl.DataFrame, cats: list[ColumnProfile], nums: list[ColumnProfile]
) -> dict[str, dict[str, list[GroupStat]]]:
    """One group_by per categorical column for every numeric column: n, mean,
    median, std, q25, q75 per group. Row set per num = cat non-null and num
    finite, exactly like the layer-1 eta-squared."""
    if not cats or not nums:
        return {}
    names = [c.name for c in nums]
    out: dict[str, dict[str, list[GroupStat]]] = {}
    for cat in cats:
        data = df.filter(pl.col(cat.name).is_not_null()).select(
            pl.col(cat.name), *[_finite_expr(n) for n in names]
        )
        aggs: list[pl.Expr] = []
        for n in names:
            col = pl.col(n)
            aggs += [
                col.count().alias(f"n:{n}"),
                col.mean().alias(f"mean:{n}"),
                col.median().alias(f"median:{n}"),
                col.std().alias(f"std:{n}"),
                _q(n, 0.25, f"q25:{n}"),
                _q(n, 0.75, f"q75:{n}"),
            ]
        groups = data.group_by(cat.name, maintain_order=True).agg(aggs)
        rows = sorted(groups.iter_rows(named=True), key=lambda r: str(r[cat.name]))
        per_num: dict[str, list[GroupStat]] = {}
        for n in names:
            stats = []
            for r in rows:
                if not r[f"n:{n}"] or r[f"mean:{n}"] is None:
                    continue
                stats.append(
                    GroupStat(
                        group=str(r[cat.name]),
                        n=int(r[f"n:{n}"]),
                        mean=float(r[f"mean:{n}"]),
                        median=float(r[f"median:{n}"]),
                        std=_finite(r[f"std:{n}"]),
                        q25=float(r[f"q25:{n}"]),
                        q75=float(r[f"q75:{n}"]),
                    )
                )
            per_num[n] = stats
        out[cat.name] = per_num
    return out


def _group_summaries(
    layer1: Evidence,
    nums: list[ColumnProfile],
    group_stats: dict[str, dict[str, list[GroupStat]]],
    definitional: set[frozenset[str]],
) -> list[GroupSummary]:
    usable = {c.name for c in nums}
    effects = [
        e
        for e in sorted(layer1.cat_num, key=lambda e: (-e.eta_squared, e.cat, e.num))
        if e.num in usable
        and e.n_total >= MIN_ROWS
        and e.eta_squared >= GROUP_SUMMARY_MIN_ETA
        and frozenset((e.cat, e.num)) not in definitional
    ]
    chosen = effects[:GROUP_SUMMARY_TOP]
    covered = {e.num for e in chosen}
    for c in nums:  # every numeric column gets at least its best grouping
        if len(chosen) >= MAX_GROUP_SUMMARIES:
            break
        if c.name in covered:
            continue
        best = next((e for e in effects if e.num == c.name), None)
        if best is not None and best.eta_squared >= GROUP_SUMMARY_MIN_ETA:
            chosen.append(best)
            covered.add(c.name)
    results = []
    for e in chosen:
        stats = group_stats.get(e.cat, {}).get(e.num, [])
        if len(stats) < 2:
            continue
        pooled = _pooled_std(stats)
        means = [g.mean for g in stats]
        top = max(stats, key=lambda g: (g.mean, g.group))
        bottom = min(stats, key=lambda g: (g.mean, g.group))
        results.append(
            GroupSummary(
                cat=e.cat,
                num=e.num,
                n_total=sum(g.n for g in stats),
                eta_squared=e.eta_squared,
                groups=stats[:MAX_GROUPS_LISTED],
                pooled_std=pooled,
                max_diff_sd=round((max(means) - min(means)) / pooled, 6) if pooled else None,
                top_group=top.group,
                bottom_group=bottom.group,
            )
        )
    return results


def _pooled_std(stats: list[GroupStat]) -> float | None:
    num = sum((g.n - 1) * g.std**2 for g in stats if g.std is not None and g.n > 1)
    den = sum(g.n - 1 for g in stats if g.std is not None and g.n > 1)
    if den <= 0:
        return None
    pooled = math.sqrt(num / den)
    return pooled if pooled > 0 else None


def _subgroup_anomalies(
    layer1: Evidence,
    group_stats: dict[str, dict[str, list[GroupStat]]],
    definitional: set[frozenset[str]],
) -> list[SubgroupAnomaly]:
    results = []
    for e in layer1.cat_num:
        if frozenset((e.cat, e.num)) in definitional:
            continue
        stats = [g for g in group_stats.get(e.cat, {}).get(e.num, []) if g.n >= ANOMALY_MIN_ROWS]
        if len(stats) < ANOMALY_MIN_GROUPS:
            continue
        pooled = _pooled_std(stats)
        if pooled is None:
            continue
        means = sorted(g.mean for g in stats)
        med = _median(means)
        mad = _median(sorted(abs(m - med) for m in means))
        if mad <= 0:
            continue
        for g in stats:
            z = (g.mean - med) / (MAD_SCALE * mad)
            diff_sd = (g.mean - med) / pooled
            if abs(z) >= ANOMALY_Z and abs(diff_sd) >= ANOMALY_MIN_DIFF_SD:
                results.append(
                    SubgroupAnomaly(
                        cat=e.cat,
                        num=e.num,
                        group=g.group,
                        n=g.n,
                        mean=g.mean,
                        others_median=med,
                        robust_z=round(z, 4),
                        diff_sd=round(diff_sd, 4),
                    )
                )
    results.sort(key=lambda a: (-abs(a.robust_z), a.cat, a.num, a.group))
    return results[:MAX_ANOMALIES]


# --- conditional relationships ---------------------------------------------------


def _conditional_relationships(
    df: pl.DataFrame,
    layer1: Evidence,
    nums: list[ColumnProfile],
    cats: list[ColumnProfile],
    correlations: Correlations | None,
    definitional: set[frozenset[str]],
) -> list[ConditionalRelationship]:
    """Per-group correlation AND slope of numeric pairs inside the groups of
    the top grouping columns. Unlike layer 1's slope heterogeneity this does
    NOT require the pair to correlate overall: a relationship that flips
    sign between groups cancels to r ≈ 0 and is exactly what must surface.
    Batched as ONE group_by per grouping column with closed-form sums, so a
    pair's row set is exactly the rows where both values are finite."""
    lo, hi = TIME_PATTERN_GROUP_RANGE
    best_eta: dict[str, float] = {}
    for e in layer1.cat_num:
        best_eta[e.cat] = max(best_eta.get(e.cat, 0.0), e.eta_squared)
    group_cols = sorted(
        (c.name for c in cats if lo <= (c.n_categories or 0) <= hi),
        key=lambda c: (-best_eta.get(c, 0.0), c),
    )[:CONDITIONAL_TOP_GROUPS]
    # numeric columns: the ones in layer 1's slope entries, then by strongest
    # overall correlation, then column order — capped
    rank: dict[str, float] = {}
    for key, value in _pearson_index(correlations).items():
        for name in key:
            rank[name] = max(rank.get(name, 0.0), abs(value))
    ordered = sorted((c.name for c in nums), key=lambda n: (-rank.get(n, 0.0), n))
    chosen: list[str] = []
    for h in layer1.slope_heterogeneity:
        for name in (h.x, h.y):
            if name in ordered and name not in chosen:
                chosen.append(name)
    for name in ordered:
        if len(chosen) >= CONDITIONAL_TOP_NUMS:
            break
        if name not in chosen:
            chosen.append(name)
    chosen = chosen[:CONDITIONAL_TOP_NUMS]
    pairs = [
        (a, b)
        for i, a in enumerate(chosen)
        for b in chosen[i + 1 :]
        if frozenset((a, b)) not in definitional
    ]
    if not group_cols or not pairs:
        return []

    results = []
    for group in group_cols:
        names = sorted({n for pair in pairs for n in pair})
        data = df.filter(pl.col(group).is_not_null()).select(
            pl.col(group), *[_finite_expr(n) for n in names]
        )
        aggs: list[pl.Expr] = []
        for x, y in pairs:
            tag = f"{x}|{y}"
            both = pl.col(x).is_not_null() & pl.col(y).is_not_null()
            xm = pl.when(both).then(pl.col(x)).otherwise(None)
            ym = pl.when(both).then(pl.col(y)).otherwise(None)
            aggs += [
                both.sum().alias(f"n:{tag}"),
                xm.sum().alias(f"sx:{tag}"),
                ym.sum().alias(f"sy:{tag}"),
                (xm * xm).sum().alias(f"sxx:{tag}"),
                (ym * ym).sum().alias(f"syy:{tag}"),
                (xm * ym).sum().alias(f"sxy:{tag}"),
            ]
        table = data.group_by(group, maintain_order=True).agg(aggs)
        rows = sorted(table.iter_rows(named=True), key=lambda r: str(r[group]))
        for x, y in pairs:
            tag = f"{x}|{y}"
            entries = []
            valid = []
            for r in rows:
                n = int(r[f"n:{tag}"])
                corr = slope = None
                if n >= MIN_SLOPE_GROUP_ROWS:
                    cxx = n * r[f"sxx:{tag}"] - r[f"sx:{tag}"] ** 2
                    cyy = n * r[f"syy:{tag}"] - r[f"sy:{tag}"] ** 2
                    cxy = n * r[f"sxy:{tag}"] - r[f"sx:{tag}"] * r[f"sy:{tag}"]
                    if cxx > 0 and cyy > 0:
                        corr = _finite(cxy / math.sqrt(cxx * cyy))
                        slope = _finite(cxy / cxx)
                if n == 0:
                    continue
                entries.append(GroupCorrelation(group=str(r[group]), n=n, corr=corr, slope=slope))
                if corr is not None:
                    valid.append((corr, n))
            if len(valid) < 2:
                continue
            corrs = [v for v, _ in valid]
            results.append(
                ConditionalRelationship(
                    x=x,
                    y=y,
                    group=group,
                    groups=entries,
                    corr_spread=round(max(corrs) - min(corrs), 6),
                    n_min=min(n for _, n in valid),
                )
            )
    results.sort(key=lambda r: (-r.corr_spread, r.x, r.y, r.group))
    return results[:MAX_CONDITIONAL]


# --- group time patterns ------------------------------------------------------------


def _group_time_patterns(
    df: pl.DataFrame,
    layer1: Evidence,
    by_name: dict[str, ColumnProfile],
    dts: list[ColumnProfile],
    suppressed: set[str],
) -> list[GroupTimePattern]:
    lo, hi = TIME_PATTERN_GROUP_RANGE
    eta_by_num: dict[str, list[tuple[float, str]]] = {}
    for e in layer1.cat_num:
        cat = by_name.get(e.cat)
        if cat is None or not (lo <= (cat.n_categories or 0) <= hi):
            continue
        eta_by_num.setdefault(e.num, []).append((-e.eta_squared, e.cat))
    time_by_dt: dict[str, list[tuple[float, str]]] = {}
    for t in layer1.time_effects:
        if t.num in suppressed:
            continue
        time_by_dt.setdefault(t.datetime_col, []).append((-t.eta_squared, t.num))

    candidates: list[tuple[str, str, str]] = []
    for dt in dts:
        nums = sorted(time_by_dt.get(dt.name, []))[:TIME_PATTERN_TOP_NUMS]
        for _, num in nums:
            for _, cat in sorted(eta_by_num.get(num, []))[:TIME_PATTERN_TOP_CATS]:
                candidates.append((dt.name, num, cat))
    candidates = candidates[:MAX_GROUP_TIME_PATTERNS]
    if not candidates:
        return []

    # the span-adaptive bucket (not layer 1's, which coarsens for degrees of
    # freedom), coarsened while the series is too long AND the coarser one
    # still keeps a usable number of points; otherwise the finer series is
    # thinned evenly (60 days -> 24 daily points, not 2 monthly ones)
    bucket_for: dict[str, TimeBucket] = {}
    for dt_name in {c[0] for c in candidates}:
        span = _datetime_span_days(df[dt_name])
        current: TimeBucket = span_bucket(span if span is not None else 0.0)
        counts: dict[TimeBucket, int] = {}
        for bucket in ("day", "month", "year"):
            counts[bucket] = int(df.select(pl.col(dt_name).dt.truncate(_BUCKET_TRUNC[bucket]).n_unique()).item())
        while counts[current] > MAX_TIME_POINTS:
            coarser = _COARSER[current]
            if coarser is None or counts[coarser] < MAX_TIME_POINTS // 2:
                break
            current = coarser
        bucket_for[dt_name] = current

    results = []
    for dt_name, num, cat in candidates:
        pattern = _one_group_time_pattern(df, dt_name, num, cat, bucket_for[dt_name])
        if pattern is not None:
            results.append(pattern)
    return results


def _one_group_time_pattern(
    df: pl.DataFrame, dt: str, num: str, cat: str, bucket: TimeBucket
) -> GroupTimePattern | None:
    base = df.filter(pl.col(dt).is_not_null() & pl.col(cat).is_not_null()).select(
        pl.col(dt).dt.truncate(_BUCKET_TRUNC[bucket]).alias(_BIN), pl.col(cat), _finite_expr(num)
    )
    if base.height < MIN_ROWS:
        return None
    agg = (
        base.group_by(_BIN, cat, maintain_order=True)
        .agg(pl.col(num).count().alias("n"), pl.col(num).mean().alias("mean"))
        .filter(pl.col("n") > 0)
        .sort(_BIN)
    )
    buckets = sorted(agg[_BIN].unique().to_list())
    keep = set(_thin_indices(len(buckets), MAX_TIME_POINTS))
    kept = {b for i, b in enumerate(buckets) if i in keep}
    series: dict[str, list[TimeSeriesPoint]] = {}
    group_rows: dict[str, int] = {}
    for r in agg.iter_rows(named=True):
        key = str(r[cat])
        group_rows[key] = group_rows.get(key, 0) + int(r["n"])
        if r[_BIN] not in kept or r["mean"] is None:
            continue
        series.setdefault(key, []).append(
            TimeSeriesPoint(bucket=r[_BIN].isoformat(), mean=float(r["mean"]), n=int(r["n"]))
        )
    # a group with too few rows, or whose series is a couple of points, is
    # noise that would only pad the prompt (stage 17.1b)
    series = {
        key: points
        for key, points in series.items()
        if group_rows.get(key, 0) >= MIN_ROWS and len(points) >= MIN_TIME_POINTS
    }
    if len(series) < 2:
        return None
    return GroupTimePattern(
        datetime_col=dt,
        num=num,
        group=cat,
        bucket=bucket,
        series=dict(sorted(series.items())),
        n_total=sum(group_rows[key] for key in series),
    )


def _thin_indices(count: int, limit: int) -> list[int]:
    if count <= limit:
        return list(range(count))
    return sorted({round(i * (count - 1) / (limit - 1)) for i in range(limit)})


# --- nonlinear signals ---------------------------------------------------------------


def _nonlinear(
    df: pl.DataFrame,
    nums: list[ColumnProfile],
    correlations: Correlations | None,
    definitional: set[frozenset[str]],
) -> list[NonlinearSignal]:
    """For every x: equal-frequency bins by rank, then ONE group_by giving
    n / mean / within-SS of every y per bin — the adjusted eta-squared of y
    over the bins and the bin means come out of the same execution. Pearson
    r² is the profile's pairwise-complete correlation."""
    names = [c.name for c in nums[:MAX_NONLINEAR_COLUMNS]]
    if len(names) < 2:
        return []
    data = df.select(_finite_expr(n) for n in names)
    counts = data.select(pl.col(n).count().alias(n) for n in names).row(0, named=True)
    uniques = data.select(pl.col(n).n_unique().alias(n) for n in names).row(0, named=True)
    pearson = _pearson_index(correlations)

    # one lazy query per x (rank -> equal-frequency bin -> one group_by), all
    # collected in a single call so polars runs them in parallel (17.1b)
    queries: list[tuple[str, list[str], pl.LazyFrame]] = []
    lazy = data.lazy()
    for x in names:
        if counts[x] < NONLINEAR_MIN_ROWS or uniques[x] < NONLINEAR_MIN_X_UNIQUE:
            continue
        ys = [y for y in names if y != x and frozenset((x, y)) not in definitional]
        if not ys:
            continue
        xs = lazy.filter(pl.col(x).is_not_null()).select(x, *ys)
        binned = xs.with_columns(
            ((pl.col(x).rank(method="average") - 1) * NONLINEAR_BINS / counts[x])
            .floor()
            .clip(0, NONLINEAR_BINS - 1)
            .cast(pl.Int32)
            .alias(_BIN)
        )
        aggs: list[pl.Expr] = [pl.col(x).min().alias("lo"), pl.col(x).max().alias("hi")]
        for y in ys:
            col = pl.col(y)
            aggs += [
                col.count().alias(f"n:{y}"),
                col.mean().alias(f"mean:{y}"),
                ((col - col.mean()) ** 2).sum().alias(f"ssw:{y}"),
            ]
        queries.append((x, ys, binned.group_by(_BIN).agg(aggs).sort(_BIN)))
    if not queries:
        return []
    collected = pl.collect_all([q for _, _, q in queries])

    results = []
    for (x, ys, _), per_bin in zip(queries, collected):
        rows = list(per_bin.iter_rows(named=True))
        for y in ys:
            bins = [
                NonlinearBin(lo=float(r["lo"]), hi=float(r["hi"]), n=int(r[f"n:{y}"]), y_mean=float(r[f"mean:{y}"]))
                for r in rows
                if r[f"n:{y}"] and r[f"mean:{y}"] is not None
            ]
            eta = _binned_eta(bins, [float(r[f"ssw:{y}"]) for r in rows if r[f"n:{y}"] and r[f"mean:{y}"] is not None])
            if eta is None or len(bins) < 3:
                continue
            n_total = sum(b.n for b in bins)
            if n_total < NONLINEAR_MIN_ROWS:
                continue
            r = pearson.get(frozenset((x, y)))
            r2 = None if r is None else round(r * r, 6)
            gap = max(eta - (r2 or 0.0), 0.0)
            means = [b.y_mean for b in bins]
            monotone = _is_monotone(means)
            results.append(
                NonlinearSignal(
                    x=x,
                    y=y,
                    n=n_total,
                    bins=bins,
                    binned_eta2=round(eta, 6),
                    r2_pearson=r2,
                    nonlinear_gap=round(gap, 6),
                    monotone=monotone,
                    shape=_curve_shape(eta, gap, monotone, means),
                )
            )
    # one entry per unordered pair: the direction whose bins explain more
    best: dict[frozenset[str], NonlinearSignal] = {}
    for s in results:
        if s.shape == "flat":
            continue
        key = frozenset((s.x, s.y))
        current = best.get(key)
        if current is None or (s.nonlinear_gap, s.binned_eta2) > (current.nonlinear_gap, current.binned_eta2):
            best[key] = s
    kept = sorted(best.values(), key=lambda s: (-s.nonlinear_gap, -s.binned_eta2, s.x, s.y))
    return kept[:MAX_NONLINEAR]


def _pearson_index(correlations: Correlations | None) -> dict[frozenset[str], float]:
    index: dict[frozenset[str], float] = {}
    if correlations is None:
        return index
    cols = correlations.columns
    for i, a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            value = correlations.matrix[i][j]
            if value is not None:
                index[frozenset((a, cols[j]))] = value
    return index


def _binned_eta(bins: list[NonlinearBin], ssw: list[float]) -> float | None:
    """Adjusted eta-squared of y over the bins from per-bin n / mean /
    within-SS: SS_total = Σ(ssw_b + n_b (mean_b − grand)²)."""
    n = sum(b.n for b in bins)
    k = len(bins)
    if k < 2 or n - k < 2:
        return None
    grand = sum(b.n * b.y_mean for b in bins) / n
    ss_within = sum(ssw)
    ss_total = ss_within + sum(b.n * (b.y_mean - grand) ** 2 for b in bins)
    if ss_total <= 0:
        return None
    return max(1 - (ss_within / (n - k)) / (ss_total / (n - 1)), 0.0)


def _is_monotone(means: list[float]) -> bool:
    """Monotone up to noise: a step against the trend smaller than
    MONOTONE_TOLERANCE of the series range does not break monotonicity."""
    if len(means) < 2:
        return True
    slack = MONOTONE_TOLERANCE * (max(means) - min(means))
    diffs = [b - a for a, b in zip(means, means[1:])]
    return all(d >= -slack for d in diffs) or all(d <= slack for d in diffs)


def _count_curve_peaks(means: list[float]) -> int:
    """Separated INTERIOR peaks of a bin-mean curve (the endpoints of a U or
    a monotone curve are never peaks): means normalised to [0, 1], a peak is
    a local maximum at least CURVE_PEAK_MIN above the minimum, and two peaks
    stay separate only when the valley between them drops to at most
    CURVE_VALLEY_MAX_RATIO of the lower one."""
    if len(means) < 3:
        return 0
    lo, hi = min(means), max(means)
    if hi <= lo:
        return 0
    norm = [(m - lo) / (hi - lo) for m in means]
    peaks = [
        i
        for i in range(1, len(norm) - 1)
        if norm[i] > norm[i - 1] and norm[i] >= norm[i + 1] and norm[i] >= CURVE_PEAK_MIN
    ]
    if len(peaks) < 2:
        return len(peaks)
    merged = [peaks[0]]
    for p in peaks[1:]:
        last = merged[-1]
        valley = min(norm[last : p + 1])
        if valley <= CURVE_VALLEY_MAX_RATIO * min(norm[last], norm[p]):
            merged.append(p)
        elif norm[p] > norm[last]:
            merged[-1] = p
    return len(merged)


def _curve_shape(eta: float, gap: float, monotone: bool, means: list[float]) -> str:
    if eta < SHAPE_FLAT_ETA:
        return "flat"
    if gap < SHAPE_LINEAR_GAP:
        return "linear"
    if monotone:
        return "monotone_nonlinear"
    if _count_curve_peaks(means) >= 2:
        return "multi_peak"
    curvature, vertex = _quadratic_fit(means)
    interior = 0 < vertex < len(means) - 1
    if curvature > 0 and interior:
        return "u_shape"
    if curvature < 0 and interior:
        return "inverted_u"
    return "other"


def _quadratic_fit(means: list[float]) -> tuple[float, float]:
    """Least-squares y = a + b·i + c·i² over bin index i; returns (c, vertex)."""
    n = len(means)
    xs = [float(i) for i in range(n)]
    s0, s1, s2, s3, s4 = n, sum(xs), sum(v**2 for v in xs), sum(v**3 for v in xs), sum(v**4 for v in xs)
    t0, t1, t2 = sum(means), sum(v * m for v, m in zip(xs, means)), sum(v * v * m for v, m in zip(xs, means))
    # solve the 3x3 normal equations by Cramer's rule
    det = s0 * (s2 * s4 - s3 * s3) - s1 * (s1 * s4 - s2 * s3) + s2 * (s1 * s3 - s2 * s2)
    if abs(det) < 1e-12:
        return 0.0, -1.0
    det_b = s0 * (t1 * s4 - s3 * t2) - t0 * (s1 * s4 - s2 * s3) + s2 * (s1 * t2 - t1 * s2)
    det_c = s0 * (s2 * t2 - t1 * s3) - s1 * (s1 * t2 - t1 * s2) + t0 * (s1 * s3 - s2 * s2)
    b, c = det_b / det, det_c / det
    if c == 0:
        return 0.0, -1.0
    return c, -b / (2 * c)


# --- change points ---------------------------------------------------------------------


def _change_points(
    df: pl.DataFrame, dts: list[ColumnProfile], nums: list[ColumnProfile]
) -> list[ChangePoint]:
    nums = nums[:MAX_NONLINEAR_COLUMNS]
    names = [c.name for c in nums]
    if not names:
        return []
    column_std = {c.name: c.std for c in nums}
    results = []
    for dt in dts:
        span = _datetime_span_days(df[dt.name])
        if span is None:
            continue
        bucket = _choose_change_bucket(df[dt.name], span_bucket(span))
        if bucket is None:
            continue
        base = df.filter(pl.col(dt.name).is_not_null()).select(
            pl.col(dt.name).dt.truncate(_BUCKET_TRUNC[bucket]).alias(_BIN),
            *[_finite_expr(n) for n in names],
        )
        aggs: list[pl.Expr] = []
        for n in names:
            aggs += [pl.col(n).count().alias(f"n:{n}"), pl.col(n).mean().alias(f"mean:{n}")]
        series = base.group_by(_BIN).agg(aggs).sort(_BIN)
        rows = list(series.iter_rows(named=True))
        for n in names:
            points = [
                (r[_BIN], float(r[f"mean:{n}"]), int(r[f"n:{n}"]))
                for r in rows
                if r[f"n:{n}"] and r[f"mean:{n}"] is not None
            ]
            found = _best_split(points)
            if found is None:
                continue
            k, before, after, effect = found
            std = column_std[n] or 0.0
            diff_sd = abs(after - before) / std if std > 0 else 0.0
            flagged = effect >= CHANGE_FLAG_EFFECT and diff_sd >= CHANGE_FLAG_DIFF_SD
            strength = "strong" if flagged else "weak" if effect >= CHANGE_FLAG_EFFECT else "none"
            results.append(
                ChangePoint(
                    datetime_col=dt.name,
                    num=n,
                    bucket=bucket,
                    n_buckets=len(points),
                    change_at=points[k][0].isoformat(),
                    before_mean=before,
                    after_mean=after,
                    effect_size=round(effect, 6),
                    diff_sd=round(diff_sd, 6),
                    n_before=sum(p[2] for p in points[:k]),
                    n_after=sum(p[2] for p in points[k:]),
                    flagged=flagged,
                    strength=strength,
                )
            )
    results.sort(key=lambda c: (-c.effect_size, c.datetime_col, c.num))
    return results[:MAX_CHANGE_POINTS]


def _choose_change_bucket(s: pl.Series, bucket: TimeBucket) -> TimeBucket | None:
    """Span-adaptive bucket refined until both segments can hold
    MIN_SEGMENT_BUCKETS buckets, coarsened when there would be too many."""
    current: TimeBucket | None = bucket
    while current is not None:
        n_buckets = s.dt.truncate(_BUCKET_TRUNC[current]).n_unique()
        if n_buckets > MAX_CHANGE_BUCKETS:
            coarser = _COARSER[current]
            if coarser is None:
                return None
            current = coarser
            continue
        if n_buckets >= 2 * MIN_SEGMENT_BUCKETS:
            return current
        current = _FINER[current]
    return None


def _best_split(points: list[tuple[Any, float, int]]) -> tuple[int, float, float, float] | None:
    """Split index k (first bucket of the second segment) maximising the
    between-segment sum of squares of the bucket means (each bucket weighted
    equally, so a busy month cannot dominate), with effect size in pooled
    within-segment SDs of the bucket means."""
    means = [p[1] for p in points]
    b = len(means)
    if b < 2 * MIN_SEGMENT_BUCKETS:
        return None
    prefix = [0.0]
    prefix_sq = [0.0]
    for m in means:
        prefix.append(prefix[-1] + m)
        prefix_sq.append(prefix_sq[-1] + m * m)
    grand = prefix[-1] / b
    best: tuple[float, int] | None = None
    for k in range(MIN_SEGMENT_BUCKETS, b - MIN_SEGMENT_BUCKETS + 1):
        m1 = prefix[k] / k
        m2 = (prefix[-1] - prefix[k]) / (b - k)
        between = k * (m1 - grand) ** 2 + (b - k) * (m2 - grand) ** 2
        if best is None or between > best[0]:
            best = (between, k)
    assert best is not None
    k = best[1]
    m1 = prefix[k] / k
    m2 = (prefix[-1] - prefix[k]) / (b - k)
    ss1 = prefix_sq[k] - k * m1 * m1
    ss2 = (prefix_sq[-1] - prefix_sq[k]) - (b - k) * m2 * m2
    pooled = math.sqrt(max(ss1 + ss2, 0.0) / (b - 2))
    if pooled <= 0:
        return None
    return k, m1, m2, abs(m2 - m1) / pooled
