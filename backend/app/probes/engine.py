"""Targeted validation probes (stage 17.2): deterministic backend checks the
LLM workflow can request when the evidence tables do not already answer a
hypothesis.

Design rules:
- Every probe is a pure function of (profiled sample, profile, request);
  the statistics reuse the layer-1 / layer-2 estimators (adjusted
  eta-squared, unweighted-means interaction share, per-group closed-form
  correlations, equal-frequency binning, single change point) so a probe
  verdict never disagrees with the evidence tables on the same columns.
- Thresholds are the same numbers the rule engine and layer 2 use; the
  verdict is effect-size first, then capped by the same confidence chain the
  recommendations pass through (stage 9 `assess` on the suggested chart), so
  a tiny sample can never earn "pass".
- The row set is the profiler's seeded sample with the profile's casts
  replayed (identical to what layer 1/2 saw); no raw rows leave this module —
  `evidence` carries per-group / per-bin / per-segment summaries only.
- Bounded: at most `max_probes` per call, duplicates collapsed, results cached
  per (dataset, profile version, request) with a per-key lock.
"""

import math
import threading
from itertools import combinations
from typing import Any

import polars as pl

from ..charts.confidence import TOP_CONFIDENCE_FLOOR, assess
from ..charts.rules import choose_time_granularity, slope_spread_threshold
from ..charts.spec import ChartSpec, validate_spec
from ..profiling.evidence import (
    MIN_SLOPE_GROUP_ROWS,
    _BUCKET_TRUNC,
    _COARSER,
    _datetime_span_days,
    _finite,
    _interaction_effect,
    adjusted_eta_squared_many,
    span_bucket,
)
from ..profiling.layer2 import (
    CHANGE_FLAG_DIFF_SD,
    CHANGE_FLAG_EFFECT,
    MAX_TIME_POINTS,
    NONLINEAR_BINS,
    NONLINEAR_MIN_ROWS,
    NONLINEAR_MIN_X_UNIQUE,
    _best_split,
    _binned_eta,
    _choose_change_bucket,
    _curve_shape,
    _finite_expr,
    _is_monotone,
    _pooled_std,
    _q,
    _thin_indices,
)
from ..profiling.models import PROFILE_VERSION, DatasetProfile, GroupStat, NonlinearBin, TimeBucket
from ..profiling.types import SAMPLE_SEED, apply_semantic_casts
from .schemas import ProbeOutcome, ProbeRejected, ProbeRequest, ProbeResult, validate_request

# --- caps and thresholds (the only numbers in this layer) --------------------

MAX_PROBES = 5  # per run_probes call; the rest is rejected with reason "cap"
MIN_ROWS = 30  # below this a probe cannot "pass" (same floor as layer 2)

# effect thresholds on the probe's effect_size: (pass, weak); the numbers
# are the ones the rule engine / layer 2 already use for the same statistic
THRESHOLDS: dict[str, dict[str, float]] = {
    "group_difference": {"pass": 0.10, "weak": 0.03},  # adjusted eta-squared (LINE_GROUP_MAIN_EFFECT_MIN)
    "distribution_difference": {"pass": 0.25, "weak": 0.12},  # max two-sample KS statistic
    "nonlinear_relationship": {"pass": 0.15, "weak": 0.08},  # nonlinear_gap (NONLINEAR_CORR_GAP)
    "time_pattern": {"pass": 0.10, "weak": 0.03},  # time eta-squared, or a flagged change point
    "interaction": {"pass": 0.10, "weak": 0.03},  # interaction share of variance
    # grouped_relationship / slope_difference: the pass threshold is dynamic
    # (slope_spread_threshold(n_min)); weak is half of it — filled per result
}
NONLINEAR_PASS_MIN_ETA = 0.15  # a gap only counts when the bins explain this much
KS_MAX_GROUPS = 8  # pairwise KS over the largest groups only
_BIN = "__probe_bin"


# --- cache -------------------------------------------------------------------


class ProbeCache:
    """Process-local (dataset, profile version, request) -> result, with a
    per-key lock so concurrent requests for the same probe compute once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key_locks: dict[tuple, threading.Lock] = {}
        self._results: dict[tuple, ProbeResult] = {}

    def get_or_compute(self, key: tuple, compute) -> tuple[ProbeResult, bool]:
        cached = self._results.get(key)
        if cached is not None:
            return cached, True
        with self._lock:
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:
            cached = self._results.get(key)
            if cached is not None:
                return cached, True
            result = compute()
            self._results[key] = result
            return result, False

    def clear(self) -> None:
        with self._lock:
            self._results.clear()
            self._key_locks.clear()


_default_cache = ProbeCache()


# --- entry point ----------------------------------------------------------------


def prepare_frame(df: pl.DataFrame, profile: DatasetProfile) -> pl.DataFrame:
    """The profiler's row set: the seeded sample (when the table was
    sampled) with the profile's casts replayed."""
    sample = df
    if profile.sampled and 0 < profile.profiled_rows < df.height:
        sample = df.sample(profile.profiled_rows, seed=SAMPLE_SEED)
    return apply_semantic_casts(sample, profile)


def run_probes(
    df: pl.DataFrame,
    profile: DatasetProfile,
    requests: list[ProbeRequest],
    max_probes: int = MAX_PROBES,
    cache: ProbeCache | None = None,
) -> list[ProbeOutcome]:
    """One outcome per request, in request order. Invalid requests are
    rejected before any computation; duplicates (same type and columns)
    return the first outcome; requests beyond `max_probes` executable ones are
    rejected with reason "cap"."""
    cache = cache if cache is not None else _default_cache
    outcomes: list[ProbeOutcome | None] = [None] * len(requests)
    seen: dict[tuple, int] = {}
    executable: list[int] = []
    for i, req in enumerate(requests):
        reason = validate_request(req, profile)
        if reason is not None:
            outcomes[i] = ProbeRejected(type=req.type, columns=req.columns, reason=reason)
            continue
        key = req.key()
        if key in seen:
            outcomes[i] = ProbeRejected(type=req.type, columns=req.columns, reason="duplicate")
            continue
        seen[key] = i
        if len(executable) >= max_probes:
            outcomes[i] = ProbeRejected(
                type=req.type, columns=req.columns, reason=f"cap: at most {max_probes} probes per run"
            )
            continue
        executable.append(i)

    frame: pl.DataFrame | None = None
    for i in executable:
        req = requests[i]
        cache_key = (profile.dataset_id, PROFILE_VERSION, *req.key())

        def compute(req=req):
            nonlocal frame
            if frame is None:
                frame = prepare_frame(df, profile)
            return _run_one(frame, profile, req)

        result, hit = cache.get_or_compute(cache_key, compute)
        outcomes[i] = result.model_copy(update={"cached": hit})
    return [o for o in outcomes if o is not None]


def _run_one(frame: pl.DataFrame, profile: DatasetProfile, req: ProbeRequest) -> ProbeResult:
    runner = _RUNNERS[req.type]
    return runner(frame, profile, req)


# --- shared pieces ------------------------------------------------------------


def _finish(
    profile: DatasetProfile,
    req: ProbeRequest,
    *,
    effect_size: float,
    effect_label: str,
    n: int,
    n_min_group: int | None,
    thresholds: dict[str, float],
    evidence: dict[str, Any],
    chart: ChartSpec,
    notes: list[str],
    effect_verdict: str | None = None,
) -> ProbeResult:
    """Effect-size verdict, then the confidence cap (stage 9 chain on the
    suggested chart), then the sample-size floor."""
    errors = validate_spec(chart, profile)
    chart_out: ChartSpec | None = chart
    if errors:
        chart_out = None
        notes = notes + [f"suggested chart rejected: {'; '.join(errors)}"]
    assessment = assess(chart, profile)
    confidence = assessment.confidence

    if effect_verdict is None:
        if effect_size >= thresholds["pass"]:
            effect_verdict = "pass"
        elif effect_size >= thresholds["weak"]:
            effect_verdict = "weak"
        else:
            effect_verdict = "fail"
    verdict = effect_verdict
    if verdict == "pass":
        if confidence.overall < TOP_CONFIDENCE_FLOOR:
            verdict = "weak"
            notes = notes + [f"capped at weak: confidence {confidence.overall:.2f} < {TOP_CONFIDENCE_FLOOR}"]
        elif n < MIN_ROWS:
            verdict = "weak"
            notes = notes + [f"capped at weak: only {n} rows"]
        elif n_min_group is not None and n_min_group < MIN_ROWS and assessment.tier_cap is not None:
            verdict = "weak"
            notes = notes + [f"capped at weak: smallest group has {n_min_group} rows"]
    return ProbeResult(
        type=req.type,
        columns=dict(req.columns),
        effect_size=round(float(effect_size), 6),
        effect_label=effect_label,
        n=int(n),
        n_min_group=n_min_group,
        confidence=confidence,
        verdict=verdict,  # type: ignore[arg-type]
        thresholds=thresholds,
        evidence=evidence,
        chart=chart_out,
        notes=notes,
    )


def _title(req: ProbeRequest) -> str:
    c = req.columns
    return {
        "group_difference": f"Mean {c.get('target')} by {c.get('group')}",
        "distribution_difference": f"{c.get('target')} distribution by {c.get('group')}",
        "grouped_relationship": f"{c.get('y')} vs {c.get('x')} by {c.get('group')}",
        "slope_difference": f"{c.get('y')} vs {c.get('x')} by {c.get('group')}",
        "nonlinear_relationship": f"{c.get('y')} vs {c.get('x')}",
        "time_pattern": f"{c.get('target')} over {c.get('time')}"
        + (f" by {c['group']}" if c.get("group") else ""),
        "interaction": f"Mean {c.get('target')} by {c.get('factor1')} and {c.get('factor2')}",
    }[req.type]


def _group_table(frame: pl.DataFrame, group: str, num: str) -> list[GroupStat]:
    """Per-group n / mean / median / std / q25 / q75 (layer-2 GroupStat form)."""
    data = frame.filter(pl.col(group).is_not_null()).select(pl.col(group), _finite_expr(num))
    col = pl.col(num)
    table = data.group_by(group, maintain_order=True).agg(
        col.count().alias("n"),
        col.mean().alias("mean"),
        col.median().alias("median"),
        col.std().alias("std"),
        _q(num, 0.25, "q25"),
        _q(num, 0.75, "q75"),
    )
    rows = sorted(table.iter_rows(named=True), key=lambda r: str(r[group]))
    return [
        GroupStat(
            group=str(r[group]),
            n=int(r["n"]),
            mean=float(r["mean"]),
            median=float(r["median"]),
            std=_finite(r["std"]),
            q25=float(r["q25"]),
            q75=float(r["q75"]),
        )
        for r in rows
        if r["n"] and r["mean"] is not None
    ]


# --- group_difference ----------------------------------------------------------


def _group_difference(frame: pl.DataFrame, profile: DatasetProfile, req: ProbeRequest) -> ProbeResult:
    group, target = req.columns["group"], req.columns["target"]
    result = adjusted_eta_squared_many(frame, group, [target])[target]
    stats = _group_table(frame, group, target)
    pooled = _pooled_std(stats)
    means = [g.mean for g in stats]
    evidence: dict[str, Any] = {
        "groups": [g.model_dump() for g in stats],
        "eta_squared": None if result is None else round(result.eta_squared, 6),
        "n_groups": None if result is None else result.n_groups,
        "pooled_std": pooled,
        "max_diff_sd": round((max(means) - min(means)) / pooled, 6) if pooled and means else None,
        "top_group": max(stats, key=lambda g: (g.mean, g.group)).group if stats else None,
        "bottom_group": min(stats, key=lambda g: (g.mean, g.group)).group if stats else None,
    }
    chart = ChartSpec(title=_title(req), type="bar", x=group, y=target, aggregation="mean")
    notes: list[str] = []
    if result is None:
        notes.append("effect undefined: fewer than two groups with rows, or no within-group variation")
    return _finish(
        profile,
        req,
        effect_size=0.0 if result is None else result.eta_squared,
        effect_label="adjusted eta-squared of target across groups",
        n=0 if result is None else result.n_total,
        n_min_group=None if result is None else result.n_min,
        thresholds=THRESHOLDS["group_difference"],
        evidence=evidence,
        chart=chart,
        notes=notes,
        effect_verdict="fail" if result is None else None,
    )


# --- grouped_relationship / slope_difference ----------------------------------


def _group_correlations(frame: pl.DataFrame, x: str, y: str, group: str) -> list[dict[str, Any]]:
    """Per-group Pearson r and OLS slope of y on x from closed-form sums in a
    single group_by (the layer-2 conditional-relationship estimator)."""
    data = frame.filter(pl.col(group).is_not_null()).select(pl.col(group), _finite_expr(x), _finite_expr(y))
    both = pl.col(x).is_not_null() & pl.col(y).is_not_null()
    xm = pl.when(both).then(pl.col(x)).otherwise(None)
    ym = pl.when(both).then(pl.col(y)).otherwise(None)
    table = data.group_by(group, maintain_order=True).agg(
        both.sum().alias("n"),
        xm.sum().alias("sx"),
        ym.sum().alias("sy"),
        (xm * xm).sum().alias("sxx"),
        (ym * ym).sum().alias("syy"),
        (xm * ym).sum().alias("sxy"),
    )
    entries = []
    for r in sorted(table.iter_rows(named=True), key=lambda r: str(r[group])):
        n = int(r["n"])
        if n == 0:
            continue
        corr = slope = None
        if n >= MIN_SLOPE_GROUP_ROWS:
            cxx = n * r["sxx"] - r["sx"] ** 2
            cyy = n * r["syy"] - r["sy"] ** 2
            cxy = n * r["sxy"] - r["sx"] * r["sy"]
            if cxx > 0 and cyy > 0:
                corr = _finite(cxy / math.sqrt(cxx * cyy))
                slope = _finite(cxy / cxx)
        entries.append({"group": str(r[group]), "n": n, "corr": corr, "slope": slope})
    return entries


def _slope_difference(frame: pl.DataFrame, profile: DatasetProfile, req: ProbeRequest) -> ProbeResult:
    x, y, group = req.columns["x"], req.columns["y"], req.columns["group"]
    entries = _group_correlations(frame, x, y, group)
    valid = [e for e in entries if e["corr"] is not None]
    overall = _finite(
        frame.select(_finite_expr(x), _finite_expr(y)).drop_nulls().select(pl.corr(x, y)).item()
    )
    chart = ChartSpec(title=_title(req), type="scatter", x=x, y=y, group_by=group)
    n_total = sum(e["n"] for e in entries)
    if len(valid) < 2:
        return _finish(
            profile,
            req,
            effect_size=0.0,
            effect_label="spread of per-group correlations (max - min)",
            n=n_total,
            n_min_group=min((e["n"] for e in entries), default=None),
            thresholds={"pass": 0.3, "weak": 0.15},
            evidence={"groups": entries, "overall_corr": overall, "spread": None, "threshold": None},
            chart=chart,
            notes=[f"fewer than two groups with at least {MIN_SLOPE_GROUP_ROWS} rows"],
            effect_verdict="fail",
        )
    corrs = [e["corr"] for e in valid]
    spread = max(corrs) - min(corrs)
    n_min = min(e["n"] for e in valid)
    threshold = slope_spread_threshold(n_min)
    thresholds = {"pass": round(threshold, 6), "weak": round(threshold / 2, 6)}
    evidence = {
        "groups": entries,
        "overall_corr": overall,
        "spread": round(spread, 6),
        "threshold": round(threshold, 6),
        "n_min": n_min,
        "strongest_group": max(valid, key=lambda e: (abs(e["corr"]), e["group"]))["group"],
        "weakest_group": min(valid, key=lambda e: (abs(e["corr"]), e["group"]))["group"],
    }
    return _finish(
        profile,
        req,
        effect_size=spread,
        effect_label="spread of per-group correlations (max - min)",
        n=n_total,
        n_min_group=n_min,
        thresholds=thresholds,
        evidence=evidence,
        chart=chart,
        notes=[],
    )


# --- nonlinear_relationship ------------------------------------------------------


def _nonlinear_relationship(frame: pl.DataFrame, profile: DatasetProfile, req: ProbeRequest) -> ProbeResult:
    x, y = req.columns["x"], req.columns["y"]
    data = frame.select(_finite_expr(x), _finite_expr(y)).drop_nulls()
    chart = ChartSpec(title=_title(req), type="scatter", x=x, y=y)
    label = "nonlinear_gap = binned eta-squared - Pearson r-squared"
    n = data.height
    if n < NONLINEAR_MIN_ROWS or data[x].n_unique() < NONLINEAR_MIN_X_UNIQUE:
        return _finish(
            profile,
            req,
            effect_size=0.0,
            effect_label=label,
            n=n,
            n_min_group=None,
            thresholds=THRESHOLDS["nonlinear_relationship"],
            evidence={"bins": [], "binned_eta2": None, "r2_pearson": None, "nonlinear_gap": None},
            chart=chart,
            notes=[f"needs at least {NONLINEAR_MIN_ROWS} rows and {NONLINEAR_MIN_X_UNIQUE} distinct x values"],
            effect_verdict="fail",
        )
    binned = data.with_columns(
        ((pl.col(x).rank(method="average") - 1) * NONLINEAR_BINS / n)
        .floor()
        .clip(0, NONLINEAR_BINS - 1)
        .cast(pl.Int32)
        .alias(_BIN)
    )
    col = pl.col(y)
    per_bin = (
        binned.group_by(_BIN)
        .agg(
            pl.col(x).min().alias("lo"),
            pl.col(x).max().alias("hi"),
            col.count().alias("n"),
            col.mean().alias("mean"),
            ((col - col.mean()) ** 2).sum().alias("ssw"),
        )
        .sort(_BIN)
    )
    rows = [r for r in per_bin.iter_rows(named=True) if r["n"] and r["mean"] is not None]
    bins = [NonlinearBin(lo=float(r["lo"]), hi=float(r["hi"]), n=int(r["n"]), y_mean=float(r["mean"])) for r in rows]
    eta = _binned_eta(bins, [float(r["ssw"]) for r in rows])
    r = _finite(data.select(pl.corr(x, y)).item())
    r2 = None if r is None else round(r * r, 6)
    if eta is None or len(bins) < 3:
        return _finish(
            profile,
            req,
            effect_size=0.0,
            effect_label=label,
            n=n,
            n_min_group=None,
            thresholds=THRESHOLDS["nonlinear_relationship"],
            evidence={"bins": [b.model_dump() for b in bins], "binned_eta2": eta, "r2_pearson": r2, "nonlinear_gap": None},
            chart=chart,
            notes=["binned effect undefined (constant y or too few bins)"],
            effect_verdict="fail",
        )
    gap = max(eta - (r2 or 0.0), 0.0)
    means = [b.y_mean for b in bins]
    monotone = _is_monotone(means)
    shape = _curve_shape(eta, gap, monotone, means)
    thresholds = THRESHOLDS["nonlinear_relationship"]
    effect_verdict = None
    notes: list[str] = []
    if gap >= thresholds["pass"] and eta < NONLINEAR_PASS_MIN_ETA:
        effect_verdict = "weak"
        notes.append(f"bins explain only {eta:.2f} of the variance (< {NONLINEAR_PASS_MIN_ETA})")
    evidence = {
        "bins": [b.model_dump() for b in bins],
        "binned_eta2": round(eta, 6),
        "r2_pearson": r2,
        "pearson_r": None if r is None else round(r, 6),
        "nonlinear_gap": round(gap, 6),
        "monotone": monotone,
        "shape": shape,
    }
    return _finish(
        profile,
        req,
        effect_size=gap,
        effect_label=label,
        n=n,
        n_min_group=None,
        thresholds=thresholds,
        evidence=evidence,
        chart=chart,
        notes=notes,
        effect_verdict=effect_verdict,
    )


# --- distribution_difference ------------------------------------------------------


def _ks_statistic(a: pl.Series, b: pl.Series) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: max |F_a - F_b| over the
    pooled sorted values (ties handled by evaluating at each distinct value)."""
    pooled = pl.concat(
        [
            pl.DataFrame({"v": a, "is_a": pl.Series([1] * len(a), dtype=pl.Int64)}),
            pl.DataFrame({"v": b, "is_a": pl.Series([0] * len(b), dtype=pl.Int64)}),
        ]
    ).sort("v")
    steps = pooled.group_by("v", maintain_order=True).agg(
        pl.col("is_a").sum().alias("na"), (1 - pl.col("is_a")).sum().alias("nb")
    )
    fa = steps["na"].cum_sum() / len(a)
    fb = steps["nb"].cum_sum() / len(b)
    return float((fa - fb).abs().max())


def _distribution_difference(frame: pl.DataFrame, profile: DatasetProfile, req: ProbeRequest) -> ProbeResult:
    group, target = req.columns["group"], req.columns["target"]
    stats = _group_table(frame, group, target)
    data = frame.filter(pl.col(group).is_not_null()).select(pl.col(group), _finite_expr(target)).drop_nulls()
    chart = ChartSpec(title=_title(req), type="box", x=group, y=target)
    label = "max two-sample KS statistic between groups"
    thresholds = THRESHOLDS["distribution_difference"]
    eligible = sorted(
        (g for g in stats if g.n >= MIN_SLOPE_GROUP_ROWS), key=lambda g: (-g.n, g.group)
    )[:KS_MAX_GROUPS]
    n_total = sum(g.n for g in stats)
    groups_out = [
        {**g.model_dump(), "iqr": round(g.q75 - g.q25, 6)} for g in stats
    ]
    if len(eligible) < 2:
        return _finish(
            profile,
            req,
            effect_size=0.0,
            effect_label=label,
            n=n_total,
            n_min_group=min((g.n for g in stats), default=None),
            thresholds=thresholds,
            evidence={"groups": groups_out, "pairs": [], "max_ks": None, "ks_pair": None},
            chart=chart,
            notes=[f"fewer than two groups with at least {MIN_SLOPE_GROUP_ROWS} rows"],
            effect_verdict="fail",
        )
    values = {
        g.group: data.filter(pl.col(group).cast(pl.String) == g.group)[target] for g in eligible
    }
    pairs = []
    for a, b in combinations([g.group for g in eligible], 2):
        pairs.append({"a": a, "b": b, "ks": round(_ks_statistic(values[a], values[b]), 6)})
    best = max(pairs, key=lambda p: (p["ks"], p["a"], p["b"]))
    evidence = {
        "groups": groups_out,
        "pairs": pairs,
        "max_ks": best["ks"],
        "ks_pair": [best["a"], best["b"]],
        "median_spread": round(max(g.median for g in eligible) - min(g.median for g in eligible), 6),
    }
    return _finish(
        profile,
        req,
        effect_size=best["ks"],
        effect_label=label,
        n=n_total,
        n_min_group=min(g.n for g in eligible),
        thresholds=thresholds,
        evidence=evidence,
        chart=chart,
        notes=[],
    )


# --- time_pattern ------------------------------------------------------------------


def _time_pattern(frame: pl.DataFrame, profile: DatasetProfile, req: ProbeRequest) -> ProbeResult:
    time, target = req.columns["time"], req.columns["target"]
    group = req.columns.get("group")
    span = _datetime_span_days(frame[time])
    thresholds = THRESHOLDS["time_pattern"]
    label = "adjusted eta-squared of target across time buckets"
    unique = int(frame[time].n_unique())
    granularity = choose_time_granularity(unique, span or 0.0)
    chart = ChartSpec(
        title=_title(req),
        type="line",
        x=time,
        y=target,
        group_by=group,
        aggregation="mean",
        time_granularity=granularity,
    )
    if span is None:
        return _finish(
            profile,
            req,
            effect_size=0.0,
            effect_label=label,
            n=0,
            n_min_group=None,
            thresholds=thresholds,
            evidence={"bucket": None, "eta_squared": None, "series": [], "change_point": None},
            chart=chart,
            notes=["fewer than two timestamps"],
            effect_verdict="fail",
        )
    # bucketed eta-squared, coarsened while the finer bucket has no within-group df
    bucket: TimeBucket | None = span_bucket(span)
    eta = None
    used: TimeBucket | None = None
    while bucket is not None and eta is None:
        bucketed = frame.select(
            pl.col(time).dt.truncate(_BUCKET_TRUNC[bucket]).alias(_BIN), _finite_expr(target)
        )
        eta = adjusted_eta_squared_many(bucketed, _BIN, [target])[target]
        used = bucket
        bucket = _COARSER[bucket]
    if eta is None or used is None:
        return _finish(
            profile,
            req,
            effect_size=0.0,
            effect_label=label,
            n=0,
            n_min_group=None,
            thresholds=thresholds,
            evidence={"bucket": None, "eta_squared": None, "series": [], "change_point": None},
            chart=chart,
            notes=["time effect undefined (no within-bucket variation)"],
            effect_verdict="fail",
        )
    # bucket-mean series (thinned to MAX_TIME_POINTS) and the single change point
    change_bucket = _choose_change_bucket(frame[time], span_bucket(span))
    series_bucket = change_bucket or used
    base = frame.filter(pl.col(time).is_not_null()).select(
        pl.col(time).dt.truncate(_BUCKET_TRUNC[series_bucket]).alias(_BIN),
        _finite_expr(target),
        *([pl.col(group)] if group else []),
    )
    col = pl.col(target)
    means = base.group_by(_BIN).agg(col.count().alias("n"), col.mean().alias("mean")).sort(_BIN)
    points = [
        (r[_BIN], float(r["mean"]), int(r["n"])) for r in means.iter_rows(named=True) if r["n"] and r["mean"] is not None
    ]
    keep = set(_thin_indices(len(points), MAX_TIME_POINTS))
    series = [
        {"bucket": p[0].isoformat(), "mean": round(p[1], 6), "n": p[2]} for i, p in enumerate(points) if i in keep
    ]
    change: dict[str, Any] | None = None
    if change_bucket is not None:
        found = _best_split(points)
        if found is not None:
            k, before, after, effect = found
            std = next((c.std for c in profile.columns if c.name == target), None) or 0.0
            diff_sd = abs(after - before) / std if std > 0 else 0.0
            change = {
                "bucket": change_bucket,
                "change_at": points[k][0].isoformat(),
                "before_mean": round(before, 6),
                "after_mean": round(after, 6),
                "effect_size": round(effect, 6),
                "diff_sd": round(diff_sd, 6),
                "n_before": sum(p[2] for p in points[:k]),
                "n_after": sum(p[2] for p in points[k:]),
                "flagged": effect >= CHANGE_FLAG_EFFECT and diff_sd >= CHANGE_FLAG_DIFF_SD,
            }
    group_series: dict[str, list[dict[str, Any]]] | None = None
    n_min_group: int | None = None
    if group:
        by_group = (
            base.filter(pl.col(group).is_not_null())
            .group_by(_BIN, group)
            .agg(col.count().alias("n"), col.mean().alias("mean"))
            .filter(pl.col("n") > 0)
            .sort(_BIN)
        )
        group_series = {}
        sizes: dict[str, int] = {}
        for r in by_group.iter_rows(named=True):
            if r["mean"] is None:
                continue
            key = str(r[group])
            sizes[key] = sizes.get(key, 0) + int(r["n"])
            bucket_index = next((i for i, p in enumerate(points) if p[0] == r[_BIN]), None)
            if bucket_index in keep:
                group_series.setdefault(key, []).append(
                    {"bucket": r[_BIN].isoformat(), "mean": round(float(r["mean"]), 6), "n": int(r["n"])}
                )
        group_series = dict(sorted(group_series.items()))
        n_min_group = min(sizes.values()) if sizes else None
    evidence = {
        "bucket": used,
        "eta_squared": round(eta.eta_squared, 6),
        "n_buckets": eta.n_groups,
        "series_bucket": series_bucket,
        "series": series,
        "change_point": change,
        "group_series": group_series,
    }
    effect_verdict = None
    notes: list[str] = []
    if change is not None and change["flagged"] and eta.eta_squared < thresholds["pass"]:
        effect_verdict = "pass"
        notes.append("passes on a flagged change point")
    return _finish(
        profile,
        req,
        effect_size=eta.eta_squared,
        effect_label=label,
        n=eta.n_total,
        n_min_group=n_min_group,
        thresholds=thresholds,
        evidence=evidence,
        chart=chart,
        notes=notes,
        effect_verdict=effect_verdict,
    )


# --- interaction -----------------------------------------------------------------------


def _interaction(frame: pl.DataFrame, profile: DatasetProfile, req: ProbeRequest) -> ProbeResult:
    f1, f2, target = req.columns["factor1"], req.columns["factor2"], req.columns["target"]
    result = _interaction_effect(frame, f1, f2, target)
    data = frame.select(f1, f2, _finite_expr(target)).drop_nulls()
    cells = (
        data.group_by(f1, f2, maintain_order=True)
        .agg(pl.len().alias("n"), pl.col(target).mean().alias("mean"))
        .sort(f1, f2)
    )
    cell_rows = [
        {"factor1": str(r[f1]), "factor2": str(r[f2]), "n": int(r["n"]), "mean": round(float(r["mean"]), 6)}
        for r in cells.iter_rows(named=True)
        if r["mean"] is not None
    ]
    chart = ChartSpec(title=_title(req), type="bar", x=f1, y=target, group_by=f2, aggregation="mean")
    label = "interaction share of variance (unweighted-means ANOVA)"
    thresholds = THRESHOLDS["interaction"]
    if result is None:
        return _finish(
            profile,
            req,
            effect_size=0.0,
            effect_label=label,
            n=data.height,
            n_min_group=min((c["n"] for c in cell_rows), default=None),
            thresholds=thresholds,
            evidence={"cells": cell_rows, "strength": None},
            chart=chart,
            notes=["interaction undefined: sparse grid, too few cells, or no variation"],
            effect_verdict="fail",
        )
    strength, n = result
    return _finish(
        profile,
        req,
        effect_size=strength,
        effect_label=label,
        n=n,
        n_min_group=min((c["n"] for c in cell_rows), default=None),
        thresholds=thresholds,
        evidence={"cells": cell_rows, "strength": round(strength, 6), "n_cells": len(cell_rows)},
        chart=chart,
        notes=[],
    )


_RUNNERS = {
    "group_difference": _group_difference,
    "grouped_relationship": _slope_difference,
    "slope_difference": _slope_difference,
    "nonlinear_relationship": _nonlinear_relationship,
    "distribution_difference": _distribution_difference,
    "time_pattern": _time_pattern,
    "interaction": _interaction,
}
