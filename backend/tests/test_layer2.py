"""Evidence layer 2 (stage 17.1): planted positive / negative cases per
summary kind, determinism, JSON round trip, and the guarantee that layer 1
and the rule engine are untouched."""

import math
import random
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from app.charts.rules import recommend_charts
from app.datasets.loader import load_dataframe
from app.profiling import layer2
from app.profiling.evidence import compute_evidence
from app.profiling.layer2 import compute_layer2
from app.profiling.models import DatasetProfile
from app.profiling.profiler import profile_dataset

BIG = 10**6
DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"


def _profile(df: pl.DataFrame) -> DatasetProfile:
    return profile_dataset(df, "0" * 32, BIG)


def _l2(df: pl.DataFrame):
    return _profile(df).evidence.layer2


def _dates(n: int, start: datetime = datetime(2024, 1, 1), step: timedelta = timedelta(days=1)):
    return [start + i * step for i in range(n)]


# --- distributions ------------------------------------------------------------


def test_distribution_bimodal_vs_normal() -> None:
    rng = random.Random(1)
    n = 2000
    bimodal = [rng.gauss(0, 1) if i % 2 else rng.gauss(8, 1) for i in range(n)]
    normal = [rng.gauss(0, 1) for _ in range(n)]
    right = [rng.expovariate(1.0) for _ in range(n)]
    out = _l2(pl.DataFrame({"bimodal": bimodal, "normal": normal, "right": right}))
    by = {d.column: d for d in out.distributions}
    assert set(by) == {"bimodal", "normal", "right"}

    assert by["bimodal"].modes == 2
    assert by["bimodal"].multimodal_signal is True
    assert by["bimodal"].bimodality_coefficient > layer2.BIMODALITY_THRESHOLD

    assert by["normal"].modes == 1
    assert by["normal"].multimodal_signal is False
    assert by["normal"].shape == "symmetric"
    assert by["normal"].bimodality_coefficient < layer2.BIMODALITY_THRESHOLD

    assert by["right"].shape == "right_skewed"
    assert by["right"].multimodal_signal is False

    d = by["normal"]
    assert d.n == n
    assert d.p05 < d.p25 < d.p50 < d.p75 < d.p95
    assert len(d.bins) == layer2.DIST_BINS
    assert math.isclose(sum(d.bins) + d.outside_ratio, 1.0, abs_tol=1e-6)
    assert d.bin_lo < d.bin_hi


def test_distribution_discrete_values_are_not_multimodal() -> None:
    # four discount levels look like four "peaks" but are just discrete values
    df = pl.DataFrame({"discount": [0.0, 0.1, 0.2, 0.3] * 100, "v": [float(i) for i in range(400)]})
    out = _l2(df)
    by = {d.column: d for d in out.distributions}
    assert by["discount"].modes >= 2
    assert by["discount"].multimodal_signal is False


def test_distribution_bins_use_robust_range_and_report_outside_share() -> None:
    rng = random.Random(2)
    values = [rng.gauss(50, 5) for _ in range(500)] + [10_000.0] * 5  # extreme tail
    out = _l2(pl.DataFrame({"v": values}))
    d = out.distributions[0]
    assert d.robust_range is not None
    assert d.bin_hi < 1000  # the window is the robust range, not min..max
    assert d.outside_ratio == pytest.approx(5 / 505, abs=1e-6)
    assert d.modes == 1


def test_distribution_skips_short_and_constant_columns() -> None:
    df = pl.DataFrame({"short": [1.0, 2.0, 3.0] + [None] * 40, "const": [5.0] * 43, "v": [float(i) for i in range(43)]})
    out = _l2(df)
    assert [d.column for d in out.distributions] == ["v"]


def test_count_peaks_rules() -> None:
    assert layer2._count_peaks([0.02, 0.05, 0.1, 0.18, 0.3, 0.2, 0.1, 0.03, 0.01, 0.01]) == 1
    assert layer2._count_peaks([0.25, 0.15, 0.05, 0.02, 0.01, 0.01, 0.02, 0.05, 0.15, 0.29]) == 2
    # two bumps with a shallow valley are one hill
    assert layer2._count_peaks([0.05, 0.2, 0.18, 0.19, 0.2, 0.15, 0.03, 0.0, 0.0, 0.0]) == 1
    # a plateau counts once
    assert layer2._count_peaks([0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05, 0.0, 0.0]) == 1
    # a tiny bump below the share floor is not a peak
    assert layer2._count_peaks([0.3, 0.3, 0.2, 0.1, 0.02, 0.0, 0.05, 0.03, 0.0, 0.0]) == 1


# --- group summaries and anomalies --------------------------------------------


def _grouped_df(shift: float, seed: int = 3, n: int = 600) -> pl.DataFrame:
    rng = random.Random(seed)
    groups = ["a", "b", "c", "d", "e", "f"]
    cat = [groups[i % 6] for i in range(n)]
    value = [rng.gauss(0, 1) + (shift if cat[i] == "f" else 0.0) for i in range(n)]
    return pl.DataFrame({"g": cat, "v": value, "noise": [rng.gauss(0, 1) for _ in range(n)]})


def test_group_summary_reports_per_group_stats_and_effect() -> None:
    out = _l2(_grouped_df(shift=3.0))
    summary = next(s for s in out.group_summaries if s.cat == "g" and s.num == "v")
    assert [g.group for g in summary.groups] == ["a", "b", "c", "d", "e", "f"]
    assert all(g.n == 100 for g in summary.groups)
    assert summary.n_total == 600
    assert summary.top_group == "f"
    assert summary.max_diff_sd == pytest.approx(3.0, abs=0.6)
    f = summary.groups[-1]
    assert f.mean == pytest.approx(3.0, abs=0.4)
    assert f.q25 < f.median < f.q75
    assert summary.eta_squared > 0.4


def test_group_summary_fallback_needs_a_minimal_effect() -> None:
    # every numeric column gets its best grouping ONLY when that grouping
    # explains at least GROUP_SUMMARY_MIN_ETA (17.1b): a pure-noise column
    # against the same 6 groups is left out, a small real shift is kept
    rng = random.Random(30)
    n = 6000
    groups = ["a", "b", "c", "d", "e", "f"]
    cat = [groups[i % 6] for i in range(n)]
    df = pl.DataFrame(
        {
            "g": cat,
            "v": [rng.gauss(0, 1) + (3.0 if cat[i] == "f" else 0.0) for i in range(n)],
            "small": [rng.gauss(0, 1) + (0.4 if cat[i] == "f" else 0.0) for i in range(n)],  # eta ~ 0.02
            "noise": [rng.gauss(0, 1) for _ in range(n)],
        }
    )
    profile = _profile(df)
    eta = {(e.cat, e.num): e.eta_squared for e in profile.evidence.cat_num}
    assert eta[("g", "noise")] < layer2.GROUP_SUMMARY_MIN_ETA
    assert eta[("g", "small")] >= layer2.GROUP_SUMMARY_MIN_ETA
    assert {s.num for s in profile.evidence.layer2.group_summaries} == {"v", "small"}


def test_subgroup_anomaly_detected_and_absent_without_shift() -> None:
    shifted = _l2(_grouped_df(shift=3.0)).subgroup_anomalies
    assert [(a.cat, a.num, a.group) for a in shifted] == [("g", "v", "f")]
    assert shifted[0].robust_z >= layer2.ANOMALY_Z
    assert shifted[0].diff_sd >= layer2.ANOMALY_MIN_DIFF_SD
    assert shifted[0].n == 100

    assert _l2(_grouped_df(shift=0.0)).subgroup_anomalies == []


def test_subgroup_anomaly_needs_a_gap_in_data_units() -> None:
    # six groups whose means sit within a hair of each other: a tiny MAD makes
    # the robust z large, but the gap is far below half a within-group SD
    rng = random.Random(4)
    n = 6000
    cat = [f"g{i % 6}" for i in range(n)]
    value = [rng.gauss(0, 1) + (0.05 if cat[i] == "g5" else 0.0) for i in range(n)]
    out = _l2(pl.DataFrame({"g": cat, "v": value}))
    assert out.subgroup_anomalies == []


def test_subgroup_anomaly_needs_enough_groups_and_rows() -> None:
    rng = random.Random(5)
    n = 400
    cat = [["a", "b", "c", "d"][i % 4] for i in range(n)]  # only 4 groups
    value = [rng.gauss(0, 1) + (4.0 if cat[i] == "d" else 0.0) for i in range(n)]
    assert _l2(pl.DataFrame({"g": cat, "v": value})).subgroup_anomalies == []


# --- definitional structure is excluded -----------------------------------------


def test_layer2_excludes_derived_pairs_and_duplicate_columns() -> None:
    rng = random.Random(6)
    n = 800
    price = [rng.uniform(10, 100) for _ in range(n)]
    qty = [rng.randint(1, 8) for _ in range(n)]
    region = [["N", "S", "E", "W"][i % 4] for i in range(n)]
    df = pl.DataFrame(
        {
            "region": region,
            "price": price,
            "price_usd": [p * 1.1 for p in price],  # near-duplicate of price
            "qty": qty,
            "sales": [p * q for p, q in zip(price, qty)],  # derived
        }
    )
    profile = _profile(df)
    assert profile.evidence.derived_columns[0].target == "sales"
    assert profile.evidence.near_duplicate_groups[0].duplicates == ["price_usd"]
    out = profile.evidence.layer2
    assert "price_usd" not in {d.column for d in out.distributions}
    assert all(s.num != "price_usd" for s in out.group_summaries)
    assert all({s.x, s.y} != {"sales", "price"} and {s.x, s.y} != {"sales", "qty"} for s in out.nonlinear)
    assert all({s.x, s.y} != {"price", "price_usd"} for s in out.nonlinear)
    assert all(frozenset((s.cat, s.num)) != frozenset(("qty", "sales")) for s in out.group_summaries)


# --- conditional relationships ----------------------------------------------------


def test_conditional_relationship_slope_flips_by_group() -> None:
    rng = random.Random(7)
    n = 900
    g = [["up", "flat", "down"][i % 3] for i in range(n)]
    x = [rng.uniform(0, 10) for _ in range(n)]
    slope = {"up": 2.0, "flat": 0.0, "down": -2.0}
    y = [slope[g[i]] * x[i] + rng.gauss(0, 1) for i in range(n)]
    out = _l2(pl.DataFrame({"g": g, "x": x, "y": y}))
    rel = next(r for r in out.conditional_relationships if {r.x, r.y} == {"x", "y"} and r.group == "g")
    by = {e.group: e for e in rel.groups}
    assert [e.group for e in rel.groups] == ["down", "flat", "up"]
    assert all(e.n == 300 for e in rel.groups)
    assert by["up"].slope == pytest.approx(2.0, abs=0.15)
    assert by["down"].slope == pytest.approx(-2.0, abs=0.15)
    assert abs(by["flat"].slope) < 0.15
    assert by["up"].corr > 0.9 and by["down"].corr < -0.9
    assert rel.corr_spread > 1.8
    assert rel.n_min == 300


def test_conditional_relationship_marks_small_groups() -> None:
    rng = random.Random(8)
    n = 620
    g = ["big" if i < 600 else "tiny" for i in range(n)]
    x = [rng.uniform(0, 10) for _ in range(n)]
    y = [x[i] * (1 if g[i] == "big" else -1) + rng.gauss(0, 1) for i in range(n)]
    df = pl.DataFrame({"g": g, "x": x, "y": y, "h": [["p", "q"][i % 2] for i in range(n)]})
    # the layer-1 slope entry needs two groups with >= 30 rows; use h for that
    y2 = [x[i] * (1 if df["h"][i] == "p" else -1) + rng.gauss(0, 1) for i in range(n)]
    out = _l2(df.with_columns(pl.Series("y", y2)))
    for rel in out.conditional_relationships:
        for e in rel.groups:
            if e.n < layer2.MIN_SLOPE_GROUP_ROWS:
                assert e.corr is None and e.slope is None


# --- nonlinear signals ---------------------------------------------------------------


def _curve_df(seed: int = 9, n: int = 1500) -> pl.DataFrame:
    rng = random.Random(seed)
    x = [rng.uniform(-3, 3) for _ in range(n)]
    return pl.DataFrame(
        {
            "x": x,
            "u": [v * v + rng.gauss(0, 0.5) for v in x],  # U shape, Pearson ~ 0
            "cap": [-v * v + rng.gauss(0, 0.5) for v in x],  # inverted U
            "lin": [2 * v + rng.gauss(0, 0.5) for v in x],  # linear
            "sat": [math.tanh(v) * 3 + rng.gauss(0, 0.2) for v in x],  # monotone, saturating
            "noise": [rng.gauss(0, 1) for _ in range(n)],
        }
    )


def test_nonlinear_shapes_and_ranking() -> None:
    out = _l2(_curve_df())
    by = {frozenset((s.x, s.y)): s for s in out.nonlinear}
    assert len(by) == len(out.nonlinear)  # each pair appears in one direction only
    u = by[frozenset(("x", "u"))]
    assert (u.x, u.y) == ("x", "u")  # the explanatory direction wins
    assert u.shape == "u_shape"
    assert u.r2_pearson < 0.05
    assert u.binned_eta2 > 0.8
    assert u.nonlinear_gap > 0.7
    assert u.monotone is False
    assert len(u.bins) == layer2.NONLINEAR_BINS
    assert all(b.lo <= b.hi for b in u.bins)
    assert [b.lo for b in u.bins] == sorted(b.lo for b in u.bins)
    assert u.n == 1500

    assert by[frozenset(("x", "cap"))].shape == "inverted_u"
    lin = by[frozenset(("x", "lin"))]
    assert lin.shape == "linear"
    assert lin.nonlinear_gap < layer2.SHAPE_LINEAR_GAP
    assert lin.r2_pearson > 0.9
    sat = by[frozenset(("x", "sat"))]
    assert sat.shape == "monotone_nonlinear"
    assert sat.monotone is True
    # flat pairs are dropped
    assert frozenset(("x", "noise")) not in by
    assert not any("noise" in key for key in by)
    # the strongest non-linear gap ranks first, linear pairs last
    assert out.nonlinear[0].y in {"u", "cap"}
    assert out.nonlinear[-1].shape == "linear"


def test_nonlinear_multi_peak_is_separated_from_u_shapes() -> None:
    # two humps (a commuting profile) vs one hump vs a U: only the first is multi_peak
    rng = random.Random(31)
    n = 3000
    x = [rng.uniform(0, 24) for _ in range(n)]

    def humps(v: float) -> float:
        return 5 * math.exp(-((v - 8) ** 2) / 2) + 6 * math.exp(-((v - 18) ** 2) / 2)

    df = pl.DataFrame(
        {
            "x": x,
            "two": [humps(v) + rng.gauss(0, 0.3) for v in x],
            "one": [6 * math.exp(-((v - 12) ** 2) / 8) + rng.gauss(0, 0.3) for v in x],
            "u": [(v - 12) ** 2 / 10 + rng.gauss(0, 0.3) for v in x],
        }
    )
    by = {s.y: s for s in _l2(df).nonlinear if s.x == "x"}
    assert by["two"].shape == "multi_peak"
    assert by["one"].shape == "inverted_u"
    assert by["u"].shape == "u_shape"
    # the peak counter itself: endpoints never count, a shallow dip does not split a hill
    assert layer2._count_curve_peaks([5, 3, 2, 1, 2, 3, 5]) == 0  # U
    assert layer2._count_curve_peaks([0, 3, 5, 3, 0, 3, 6, 3, 0]) == 2
    assert layer2._count_curve_peaks([0, 3, 5, 4.6, 5, 3, 0]) == 1  # dip of 8%: one hill
    assert layer2._count_curve_peaks([0, 1, 2, 3, 4, 5, 6]) == 0  # monotone


def test_monotone_tolerates_small_noise_steps() -> None:
    assert layer2._is_monotone([0, 1, 2, 3, 4]) is True
    assert layer2._is_monotone([0, 1, 2, 1.9, 4]) is True  # a 2.5%-of-range dip is noise
    assert layer2._is_monotone([0, 1, 2, 1.0, 4]) is False  # a 25% drop is not
    assert layer2._is_monotone([4, 3, 2, 2.1, 0]) is True


def test_nonlinear_needs_enough_rows_and_distinct_x() -> None:
    rng = random.Random(10)
    small = pl.DataFrame({"x": [rng.uniform(-3, 3) for _ in range(80)]}).with_columns(
        (pl.col("x") ** 2).alias("u")
    )
    assert _l2(small).nonlinear == []
    few_levels = pl.DataFrame({"x": [float(i % 4) for i in range(400)]}).with_columns(
        ((pl.col("x") - 1.5) ** 2 + pl.Series([rng.gauss(0, 0.1) for _ in range(400)])).alias("u")
    )
    assert _l2(few_levels).nonlinear == []


def test_binned_eta_matches_layer1_formula() -> None:
    from app.profiling.evidence import adjusted_eta_squared

    df = _curve_df().select("x", "u")
    n = df.height
    binned = df.with_columns(
        ((pl.col("x").rank(method="average") - 1) * layer2.NONLINEAR_BINS / n)
        .floor()
        .clip(0, layer2.NONLINEAR_BINS - 1)
        .cast(pl.Int32)
        .alias("b")
    )
    reference = adjusted_eta_squared(binned, "b", "u")[0]
    signal = next(s for s in _l2(df).nonlinear if s.y == "u")
    assert signal.binned_eta2 == pytest.approx(reference, abs=1e-6)


# --- change points and group time patterns ---------------------------------------


def _series_df(step_at: int | None, n_days: int = 400, seed: int = 11) -> pl.DataFrame:
    rng = random.Random(seed)
    dates = _dates(n_days)
    level = [rng.gauss(0, 1) + (5.0 if step_at is not None and i >= step_at else 0.0) for i in range(n_days)]
    return pl.DataFrame(
        {
            "day": dates,
            "level": level,
            "stationary": [rng.gauss(0, 1) for _ in range(n_days)],
            "g": [["a", "b"][i % 2] for i in range(n_days)],
        }
    )


def test_change_point_found_at_the_step_and_not_on_stationary_series() -> None:
    out = _l2(_series_df(step_at=250))
    by = {c.num: c for c in out.change_points}
    cp = by["level"]
    assert cp.flagged is True
    assert cp.bucket == "month"
    assert cp.change_at.startswith("2024-09")  # day 250 is 2024-09-07; monthly buckets split at September
    assert cp.after_mean - cp.before_mean == pytest.approx(5.0, abs=0.8)
    assert cp.effect_size >= layer2.CHANGE_FLAG_EFFECT
    assert cp.diff_sd >= layer2.CHANGE_FLAG_DIFF_SD
    assert cp.n_before + cp.n_after == 400

    assert by["stationary"].flagged is False
    assert cp.strength == "strong"
    assert by["stationary"].strength == "none"


def test_change_point_clear_but_small_is_graded_weak() -> None:
    # a step that is unmistakable against the bucket-mean noise but tiny in
    # data units (0.15 column SDs): effect_size >= 1.5, diff_sd < 0.3 -> "weak"
    rng = random.Random(12)
    n = 8000  # hourly rows over ~11 months: ~720 rows per monthly bucket
    level = [rng.gauss(0, 1) + (0.2 if i >= n // 2 else 0.0) for i in range(n)]
    df = pl.DataFrame({"day": _dates(n, step=timedelta(hours=1)), "level": level})
    cp = next(c for c in _l2(df).change_points if c.num == "level")
    assert cp.effect_size >= layer2.CHANGE_FLAG_EFFECT, cp
    assert cp.diff_sd < layer2.CHANGE_FLAG_DIFF_SD
    assert cp.flagged is False
    assert cp.strength == "weak"


def test_change_point_needs_enough_buckets() -> None:
    out = _l2(_series_df(step_at=3, n_days=6))
    assert out.change_points == []


def test_group_time_pattern_series_per_group() -> None:
    df = _series_df(step_at=None, n_days=60).with_columns(
        (pl.col("level") + pl.when(pl.col("g") == "b").then(4.0).otherwise(0.0)).alias("level")
    )
    out = _l2(df)
    pattern = next(p for p in out.group_time_patterns if p.num == "level" and p.group == "g")
    assert set(pattern.series) == {"a", "b"}
    assert pattern.bucket == "day"  # 60 days: thinned to 24 points, not coarsened to 2 months
    assert all(len(points) <= layer2.MAX_TIME_POINTS for points in pattern.series.values())
    assert all(p.n >= 1 for points in pattern.series.values() for p in points)
    assert pattern.n_total == 60
    mean_a = sum(p.mean for p in pattern.series["a"]) / len(pattern.series["a"])
    mean_b = sum(p.mean for p in pattern.series["b"]) / len(pattern.series["b"])
    assert mean_b - mean_a == pytest.approx(4.0, abs=1.0)
    buckets = [p.bucket for p in pattern.series["a"]]
    assert buckets == sorted(buckets)


def test_group_time_pattern_drops_tiny_groups_and_short_series() -> None:
    # 120 daily rows: group "a" 100 rows, "b" 15 rows (< MIN_ROWS), "c" 5 rows
    # spread over 3 days (< MIN_TIME_POINTS) -> only groups with substance remain
    rng = random.Random(13)
    n_days = 120
    g = ["a"] * 100 + ["b"] * 15 + ["c"] * 5
    df = pl.DataFrame(
        {
            "day": _dates(n_days),
            "level": [rng.gauss(0, 1) + (2.0 if g[i] == "b" else 0.0) for i in range(n_days)],
            "g": g,
            "h": [["p", "q"][i % 2] for i in range(n_days)],
        }
    )
    out = _l2(df)
    for p in out.group_time_patterns:
        for key, points in p.series.items():
            assert len(points) >= layer2.MIN_TIME_POINTS
            assert sum(pt.n for pt in points) <= p.n_total
        if p.group == "g":
            assert "b" not in p.series and "c" not in p.series
    # a pattern with fewer than two surviving groups is dropped entirely
    assert all(len(p.series) >= 2 for p in out.group_time_patterns)


def test_thin_indices_are_even_and_bounded() -> None:
    assert layer2._thin_indices(10, 24) == list(range(10))
    thinned = layer2._thin_indices(100, 24)
    assert len(thinned) == 24 and thinned[0] == 0 and thinned[-1] == 99


# --- invariants -----------------------------------------------------------------------


def test_layer2_is_deterministic_and_round_trips() -> None:
    df = _curve_df().with_columns(pl.Series("g", [["a", "b", "c"][i % 3] for i in range(1500)]))
    first = _profile(df)
    second = _profile(df)
    assert first.evidence.layer2 == second.evidence.layer2
    reloaded = DatasetProfile.model_validate_json(first.model_dump_json())
    assert reloaded.evidence.layer2 == first.evidence.layer2
    assert first.profile_version == 11


def test_layer1_and_rules_unchanged_by_layer2() -> None:
    df = _curve_df().with_columns(pl.Series("g", [["a", "b", "c"][i % 3] for i in range(1500)]))
    profile = _profile(df)
    # the rule engine never reads layer 2: blanking it changes nothing
    with_layer2 = [r.model_dump() for r in recommend_charts(profile)]
    stripped = profile.model_copy(deep=True)
    stripped.evidence.layer2 = type(profile.evidence.layer2)()
    without_layer2 = [r.model_dump() for r in recommend_charts(stripped)]
    assert with_layer2 == without_layer2
    # layer 1 is computed exactly as before layer 2 existed
    from app.profiling.types import apply_casts

    casts = {c.name: c.cast_params for c in profile.columns if c.cast_params is not None}
    standalone = compute_evidence(apply_casts(df, casts), profile.columns, profile.correlations)
    assert standalone.model_dump(exclude={"layer2"}) == profile.evidence.model_dump(exclude={"layer2"})


def _assert_close(a, b, tol: float = 1e-6) -> None:
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys()
        for key in a:
            _assert_close(a[key], b[key], tol)
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b)
        for x, y in zip(a, b):
            _assert_close(x, y, tol)
    elif isinstance(a, float) and isinstance(b, float):
        assert a == pytest.approx(b, rel=tol, abs=tol)
    else:
        assert a == b


def test_layer2_structure_survives_row_shuffling() -> None:
    # row order only changes floating-point summation order: every label,
    # count, shape and ordering is identical and every number agrees to 1e-6
    df = _curve_df().with_columns(
        pl.Series("g", [["a", "b", "c"][i % 3] for i in range(1500)]),
        pl.Series("day", _dates(1500)),
    )
    ordered = _profile(df).evidence.layer2.model_dump()
    shuffled = _profile(df.sample(fraction=1.0, shuffle=True, seed=5)).evidence.layer2.model_dump()
    _assert_close(ordered, shuffled)


def test_layer2_counts_exclude_null_and_non_finite_rows() -> None:
    rng = random.Random(14)
    n = 600
    g = [["a", "b", "c"][i % 3] for i in range(n)]
    x = [rng.uniform(0, 10) for _ in range(n)]
    v = [x[i] ** 2 + (20.0 if g[i] == "c" else 0.0) + rng.gauss(0, 1) for i in range(n)]
    v[0] = None
    v[1] = float("nan")
    v[2] = float("inf")
    v[3] = float("-inf")
    df = pl.DataFrame({"g": g, "x": x, "v": v, "day": _dates(n)})
    out = _l2(df)
    dist = next(d for d in out.distributions if d.column == "v")
    assert dist.n == n - 4
    summary = next(s for s in out.group_summaries if s.num == "v")
    assert summary.n_total == n - 4
    signal = next(s for s in out.nonlinear if {s.x, s.y} == {"x", "v"})
    assert signal.n == n - 4
    cp = next(c for c in out.change_points if c.num == "v")
    assert cp.n_before + cp.n_after == n - 4


def test_layer2_respects_every_cap_and_skips_id_columns() -> None:
    rng = random.Random(15)
    n = 1200
    frame: dict = {"record_id": list(range(1, n + 1))}  # identifier: never summarised
    for i in range(12):  # 12 independent U-shaped pairs so the nonlinear cap binds
        base = [rng.uniform(0, 10) for _ in range(n)]
        frame[f"b{i:02d}"] = base
        frame[f"c{i:02d}"] = [(b - 5) ** 2 + rng.gauss(0, 1) for b in base]
    for i in range(30):  # 30 noise columns push every cap (54 numeric > MAX_DISTRIBUTIONS)
        frame[f"s{i:02d}"] = [rng.gauss(0, 1) for _ in range(n)]
    for k in range(6):
        frame[f"cat{k}"] = [f"k{(i + k) % 6}" for i in range(n)]
    frame["day"] = _dates(n)
    out = _l2(pl.DataFrame(frame))
    assert len(out.distributions) <= layer2.MAX_DISTRIBUTIONS
    assert len(out.group_summaries) <= layer2.MAX_GROUP_SUMMARIES
    assert len(out.conditional_relationships) <= layer2.MAX_CONDITIONAL
    assert len(out.group_time_patterns) <= layer2.MAX_GROUP_TIME_PATTERNS
    assert len(out.nonlinear) == layer2.MAX_NONLINEAR
    assert len(out.change_points) <= layer2.MAX_CHANGE_POINTS
    assert len(out.subgroup_anomalies) <= layer2.MAX_ANOMALIES
    names = {d.column for d in out.distributions}
    names |= {s.num for s in out.group_summaries}
    names |= {s.x for s in out.nonlinear} | {s.y for s in out.nonlinear}
    names |= {c.num for c in out.change_points}
    names |= {r.x for r in out.conditional_relationships} | {r.y for r in out.conditional_relationships}
    assert "record_id" not in names


def test_layer2_empty_on_text_only_and_tiny_frames() -> None:
    text = pl.DataFrame({"note": [f"row {i}" for i in range(50)]})
    out = _l2(text)
    assert out == type(out)()
    tiny = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "g": ["a", "b", "a", "b"]})
    out = _l2(tiny)
    assert out.distributions == [] and out.nonlinear == [] and out.change_points == []


# --- real bike-sharing shape ----------------------------------------------------------


def _hour_like() -> pl.DataFrame:
    path = DATASET_DIR / "hour_like.csv"
    if not path.exists():
        import subprocess
        import sys

        subprocess.run([sys.executable, str(DATASET_DIR / "syn" / "hour_like.py")], check=False)
    if not path.exists():
        pytest.skip("hour_like.csv missing and could not be generated")
    return load_dataframe(path.read_bytes(), path.name)


def test_hour_like_hourly_demand_is_a_nonlinear_signal() -> None:
    out = _l2(_hour_like())
    by = {(s.x, s.y): s for s in out.nonlinear}
    signal = by.get(("hr", "cnt")) or by.get(("cnt", "hr"))
    assert signal is not None, [(s.x, s.y, s.shape) for s in out.nonlinear]
    # the commuting double peak (17.1b): two separated interior peaks, not an inverted U
    assert signal.shape == "multi_peak", [round(b.y_mean, 1) for b in signal.bins]
    assert layer2._count_curve_peaks([b.y_mean for b in signal.bins]) >= 2
    assert signal.nonlinear_gap > 0.1
    # the near-duplicate atemp never appears; cnt = casual + registered never as a pair
    names = {s.x for s in out.nonlinear} | {s.y for s in out.nonlinear} | {d.column for d in out.distributions}
    assert "atemp" not in names
    assert all({s.x, s.y} not in ({"cnt", "casual"}, {"cnt", "registered"}) for s in out.nonlinear)
