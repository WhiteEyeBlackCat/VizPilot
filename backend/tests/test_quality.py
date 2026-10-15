"""Stage 9 #6: robust statistics, suspected sentinels, extreme values."""

import math
import random

import polars as pl
import pytest

from app.charts.confidence import assess
from app.charts.spec import ChartSpec
from app.profiling.models import DatasetProfile, RobustRange, SentinelCandidate
from app.profiling.profiler import profile_dataset
from app.profiling.quality import CONCENTRATED_SHARE, MIN_ROBUST_N, Z_SENTINEL, robust_quality

BIG = 10**6


def _q(values: list) -> dict:
    return robust_quality(pl.Series(values, dtype=pl.Float64))


def _sentinel_values(q: dict) -> dict[float, list[str]]:
    return {s.value: s.signals for s in q.get("suspected_sentinels", [])}


def test_clean_small_series_flags_nothing() -> None:
    q = _q([20, 21, 19, 24, 22])
    assert q["suspected_sentinels"] == [] and q["extreme_value_count"] == 0
    assert q["q1"] == 20 and q["q3"] == 22 and q["iqr"] == 2 and q["mad"] == 1
    assert q["robust_range"].lo == 19 and q["robust_range"].hi == 24  # clipped to the data range


def test_repeated_pattern_value_is_suspected_sentinel() -> None:
    q = _q([20, 21, -999, 22, -999])
    # q1 is -999 itself here, so the "scale" reference (100 x max(|q1|,|q3|,s))
    # cannot fire: contamination only makes that signal more conservative;
    # repeated + pattern are enough
    assert _sentinel_values(q) == {-999.0: ["extreme", "repeated", "pattern"]}
    assert q["sentinel_row_count"] == 2 and q["extreme_value_count"] == 0
    # fences come from the sentinel-free pass: the range is the clean data
    assert q["robust_range"].lo == 20 and q["robust_range"].hi == 22
    assert q["q1"] == 20.5 and q["q3"] == 21.5  # clean-pass quantiles


def test_single_legit_extreme_is_extreme_value_not_sentinel() -> None:
    q = _q([100, 120, 90, 110, 1_000_000])
    assert q["suspected_sentinels"] == []
    assert q["extreme_value_count"] == 1 and q["extreme_value_ratio"] == pytest.approx(0.2)
    assert q["robust_z_max"] > Z_SENTINEL  # extreme, but only the "scale" signal


def test_negative_centered_normal_is_not_flagged() -> None:
    rng = random.Random(3)
    q = _q([rng.gauss(-999, 5) for _ in range(1000)])
    assert q["suspected_sentinels"] == [] and q["sentinel_row_count"] == 0


def test_majority_999_is_not_flagged() -> None:
    rng = random.Random(5)
    values = [999.0] * 600 + [rng.gauss(1000, 5) for _ in range(400)]
    q = _q(values)
    assert q["mad"] == 0  # MAD collapses -> IQR fallback scale
    assert q["suspected_sentinels"] == []


def test_minus_one_flags_and_zero_fill_are_not_flagged() -> None:
    q = _q([1, 2, 3, -1, 5, -1, 7, 8, -1, 10] * 10)
    assert q["suspected_sentinels"] == [] and q["extreme_value_count"] == 0
    q = _q([20, 21, 0, 22, 23, 0, 19, 0] * 20)  # zeros at 37%
    assert q["suspected_sentinels"] == [] and q["extreme_value_count"] == 0


def test_single_9999_among_temperatures_is_suspected() -> None:
    rng = random.Random(7)
    values = [round(rng.gauss(25, 2), 1) for _ in range(999)] + [9999.0]
    q = _q(values)
    assert _sentinel_values(q) == {9999.0: ["extreme", "pattern", "scale"]}
    assert q["sentinel_row_count"] == 1


def test_constant_and_discrete_columns_have_no_detection() -> None:
    q = _q([7.0] * 50)
    assert q["robust_z_max"] is None and q["robust_range"].lo == q["robust_range"].hi == 7.0
    assert q["suspected_sentinels"] == [] and q["extreme_value_count"] == 0
    q = _q([i % 5 + 1 for i in range(500)])
    assert q["suspected_sentinels"] == [] and q["extreme_value_count"] == 0


def test_too_few_values_yield_no_robust_block() -> None:
    assert _q([1, 2, 3, 4]) == {}
    assert MIN_ROBUST_N == 5 and _q([1, 2, 3, 4, 5]) != {}


def test_uniform_integers_flag_nothing() -> None:
    q = _q(list(range(1, 1001)))
    assert q["suspected_sentinels"] == [] and q["extreme_value_count"] == 0
    assert q["robust_range"].lo == 1 and q["robust_range"].hi == 1000


def test_thirty_percent_sentinels_survive_mad_and_fences_stay_clean() -> None:
    rng = random.Random(11)
    values = [round(rng.gauss(25, 2), 1) for _ in range(700)] + [9999.0] * 150 + [-999.0] * 150
    q = _q(values)
    flagged = _sentinel_values(q)
    assert set(flagged) == {9999.0, -999.0}
    assert all("repeated" in s and "pattern" in s for s in flagged.values())
    assert q["sentinel_row_count"] == 300
    assert 15 < q["robust_range"].lo and q["robust_range"].hi < 35  # not dragged by the sentinels
    assert q["extreme_value_count"] <= 5  # only genuine normal-tail rows, if any


def test_heavy_tail_is_extreme_values_only() -> None:
    rng = random.Random(13)
    q = _q([rng.lognormvariate(4, 1) for _ in range(2000)])
    assert q["suspected_sentinels"] == []
    assert 0.01 < q["extreme_value_ratio"] < 0.1


def test_profile_excludes_nan_inf_and_fills_numeric_quality_only() -> None:
    df = pl.DataFrame(
        {
            "temp": [25.0, 26.0, float("nan"), float("inf"), 24.0, 9999.0, 9999.0, 25.5, None, 24.5],
            "city": ["a", "b"] * 5,
        }
    )
    profile = profile_dataset(df, "0" * 32, BIG)
    temp = next(c for c in profile.columns if c.name == "temp")
    q = temp.quality
    assert (q.missing_count, q.invalid_count, q.valid_count) == (1, 2, 7)
    assert [s.value for s in q.suspected_sentinels] == [9999.0] and q.sentinel_row_count == 2
    assert temp.max == 9999.0  # raw stats untouched
    city = next(c for c in profile.columns if c.name == "city").quality
    assert city.q1 is None and city.robust_range is None and city.suspected_sentinels == []
    reloaded = DatasetProfile.model_validate_json(profile.model_dump_json())
    assert reloaded == profile


def test_sampled_profile_quality_counts_are_sample_level() -> None:
    rng = random.Random(17)
    values = [rng.gauss(25, 2) for _ in range(4000)]
    for i in range(0, 4000, 100):
        values[i] = -999.0
    profile = profile_dataset(pl.DataFrame({"temp": values}), "0" * 32, sample_threshold=1000)
    q = next(c for c in profile.columns if c.name == "temp").quality
    assert profile.sampled and q.profiled_rows == 1000
    assert q.sentinel_row_count <= 1000 and q.suspected_sentinels[0].value == -999.0
    assert q.sentinel_row_count == q.suspected_sentinels[0].count


# --- audit 2026-10: mode-dominated (MAD == 0) columns ------------------------
# old -> new:
#   A [0]x990 + [999999]x10        sentinel missed, range (0, 999999) -> 999999 suspected
#   F [0]x990 + [1]x5 + [999999]x5 sentinel missed                     -> 999999 suspected, range (0, 1)
#   H [0]x700 + 1..300             99 "extreme" counts (9.9%)          -> 0 extremes, range (0, 300)
#   B / C / D / I                  same verdicts, plus concentration labels


def _legacy_robust_quality(finite: pl.Series) -> dict:
    """Frozen copy of the pre-audit implementation (PROFILE_VERSION 14); the
    MAD > 0 path must stay byte-identical to it."""
    v = finite.cast(pl.Float64)
    n = len(v)
    if n < 5:
        return {}

    def quantiles(s):
        a = float(s.quantile(0.25, interpolation="linear"))
        b = float(s.quantile(0.75, interpolation="linear"))
        return a, b, b - a

    median = float(v.median())
    mad = float((v - median).abs().median())
    q1, q3, iqr = quantiles(v)
    scale = 1.4826 * mad
    if scale == 0:
        scale = iqr / 1.349
    if scale == 0:
        return {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "mad": mad,
            "robust_z_max": None,
            "extreme_value_count": 0,
            "extreme_value_ratio": 0.0,
            "suspected_sentinels": [],
            "sentinel_row_count": 0,
            "robust_range": RobustRange(lo=float(v.min()), hi=float(v.max())),
        }
    z = (v - median).abs() / scale
    robust_z_max = float(z.max())
    sentinels = []
    candidates = v.filter(z > 10.0)
    if len(candidates):
        scale_reference = 100.0 * max(abs(q1), abs(q3), scale)
        for value, count in candidates.value_counts().rows():
            signals = ["extreme"]
            if count >= 2 and count >= 0.002 * n:
                signals.append("repeated")
            if value == math.floor(value) and abs(value) in {999.0, 9999.0, 99999.0, 999999.0}:
                signals.append("pattern")
            if abs(value) >= scale_reference:
                signals.append("scale")
            if len(signals) - 1 >= 2:
                sentinels.append(SentinelCandidate(value=float(value), count=int(count), signals=signals))
    sentinels.sort(key=lambda s: (-s.count, s.value))
    clean = v.filter(~v.is_in([s.value for s in sentinels])) if sentinels else v
    q1c, q3c, iqrc = quantiles(clean)
    lo, hi = q1c - 3.0 * iqrc, q3c + 3.0 * iqrc
    extreme_count = int(((clean < lo) | (clean > hi)).sum())
    return {
        "q1": q1c,
        "q3": q3c,
        "iqr": iqrc,
        "mad": mad,
        "robust_z_max": robust_z_max,
        "extreme_value_count": extreme_count,
        "extreme_value_ratio": extreme_count / n,
        "suspected_sentinels": sentinels,
        "sentinel_row_count": n - len(clean),
        "robust_range": RobustRange(lo=max(float(clean.min()), lo), hi=min(float(clean.max()), hi)),
    }


def _seeded_distributions() -> dict[str, list[float]]:
    rng = random.Random(2026)
    return {
        "normal": [rng.gauss(50, 10) for _ in range(1000)],
        "lognormal": [rng.lognormvariate(3, 1.2) for _ in range(1500)],
        "uniform": [rng.uniform(-5, 5) for _ in range(800)],
        "exponential": [rng.expovariate(0.1) for _ in range(1200)],
        "bimodal": [rng.gauss(0, 1) if i % 2 else rng.gauss(20, 2) for i in range(1000)],
        "cauchy_like": [rng.gauss(0, 1) / max(abs(rng.gauss(0, 1)), 1e-3) for _ in range(1000)],
        "normal_9999": [round(rng.gauss(25, 2), 1) for _ in range(990)] + [9999.0] * 10,
        "normal_neg999": [-999.0] * 20 + [rng.gauss(50, 10) for _ in range(980)],
        "thirty_pct_sentinels": [rng.gauss(25, 2) for _ in range(700)] + [9999.0] * 150 + [-999.0] * 150,
        "discrete_1_5": [i % 5 + 1 for i in range(500)],
        "zeros_37pct": [20, 21, 0, 22, 23, 0, 19, 0] * 20,
        "ints_40pct_zero": [0] * 400 + [rng.randint(1, 50) for _ in range(600)],
        "tiny": [3.0, 1.0, 4.0, 1.5, 9.0],
        "heavy_single_spike": [100, 120, 90, 110, 1_000_000],
    }


@pytest.mark.parametrize("name", list(_seeded_distributions()))
def test_mad_positive_path_is_unchanged(name) -> None:
    series = pl.Series(_seeded_distributions()[name], dtype=pl.Float64)
    assert float((series - series.median()).abs().median()) > 0
    new, old = robust_quality(series), _legacy_robust_quality(series)
    assert new == old
    assert "mode_value" not in new and "concentration" not in new  # ColumnQuality defaults stay None


def _summary(q: dict) -> tuple:
    return (
        [(s.value, s.count, s.signals) for s in q["suspected_sentinels"]],
        q["sentinel_row_count"],
        q["extreme_value_count"],
        (q["robust_range"].lo, q["robust_range"].hi),
        q.get("concentration"),
    )


_TAIL = [0] * 600 + [10, 11, 12, 13, 14, 15, 16, 17, 18, 19] * 39 + list(range(20, 30)) + [5000]

MODE_CASES = {
    # missed sentinels (audit bug) — now found; the window excludes them
    "A_zero_999999": (
        [0] * 990 + [999999] * 10,
        ([(999999.0, 10, ["extreme", "repeated", "pattern"])], 10, 0, (0.0, 0.0), "concentrated"),
    ),
    "F_zero_one_999999": (
        [0] * 990 + [1] * 5 + [999999] * 5,
        ([(999999.0, 5, ["extreme", "repeated", "pattern", "scale"])], 5, 0, (0.0, 1.0), "concentrated"),
    ),
    "L_zero_neg9999": (
        [0] * 990 + [-9999] * 10,
        ([(-9999.0, 10, ["extreme", "repeated", "pattern"])], 10, 0, (0.0, 0.0), "concentrated"),
    ),
    # two sentinels: peeled outside-in, so 999999 does not hide the inner -999
    "O_two_sentinels": (
        [0] * 980 + [999999] * 10 + [-999] * 10,
        (
            [(-999.0, 10, ["extreme", "repeated", "pattern"]), (999999.0, 10, ["extreme", "repeated", "pattern", "scale"])],
            20,
            0,
            (0.0, 0.0),
            "concentrated",
        ),
    ),
    # nonzero mode gives "scale" a reference: repeated + 123x the mode
    "M_mode100_12345": (
        [100] * 980 + [12345] * 20,
        ([(12345.0, 20, ["extreme", "repeated", "scale"])], 20, 0, (100.0, 100.0), "concentrated"),
    ),
    # zero-inflated / rare legitimate values: nothing flagged
    "C_zero_1_to_5": ([0] * 995 + [1, 2, 3, 4, 5], ([], 0, 0, (0.0, 5.0), "concentrated")),
    "D_zero_1_to_100": ([0] * 900 + list(range(1, 101)), ([], 0, 0, (0.0, 100.0), "concentrated")),
    "H_zero_1_to_300": ([0] * 700 + list(range(1, 301)), ([], 0, 0, (0.0, 300.0), "mode_dominated")),
    "N_rare_events": (
        [0] * 990 + [50, 120, 300, 450, 800, 1200, 2000, 2600, 3700, 5000],
        ([], 0, 0, (0.0, 5000.0), "concentrated"),
    ),
    # non-pattern repeated values next to an all-zero bulk: no magnitude to be
    # inconsistent with (no "scale" signal) -> conservative, not sentinels
    "K_zero_minus1": ([0] * 990 + [-1] * 10, ([], 0, 0, (-1.0, 0.0), "concentrated")),
    "J_zero_12345": ([0] * 980 + [12345] * 20, ([], 0, 0, (0.0, 12345.0), "concentrated")),
    "I_seven_minus1": ([7] * 990 + [-1] * 10, ([], 0, 0, (-1.0, 7.0), "concentrated")),
    # constant
    "B_constant": ([5] * 1000, ([], 0, 0, (5.0, 5.0), "constant")),
    "constant_tiny": ([2.5] * 5, ([], 0, 0, (2.5, 2.5), "constant")),
    # a genuine far value among the off-mode values: Tukey on T, not on the mode
    "tail_off_mode": (_TAIL, ([], 0, 1, (0.0, 32.0), "mode_dominated")),  # far-out fence q3 + 3 IQR = 32
    # float data, negative mode
    "float_neg_mode": (
        [-2.5] * 950 + [-2.25, -2.0, -1.75, -1.5, -1.0] * 10,
        ([], 0, 0, (-2.5, -1.0), "concentrated"),
    ),
}


@pytest.mark.parametrize("name", list(MODE_CASES))
def test_mode_dominated_cases(name) -> None:
    values, expected = MODE_CASES[name]
    q = _q(values)
    assert q["mad"] == 0 and q["robust_z_max"] is None
    assert _summary(q) == expected
    series = pl.Series(values, dtype=pl.Float64)
    mode = float(series.median())
    assert q["mode_value"] == mode
    assert q["mode_share"] == pytest.approx(float((series == mode).sum()) / len(values))
    assert (q["concentration"] == "concentrated") == (CONCENTRATED_SHARE <= q["mode_share"] < 1)


def test_zero_inflated_counts_no_longer_report_ten_percent_extremes() -> None:
    values = [0] * 700 + list(range(1, 301))
    assert _legacy_robust_quality(pl.Series(values, dtype=pl.Float64))["extreme_value_count"] == 99
    assert _q(values)["extreme_value_count"] == 0


def test_majority_mode_with_spread_rest() -> None:
    rng = random.Random(41)
    q = _q([5.0] * 600 + [rng.uniform(0, 10) for _ in range(400)])
    assert q["suspected_sentinels"] == [] and q["extreme_value_count"] == 0
    assert q["concentration"] == "mode_dominated" and q["mode_value"] == 5.0
    assert 0 < q["robust_range"].lo < 0.1 and 9.9 < q["robust_range"].hi < 10


def test_zero_inflated_pattern_value_inside_the_bulk_is_not_a_sentinel() -> None:
    # 999 recurs legitimately among counts up to ~2000: not far from the off-mode bulk
    q = _q([0] * 600 + [999] * 3 + [i * 5 for i in range(1, 398)])
    assert q["suspected_sentinels"] == [] and q["extreme_value_count"] == 0


def test_secondary_mode_off_mode_never_collapses_fences() -> None:
    # the off-mode values are themselves concentrated (IQR 0): no fences, no extremes
    q = _q([0] * 900 + [1] * 95 + [2, 3, 4, 5, 6])
    assert q["extreme_value_count"] == 0 and (q["robust_range"].lo, q["robust_range"].hi) == (0.0, 6.0)


def test_pass_two_guard_when_sentinel_free_values_are_mode_dominated() -> None:
    # MAD > 0 overall (45% on 5) but with the -999 sentinels removed 5 holds
    # 64% of the rest: the old fences collapsed to [5, 5] and called all 25
    # other values extreme
    values = [5.0] * 45 + [-999.0] * 30 + [1, 2, 3, 4] * 3 + list(range(6, 19))
    series = pl.Series(values, dtype=pl.Float64)
    assert float((series - series.median()).abs().median()) > 0
    old = _legacy_robust_quality(series)
    assert old["iqr"] == 0 and old["extreme_value_count"] == 25
    q = robust_quality(series)
    assert [s.value for s in q["suspected_sentinels"]] == [-999.0] and q["sentinel_row_count"] == 30
    assert q["extreme_value_count"] == 0
    assert (q["robust_range"].lo, q["robust_range"].hi) == (1.0, 18.0)


def test_all_null_and_tiny_columns_have_no_robust_block() -> None:
    assert robust_quality(pl.Series([], dtype=pl.Float64)) == {}
    assert _q([0, 0, 0, 999999]) == {}
    df = pl.DataFrame({"x": pl.Series([None] * 20, dtype=pl.Float64), "y": [float(i) for i in range(20)]})
    x = next(c for c in profile_dataset(df, "0" * 32, BIG).columns if c.name == "x").quality
    assert x.robust_range is None and x.concentration is None and x.suspected_sentinels == []


def _hist_warnings(values: list) -> list:
    profile = profile_dataset(pl.DataFrame({"v": [float(x) for x in values]}), "0" * 32, BIG)
    return assess(ChartSpec(title="t", type="histogram", x="v"), profile).warnings


def test_confidence_false_positive_extreme_warning_vanishes() -> None:
    warnings = _hist_warnings([0] * 700 + list(range(1, 301)))
    assert [w for w in warnings if w.code in ("extreme_value", "suspected_sentinel")] == []


def test_confidence_missed_sentinel_is_now_suspected() -> None:
    warnings = _hist_warnings([0] * 990 + [999999] * 10)
    (warning,) = [w for w in warnings if w.code == "suspected_sentinel"]
    assert "999999 x10" in warning.message and warning.severity == "severe"
    assert [w for w in warnings if w.code == "extreme_value"] == []


def test_confidence_extreme_message_names_the_mode() -> None:
    (warning,) = [w for w in _hist_warnings(_TAIL) if w.code == "extreme_value"]
    assert "values other than 0" in warning.message and warning.severity == "info"


def test_profile_round_trip_keeps_mode_fields() -> None:
    profile = profile_dataset(pl.DataFrame({"v": [0.0] * 990 + [999999.0] * 10}), "0" * 32, BIG)
    q = next(c for c in profile.columns if c.name == "v").quality
    assert (q.mode_value, q.mode_share, q.concentration) == (0.0, 0.99, "concentrated")
    assert DatasetProfile.model_validate_json(profile.model_dump_json()) == profile
