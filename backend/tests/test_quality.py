"""Stage 9 #6: robust statistics, suspected sentinels, extreme values."""

import random

import polars as pl
import pytest

from app.profiling.models import DatasetProfile
from app.profiling.profiler import profile_dataset
from app.profiling.quality import MIN_ROBUST_N, Z_SENTINEL, robust_quality

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
