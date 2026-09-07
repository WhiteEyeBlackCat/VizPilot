from datetime import datetime, timedelta

import polars as pl
import pytest

from app.charts.spec import ChartSpec, validate_spec
from app.profiling.profiler import profile_dataset

BIG = 10**6


def _profile(df: pl.DataFrame):
    return profile_dataset(df, "0" * 32, BIG)


@pytest.fixture(scope="module")
def profile():
    n = 60
    df = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)],
            "ts_dup": [datetime(2024, 1, 1) + timedelta(days=i // 2) for i in range(n)],
            "city": ["North", "South", "East"] * 20,
            "cat25": [f"g{i % 25:02d}" for i in range(n)],
            "const_cat": ["only"] * n,
            "value": [float(i) for i in range(n)],
            "value2": [2.0 * i + 5 for i in range(n)],
            "rating": [i % 5 + 1 for i in range(n)],
            "user_id": [f"u{i:03d}" for i in range(n)],
            "note": [f"a reasonably long free-form comment number {i}" for i in range(n)],
            "flag": [True, False] * 30,
        }
    )
    p = _profile(df)
    types = {c.name: c.semantic_type for c in p.columns}
    assert types["ts"] == "datetime" and types["city"] == "categorical"
    assert types["rating"] == "categorical" and types["user_id"] == "id" and types["note"] == "text"
    return p


@pytest.fixture(scope="module")
def wide_cat_profile():
    # 1300 rows so a 60-category string column still counts as categorical
    n = 1300
    return _profile(
        pl.DataFrame({"code": [f"c{i % 60:02d}" for i in range(n)], "value": [float(i % 97) for i in range(n)]})
    )


def _errors(profile, **kwargs) -> list[str]:
    return validate_spec(ChartSpec(title="t", **kwargs), profile)


def _assert_error(profile, fragment: str, **kwargs) -> None:
    errors = _errors(profile, **kwargs)
    assert any(fragment in e for e in errors), f"expected '{fragment}' in {errors}"


# --- valid specs ------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(type="line", x="ts", y="value"),
        dict(type="line", x="ts", y="value", time_granularity="month", aggregation="mean"),
        dict(type="line", x="ts_dup", y="value", aggregation="mean"),  # duplicate x + agg
        dict(type="line", x="value", y="value2"),  # ordered numeric x
        dict(type="line", x="ts", y="rating"),  # numeric-backed categorical y
        dict(type="line", x="ts", y="value", group_by="city"),
        dict(type="bar", x="city", y="value", aggregation="mean"),
        dict(type="bar", x="city", aggregation="count"),
        dict(type="bar", x="city", y="rating", aggregation="median", top_n=10),
        dict(type="bar", x="city", y="value", aggregation="sum", group_by="flag"),
        dict(type="scatter", x="value", y="value2"),
        dict(type="scatter", x="value", y="value2", group_by="city"),
        dict(type="histogram", x="value"),
        dict(type="histogram", x="value", bins=30, group_by="flag"),
        dict(type="box", x="city", y="value"),
        dict(type="box", y="value"),  # x optional
        dict(type="box", x="cat25", y="rating"),  # 25 categories within box's 2..50
        dict(type="heatmap"),
    ],
)
def test_valid_specs(profile, kwargs) -> None:
    assert _errors(profile, **kwargs) == []


# --- invalid specs: line ----------------------------------------------------


def test_line_categorical_x(profile) -> None:
    _assert_error(profile, "must be datetime or numeric", type="line", x="city", y="value")


def test_line_missing_axes(profile) -> None:
    errors = _errors(profile, type="line")
    assert any("requires x" in e for e in errors) and any("requires y" in e for e in errors)


def test_line_granularity_without_aggregation(profile) -> None:
    _assert_error(
        profile, "aggregation is required", type="line", x="ts", y="value", time_granularity="month"
    )


def test_line_raw_with_duplicate_x_needs_aggregation(profile) -> None:
    _assert_error(profile, "duplicate values", type="line", x="ts_dup", y="value")
    _assert_error(
        profile, "duplicate values", type="line", x="ts_dup", y="value", time_granularity="raw"
    )


def test_time_granularity_on_numeric_x(profile) -> None:
    _assert_error(
        profile,
        "requires a datetime x",
        type="line", x="value", y="value2", time_granularity="day", aggregation="mean",
    )


def test_time_granularity_on_non_line(profile) -> None:
    _assert_error(
        profile,
        "only valid for line",
        type="bar", x="city", y="value", aggregation="mean", time_granularity="month",
    )


# --- invalid specs: bar -----------------------------------------------------


def test_bar_count_cannot_have_mean(profile) -> None:
    _assert_error(profile, "aggregation='count'", type="bar", x="city", aggregation="mean")


def test_bar_numeric_x(profile) -> None:
    _assert_error(profile, "must be categorical or boolean", type="bar", x="value", y="value2", aggregation="mean")


def test_bar_y_without_aggregation(profile) -> None:
    _assert_error(profile, "requires aggregation", type="bar", x="city", y="value")


def test_group_by_too_many_categories(profile) -> None:
    _assert_error(profile, "categories", type="bar", x="city", y="value", aggregation="mean", group_by="cat25")


# --- invalid specs: scatter / histogram -------------------------------------


def test_scatter_with_aggregation(profile) -> None:
    _assert_error(profile, "must not have an aggregation", type="scatter", x="value", y="value2", aggregation="mean")


def test_scatter_same_column(profile) -> None:
    _assert_error(profile, "different columns", type="scatter", x="value", y="value")


def test_line_same_x_and_y(profile) -> None:
    # identity chart: x==y must be rejected for every type, not just scatter
    _assert_error(profile, "different columns", type="line", x="value", y="value")


def test_group_by_same_as_y(profile) -> None:
    # grouping by the very column being aggregated is meaningless
    _assert_error(
        profile, "different columns",
        type="bar", x="city", y="rating", group_by="rating", aggregation="mean",
    )


def test_group_by_same_as_x(profile) -> None:
    _assert_error(
        profile, "different columns",
        type="bar", x="city", y="value", group_by="city", aggregation="mean",
    )


def test_scatter_categorical_axis(profile) -> None:
    _assert_error(profile, "must be numeric", type="scatter", x="city", y="value")


def test_histogram_with_y(profile) -> None:
    _assert_error(profile, "must not have y", type="histogram", x="value", y="value2")


def test_histogram_bins_out_of_range(profile) -> None:
    _assert_error(profile, "bins must be within", type="histogram", x="value", bins=300)
    _assert_error(profile, "bins must be within", type="histogram", x="value", bins=4)


def test_bins_on_non_histogram(profile) -> None:
    _assert_error(profile, "only valid for histogram", type="bar", x="city", y="value", aggregation="mean", bins=30)


def test_top_n_rules(profile) -> None:
    _assert_error(profile, "top_n must be within", type="bar", x="city", y="value", aggregation="mean", top_n=2)
    _assert_error(profile, "only valid for bar", type="histogram", x="value", top_n=10)


# --- invalid specs: box / heatmap -------------------------------------------


def test_box_with_group_by(profile) -> None:
    _assert_error(profile, "must not have group_by", type="box", x="city", y="value", group_by="flag")


def test_box_single_category_x(profile) -> None:
    _assert_error(profile, "2..50 categories", type="box", x="const_cat", y="value")


def test_box_too_many_categories(wide_cat_profile) -> None:
    _assert_error(wide_cat_profile, "2..50 categories", type="box", x="code", y="value")


def test_box_with_aggregation(profile) -> None:
    _assert_error(profile, "must not have an aggregation", type="box", x="city", y="value", aggregation="mean")


def test_heatmap_with_fields(profile) -> None:
    assert _errors(profile, type="heatmap", x="value") == ["heatmap must not have x"]
    _assert_error(profile, "must not have aggregation", type="heatmap", aggregation="mean")


def test_heatmap_requires_two_numeric_columns() -> None:
    p = _profile(pl.DataFrame({"city": ["a", "b", "c"] * 10}))
    _assert_error(p, "at least 2 numeric columns", type="heatmap")


# --- common rules -----------------------------------------------------------


def test_nonexistent_column(profile) -> None:
    _assert_error(profile, "does not exist", type="histogram", x="ghost")


def test_id_and_text_columns_banned_as_axes(profile) -> None:
    _assert_error(profile, "cannot be used as a chart axis", type="histogram", x="user_id")
    _assert_error(profile, "cannot be used as a chart axis", type="box", y="note")
    _assert_error(profile, "cannot be used as a chart axis", type="line", x="ts", y="value", group_by="note")


def test_errors_are_collected_not_short_circuited(profile) -> None:
    errors = _errors(profile, type="scatter", x="city", y="note", aggregation="mean")
    assert len(errors) >= 3  # bad x type, banned y, forbidden aggregation
