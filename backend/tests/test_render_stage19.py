"""Stage 19 renderers and category order: every payload is aggregated by the
backend, ordered by the column's natural order when it has one, and shaped
so the adapter never re-computes a statistic."""

import math

import polars as pl
import pytest

from app.charts.render import RenderError, render_chart
from app.charts.spec import ChartSpec, validate_spec
from app.profiling.profiler import profile_dataset
from app.profiling.types import apply_semantic_casts

BIG = 10**6


def _render(df: pl.DataFrame, **kwargs):
    profile = profile_dataset(df, "0" * 32, BIG)
    casted = apply_semantic_casts(df, profile)
    spec = ChartSpec(title="t", **kwargs)
    assert validate_spec(spec, profile) == [], validate_spec(spec, profile)
    return render_chart(casted, spec, profile)


def _cyclic_df(n: int = 480) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "mnth": [(i % 12) + 1 for i in range(n)],
            "weekday": [["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i % 7] for i in range(n)],
            "region": [["North", "South", "East", "West"][i % 4] for i in range(n)],
            "flag": [i % 2 == 0 for i in range(n)],
            "temp": [10 + 10 * math.sin((i % 12) / 12 * 2 * math.pi) + (i % 5) * 0.1 for i in range(n)],
            "level": [["low", "medium", "high"][i % 3] for i in range(n)],
        }
    )


# --- natural order on the classic charts --------------------------------------


def test_bar_over_a_month_keeps_month_order_by_default() -> None:
    data = _render(_cyclic_df(), type="bar", x="mnth", y="temp", aggregation="mean")["chart_data"]
    assert data["categories"] == list(range(1, 13))
    # values travel with their month: July (7) is the warmest in the planted sine
    values = data["series"][0]["values"]
    assert values.index(max(values)) == 3  # month 4: sin peaks at i%12 == 3
    ranked = _render(_cyclic_df(), type="bar", x="mnth", y="temp", aggregation="mean", sort="value")["chart_data"]
    assert ranked["categories"][0] == 4 and ranked["categories"] != list(range(1, 13))


def test_bar_over_weekday_names_and_ordinal_labels() -> None:
    days = _render(_cyclic_df(), type="bar", x="weekday", y="temp", aggregation="mean")["chart_data"]
    assert days["categories"] == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    levels = _render(_cyclic_df(), type="box", x="level", y="temp")["chart_data"]
    assert [g["name"] for g in levels["groups"]] == ["low", "medium", "high"]
    # a nominal category keeps the pre-stage-19 behaviour: ranked bar, alphabetical box
    nominal = _render(_cyclic_df(), type="bar", x="region", y="temp", aggregation="mean")["chart_data"]
    values = nominal["series"][0]["values"]
    assert values == sorted(values, reverse=True)
    assert [g["name"] for g in _render(_cyclic_df(), type="box", x="region", y="temp")["chart_data"]["groups"]] == ["East", "North", "South", "West"]
    natural = _render(_cyclic_df(), type="bar", x="region", y="temp", aggregation="mean", sort="natural")["chart_data"]
    assert natural["categories"] == ["East", "North", "South", "West"]


def test_bar_top_n_keeps_the_largest_then_orders_them() -> None:
    data = _render(_cyclic_df(), type="bar", x="mnth", y="temp", aggregation="mean", top_n=3)["chart_data"]
    assert data["truncated"] is True and data["categories"] == sorted(data["categories"])
    assert len(data["categories"]) == 3


def test_line_over_an_ordered_category() -> None:
    result = _render(_cyclic_df(), type="line", x="weekday", y="temp", aggregation="mean")
    (series,) = result["chart_data"]["series"]
    assert series["x"] == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    assert len(series["y"]) == 7 and result["n_points"] == 7
    months = _render(_cyclic_df(), type="line", x="mnth", y="temp", aggregation="mean")["chart_data"]["series"][0]
    assert months["x"] == list(range(1, 13))


# --- density family -----------------------------------------------------------------


def test_violin_raincloud_ridgeline_share_one_payload_shape() -> None:
    df = _cyclic_df()
    for kind in ("violin", "raincloud", "ridgeline"):
        result = _render(df, type=kind, x="level", y="temp")
        data = result["chart_data"]
        assert [g["name"] for g in data["groups"]] == ["low", "medium", "high"]
        assert len(data["grid"]) == 64
        for g in data["groups"]:
            assert len(g["density"]) == 64 and max(g["density"]) == pytest.approx(1.0)
            assert g["q1"] <= g["median"] <= g["q3"] and g["count"] == 160
            assert ("points" in g) == (kind == "raincloud")
        assert data["y_label"] == "temp"
    rain = _render(df, type="raincloud", x="level", y="temp")["chart_data"]
    assert all(len(g["points"]) == 160 for g in rain["groups"])  # under the cap: every point
    single = _render(df, type="violin", y="temp")["chart_data"]
    assert [g["name"] for g in single["groups"]] == ["all"]


def test_raincloud_points_are_sampled_deterministically_above_the_cap() -> None:
    df = pl.DataFrame({"g": ["a", "b"] * 500, "v": [float(i) for i in range(1000)]})
    first = _render(df, type="raincloud", x="g", y="v")
    second = _render(df, type="raincloud", x="g", y="v")
    assert first["sampled"] is True and first == second
    assert all(len(g["points"]) == 200 for g in first["chart_data"]["groups"])


# --- dot_interval ------------------------------------------------------------------


def test_dot_interval_mean_and_normal_interval() -> None:
    df = pl.DataFrame({"g": ["a"] * 4 + ["b"] * 4 + ["c"], "v": [1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 10.0, 10.0, 7.0]})
    groups = _render(df, type="dot_interval", x="g", y="v")["chart_data"]["groups"]
    a, b, c = groups
    assert a["name"] == "a" and a["mean"] == 2.5 and a["n"] == 4
    half = 1.96 * pl.Series([1.0, 2.0, 3.0, 4.0]).std() / 2
    assert a["lo"] == pytest.approx(2.5 - half) and a["hi"] == pytest.approx(2.5 + half)
    assert b["lo"] == pytest.approx(10.0) and b["hi"] == pytest.approx(10.0)  # zero spread
    assert c["n"] == 1 and c["lo"] is None and c["hi"] is None


# --- dumbbell / slopegraph -----------------------------------------------------------


def test_two_state_payload_and_exactly_two_states() -> None:
    df = _cyclic_df()
    result = _render(df, type="dumbbell", x="region", y="temp", group_by="flag", aggregation="mean")
    data = result["chart_data"]
    assert data["categories"] == ["East", "North", "South", "West"]
    assert data["states"] == ["False", "True"]
    assert [s["name"] for s in data["series"]] == ["False", "True"]
    # named states keep their reading order, not the alphabet
    periods = df.with_columns(pl.when(pl.col("flag")).then(pl.lit("after")).otherwise(pl.lit("before")).alias("period"))
    ordered = _render(periods, type="dumbbell", x="region", y="temp", group_by="period", aggregation="mean")["chart_data"]
    assert ordered["states"] == ["before", "after"]
    assert all(len(s["values"]) == 4 for s in data["series"])
    slope = _render(df, type="slopegraph", x="level", y="temp", group_by="flag", aggregation="mean")["chart_data"]
    assert slope["categories"] == ["low", "medium", "high"]
    # a three-state column is rejected by the spec gate before it reaches render
    profile = profile_dataset(df, "0" * 32, BIG)
    errors = validate_spec(ChartSpec(title="t", type="dumbbell", x="region", y="temp", group_by="level", aggregation="mean"), profile)
    assert any("exactly 2 categories" in e for e in errors)


# --- hexbin -----------------------------------------------------------------------


def test_hexbin_counts_every_point_once() -> None:
    n = 6000
    df = pl.DataFrame({"x": [math.sin(i) * 3 for i in range(n)], "y": [math.cos(i * 0.7) * 2 for i in range(n)]})
    result = _render(df, type="hexbin", x="x", y="y")
    data = result["chart_data"]
    assert sum(c[2] for c in data["cells"]) == n == data["n"]
    assert data["columns"] == 40 and data["rows"] >= 1 and data["dx"] > 0 and data["dy"] > 0
    assert data["cells"] == sorted(data["cells"])  # deterministic order
    assert result["n_points"] == len(data["cells"]) < n


# --- polar / rose reuse the bar aggregate in natural order ------------------------------


def test_polar_line_and_rose_over_a_cycle() -> None:
    df = _cyclic_df()
    # a 24-hour cycle is never rank-truncated to the bar's default top 20
    hours = pl.DataFrame({"hr": [i % 24 for i in range(480)], "cnt": [float(i % 7) for i in range(480)]})
    assert _render(hours, type="rose", x="hr", y="cnt", aggregation="sum")["chart_data"]["categories"] == list(range(24))
    # a rose encodes a magnitude: a mean is rejected (polar_line takes it)
    hp = profile_dataset(hours, "0" * 32, BIG)
    assert any("magnitude" in e for e in validate_spec(ChartSpec(title="t", type="rose", x="hr", y="cnt", aggregation="mean"), hp))
    assert validate_spec(ChartSpec(title="t", type="polar_line", x="hr", y="cnt", aggregation="mean"), hp) == []
    polar = _render(df, type="polar_line", x="mnth", y="temp", aggregation="mean")["chart_data"]
    assert polar["categories"] == list(range(1, 13))
    rose = _render(df, type="rose", x="weekday", aggregation="count")["chart_data"]
    assert rose["categories"] == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    assert rose["series"][0]["name"] == "count"
    profile = profile_dataset(df, "0" * 32, BIG)
    errors = validate_spec(ChartSpec(title="t", type="rose", x="region", y="temp", aggregation="sum"), profile)
    assert any("must be cyclic" in e for e in errors)  # region → sales is never a rose


# --- matrix heatmap ---------------------------------------------------------------------


def test_matrix_heatmap_orders_both_dimensions() -> None:
    df = _cyclic_df()
    data = _render(df, type="heatmap", x="mnth", group_by="level", y="temp", aggregation="mean")["chart_data"]
    assert data["columns"] == [str(m) for m in range(1, 13)]
    assert data["rows"] == ["low", "medium", "high"]
    assert len(data["matrix"]) == 3 and all(len(r) == 12 for r in data["matrix"])
    assert data["y_label"] == "mean(temp)"
    # the correlation matrix is untouched (it needs two numeric columns)
    corr = _render(df.with_columns((pl.col("temp") * 2 + pl.col("mnth")).alias("other")), type="heatmap")["chart_data"]
    assert "rows" not in corr and corr["y_label"] == "correlation"


def test_lollipop_is_the_bar_aggregate() -> None:
    df = _cyclic_df()
    assert _render(df, type="lollipop", x="region", y="temp", aggregation="mean")["chart_data"] == _render(
        df, type="bar", x="region", y="temp", aggregation="mean"
    )["chart_data"]


def test_render_error_shape_unchanged() -> None:
    with pytest.raises(RenderError):
        raise RenderError("x")
