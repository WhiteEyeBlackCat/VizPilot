from datetime import datetime, timedelta

import polars as pl
import pytest
from fastapi.testclient import TestClient

import app.charts.render as render_module
from app.charts.render import CastedFrameCache, render_chart
from app.charts.spec import ChartSpec, validate_spec
from app.profiling.profiler import profile_dataset
from app.profiling.types import apply_semantic_casts

BIG = 10**6
D = datetime(2024, 1, 1)


def _render(df: pl.DataFrame, **kwargs):
    profile = profile_dataset(df, "0" * 32, BIG)
    casted = apply_semantic_casts(df, profile)
    spec = ChartSpec(title="t", **kwargs)
    assert validate_spec(spec, profile) == [], "test fixture spec must be valid"
    return render_chart(casted, spec, profile)


# --- line -------------------------------------------------------------------


def test_line_raw_sorted_by_x() -> None:
    df = pl.DataFrame({"ts": [D + timedelta(days=1), D, D + timedelta(days=2)], "v": [10.0, 20.0, 30.0]})
    result = _render(df, type="line", x="ts", y="v")
    (series,) = result["chart_data"]["series"]
    assert series["name"] == "v"
    assert series["x"] == ["2024-01-01T00:00:00", "2024-01-02T00:00:00", "2024-01-03T00:00:00"]
    assert series["y"] == [20.0, 10.0, 30.0]
    assert result["n_points"] == 3 and result["sampled"] is False
    assert result["spec"]["time_granularity"] is None  # echo untouched


def test_line_month_truncate_mean() -> None:
    df = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 5), datetime(2024, 1, 20), datetime(2024, 2, 10)],
            "v": [1.0, 3.0, 10.0],
        }
    )
    result = _render(df, type="line", x="ts", y="v", time_granularity="month", aggregation="mean")
    (series,) = result["chart_data"]["series"]
    assert series["x"] == ["2024-01-01T00:00:00", "2024-02-01T00:00:00"]
    assert series["y"] == [2.0, 10.0]
    assert result["chart_data"]["y_label"] == "mean(v)"


def test_line_grouped_count_series() -> None:
    df = pl.DataFrame(
        {
            "ts": [D, D, D + timedelta(days=1), D],
            "g": ["a", "b", "a", "a"],
            "v": [1.0, 2.0, 3.0, 4.0],
        }
    )
    result = _render(df, type="line", x="ts", y="v", group_by="g", aggregation="count")
    series = result["chart_data"]["series"]
    assert [s["name"] for s in series] == ["a", "b"]  # lexicographic
    assert series[0]["x"] == ["2024-01-01T00:00:00", "2024-01-02T00:00:00"]
    assert series[0]["y"] == [2, 1]
    assert series[1]["y"] == [1]
    assert result["chart_data"]["y_label"] == "count"


def test_line_raw_over_limit_downgrades_and_echoes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render_module, "LINE_MAX_POINTS", 100)
    df = pl.DataFrame(
        {"ts": [D + timedelta(days=i) for i in range(600)], "v": [float(i % 13) for i in range(600)]}
    )
    result = _render(df, type="line", x="ts", y="v")
    # blocking #2: echoed spec reflects the granularity actually rendered
    assert result["spec"]["time_granularity"] == "week"
    assert result["spec"]["aggregation"] == "mean"
    assert 0 < result["n_points"] <= 100
    assert result["chart_data"]["y_label"] == "mean(v)"


def test_line_empty_after_null_drop() -> None:
    df = pl.DataFrame({"ts": [D, None], "v": [None, 5.0]})
    result = _render(df, type="line", x="ts", y="v")
    assert result["chart_data"]["series"] == [] and result["n_points"] == 0


def test_line_y_label_without_aggregation() -> None:
    df = pl.DataFrame({"ts": [D, D + timedelta(days=1)], "v": [1.0, 2.0]})
    assert _render(df, type="line", x="ts", y="v")["chart_data"]["y_label"] == "v"


# --- bar --------------------------------------------------------------------


def test_bar_mean_ordering_and_values() -> None:
    df = pl.DataFrame({"c": ["a", "a", "b"], "v": [1.0, 3.0, 10.0]})
    result = _render(df, type="bar", x="c", y="v", aggregation="mean")
    data = result["chart_data"]
    assert data["categories"] == ["b", "a"]  # by aggregate desc
    assert data["series"] == [{"name": "v", "values": [10.0, 2.0]}]
    assert data["truncated"] is False
    assert data["y_label"] == "mean(v)"


def test_bar_count_naming() -> None:
    df = pl.DataFrame({"c": ["a", "a", "b"], "v": [1.0, 2.0, 3.0]})
    result = _render(df, type="bar", x="c", aggregation="count")
    data = result["chart_data"]
    assert data["categories"] == ["a", "b"]
    assert data["series"] == [{"name": "count", "values": [2, 1]}]
    assert data["y_label"] == "count"


def test_bar_top_n_truncation() -> None:
    df = pl.DataFrame(
        {"c": [f"c{i}" for i in range(5) for _ in range(2)], "v": [float(5 - i) for i in range(5) for _ in range(2)]}
    )
    result = _render(df, type="bar", x="c", y="v", aggregation="mean", top_n=3)
    data = result["chart_data"]
    assert data["categories"] == ["c0", "c1", "c2"]
    assert data["truncated"] is True


def test_bar_grouped_null_alignment() -> None:
    df = pl.DataFrame({"c": ["a", "a", "b"], "g": ["g1", "g2", "g1"], "v": [1.0, 5.0, 10.0]})
    result = _render(df, type="bar", x="c", y="v", aggregation="mean", group_by="g")
    data = result["chart_data"]
    # ranking ignores group: mean(a)=3, mean(b)=10 -> [b, a]
    assert data["categories"] == ["b", "a"]
    assert data["series"] == [
        {"name": "g1", "values": [10.0, 1.0]},
        {"name": "g2", "values": [None, 5.0]},  # missing (b, g2) cell -> null
    ]
    assert result["n_points"] == 3  # nulls not counted


def test_bar_ranks_by_absolute_value() -> None:
    df = pl.DataFrame({"c": ["neg", "neg", "pos"], "v": [-100.0, -100.0, 5.0]})
    result = _render(df, type="bar", x="c", y="v", aggregation="mean")
    assert result["chart_data"]["categories"] == ["neg", "pos"]
    assert result["chart_data"]["series"][0]["values"] == [-100.0, 5.0]


def test_bar_empty_after_null_drop() -> None:
    df = pl.DataFrame({"c": ["a", None], "v": [None, 2.0]})  # every row loses one axis
    result = _render(df, type="bar", x="c", y="v", aggregation="mean")
    assert result["chart_data"]["categories"] == [] and result["chart_data"]["series"] == []
    assert result["n_points"] == 0


# --- scatter ----------------------------------------------------------------


def test_scatter_drops_nulls_and_names_series() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, None, 4.0], "y": [2.0, 4.0, 6.0, None]})
    result = _render(df, type="scatter", x="x", y="y")
    (series,) = result["chart_data"]["series"]
    assert series == {"name": "y", "x": [1.0, 2.0], "y": [2.0, 4.0]}
    assert result["sampled"] is False and result["n_points"] == 2


def test_scatter_sampling_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render_module, "SCATTER_MAX_POINTS", 50)
    df = pl.DataFrame({"x": [float(i) for i in range(200)], "y": [float(i * 2) for i in range(200)]})
    r1 = _render(df, type="scatter", x="x", y="y")
    r2 = _render(df, type="scatter", x="x", y="y")
    assert r1["sampled"] is True and r1["n_points"] == 50
    assert r1 == r2  # seeded sampling


def test_scatter_grouped_series_sorted() -> None:
    df = pl.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 1.0, 4.0, 3.0], "g": ["b", "a", "b", "a"]}
    )
    result = _render(df, type="scatter", x="x", y="y", group_by="g")
    assert [s["name"] for s in result["chart_data"]["series"]] == ["a", "b"]
    assert result["chart_data"]["series"][0]["x"] == [2.0, 4.0]


# --- histogram --------------------------------------------------------------


def test_histogram_hand_calculated_bins() -> None:
    df = pl.DataFrame({"v": [float(i) for i in range(10)]})
    result = _render(df, type="histogram", x="v", bins=5)
    bins = result["chart_data"]["bins"]
    assert bins["edges"][0] == 0.0 and bins["edges"][-1] == 9.0 and len(bins["edges"]) == 6
    assert bins["counts"] == [2, 2, 2, 2, 2]
    assert result["chart_data"]["y_label"] == "count"


def test_histogram_counts_exclude_null_and_nan() -> None:
    df = pl.DataFrame({"v": [1.0, 2.0, None, float("nan"), 3.0]})
    result = _render(df, type="histogram", x="v")
    assert sum(result["chart_data"]["bins"]["counts"]) == 3


def test_histogram_constant_column_single_bucket() -> None:
    df = pl.DataFrame({"v": [5.0] * 8})
    bins = _render(df, type="histogram", x="v")["chart_data"]["bins"]
    assert bins["counts"] == [8]
    assert bins["edges"] == [4.5, 5.5]


def test_histogram_grouped_shares_edges() -> None:
    df = pl.DataFrame({"v": [float(i) for i in range(10)], "g": ["a"] * 5 + ["b"] * 5})
    result = _render(df, type="histogram", x="v", bins=5, group_by="g")
    data = result["chart_data"]
    assert data["bins"]["counts"] == [2, 2, 2, 2, 2]
    a, b = data["series"]
    assert a["name"] == "a" and a["edges"] == data["bins"]["edges"] and a["counts"] == [2, 2, 1, 0, 0]
    assert b["name"] == "b" and b["edges"] == data["bins"]["edges"] and b["counts"] == [0, 0, 1, 2, 2]


# --- box --------------------------------------------------------------------


def test_box_hand_calculated_summary() -> None:
    df = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 100.0]})
    (group,) = _render(df, type="box", y="v")["chart_data"]["groups"]
    assert group["name"] == "all"
    assert (group["q1"], group["median"], group["q3"]) == (2.0, 3.0, 4.0)  # linear interpolation
    assert (group["lower_fence"], group["upper_fence"]) == (-1.0, 7.0)
    assert group["outliers"] == [100.0]
    assert group["mean"] == 22.0 and group["count"] == 5


def test_box_groups_sorted_by_category() -> None:
    df = pl.DataFrame({"c": ["b", "a", "b", "a"], "v": [4.0, 1.0, 6.0, 3.0]})
    groups = _render(df, type="box", x="c", y="v")["chart_data"]["groups"]
    assert [g["name"] for g in groups] == ["a", "b"]
    assert groups[0]["median"] == 2.0 and groups[1]["median"] == 5.0


def test_box_outlier_cap_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render_module, "BOX_MAX_OUTLIERS", 5)
    df = pl.DataFrame({"v": [0.0] * 100 + [1000.0 + i for i in range(20)]})
    r1 = _render(df, type="box", y="v")
    r2 = _render(df, type="box", y="v")
    assert len(r1["chart_data"]["groups"][0]["outliers"]) == 5
    assert r1["sampled"] is True and r1 == r2


# --- heatmap ----------------------------------------------------------------


def test_heatmap_uses_profile_correlations() -> None:
    df = pl.DataFrame(
        {"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0], "c": [3.0, 1.0, 2.0]}
    )
    profile = profile_dataset(df, "0" * 32, BIG)
    result = render_chart(df, ChartSpec(title="t", type="heatmap"), profile)
    data = result["chart_data"]
    assert data["columns"] == profile.correlations.columns
    assert data["matrix"] == profile.correlations.matrix
    assert data["y_label"] == "correlation"
    assert result["n_points"] == 9


# --- cast cache -------------------------------------------------------------


def test_casted_frame_cache_single_slot() -> None:
    df = pl.DataFrame({"amount": ["1,000", "2,000"], "v": [1.0, 2.0]})
    profile = profile_dataset(df, "0" * 32, BIG)
    cache = CastedFrameCache()
    first = cache.get("a" * 32, df, profile)
    assert first.schema["amount"] == pl.Float64  # cast replayed
    assert cache.get("a" * 32, df, profile) is first  # cached object
    assert cache.get("b" * 32, df, profile) is not first  # slot replaced


# --- API --------------------------------------------------------------------

CSV = (
    "date,city,amount\n"
    "05/01/2024,North,\"$1,000\"\n"
    "06/01/2024,South,\"$3,000\"\n"
    "07/01/2024,North,\"$2,000\"\n"
    "08/01/2024,South,\"$5,000\"\n"
)


def _upload(client: TestClient) -> str:
    resp = client.post("/api/datasets", files={"file": ("sales.csv", CSV.encode())})
    assert resp.status_code == 200, resp.text
    return resp.json()["dataset_id"]


def _post_render(client: TestClient, dataset_id: str, **spec):
    return client.post("/api/charts/render", json={"dataset_id": dataset_id, "spec": {"title": "t", **spec}})


def test_render_api_line_with_cast_replay(client: TestClient) -> None:
    dataset_id = _upload(client)
    resp = _post_render(client, dataset_id, type="line", x="date", y="amount")
    assert resp.status_code == 200, resp.text
    (series,) = resp.json()["chart_data"]["series"]
    assert series["y"] == [1000.0, 3000.0, 2000.0, 5000.0]  # currency strings replayed
    assert series["x"][0].startswith("2024-01-05")


def test_render_api_bar_mean(client: TestClient) -> None:
    dataset_id = _upload(client)
    resp = _post_render(client, dataset_id, type="bar", x="city", y="amount", aggregation="mean")
    data = resp.json()["chart_data"]
    assert data["categories"] == ["South", "North"]  # 4000 > 1500
    assert data["series"] == [{"name": "amount", "values": [4000.0, 1500.0]}]


def test_render_api_invalid_spec_and_pie_share_422_shape(client: TestClient) -> None:
    dataset_id = _upload(client)
    bad = _post_render(client, dataset_id, type="scatter", x="city", y="amount", aggregation="mean")
    pie = _post_render(client, dataset_id, type="pie", x="city")
    assert bad.status_code == pie.status_code == 422
    for resp in (bad, pie):
        body = resp.json()
        assert set(body) == {"detail"} and set(body["detail"]) == {"errors"}
        assert isinstance(body["detail"]["errors"], list) and body["detail"]["errors"]
    assert len(bad.json()["detail"]["errors"]) >= 2  # all errors collected


def test_render_api_404(client: TestClient) -> None:
    spec = {"title": "t", "type": "histogram", "x": "v"}
    assert client.post("/api/charts/render", json={"dataset_id": "nope", "spec": spec}).status_code == 404
    assert client.post("/api/charts/render", json={"dataset_id": "0" * 32, "spec": spec}).status_code == 404
    assert client.post("/api/charts/render", json={"spec": spec}).status_code == 404


def test_render_api_no_nan_or_infinity(client: TestClient) -> None:
    csv = "v\n1.5\nNaN\ninf\n2.5\n3.5\n"
    dataset_id = client.post("/api/datasets", files={"file": ("n.csv", csv.encode())}).json()["dataset_id"]
    resp = _post_render(client, dataset_id, type="histogram", x="v")
    assert resp.status_code == 200
    assert "NaN" not in resp.text and "Infinity" not in resp.text
    assert sum(resp.json()["chart_data"]["bins"]["counts"]) == 3


def test_numeric_x_line_stride_sampled() -> None:
    # 30k unique numeric x values: no granularity to downgrade to, so the
    # series is stride-sampled instead of hitting the hard point limit
    import polars as pl

    from app.charts.render import LINE_MAX_POINTS, render_chart
    from app.charts.spec import ChartSpec
    from app.profiling.profiler import profile_dataset

    n = 30_000
    df = pl.DataFrame({"a": [float(i) for i in range(n)], "b": [float(i % 7) for i in range(n)]})
    profile = profile_dataset(df, "0" * 32, 100_000)
    spec = ChartSpec(title="t", type="line", x="a", y="b")
    chart_data, sampled, n_points = (
        render_chart(df, spec, profile)["chart_data"],
        render_chart(df, spec, profile)["sampled"],
        render_chart(df, spec, profile)["n_points"],
    )
    assert sampled is True
    assert n_points <= LINE_MAX_POINTS
    xs = chart_data["series"][0]["x"]
    assert xs == sorted(xs) and xs[0] == 0.0
