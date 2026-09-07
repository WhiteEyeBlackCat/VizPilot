from collections import Counter
from datetime import datetime, timedelta

import polars as pl
import pytest
from fastapi.testclient import TestClient

from app.charts.rules import choose_time_granularity, recommend_charts
from app.charts.spec import validate_spec
from app.profiling.profiler import profile_dataset

BIG = 10**6


def _profile(df: pl.DataFrame):
    return profile_dataset(df, "0" * 32, BIG)


def _mixed_df(n: int = 60) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)],
            "city": ["North", "South", "East"] * (n // 3),
            "value": [float(i) for i in range(n)],
            "value2": [2.0 * i for i in range(n)],
            "value3": [float(i % 2) for i in range(n)],  # ~uncorrelated with value
            "cat25": [f"g{i % 25:02d}" for i in range(n)],
            "user_id": [f"u{i:03d}" for i in range(n)],
            "note": [f"a reasonably long free-form comment number {i}" for i in range(n)],
            "mostly_null": [float(i) if i < n // 5 else None for i in range(n)],
        }
    )


def _wide_df(n: int = 40) -> pl.DataFrame:
    data = {f"n{k}": [float(i * (k + 1)) for i in range(n)] for k in range(8)}
    data["ts"] = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    data["city"] = ["a", "b", "c", "d"] * (n // 4)
    return pl.DataFrame(data)


@pytest.fixture(scope="module")
def mixed_recs():
    profile = _profile(_mixed_df())
    return profile, recommend_charts(profile)


# --- rule outputs on a mixed dataset ----------------------------------------


def test_covers_all_chart_types(mixed_recs) -> None:
    _, recs = mixed_recs
    assert {r.spec.type for r in recs} == {"line", "bar", "box", "scatter", "histogram", "heatmap"}


def test_all_outputs_pass_validation(mixed_recs) -> None:
    profile, recs = mixed_recs
    for rec in recs:
        assert validate_spec(rec.spec, profile) == [], rec.spec


def test_skips_id_text_and_high_missing_columns(mixed_recs) -> None:
    _, recs = mixed_recs
    for rec in recs:
        used = {rec.spec.x, rec.spec.y, rec.spec.group_by}
        assert used.isdisjoint({"user_id", "note", "mostly_null"})


def test_high_cardinality_categorical_not_used(mixed_recs) -> None:
    _, recs = mixed_recs
    assert all(r.spec.x != "cat25" for r in recs if r.spec.type in ("bar", "box"))


def test_line_charts_scored_by_time_effect(mixed_recs) -> None:
    _, recs = mixed_recs
    lines = [r for r in recs if r.spec.type == "line"]
    assert lines and all(r.spec.x == "ts" for r in lines)
    # stage7: grouping now requires interaction/main-effect evidence — this
    # dataset has none (city is unrelated to the numeric columns)
    assert all(r.spec.group_by is None for r in lines)
    # the time effect drives ranking: trending series above the noise series
    noise = next(r for r in lines if r.spec.y == "value3")
    trending = [r for r in lines if r.spec.y in ("value", "value2")]
    assert trending and all(r.score > noise.score for r in trending)
    assert noise.score == pytest.approx(0.55)  # 0.5 + daily regularity bonus only


def test_count_bar_present(mixed_recs) -> None:
    _, recs = mixed_recs
    assert any(
        r.spec.type == "bar" and r.spec.y is None and r.spec.aggregation == "count" for r in recs
    )


def test_scatter_uses_correlation(mixed_recs) -> None:
    _, recs = mixed_recs
    scatters = [r for r in recs if r.spec.type == "scatter"]
    assert len(scatters) == 1  # only |corr(value, value2)| >= 0.3
    top = scatters[0]
    assert {top.spec.x, top.spec.y} == {"value", "value2"}
    # stage7: grouping needs slope-heterogeneity evidence; the correlation is
    # identical inside every city group, so the scatter stays ungrouped
    assert top.spec.group_by is None
    assert top.score == pytest.approx(0.5 + 0.4 * 1.0)


def test_dedup_and_priorities(mixed_recs) -> None:
    _, recs = mixed_recs
    keys = [(r.spec.type, r.spec.x, r.spec.y, r.spec.group_by) for r in recs]
    assert len(keys) == len(set(keys))
    assert [r.spec.priority for r in recs] == list(range(1, len(recs) + 1))
    assert all(a.score >= b.score for a, b in zip(recs, recs[1:]))


# --- scatter ranking follows |corr| -----------------------------------------


def test_higher_correlation_ranks_scatter_higher() -> None:
    n = 60
    df = pl.DataFrame(
        {
            "a": [float(i) for i in range(n)],
            "b": [2.0 * i + 1 for i in range(n)],  # corr(a, b) = 1.0
            "c1": [100.0, 200.0, 300.0, 400.0] * (n // 4),
            "c2": [200.0, 100.0, 400.0, 300.0] * (n // 4),  # corr(c1, c2) = 0.6
        }
    )
    recs = recommend_charts(_profile(df))
    scatters = [r for r in recs if r.spec.type == "scatter"]
    pairs = [{r.spec.x, r.spec.y} for r in scatters]
    assert pairs[0] == {"a", "b"} and {"c1", "c2"} in pairs
    strong, weak = scatters[0], scatters[pairs.index({"c1", "c2"})]
    assert strong.score == pytest.approx(0.9)
    assert weak.score == pytest.approx(0.5 + 0.4 * 0.6)
    assert strong.spec.priority < weak.spec.priority


# --- granularity ------------------------------------------------------------


@pytest.mark.parametrize(
    "unique_count,span_days,expected",
    [
        (400, 100, "raw"),
        (5000, 0.5, "raw"),  # intraday: "day" would collapse to one point
        (5000, 1.9, "raw"),
        (501, 450, "day"),
        (600, 2000, "week"),  # 2000/7 = 285 <= 500
        (600, 5000, "month"),  # 5000/7 > 500, 5000/30 = 167
        (600, 40000, "month"),  # every estimate > 500 -> month fallback
    ],
)
def test_choose_time_granularity(unique_count: int, span_days: float, expected: str) -> None:
    assert choose_time_granularity(unique_count, span_days) == expected


def test_long_series_gets_coarse_granularity_with_mean() -> None:
    n = 600  # 600 unique days: raw and day > 500 -> week
    df = pl.DataFrame(
        {
            "ts": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)],
            "value": [float(i % 37) for i in range(n)],
        }
    )
    profile = _profile(df)
    lines = [r for r in recommend_charts(profile) if r.spec.type == "line"]
    assert lines[0].spec.time_granularity == "week"
    assert lines[0].spec.aggregation == "mean"
    assert validate_spec(lines[0].spec, profile) == []


# --- caps and diversity -----------------------------------------------------


def test_diversity_and_overall_cap() -> None:
    recs = recommend_charts(_profile(_wide_df()))
    assert len(recs) == 12  # 16 type-capped candidates -> overall cap
    by_type = Counter(r.spec.type for r in recs)
    assert all(count <= 3 for count in by_type.values())
    assert len(by_type) >= 4  # a wide dataset must not collapse into one type


def test_empty_recommendations_for_unusable_dataset() -> None:
    df = pl.DataFrame(
        {
            "user_id": [f"u{i:04d}" for i in range(30)],
            "note": [f"quite a long unique piece of free text number {i}" for i in range(30)],
        }
    )
    assert recommend_charts(_profile(df)) == []


# --- property-style: every rule output validates on varied datasets ----------


@pytest.mark.parametrize(
    "df",
    [
        _mixed_df(),
        _wide_df(),
        pl.DataFrame({"only_num": [float(i) for i in range(20)]}),
        pl.DataFrame({"city": ["a", "b", "c"] * 10}),
        pl.DataFrame(
            {
                "ts": [datetime(2024, 1, 1) + timedelta(days=i // 3) for i in range(30)],
                "value": [float(i) for i in range(30)],  # duplicated timestamps (long format)
            }
        ),
        pl.DataFrame({"flag": [True, False] * 15, "value": [float(i) for i in range(30)]}),
    ],
    ids=["mixed", "wide", "single-numeric", "categorical-only", "duplicated-ts", "boolean"],
)
def test_property_all_outputs_validate(df: pl.DataFrame) -> None:
    profile = _profile(df)
    for rec in recommend_charts(profile):
        assert validate_spec(rec.spec, profile) == [], rec.spec
        assert rec.source == "rules"
        assert 0 < rec.score <= 1.0


# --- API --------------------------------------------------------------------

CSV = (
    "date,city,value,value2\n"
    + "\n".join(
        f"2024-01-{i + 1:02d},{'North South East'.split()[i % 3]},{i}.0,{2 * i}.0" for i in range(12)
    )
    + "\n"
)


def test_recommendations_endpoint_shape(client: TestClient) -> None:
    dataset_id = client.post("/api/datasets", files={"file": ("m.csv", CSV.encode())}).json()["dataset_id"]
    resp = client.get(f"/api/datasets/{dataset_id}/recommendations")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"charts", "insights", "message"}
    assert body["insights"] == [] and body["message"] is None
    assert len(body["charts"]) >= 3
    first = body["charts"][0]
    assert set(first) == {"spec", "score", "source", "tier"}
    assert first["source"] == "rules"
    assert first["tier"] in ("top", "secondary", "exploratory")
    assert first["spec"]["priority"] == 1
    assert len({c["spec"]["type"] for c in body["charts"]}) >= 2
    assert "NaN" not in resp.text and "Infinity" not in resp.text


def test_recommendations_endpoint_404(client: TestClient) -> None:
    assert client.get(f"/api/datasets/{'0' * 32}/recommendations").status_code == 404
    assert client.get("/api/datasets/nope/recommendations").status_code == 404


def test_recommendations_empty_with_message(client: TestClient) -> None:
    csv = "note\n" + "\n".join(f"quite a long unique piece of user feedback number {i}" for i in range(30)) + "\n"
    dataset_id = client.post("/api/datasets", files={"file": ("t.csv", csv.encode())}).json()["dataset_id"]
    body = client.get(f"/api/datasets/{dataset_id}/recommendations").json()
    assert body["charts"] == []
    assert isinstance(body["message"], str) and body["message"]
