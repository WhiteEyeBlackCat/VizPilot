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
    # exact key sets: the response shape is the frontend contract (stage 9
    # added the additive `warnings` / `confidence` fields)
    assert set(body) == {"charts", "insights", "message", "warnings"}
    assert body["insights"] == [] and body["message"] is None and body["warnings"] == []
    assert len(body["charts"]) >= 3
    first = body["charts"][0]
    assert set(first) == {"spec", "score", "source", "tier", "confidence", "warnings"}
    assert first["confidence"]["n_source"] in ("exact", "estimated")
    assert "tier_cap" not in first
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


# --- stage 9 #6: the histogram skew bonus is not earned by suspected sentinels


def test_histogram_skew_bonus_not_earned_by_sentinels() -> None:
    import random

    from app.charts.rules import evaluate_llm_spec
    from app.charts.spec import ChartSpec
    from app.profiling.models import SentinelCandidate

    rng = random.Random(41)
    values = [rng.gauss(25, 2) for _ in range(585)] + [9999.0] * 15  # skew manufactured by a sentinel
    genuine = [rng.lognormvariate(0, 1) for _ in range(600)]  # genuinely skewed
    profile = _profile(pl.DataFrame({"t": values, "rev": genuine}))
    t = next(c for c in profile.columns if c.name == "t")
    assert t.skewness is not None and abs(t.skewness) > 1
    t.quality.suspected_sentinels = [SentinelCandidate(value=9999.0, count=15, signals=["extreme", "repeated", "pattern"])]
    recs = recommend_charts(profile)
    hist_t = next(r for r in recs if r.spec.type == "histogram" and r.spec.x == "t")
    hist_rev = next(r for r in recs if r.spec.type == "histogram" and r.spec.x == "rev")
    # compare base scores: the confidence layer may also discount either
    # column (rev's heavy tail is a legitimate extreme_value case)
    assert hist_t.score / hist_t.confidence.overall == pytest.approx(0.5)  # no bonus
    assert hist_rev.score / hist_rev.confidence.overall == pytest.approx(0.55)  # genuine skew keeps it
    assert evaluate_llm_spec(ChartSpec(title="h", type="histogram", x="t"), profile)[0] == 0.5
    assert evaluate_llm_spec(ChartSpec(title="h", type="histogram", x="rev"), profile)[0] == 0.55


# --- derived columns: definitional pairs sink (stage 13) --------------------

from app.charts.rules import (  # noqa: E402
    DERIVED_SCORE_FACTOR,
    Recommendation,
    derived_column_warnings,
    evaluate_llm_spec,
)
from app.charts.spec import ChartSpec  # noqa: E402


def _derived_df(n: int = 400) -> pl.DataFrame:
    import random

    rng = random.Random(21)
    price = [round(rng.uniform(50, 5000), 2) for _ in range(n)]
    qty = [rng.randrange(1, 11) for _ in range(n)]
    u = [rng.uniform(0, 10) for _ in range(n)]
    return pl.DataFrame(
        {
            "region": [["N", "S", "E"][i % 3] for i in range(n)],
            "unit_price": price,
            "quantity": qty,
            "sales": [round(p * q, 2) for p, q in zip(price, qty)],
            "u": u,
            "v": [x + rng.gauss(0, 2) for x in u],  # r ~ 0.8: a real relationship
        }
    )


@pytest.fixture(scope="module")
def derived_recs():
    profile = _profile(_derived_df())
    return profile, recommend_charts(profile)


def test_derived_pairs_are_capped_exploratory_and_explained(derived_recs) -> None:
    profile, recs = derived_recs
    assert [d.formula for d in profile.evidence.derived_columns] == ["unit_price × quantity"]
    definitional = [
        r for r in recs if r.spec.x and r.spec.y and {r.spec.x, r.spec.y} in ({"unit_price", "sales"}, {"quantity", "sales"})
    ]
    assert definitional, "the price/quantity vs sales charts must still be listed, just demoted"
    for rec in definitional:
        assert rec.tier == "exploratory"
        assert rec.spec.reason.startswith("sales is computed as unit_price × quantity; this relationship is definitional")
        assert [w.code for w in rec.warnings] == ["derived_relationship"]
        assert rec.warnings[0].severity == "info" and rec.warnings[0].meta["target"] == "sales"
    for rec in recs:
        if rec not in definitional:
            assert "definitional" not in rec.spec.reason
            assert all(w.code != "derived_relationship" for w in rec.warnings)


def test_derived_scatter_ranks_after_real_relationships(derived_recs) -> None:
    _, recs = derived_recs
    scatters = [r for r in recs if r.spec.type == "scatter"]
    assert {scatters[0].spec.x, scatters[0].spec.y} == {"u", "v"}
    derived = next(r for r in scatters if {r.spec.x, r.spec.y} == {"unit_price", "sales"})
    # score = (0.5 + 0.4 * strength) x confidence x DERIVED_SCORE_FACTOR
    base = derived.score / derived.confidence.overall / DERIVED_SCORE_FACTOR
    assert base == pytest.approx(0.5 + 0.4 * max(abs(v) for v in [
        _pearson(derived_recs[0], "unit_price", "sales"),
        derived_recs[0].evidence.num_num_spearman.matrix[
            derived_recs[0].evidence.num_num_spearman.columns.index("unit_price")
        ][derived_recs[0].evidence.num_num_spearman.columns.index("sales")],
    ]))


def _pearson(profile, a, b):
    corr = profile.correlations
    return corr.matrix[corr.columns.index(a)][corr.columns.index(b)]


def test_llm_hypothesis_on_derived_pair_is_unverified(derived_recs) -> None:
    profile, _ = derived_recs
    spec = ChartSpec(title="t", type="scatter", x="unit_price", y="sales")
    assert evaluate_llm_spec(spec, profile) == (0.5, "unverified")
    spec = ChartSpec(title="t", type="bar", x="quantity", y="sales", aggregation="mean")
    assert evaluate_llm_spec(spec, profile) == (0.5, "unverified")
    # a real pair keeps its evidence grade
    assert evaluate_llm_spec(ChartSpec(title="t", type="scatter", x="u", y="v"), profile)[1] == "strong"


def test_derived_dataset_warning(derived_recs) -> None:
    profile, _ = derived_recs
    warnings = derived_column_warnings(profile)
    assert [(w.code, w.severity, w.meta["target"], w.meta["components"]) for w in warnings] == [
        ("derived_column", "info", "sales", ["unit_price", "quantity"])
    ]
    assert "computed as unit_price × quantity" in warnings[0].message


def test_near_copy_is_disclosed_but_not_demoted(mixed_recs) -> None:
    # value2 = 2 * value ranks identically: the pair keeps its evidence
    # score and tier (the transform is unknown), it just says so
    _, recs = mixed_recs
    scatter = next(r for r in recs if r.spec.type == "scatter" and {r.spec.x, r.spec.y} == {"value", "value2"})
    assert scatter.tier == "top"
    assert scatter.spec.reason.startswith("value2 is nearly a transformed copy of value")
    assert [(w.code, w.meta["kind"]) for w in scatter.warnings] == [("derived_relationship", "near_copy")]
