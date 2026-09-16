"""Stage 19 end-to-end cases: evidence + roles → intent → chart, through
recommend_charts and the renderer, on planted datasets."""

import math
import random
from datetime import datetime, timedelta

import polars as pl
import pytest

from app.charts.render import render_chart
from app.charts.rules import Recommendation, recommend_charts
from app.profiling.profiler import profile_dataset
from app.profiling.types import apply_semantic_casts
from tests.test_recommendation_quality import _real_dataset

BIG = 10**6
FANCY = {"violin", "raincloud", "ridgeline", "dot_interval", "lollipop", "dumbbell", "slopegraph", "hexbin", "polar_line", "rose"}
DENSITY = {"violin", "raincloud", "ridgeline"}


def _recs(df: pl.DataFrame):
    profile = profile_dataset(df, "0" * 32, BIG)
    return profile, recommend_charts(profile)


def _find(recs: list[Recommendation], **match) -> Recommendation | None:
    for rec in recs:
        if all(getattr(rec.spec, k) == v for k, v in match.items()):
            return rec
    return None


def _all_types(recs: list[Recommendation]) -> set[str]:
    return {r.spec.type for r in recs} | {a.type for r in recs for a in r.alternatives}


# --- 1. month × temperature: cyclic, never value-sorted ---------------------------


def _month_df(heteroscedastic: bool, n: int = 1200) -> pl.DataFrame:
    rng = random.Random(1)
    rows = []
    for i in range(n):
        m = (i % 12) + 1
        spread = (0.5 + 2.5 * (m / 12)) if heteroscedastic else 1.0
        rows.append({"mnth": m, "temp": 15 + 10 * math.sin((m - 1) / 12 * 2 * math.pi) + rng.gauss(0, spread), "region": ["N", "S"][i % 2]})
    return pl.DataFrame(rows)


def test_month_temperature_is_a_cycle_in_month_order() -> None:
    profile, recs = _recs(_month_df(heteroscedastic=False))
    rec = _find(recs, x="mnth", y="temp")
    assert rec is not None and rec.intent == "cyclic_pattern"
    assert rec.spec.type == "line" and rec.spec.sort == "natural" and rec.spec.aggregation == "mean"
    assert "polar_line" in {a.type for a in rec.alternatives}
    assert all(r.spec.sort != "value" for r in recs if r.spec.x == "mnth")
    rendered = render_chart(apply_semantic_casts(_month_df(False), profile), rec.spec, profile)
    assert rendered["chart_data"]["series"][0]["x"] == list(range(1, 13))
    polar = next(a for a in rec.alternatives if a.type == "polar_line")
    assert render_chart(apply_semantic_casts(_month_df(False), profile), polar, profile)["chart_data"]["categories"] == list(range(1, 13))


def test_month_temperature_with_spread_evidence_becomes_a_distribution_by_group() -> None:
    profile, recs = _recs(_month_df(heteroscedastic=True))
    rec = _find(recs, x="mnth", y="temp")
    assert rec is not None and rec.intent == "distribution_by_group"
    assert rec.spec.type in ("raincloud", "ridgeline") and rec.spec.sort == "natural"
    assert {"box"} <= {a.type for a in rec.alternatives}
    rendered = render_chart(apply_semantic_casts(_month_df(True), profile), rec.spec, profile)
    assert [g["name"] for g in rendered["chart_data"]["groups"]] == [str(m) for m in range(1, 13)]


# --- 2. region × sales: nominal group comparison, never a rose ----------------------


def test_region_sales_is_nominal_and_never_polar() -> None:
    rng = random.Random(2)
    offsets = {"North": -1.5, "Central": 0.0, "South": 1.5, "East": 0.5}
    df = pl.DataFrame(
        {
            "region": [list(offsets)[i % 4] for i in range(800)],
            "sales": [offsets[list(offsets)[i % 4]] + rng.gauss(0, 1) for i in range(800)],
        }
    )
    _, recs = _recs(df)
    rec = _find(recs, x="region", y="sales")
    assert rec is not None and rec.intent == "group_comparison" and rec.spec.type == "bar"
    assert rec.tier == "top"
    assert not ({"rose", "polar_line"} & _all_types(recs))


# --- 3 / 4. numeric × numeric: hexbin only under overplotting ----------------------------


def _xy_df(n: int) -> pl.DataFrame:
    rng = random.Random(3)
    x = [rng.gauss(0, 1) for _ in range(n)]
    return pl.DataFrame({"x": x, "y": [v * 0.8 + rng.gauss(0, 0.6) for v in x], "z": [rng.gauss(0, 1) for _ in range(n)]})


def test_large_numeric_pair_prefers_hexbin() -> None:
    _, recs = _recs(_xy_df(6000))
    rec = _find(recs, x="x", y="y")
    assert rec is not None and rec.intent == "numeric_relationship"
    assert rec.spec.type == "hexbin" and {a.type for a in rec.alternatives} == {"scatter"}


def test_small_numeric_pair_stays_a_scatter() -> None:
    _, recs = _recs(_xy_df(300))
    rec = _find(recs, x="x", y="y")
    assert rec is not None and rec.spec.type == "scatter"
    assert {a.type for a in rec.alternatives} <= {"hexbin"}
    _, tiny_recs = _recs(_xy_df(120))
    assert _find(tiny_recs, x="x", y="y").alternatives == []


# --- 5. before / after × category: two states ------------------------------------------


def test_before_after_by_category_is_a_dumbbell() -> None:
    rng = random.Random(5)
    rows = []
    for i in range(1200):
        cat = ["A", "B", "C", "D"][i % 4]
        period = "before" if rng.random() < 0.5 else "after"  # crossed with the category
        lift = {"A": 3.0, "B": 0.0, "C": -2.0, "D": 1.0}[cat] if period == "after" else 0.0
        rows.append({"category": cat, "period": period, "value": {"A": 5, "B": 6, "C": 7, "D": 8}[cat] + lift + rng.gauss(0, 1)})
    _, recs = _recs(pl.DataFrame(rows))
    rec = _find(recs, x="category", y="value", group_by="period")
    assert rec is not None and rec.intent == "two_state_comparison"
    assert rec.spec.type == "dumbbell" and rec.spec.aggregation == "mean"
    assert "slopegraph" in {a.type for a in rec.alternatives}


# --- 6. ordered groups × distribution ---------------------------------------------------


def test_ordered_groups_with_different_spreads_get_a_ridge_or_raincloud() -> None:
    rng = random.Random(6)
    levels = ["low", "medium", "high", "very high"]
    df = pl.DataFrame(
        {
            "level": [levels[i % 4] for i in range(1600)],
            "score": [{"low": 10, "medium": 12, "high": 14, "very high": 16}[levels[i % 4]] + rng.gauss(0, 0.4 + 1.2 * (i % 4)) for i in range(1600)],
        }
    )
    profile, recs = _recs(df)
    rec = _find(recs, x="level", y="score")
    assert rec is not None and rec.intent == "distribution_by_group"
    assert rec.spec.type in ("raincloud", "ridgeline") and {"ridgeline", "raincloud", "box"} <= {rec.spec.type} | {a.type for a in rec.alternatives}
    rendered = render_chart(apply_semantic_casts(df, profile), rec.spec, profile)
    assert [g["name"] for g in rendered["chart_data"]["groups"]] == levels


# --- 7 / 8. tiny and noise: no misleading charts ----------------------------------------------


def test_tiny_dataset_never_gets_a_density_chart() -> None:
    _, recs = _recs(_real_dataset("tiny_dataset.csv"))
    assert recs and not (DENSITY & _all_types(recs))
    assert all(r.tier != "top" for r in recs)


def test_pure_noise_gets_no_fake_recommendation_from_the_new_charts() -> None:
    _, recs = _recs(_real_dataset("noise.csv"))
    assert all(r.tier != "top" for r in recs)
    assert all(r.spec.type not in FANCY - {"lollipop"} for r in recs if r.tier == "secondary")


# --- 9 / 10. existing datasets: suppressions intact, cycles found -------------------------------


def test_one_recommendation_per_finding_even_for_a_numeric_cycle() -> None:
    # verifier round 1: hr × cnt appeared as a cyclic line AND a hexbin
    rng = random.Random(9)
    df = pl.DataFrame({"hr": [i % 24 for i in range(6000)], "cnt": [50 + 40 * math.sin(i % 24 / 24 * 2 * math.pi) + rng.gauss(0, 5) for i in range(6000)], "temp": [rng.gauss(0, 1) for _ in range(6000)]})
    _, recs = _recs(df)
    keys = [(r.spec.x, r.spec.y, r.spec.group_by) for r in recs]
    assert len(keys) == len(set(keys))
    hr = [r for r in recs if r.spec.x == "hr" and r.spec.y == "cnt"]
    assert len(hr) == 1 and hr[0].intent == "cyclic_pattern" and hr[0].spec.type == "line"
    assert not any(r.spec.type in ("scatter", "hexbin") and "hr" in (r.spec.x, r.spec.y) for r in recs)


def test_hour_like_cycles_are_top_and_suppressions_hold() -> None:
    _, recs = _recs(_real_dataset("hour_like.csv"))
    cycles = [r for r in recs if r.intent == "cyclic_pattern"]
    assert cycles and all(r.spec.sort == "natural" and r.spec.type == "line" for r in cycles)
    assert any(r.spec.x == "hr" and r.tier == "top" for r in cycles)
    assert any(r.spec.x == "mnth" for r in cycles)
    for r in recs:
        if r.spec.type != "heatmap":
            assert "atemp" not in (r.spec.x, r.spec.y, r.spec.group_by)
        assert not ({r.spec.x, r.spec.y} in ({"cnt", "casual"}, {"cnt", "registered"}) and r.tier == "top")


def test_sales_basic_formula_still_never_top() -> None:
    _, recs = _recs(_real_dataset("sales_basic.csv"))
    assert all(r.tier != "top" for r in recs)
    for r in recs:
        if r.spec.x and r.spec.y and {r.spec.x, r.spec.y} <= {"sales", "unit_price", "quantity", "discount"} and "sales" in (r.spec.x, r.spec.y):
            assert r.tier == "exploratory" and "definitional" in r.spec.reason
    assert not ({"rose", "polar_line"} & _all_types(recs))


def test_recommendations_are_deterministic_with_intents() -> None:
    df = _month_df(False)
    first = [r.model_dump() for r in _recs(df)[1]]
    second = [r.model_dump() for r in _recs(df)[1]]
    assert first == second and all("intent" in r and "alternatives" in r for r in first)
