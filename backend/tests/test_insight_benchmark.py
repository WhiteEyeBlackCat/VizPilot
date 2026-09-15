"""Stage 17.4 insight benchmark: eight planted scenarios run through the
whole two-stage pipeline (rules -> hallucination gate -> coverage -> probes
-> conditional LLM #2) with a fake provider, so every verdict has a known
right answer.

1. sales_basic      formula restatements are not insights; zero is honest
2. hour_like        cnt = casual + registered and temp ≈ atemp never lead;
                    rider-behaviour patterns do
3. nonlinear        Pearson-weak U shape is found; the reverse claim is not
4. interaction      overall effects are ordinary, one cell is not -> found
5. noise            independent columns -> zero insights, whatever is asked
6. tiny             n = 4 -> nothing is a finding
7. probe failure    an uncovered claim the data does not support is dropped
8. hallucination    unknown columns / types / numbers never reach the user

The real model (qwen2.5:14b) is run by hand and recorded in
`.claude/docs/stages/stage17.4.md`; LLM proposals vary between runs, so
those runs are not tests."""

import random
from datetime import datetime, timedelta

import polars as pl
import pytest

from app.llm.service import NO_PATTERNS_MESSAGE, RecommendationService
from app.profiling.profiler import profile_dataset
from tests.test_llm_workflow import BIG, TwoStageFake, _h, _noise_frame, _planted
from tests.test_recommendation_quality import _real_dataset


def _run(df: pl.DataFrame, hypotheses: list[dict], final: dict | None = None, seed: str = "7"):
    profile = profile_dataset(df, seed * 32, BIG)
    provider = TwoStageFake(hypotheses, None, final)
    result = RecommendationService(provider).get(profile, use_llm=True, df=df, include_debug=True)
    return profile, result, provider


def _scatter(x: str, y: str) -> dict:
    return {"title": f"{y} vs {x}", "type": "scatter", "x": x, "y": y, "priority": 1}


def _top_pairs(result) -> list[frozenset]:
    return [frozenset(filter(None, (c["spec"]["x"], c["spec"]["y"]))) for c in result["charts"] if c["tier"] == "top"]


def _reasons(result) -> list[str]:
    return [d["reason"] for d in result["debug"]["dropped"]]


# --- 1. sales_basic ---------------------------------------------------------------------


def _sales_basic(n: int = 1000) -> pl.DataFrame:
    """Same structure as dataset/syn/sales_basic.py (which needs pandas):
    sales = unit_price × quantity × (1 − discount) exactly; region and
    category carry no signal."""
    rng = random.Random(42)
    unit_price = [round(rng.uniform(50, 5000), 2) for _ in range(n)]
    quantity = [rng.randint(1, 10) for _ in range(n)]
    discount = [rng.choice([0, 0.05, 0.1, 0.2]) for _ in range(n)]
    return pl.DataFrame(
        {
            "order_id": list(range(1, n + 1)),
            "order_date": [datetime(2025, 1, 1) + timedelta(days=rng.randint(0, 364)) for _ in range(n)],
            "region": [rng.choice(["North", "Central", "South", "East"]) for _ in range(n)],
            "category": [rng.choice(["Electronics", "Clothing", "Food", "Home"]) for _ in range(n)],
            "unit_price": unit_price,
            "quantity": quantity,
            "discount": discount,
            "sales": [round(p * q * (1 - d), 2) for p, q, d in zip(unit_price, quantity, discount)],
        }
    )


def test_sales_basic_formula_is_not_a_finding_and_zero_is_honest() -> None:
    df = _sales_basic()
    hypotheses = [
        _h("Sales rise with quantity.", None, None, _scatter("quantity", "sales"), 5),
        _h("Sales grow non-linearly with unit_price.", "nonlinear_relationship", {"x": "unit_price", "y": "sales"}, None, 4),
        _h("Discount level changes sales.", "group_difference", {"group": "discount", "target": "sales"}, None, 4),
        _h("Sales differ by region.", "group_difference", {"group": "region", "target": "sales"}, None, 3),
    ]
    profile, result, provider = _run(df, hypotheses)
    assert [d.target for d in profile.evidence.derived_columns] == ["sales"]
    # every hypothesis fell: three restate the definition, region has no effect
    assert result["insights"] == [] and result["message"] == NO_PATTERNS_MESSAGE
    assert provider.final_calls == 0
    reasons = _reasons(result)
    assert len(reasons) == 4
    # the two probe requests on the formula are named as such; the chart-only
    # one loses its chart to the derived cap and has nothing left to verify
    assert sum("definitional: sales = unit_price × quantity × (1 − discount)" in r for r in reasons) == 2
    assert any("nothing to verify" in r for r in reasons)
    assert any("region" in d["statement"] and "existing evidence fails" in d["reason"] and "= 0.000" in d["reason"] for d in result["debug"]["dropped"])
    # the definition never leads the chart list either
    components = {"unit_price", "quantity", "discount"}
    assert not any("sales" in pair and pair & components for pair in _top_pairs(result))
    assert any(w["code"] == "derived_column" for w in result["warnings"])


# --- 2. hour_like ----------------------------------------------------------------------


def test_hour_like_definitions_do_not_lead_but_rider_patterns_do() -> None:
    df = _real_dataset("hour_like.csv")
    hypotheses = [
        _h("cnt is highly correlated with registered.", None, None, _scatter("registered", "cnt"), 5),
        _h("temp and atemp move together.", None, None, _scatter("temp", "atemp"), 4),
        _h("Casual rides peak around midday, an inverted U over the hour.", "nonlinear_relationship", {"x": "hr", "y": "casual"}, None, 5),
        _h("Registered rides show a commuter double peak by hour.", "nonlinear_relationship", {"x": "hr", "y": "registered"}, None, 4),
        _h("Temperature differs by month.", "group_difference", {"group": "mnth", "target": "temp"}, None, 3),
    ]
    profile, result, provider = _run(df, hypotheses)
    assert {d.target for d in profile.evidence.derived_columns} >= {"cnt", "atemp"}
    dropped = {d["id"]: d["reason"] for d in result["debug"]["dropped"]}
    assert dropped[1] == "definitional: cnt = casual + registered"
    assert "nothing to verify" in dropped[2]  # the temp/atemp chart fell to the near-duplicate cap
    texts = [i["text"] for i in result["insights"]]
    assert len(result["insights"]) == 3 and provider.final_calls == 1
    assert all(i["validation"] == "existing_evidence" and i["supported"] == "strong" for i in result["insights"])
    covered = {c["id"]: c for c in result["debug"]["covered_by_existing_evidence"]}
    assert covered[3]["test"] == "nonlinear_relationship" and covered[3]["effect"] > 0.7
    assert covered[5]["test"] == "group_difference" and 0.3 < covered[5]["effect"] < 0.4
    assert not any("cnt" in t or "atemp" in t for t in texts)
    # the formula and the near-copy never occupy the top tier
    for pair in _top_pairs(result):
        assert pair not in ({"cnt", "casual"}, {"cnt", "registered"}, {"temp", "atemp"})


# --- 3. planted nonlinear -------------------------------------------------------------


def test_pearson_weak_u_shape_is_found_but_only_in_its_direction() -> None:
    df = _planted()
    profile, result, provider = _run(
        df,
        [
            _h("u is a U shape in x.", "nonlinear_relationship", {"x": "x", "y": "u"}, None, 5),
            _h("x is a U shape in u.", "nonlinear_relationship", {"x": "u", "y": "x"}, None, 4),
        ],
    )
    (shape,) = [s for s in profile.evidence.layer2.nonlinear if (s.x, s.y) == ("x", "u")]
    assert shape.shape == "u_shape" and (shape.r2_pearson or 0.0) < 0.1 and shape.binned_eta2 > 0.5
    (insight,) = result["insights"]
    assert insight["validation"] == "existing_evidence" and insight["supported"] == "strong"
    assert insight["effect"]["label"].startswith("nonlinear_gap")
    # the reverse direction is not in the table: it ran its own probe and failed
    (probe,) = result["debug"]["probes"]
    assert probe["type"] == "nonlinear_relationship" and probe["verdict"] == "fail"
    assert any("probe failed" in r for r in _reasons(result))
    # a single finding the tables already held is worded by the template; LLM #2 is not needed
    assert provider.final_calls == 0 and result["debug"]["wording"][0]["source"] == "template"


# --- 4. planted interaction -----------------------------------------------------------


def _interaction_frame(n: int = 600) -> pl.DataFrame:
    """y is ordinary in every cell except (segment c, flag true): each main
    effect alone is modest, the cell is not."""
    rng = random.Random(9)
    seg = [rng.choice(["a", "b", "c"]) for _ in range(n)]
    flag = [rng.random() < 0.5 for _ in range(n)]
    y = [rng.gauss(10, 1) + (6 if s == "c" and f else 0) for s, f in zip(seg, flag)]
    return pl.DataFrame({"segment": seg, "flag": flag, "y": y, "noise": [rng.gauss(0, 1) for _ in range(n)]})


def test_interaction_and_sign_flip_are_detected() -> None:
    df = _interaction_frame()
    profile, result, provider = _run(df, [_h("segment and flag interact on y.", "interaction", {"factor1": "segment", "factor2": "flag", "target": "y"}, None, 5)])
    # neither main effect is remarkable on its own
    assert all(e.eta_squared < 0.3 for e in profile.evidence.cat_num if e.num == "y")
    # the layer-1 interaction table already holds the cell, so no probe is spent
    (cover,) = result["debug"]["covered_by_existing_evidence"]
    assert cover["test"] == "interaction" and cover["verdict"] == "pass" and cover["effect"] > 0.3
    assert result["debug"]["probes"] == []
    (insight,) = result["insights"]
    assert insight["validation"] == "existing_evidence" and insight["supported"] == "strong"
    assert insight["effect"]["label"].startswith("interaction share of variance")
    assert provider.final_calls == 0

    # the planted sign flip (w on z by segment): overall correlation ~0, the tables see it
    planted = _planted()
    _, result, provider = _run(planted, [_h("w's relation to z flips by segment.", "slope_difference", {"x": "z", "y": "w", "group": "segment"}, None, 4)])
    (insight,) = result["insights"]
    assert insight["validation"] in ("existing_evidence", "probe") and insight["supported"] == "strong"


# --- 5. noise ------------------------------------------------------------------------


def test_noise_yields_zero_insights_for_every_kind_of_claim() -> None:
    df = _noise_frame()
    hypotheses = [
        _h("n0 differs by segment.", "group_difference", {"group": "segment", "target": "n0"}, None, 5),
        _h("n1 relates to n2 in a curve.", "nonlinear_relationship", {"x": "n1", "y": "n2"}, None, 5),
        _h("n3 drifts over time.", "time_pattern", {"time": "ts", "target": "n3"}, None, 4),
        _h("n4 vs n5 differs by flag.", "slope_difference", {"x": "n4", "y": "n5", "group": "flag"}, None, 4),
        _h("segment and region interact on n6.", "interaction", {"factor1": "segment", "factor2": "region", "target": "n6"}, None, 3),
    ]
    _, result, provider = _run(df, hypotheses)
    assert result["insights"] == [] and result["message"] == NO_PATTERNS_MESSAGE
    assert provider.final_calls == 0
    # chart-only claims go through the same thresholds
    _, result, provider = _run(df, [_h("u0 tracks u1.", None, None, _scatter("u0", "u1"), 4), _h("n7 by region.", None, None, {"title": "n7 by region", "type": "bar", "x": "region", "y": "n7", "aggregation": "mean", "priority": 1}, 4)])
    assert result["insights"] == [] and provider.final_calls == 0
    assert all(c["source"] == "rules" for c in result["charts"])


# --- 6. tiny -------------------------------------------------------------------------


def test_tiny_dataset_has_no_high_confidence_finding() -> None:
    df = pl.DataFrame({"id": [1, 2, 3, 4], "name": ["a", "b", "c", "d"], "value": [10.0, 200.0, 30.0, 400.0], "flag": [True, False, True, False]})
    _, result, provider = _run(df, [_h("value is far higher when flag is false.", "group_difference", {"group": "flag", "target": "value"}, None, 5)])
    assert result["insights"] == [] and result["message"] == NO_PATTERNS_MESSAGE
    assert any("fewer than 30 rows" in r for r in _reasons(result))
    assert provider.final_calls == 0
    assert all(c["tier"] != "top" for c in result["charts"])


# --- 7. probe failure -----------------------------------------------------------------


def test_unsupported_claims_are_probed_and_dropped() -> None:
    df = _planted()
    hypotheses = [
        _h("n1 depends on n0 in a curve.", "nonlinear_relationship", {"x": "n0", "y": "n1"}, None, 5),
        _h("n2's distribution differs by segment.", "distribution_difference", {"group": "segment", "target": "n2"}, None, 4),
        _h("n3 shifts over the period.", "time_pattern", {"time": "ts", "target": "n3"}, None, 3),
    ]
    _, result, provider = _run(df, hypotheses)
    assert result["insights"] == [] and result["message"] == NO_PATTERNS_MESSAGE
    # the curve and the distribution claims are not in the tables: probed and failed;
    # the time claim is answered by the layer-1 time buckets: no probe spent, still dropped
    probes = result["debug"]["probes"]
    assert [p["type"] for p in probes] == ["nonlinear_relationship", "distribution_difference"]
    assert all(p["verdict"] == "fail" for p in probes)
    reasons = _reasons(result)
    assert len(reasons) == 3 and sum("probe failed" in r for r in reasons) == 2
    assert any("existing evidence fails" in r and "time buckets" in r for r in reasons)
    assert provider.final_calls == 0


# --- 8. hallucination -------------------------------------------------------------------


def test_hallucinated_hypotheses_are_caught_and_numbers_are_never_the_llms() -> None:
    df = _planted()
    hypotheses = [
        _h("revenue differs by segment.", "group_difference", {"group": "segment", "target": "revenue"}, None, 5),
        _h("sales follow a seasonal cycle.", "seasonality", {"time": "ts", "target": "sales"}, None, 5),
        _h("sales differ by x.", "group_difference", {"group": "x", "target": "sales"}, None, 4),
        _h("customer_tier explains sales.", "group_difference", {"group": "segment", "target": "sales"}, None, 4),
        _h("segment explains 87% of sales (eta² = 0.87, n = 12000).", "group_difference", {"group": "segment", "target": "sales"}, None, 3),
    ]
    _, result, provider = _run(df, hypotheses)
    reasons = _reasons(result)
    assert len(reasons) == 4
    assert any("not a column" in r for r in reasons)  # revenue
    assert any("unknown probe type" in r for r in reasons)  # seasonality
    assert any("role 'group' needs a categorical column, 'x' is numeric" in r for r in reasons)
    assert any("unknown columns ['customer_tier']" in r for r in reasons)
    # the one real claim survives, with the backend's numbers and none of the LLM's
    (insight,) = result["insights"]
    assert insight["validation"] == "existing_evidence" and insight["supported"] == "strong"
    assert "87" not in insight["text"] and "12000" not in insight["text"] and "12,000" not in insight["text"]
    assert 0.8 < insight["effect"]["value"] < 0.87 and "0.84" in insight["text"] and "n=600" in insight["text"]
    (wording,) = result["debug"]["wording"]
    assert wording["source"] == "neutral"
    assert any("number 87 is not in the validated evidence" in e for e in result["debug"]["errors"])
    assert provider.final_calls == 0
