"""LLM merge semantics carried over from stages 5/8/13/14 under the two-stage
workflow (stage 17.3): bare chart suggestions still go through the evidence
merge; a claim is now a hypothesis that must be verified (existing evidence
or a probe) before it becomes an insight."""

import math
from datetime import datetime, timedelta

import polars as pl
import pytest
from fastapi.testclient import TestClient

from app.charts.rules import TOP_SCORE_FLOOR, derived_column_warnings, recommend_charts
from app.config import Settings
from app.llm.provider import LLMError
from app.llm.schemas import FinalResponse, HypothesisResponse, LLMUsage
from app.llm.service import NO_PATTERNS_MESSAGE, RecommendationService, _parse_hypotheses
from app.main import create_app
from app.profiling.profiler import profile_dataset

BIG = 10**6


class FakeProvider:
    """Two-stage fake: `hypotheses` answers LLM #1, `final` answers LLM #2;
    both calls are counted so tests can pin the trigger rules."""

    def __init__(self, hypotheses=None, final=None, error: LLMError | None = None, final_error: LLMError | None = None) -> None:
        if isinstance(hypotheses, dict):
            hypotheses = HypothesisResponse.model_validate(hypotheses)
        elif isinstance(hypotheses, list):
            hypotheses = HypothesisResponse(hypotheses=hypotheses)
        self.response = hypotheses or HypothesisResponse()
        if isinstance(final, dict):
            final = FinalResponse.model_validate(final)
        self.final = final or FinalResponse()
        self.error = error
        self.final_error = final_error
        self.calls = 0
        self.final_calls = 0
        self.seen_validated = None

    def generate_hypotheses(self, profile, rule_candidates):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.response, LLMUsage(prompt_chars=10, completion_chars=5, latency_ms=1.0)

    def finalize_insights(self, profile, validated):
        self.final_calls += 1
        self.seen_validated = validated
        if self.final_error is not None:
            raise self.final_error
        return self.final, LLMUsage(prompt_chars=10, completion_chars=5, latency_ms=1.0)


def _frame(n: int = 60) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)],
            "city": ["a", "b", "c"] * (n // 3),
            "value": [float(i) for i in range(n)],
            # strong (r ~ 0.96) but not a near-duplicate (stage 14 suppresses copies)
            "value2": [2.0 * i + 15 * math.sin(i) for i in range(n)],
            "value3": [float(i % 2) for i in range(n)],  # ~uncorrelated
            "rating": [i % 5 + 1 for i in range(n)],  # numeric-backed categorical
        }
    )


@pytest.fixture(scope="module")
def df():
    return _frame()  # spans two months so the time-effect evidence is computable


@pytest.fixture(scope="module")
def profile(df):
    return profile_dataset(df, "0" * 32, BIG)


def _service(hypotheses=None, final=None, error=None) -> tuple[RecommendationService, FakeProvider]:
    provider = FakeProvider(hypotheses, final, error)
    return RecommendationService(provider), provider


def _chart_by(result, type_, x=None, y=None, group=None):
    for c in result["charts"]:
        s = c["spec"]
        if s["type"] == type_ and s["x"] == x and s["y"] == y and s["group_by"] == group:
            return c
    return None


def _hyp(statement: str, chart: dict | None, **extra) -> dict:
    return {"statement": statement, "chart": chart, "importance": 3, **extra}


NEW_CHART = {
    "title": "value vs value3",
    "type": "scatter",
    "x": "value",
    "y": "value3",
    "reason": "the rules skipped this weak pair",
    "priority": 1,
}


def test_rules_only_path_identical_to_stage3(profile) -> None:
    service, provider = _service({"hypotheses": [_hyp("ignored", NEW_CHART)], "charts": [NEW_CHART]})
    result = service.get(profile, use_llm=False)
    expected = [rec.model_dump() for rec in recommend_charts(profile)]
    # no identity and no near-duplicate in this fixture: no dataset warning
    assert derived_column_warnings(profile) == []
    assert result == {"charts": expected, "insights": [], "message": None, "warnings": []}
    assert provider.calls == 0


def test_new_llm_chart_scored_by_evidence_not_fixed(profile) -> None:
    service, _ = _service({"hypotheses": ["a bare string carries no testable claim"], "charts": [NEW_CHART]})
    result = service.get(profile, use_llm=True)
    first = result["charts"][0]  # LLM priority still leads the display order
    assert first["source"] == "llm"
    assert (first["spec"]["x"], first["spec"]["y"]) == ("value", "value3")
    # stage8: evidence score replaces the fixed 0.75 — |corr| ~ 0 -> ~0.5
    assert 0.45 <= first["score"] < 0.6
    assert first["tier"] != "top"  # weak evidence is capped below top
    priorities = [c["spec"]["priority"] for c in result["charts"]]
    assert priorities == list(range(1, len(priorities) + 1))
    # stage 17.3: a plain string is not a hypothesis -> no insight, honest message
    assert result["insights"] == []
    assert result["message"] == NO_PATTERNS_MESSAGE


def test_dedup_keeps_rules_spec_adopts_llm_reason(profile) -> None:
    rules = recommend_charts(profile)
    target = next(r for r in rules if r.spec.type == "line")
    mention = {
        "title": "re-ranked line",
        "type": "line",
        "x": target.spec.x,
        "y": target.spec.y,
        "group_by": target.spec.group_by,
        "aggregation": target.spec.aggregation,
        "reason": "the LLM's better reason",
        "priority": 1,
    }
    service, _ = _service({"charts": [mention]})
    result = service.get(profile, use_llm=True)
    first = result["charts"][0]
    assert first["source"] == "rules"  # rules spec kept
    assert first["spec"]["reason"] == "the LLM's better reason"
    assert first["spec"]["time_granularity"] == target.spec.time_granularity
    assert first["spec"]["title"] == target.spec.title


def test_two_layer_sort_key(profile) -> None:
    rules = recommend_charts(profile)
    target = next(r for r in rules if r.spec.type == "heatmap")
    mention = {"title": "hm", "type": "heatmap", "reason": "context", "priority": 2}
    service, _ = _service({"charts": [dict(NEW_CHART, priority=1), mention]})
    result = service.get(profile, use_llm=True)
    charts = result["charts"]
    # layer 0 in LLM-priority order, then layer 1 by score
    assert charts[0]["spec"]["type"] == "scatter" and charts[0]["source"] == "llm"
    assert charts[1]["spec"]["type"] == "heatmap" and charts[1]["source"] == "rules"
    layer1_scores = [c["score"] for c in charts[2:]]
    assert layer1_scores == sorted(layer1_scores, reverse=True)
    assert target.spec.title == charts[1]["spec"]["title"]


def test_llm_repeating_only_rules_reorders_them(profile) -> None:
    rules = recommend_charts(profile)
    picks = [rules[5], rules[2], rules[7]]
    mentions = [
        {
            "title": f"pick {i}",
            "type": r.spec.type,
            "x": r.spec.x,
            "y": r.spec.y,
            "group_by": r.spec.group_by,
            "aggregation": r.spec.aggregation,
            "reason": "llm",
            "priority": i + 1,
        }
        for i, r in enumerate(picks)
    ]
    service, _ = _service({"charts": mentions})
    charts = service.get(profile, use_llm=True)["charts"]
    lead = [(c["spec"]["type"], c["spec"]["x"], c["spec"]["y"]) for c in charts[:3]]
    assert lead == [(r.spec.type, r.spec.x, r.spec.y) for r in picks]
    assert all(c["source"] == "rules" for c in charts)


def test_mixed_valid_and_invalid_items_tolerated(profile) -> None:
    charts = [
        42,  # not even an object
        {"title": "p", "type": "pie", "x": "city"},  # bad chart type
        {"title": "", "type": "histogram", "x": "value"},  # empty title
        {"title": "ghost", "type": "histogram", "x": "no_such_column"},  # bad column
        {"title": "cities", "type": "histogram", "x": "city"},  # wrong axis type
        NEW_CHART,  # the only valid one
    ]
    service, _ = _service({"charts": charts})
    result = service.get(profile, use_llm=True)
    llm_charts = [c for c in result["charts"] if c["source"] == "llm"]
    assert len(llm_charts) == 1 and llm_charts[0]["spec"]["title"] == "value vs value3"
    assert result["message"] == NO_PATTERNS_MESSAGE  # one chart survived, but no verified claim


def test_all_invalid_items_fall_back_with_message(profile) -> None:
    service, _ = _service({"charts": [{"title": "p", "type": "pie", "x": "city"}]})
    result = service.get(profile, use_llm=True)
    assert result["charts"] == [rec.model_dump() for rec in recommend_charts(profile)]
    assert "invalid" in result["message"] and "rule-based" in result["message"]


def test_provider_error_falls_back_with_category(profile) -> None:
    service, _ = _service(error=LLMError("timeout", "too slow"))
    result = service.get(profile, use_llm=True)
    assert result["charts"] == [rec.model_dump() for rec in recommend_charts(profile)]
    assert result["insights"] == []
    assert "timeout" in result["message"] and "rule-based" in result["message"]


def _strong_line_mention(profile) -> dict:
    rules = recommend_charts(profile)
    strong_line = next(r for r in rules if r.spec.type == "line" and r.score >= TOP_SCORE_FLOOR)
    return {
        "title": "trend",
        "type": "line",
        "x": strong_line.spec.x,
        "y": strong_line.spec.y,
        "group_by": strong_line.spec.group_by,
        "aggregation": strong_line.spec.aggregation,
        "reason": "clear upward trend",
        "priority": 1,
    }


def test_insights_capped_and_truncated(profile) -> None:
    chart = _strong_line_mention(profile)
    hypotheses = [_hyp(f"i{n}" + "x" * 400, chart) for n in range(7)]
    service, provider = _service({"hypotheses": hypotheses})
    result = service.get(profile, use_llm=True, include_debug=True)
    insights = result["insights"]
    assert len(insights) == 5  # cap kept from stage 5 (now the hypothesis cap)
    assert all(len(i["text"]) == 300 for i in insights)  # truncation kept
    assert all(i["supported"] == "strong" for i in insights)
    assert [d["reason"] for d in result["debug"]["dropped"]] == ["beyond the hypothesis cap"] * 2
    # several validated findings answered by the evidence tables -> one LLM #2 call
    assert provider.final_calls == 1


def test_cache_per_dataset_and_per_llm_flag(profile) -> None:
    service, provider = _service({"hypotheses": [_hyp("cached", NEW_CHART)], "charts": [NEW_CHART]})
    first = service.get(profile, use_llm=True)
    assert service.get(profile, use_llm=True) == first  # cache hit, no second call
    assert provider.calls == 1
    rules_only = service.get(profile, use_llm=False)
    assert provider.calls == 1  # llm=false path never touches the provider
    assert rules_only["insights"] == []


# --- stage 8: insight -> chart contract, now through the coverage check ----


def test_bare_correlation_claim_is_probed_and_dropped(profile, df) -> None:
    # stage 17.3: "X and Y move together" carries no analytical value by
    # itself; a plain scatter implies a non-linear-dependence probe, which
    # finds nothing here (60 rows, |corr| ~ 0) -> no insight, no LLM chart
    service, provider = _service({"hypotheses": [_hyp("weak pair moves together", NEW_CHART)]})
    result = service.get(profile, use_llm=True, df=df, include_debug=True)
    assert result["insights"] == [] and result["message"] == NO_PATTERNS_MESSAGE
    assert _chart_by(result, "scatter", x="value", y="value3") is None
    (probe,) = result["debug"]["probes"]
    assert probe["type"] == "nonlinear_relationship" and probe["verdict"] == "fail"
    assert provider.final_calls == 0


def test_insight_chart_priority_points_to_final_chart(profile) -> None:
    mention = _strong_line_mention(profile)
    service, provider = _service({"hypotheses": [_hyp("value climbs over time", mention)]})
    result = service.get(profile, use_llm=True)
    (insight,) = result["insights"]
    chart = _chart_by(result, "line", x=mention["x"], y=mention["y"])
    assert chart is not None
    assert insight["chart_priority"] == chart["spec"]["priority"]
    assert insight["supported"] == "strong" and insight["validation"] == "existing_evidence"
    assert insight["effect"]["label"].startswith("adjusted eta-squared") and insight["effect"]["n"] == 60
    # a single finding the tables already answer: no LLM #2, template wording
    assert provider.final_calls == 0
    assert insight["text"].startswith("value climbs over time (")


def test_insight_chart_deduped_into_rules_chart(profile) -> None:
    mention = _strong_line_mention(profile)
    service, _ = _service({"hypotheses": [_hyp("value trends up", mention)]})
    result = service.get(profile, use_llm=True)
    (insight,) = result["insights"]
    merged = _chart_by(result, "line", x=mention["x"], y=mention["y"])
    assert merged is not None and merged["source"] == "rules"
    assert insight["chart_priority"] == merged["spec"]["priority"]
    assert insight["supported"] == "strong"


@pytest.mark.parametrize(
    "bad_chart",
    [
        {"title": "p", "type": "pie", "x": "city"},  # invalid type
        {"title": "g", "type": "histogram", "x": "ghost"},  # nonexistent column
        "a bar chart of sales",  # not even an object
    ],
    ids=["pie", "ghost-column", "non-dict"],
)
def test_insight_with_rejected_chart_is_dropped(profile, bad_chart) -> None:
    service, _ = _service(
        {
            "hypotheses": [
                _hyp("bogus claim", bad_chart),
                _hyp("chartless and untestable", None),
            ]
        }
    )
    result = service.get(profile, use_llm=True, include_debug=True)
    # stage 17.3: a claim with nothing to verify is dropped, not kept as "unverified"
    assert result["insights"] == []
    assert result["message"] == NO_PATTERNS_MESSAGE
    reasons = [d["reason"] for d in result["debug"]["dropped"]]
    assert len(reasons) == 2 and all("verify" in r for r in reasons)


def test_unverifiable_combination_is_probed_and_dropped_when_it_fails(profile, df) -> None:
    # bar y=rating: valid (numeric-backed categorical) but outside the
    # evidence scan -> the chart implies a group_difference probe; the data
    # shows no effect -> the claim is dropped (stage 8 kept it as "unverified")
    chart = {
        "title": "rating by city",
        "type": "bar",
        "x": "city",
        "y": "rating",
        "aggregation": "mean",
        "reason": "cities rate differently",
        "priority": 1,
    }
    service, provider = _service({"hypotheses": [_hyp("ratings differ by city", chart)]})
    result = service.get(profile, use_llm=True, df=df, include_debug=True)
    assert result["insights"] == []
    assert _chart_by(result, "bar", x="city", y="rating") is None
    (probe,) = result["debug"]["probes"]
    assert probe["type"] == "group_difference" and probe["columns"] == {"group": "city", "target": "rating"}
    assert probe["verdict"] == "fail"
    assert provider.final_calls == 0

    # without a data frame the probe cannot run: dropped with a clear reason
    service2, _ = _service({"hypotheses": [_hyp("ratings differ by city", chart)]})
    result2 = service2.get(profile, use_llm=True, include_debug=True)
    assert result2["insights"] == []
    assert "no data frame" in result2["debug"]["dropped"][0]["reason"]


def test_weak_llm_charts_cannot_evict_strong_rules_from_top(profile) -> None:
    # blocking #2: three LLM charts occupy the display front, but tiers are
    # assigned by score — the strong rules chart keeps its top badge
    weak_charts = [
        dict(NEW_CHART, priority=1),
        {
            "title": "v2 vs v3",
            "type": "scatter",
            "x": "value2",
            "y": "value3",
            "reason": "?",
            "priority": 2,
        },
        {
            # perfect correlation but an unsupported group split -> weak cap,
            # even though the evidence score itself is high
            "title": "grouped",
            "type": "scatter",
            "x": "value",
            "y": "value2",
            "group_by": "city",
            "reason": "?",
            "priority": 3,
        },
    ]
    service, _ = _service({"charts": weak_charts})
    result = service.get(profile, use_llm=True)
    charts = result["charts"]
    # the surviving LLM charts monopolize the display front (the per-type cap
    # may cut one of the three weak scatters — that is the diversity defense)
    leading = [c for c in charts if c["source"] == "llm"]
    assert len(leading) >= 2
    assert [c["source"] for c in charts[: len(leading)]] == ["llm"] * len(leading)
    assert all(c["tier"] != "top" for c in leading)
    top = [c for c in charts if c["tier"] == "top"]
    assert top and all(c["source"] == "rules" for c in top)
    assert all(c["score"] >= TOP_SCORE_FLOOR for c in top)
    # the high-score grouped scatter is capped at secondary, not top
    grouped = _chart_by(result, "scatter", x="value", y="value2", group="city")
    assert grouped is not None and grouped["score"] > 0.85 and grouped["tier"] == "secondary"


def test_resuggested_capped_chart_stays_out_and_claim_fails_on_evidence(profile) -> None:
    # critique #4: the diversity caps cut bar(city, value3) from the rules
    # list; the LLM re-suggesting it must not bypass the caps. Under stage
    # 17.3 the claim itself is checked first: city has no effect on value3
    # (eta-squared 0) -> dropped, and no chart is added either way
    rules_keys = {(r.spec.type, r.spec.x, r.spec.y) for r in recommend_charts(profile)}
    assert ("bar", "city", "value3") not in rules_keys
    chart = {
        "title": "v3 by city",
        "type": "bar",
        "x": "city",
        "y": "value3",
        "aggregation": "mean",
        "reason": "?",
        "priority": 1,
    }
    service, _ = _service({"hypotheses": [_hyp("v3 differs by city", chart)]})
    result = service.get(profile, use_llm=True, include_debug=True)
    assert _chart_by(result, "bar", x="city", y="value3") is None  # still capped out
    assert result["insights"] == []
    (dropped,) = result["debug"]["dropped"]
    assert "existing evidence fails" in dropped["reason"]


def test_malformed_hypotheses_do_not_invalidate_response() -> None:
    # per-item tolerance carried over from stage 5, now hypothesis-aware
    assert [h.statement for h in _parse_hypotheses([{"statement": "ok"}, 42, None, {"statement": "  "}])] == ["ok"]
    assert _parse_hypotheses("a string") == []
    assert _parse_hypotheses({"not": "a list"}) == []  # a lone object is tolerated only with a statement
    (legacy,) = _parse_hypotheses([{"text": "paired", "chart": {"type": "bar"}}])  # stage-8 shape
    assert legacy.statement == "paired" and legacy.chart == {"type": "bar"}
    (extra,) = _parse_hypotheses([{"statement": "s", "importance": 5, "note": "ignored extra"}])
    assert extra.importance == 5


# --- API integration --------------------------------------------------------

CSV = (
    "date,city,value,value2,value3\n"
    + "\n".join(
        f"2024-01-{i + 1:02d},{'a b c'.split()[i % 3]},{i}.0,{2 * i}.0,{i % 2}.0" for i in range(12)
    )
    + "\n"
)


def _llm_client(tmp_path, hypotheses) -> TestClient:
    settings = Settings(data_dir=tmp_path / "data", llm_enabled=True, llm_model="fake")
    app = create_app(settings)
    app.state.recommendations = RecommendationService(FakeProvider(hypotheses))
    return TestClient(app)


def test_endpoint_with_llm_and_llm_false_switch(tmp_path) -> None:
    line = {"title": "value over date", "type": "line", "x": "date", "y": "value", "aggregation": "mean", "priority": 1}
    client = _llm_client(
        tmp_path, {"hypotheses": [_hyp("value climbs over the month.", line)], "charts": [dict(NEW_CHART)]}
    )
    dataset_id = client.post("/api/datasets", files={"file": ("d.csv", CSV.encode())}).json()["dataset_id"]

    body = client.get(f"/api/datasets/{dataset_id}/recommendations").json()
    # 12 rows: a real trend, but below the MIN_ROWS floor no claim becomes an insight
    assert body["insights"] == [] and body["message"] == NO_PATTERNS_MESSAGE
    assert any(c["source"] == "llm" for c in body["charts"])  # bare chart suggestions still merge
    assert "debug" not in body

    debug = client.get(f"/api/datasets/{dataset_id}/recommendations", params={"debug": "1"}).json()["debug"]
    assert debug["llm_calls"] == 1 and debug["validated"] == 0 and debug["probes"] == []
    assert "fewer than 30 rows" in debug["dropped"][0]["reason"]

    rules_only = client.get(f"/api/datasets/{dataset_id}/recommendations", params={"llm": "false"}).json()
    assert rules_only["insights"] == []
    assert all(c["source"] == "rules" for c in rules_only["charts"])
    assert "debug" not in rules_only


def test_endpoint_llm_disabled_ignores_llm_param(client: TestClient) -> None:
    dataset_id = client.post("/api/datasets", files={"file": ("d.csv", CSV.encode())}).json()["dataset_id"]
    default = client.get(f"/api/datasets/{dataset_id}/recommendations").json()
    forced = client.get(f"/api/datasets/{dataset_id}/recommendations", params={"llm": "true"}).json()
    assert default == forced
    assert default["insights"] == []


# --- stage 13: derived pairs ------------------------------------------------


@pytest.fixture(scope="module")
def derived_profile():
    import random

    rng = random.Random(31)
    n = 90
    price = [round(rng.uniform(50, 5000), 2) for _ in range(n)]
    qty = [float(rng.randrange(1, 11)) for _ in range(n)]
    return profile_dataset(
        pl.DataFrame(
            {
                "ts": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)],
                "price": price,
                "qty": qty,
                "total": [round(p * q, 2) for p, q in zip(price, qty)],
                "other": [rng.gauss(0, 1) for _ in range(n)],
            }
        ),
        "1" * 32,
        BIG,
    )


def test_hypothesis_restating_a_formula_is_dropped(derived_profile) -> None:
    assert [d.formula for d in derived_profile.evidence.derived_columns] == ["price × qty"]
    chart = {
        "title": "total vs price",
        "type": "scatter",
        "x": "price",
        "y": "total",
        "reason": "totals rise with price",
        "priority": 1,
    }
    service, _ = _service({"hypotheses": [_hyp("Higher prices drive higher totals.", chart)]})
    result = service.get(derived_profile, use_llm=True, include_debug=True)
    # stage 17.3: a definitional claim never becomes an insight (stage 8 kept it as "unverified")
    assert result["insights"] == []
    assert "definitional" in result["debug"]["dropped"][0]["reason"]
    assert all(c["source"] == "rules" for c in result["charts"])
    assert [w["code"] for w in result["warnings"]] == ["derived_column"]


def test_new_llm_chart_on_derived_pair_is_capped(derived_profile) -> None:
    # axes swapped relative to the rules candidate, so this is a NEW LLM
    # chart: it goes through the same derived cap as rules charts
    chart = {
        "title": "qty by total",
        "type": "scatter",
        "x": "total",
        "y": "qty",
        "reason": "?",
        "priority": 1,
    }
    service, _ = _service({"charts": [chart]})
    result = service.get(derived_profile, use_llm=True)
    added = _chart_by(result, "scatter", x="total", y="qty")
    assert added is not None and added["source"] == "llm"
    assert added["tier"] == "exploratory"
    assert "definitional" in added["spec"]["reason"]
    assert any(w["code"] == "derived_relationship" for w in added["warnings"])


# --- stage 14: near-duplicate suppression through the LLM merge -------------


@pytest.fixture(scope="module")
def dup_profile():
    n = 90
    df = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)],
            "city": ["a", "b", "c"] * (n // 3),
            "temp": [10 + 0.3 * i + 4 * math.sin(i) for i in range(n)],
            "atemp": [9 + 0.27 * i + 3.6 * math.sin(i) + 0.05 * math.cos(3 * i) for i in range(n)],
            "hum": [60 - 0.2 * i + 3 * math.cos(i / 2) for i in range(n)],
        }
    )
    return profile_dataset(df, "0" * 32, BIG)


def test_llm_duplicate_chart_is_canonicalised_and_deduped(dup_profile) -> None:
    assert [(g.representative, g.duplicates) for g in dup_profile.evidence.near_duplicate_groups] == [
        ("temp", ["atemp"])
    ]
    rules = recommend_charts(dup_profile)
    assert all("atemp" not in (r.spec.x, r.spec.y) for r in rules if r.spec.type != "heatmap")
    rules_scatter = next(r for r in rules if r.spec.type == "scatter" and {r.spec.x, r.spec.y} == {"temp", "hum"})
    charts = [
        {"title": "hum vs atemp", "type": "scatter", "x": "atemp", "y": "hum", "reason": "llm says", "priority": 1},
        {"title": "atemp vs temp", "type": "scatter", "x": "temp", "y": "atemp", "reason": "dup", "priority": 2},
        {"title": "atemp over time", "type": "line", "x": "ts", "y": "atemp", "reason": "trend", "priority": 3},
    ]
    service, _ = _service({"charts": charts})
    result = service.get(dup_profile, use_llm=True)
    specs = [(c["spec"]["type"], c["spec"]["x"], c["spec"]["y"]) for c in result["charts"]]
    assert "atemp" not in {x for _, x, _ in specs} | {y for _, _, y in specs}
    # the LLM's atemp scatter became the rules' temp scatter (LLM wording adopted)
    kept = _chart_by(result, "scatter", x="temp", y="hum")
    assert kept is not None and kept["source"] == "rules" and kept["spec"]["reason"] == "llm says"
    assert specs.count(("scatter", "temp", "hum")) == 1
    assert result["insights"] == []  # bare chart suggestions carry no claim
    assert kept["score"] == pytest.approx(rules_scatter.score)
    # the pair chart collapsed onto temp == temp and was dropped
    assert _chart_by(result, "scatter", x="temp", y="temp") is None
    # a genuinely new chart on the duplicate is re-pointed at the representative
    line = _chart_by(result, "line", x="ts", y="temp")
    assert line is not None
    assert [(w["code"], w["meta"]["target"]) for w in result["warnings"]] == [("near_duplicate_column", "atemp")]


def test_hypothesis_on_the_duplicate_pair_is_dropped(dup_profile) -> None:
    chart = {"title": "atemp vs temp", "type": "scatter", "x": "temp", "y": "atemp", "reason": "dup", "priority": 1}
    service, _ = _service({"hypotheses": [_hyp("apparent temperature tracks temperature", chart)]})
    result = service.get(dup_profile, use_llm=True, include_debug=True)
    assert result["insights"] == []
    (dropped,) = result["debug"]["dropped"]
    assert "near-duplicate" in dropped["reason"] or "verify" in dropped["reason"]


def test_hypothesis_statement_maps_duplicate_names_to_the_representative(dup_profile) -> None:
    # "atemp" in the claim text is the duplicate of temp: the workflow speaks
    # of the representative, and the grouped claim goes through the probe path
    chart = {"title": "atemp over time by city", "type": "line", "x": "ts", "y": "atemp", "group_by": "city", "aggregation": "mean", "priority": 1}
    service, _ = _service({"hypotheses": [_hyp("atemp rises over time differently per city", chart, columns={"time": "ts", "target": "atemp", "group": "city"}, test_needed="time_pattern")]})
    result = service.get(dup_profile, use_llm=True, include_debug=True)
    (dropped,) = result["debug"]["dropped"]
    assert dropped["statement"] == "temp rises over time differently per city"
    assert "grouped time pattern" in dropped["reason"]


def test_llm_new_chart_on_duplicate_carries_substitution_warning(dup_profile) -> None:
    # a chart the rules did not produce (grouped line: city has no effect on
    # temp), requested on the duplicate: kept on the representative, says so
    chart = {
        "title": "atemp over time by city",
        "type": "line",
        "x": "ts",
        "y": "atemp",
        "group_by": "city",
        "aggregation": "mean",
        "reason": "trend per city",
        "priority": 1,
    }
    rules_keys = {(r.spec.type, r.spec.x, r.spec.y, r.spec.group_by) for r in recommend_charts(dup_profile)}
    assert ("line", "ts", "temp", "city") not in rules_keys
    service, _ = _service({"charts": [chart]})
    result = service.get(dup_profile, use_llm=True)
    line = _chart_by(result, "line", x="ts", y="temp", group="city")
    assert line is not None and line["source"] == "llm"
    assert line["spec"]["title"] == "temp over time by city"
    assert [w["meta"]["substitutions"] for w in line["warnings"] if w["code"] == "near_duplicate_substituted"] == [
        [{"from": "atemp", "to": "temp"}]
    ]
    assert _chart_by(result, "line", x="ts", y="atemp", group="city") is None
