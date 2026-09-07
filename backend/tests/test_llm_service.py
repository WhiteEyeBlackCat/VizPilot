from datetime import datetime, timedelta

import polars as pl
import pytest
from fastapi.testclient import TestClient

from app.charts.rules import TOP_SCORE_FLOOR, recommend_charts
from app.config import Settings
from app.llm.provider import LLMError
from app.llm.schemas import LLMResponse
from app.llm.service import RecommendationService, _parse_insights
from app.main import create_app
from app.profiling.profiler import profile_dataset

BIG = 10**6


class FakeProvider:
    def __init__(self, response: LLMResponse | None = None, error: LLMError | None = None) -> None:
        self.response = response or LLMResponse()
        self.error = error
        self.calls = 0

    def recommend_charts(self, profile, rule_candidates) -> LLMResponse:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture(scope="module")
def profile():
    n = 60  # spans two months so the time-effect evidence is computable
    df = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)],
            "city": ["a", "b", "c"] * (n // 3),
            "value": [float(i) for i in range(n)],
            "value2": [2.0 * i for i in range(n)],
            "value3": [float(i % 2) for i in range(n)],  # ~uncorrelated
            "rating": [i % 5 + 1 for i in range(n)],  # numeric-backed categorical
        }
    )
    return profile_dataset(df, "0" * 32, BIG)


def _service(response=None, error=None) -> tuple[RecommendationService, FakeProvider]:
    provider = FakeProvider(response, error)
    return RecommendationService(provider), provider


def _chart_by(result, type_, x=None, y=None, group=None):
    for c in result["charts"]:
        s = c["spec"]
        if s["type"] == type_ and s["x"] == x and s["y"] == y and s["group_by"] == group:
            return c
    return None


NEW_CHART = {
    "title": "value vs value3",
    "type": "scatter",
    "x": "value",
    "y": "value3",
    "reason": "the rules skipped this weak pair",
    "priority": 1,
}


def test_rules_only_path_identical_to_stage3(profile) -> None:
    service, provider = _service(LLMResponse(insights=["ignored"], charts=[NEW_CHART]))
    result = service.get(profile, use_llm=False)
    expected = [rec.model_dump() for rec in recommend_charts(profile)]
    assert result == {"charts": expected, "insights": [], "message": None}
    assert provider.calls == 0


def test_new_llm_chart_scored_by_evidence_not_fixed(profile) -> None:
    service, _ = _service(LLMResponse(insights=["a insight"], charts=[NEW_CHART]))
    result = service.get(profile, use_llm=True)
    first = result["charts"][0]  # LLM priority still leads the display order
    assert first["source"] == "llm"
    assert (first["spec"]["x"], first["spec"]["y"]) == ("value", "value3")
    # stage8: evidence score replaces the fixed 0.75 — |corr| ~ 0 -> ~0.5
    assert 0.45 <= first["score"] < 0.6
    assert first["tier"] != "top"  # weak evidence is capped below top
    assert result["message"] is None
    priorities = [c["spec"]["priority"] for c in result["charts"]]
    assert priorities == list(range(1, len(priorities) + 1))
    # legacy string insight -> kept, unverified, no chart link
    assert result["insights"] == [
        {"text": "a insight", "supported": "unverified", "chart_priority": None}
    ]


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
    service, _ = _service(LLMResponse(charts=[mention]))
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
    service, _ = _service(LLMResponse(charts=[dict(NEW_CHART, priority=1), mention]))
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
    service, _ = _service(LLMResponse(charts=mentions))
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
    service, _ = _service(LLMResponse(charts=charts))
    result = service.get(profile, use_llm=True)
    llm_charts = [c for c in result["charts"] if c["source"] == "llm"]
    assert len(llm_charts) == 1 and llm_charts[0]["spec"]["title"] == "value vs value3"
    assert result["message"] is None  # one item survived -> no fallback


def test_all_invalid_items_fall_back_with_message(profile) -> None:
    service, _ = _service(LLMResponse(charts=[{"title": "p", "type": "pie", "x": "city"}]))
    result = service.get(profile, use_llm=True)
    assert result["charts"] == [rec.model_dump() for rec in recommend_charts(profile)]
    assert "invalid" in result["message"] and "rule-based" in result["message"]


def test_provider_error_falls_back_with_category(profile) -> None:
    service, _ = _service(error=LLMError("timeout", "too slow"))
    result = service.get(profile, use_llm=True)
    assert result["charts"] == [rec.model_dump() for rec in recommend_charts(profile)]
    assert result["insights"] == []
    assert "timeout" in result["message"] and "rule-based" in result["message"]


def test_insights_capped_and_truncated(profile) -> None:
    service, _ = _service(LLMResponse(insights=[f"i{n}" + "x" * 400 for n in range(7)]))
    insights = service.get(profile, use_llm=True)["insights"]
    assert len(insights) == 5  # cap kept from stage 5
    assert all(len(i["text"]) == 300 for i in insights)  # truncation kept
    assert all(i["supported"] == "unverified" and i["chart_priority"] is None for i in insights)


def test_cache_per_dataset_and_per_llm_flag(profile) -> None:
    service, provider = _service(LLMResponse(insights=["cached"], charts=[NEW_CHART]))
    first = service.get(profile, use_llm=True)
    assert service.get(profile, use_llm=True) is first  # cache hit, no second call
    assert provider.calls == 1
    rules_only = service.get(profile, use_llm=False)
    assert provider.calls == 1  # llm=false path never touches the provider
    assert rules_only["insights"] == []


# --- stage 8: insight -> chart contract -------------------------------------


def test_insight_chart_priority_points_to_final_chart(profile) -> None:
    service, _ = _service(
        LLMResponse(insights=[{"text": "weak pair moves together", "chart": NEW_CHART}])
    )
    result = service.get(profile, use_llm=True)
    (insight,) = result["insights"]
    chart = _chart_by(result, "scatter", x="value", y="value3")
    assert chart is not None and chart["source"] == "llm"
    assert insight["chart_priority"] == chart["spec"]["priority"]
    assert insight["supported"] == "weak"  # |corr| ~ 0 -> evidence exists but weak
    assert chart["tier"] != "top"


def test_insight_chart_deduped_into_rules_chart(profile) -> None:
    rules = recommend_charts(profile)
    strong_line = next(r for r in rules if r.spec.type == "line" and r.score >= TOP_SCORE_FLOOR)
    mention = {
        "title": "trend",
        "type": "line",
        "x": strong_line.spec.x,
        "y": strong_line.spec.y,
        "group_by": strong_line.spec.group_by,
        "aggregation": strong_line.spec.aggregation,
        "reason": "clear upward trend",
        "priority": 1,
    }
    service, _ = _service(LLMResponse(insights=[{"text": "value trends up", "chart": mention}]))
    result = service.get(profile, use_llm=True)
    (insight,) = result["insights"]
    merged = _chart_by(result, "line", x=strong_line.spec.x, y=strong_line.spec.y)
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
        LLMResponse(
            insights=[
                {"text": "bogus claim", "chart": bad_chart},
                {"text": "chartless but honest", "chart": None},
            ]
        )
    )
    insights = service.get(profile, use_llm=True)["insights"]
    assert [i["text"] for i in insights] == ["chartless but honest"]
    assert insights[0]["supported"] == "unverified" and insights[0]["chart_priority"] is None


def test_unverified_combination_capped_exploratory(profile) -> None:
    # bar y=rating: valid (numeric-backed categorical) but outside the
    # evidence scan -> unverified, exploratory cap, honest reason note
    chart = {
        "title": "rating by city",
        "type": "bar",
        "x": "city",
        "y": "rating",
        "aggregation": "mean",
        "reason": "cities rate differently",
        "priority": 1,
    }
    service, _ = _service(LLMResponse(insights=[{"text": "ratings differ by city", "chart": chart}]))
    result = service.get(profile, use_llm=True)
    (insight,) = result["insights"]
    assert insight["supported"] == "unverified"
    rec = _chart_by(result, "bar", x="city", y="rating")
    assert rec is not None
    assert rec["tier"] == "exploratory"
    assert "not verified against the data" in rec["spec"]["reason"]


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
    service, _ = _service(LLMResponse(charts=weak_charts))
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


def test_resuggested_capped_chart_stays_out_but_insight_survives(profile) -> None:
    # critique #4: the diversity caps cut bar(city, value3) from the rules
    # list; the LLM re-suggesting it must not bypass the caps — the insight
    # survives with chart_priority=null (blocking #1 path 3)
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
    service, _ = _service(LLMResponse(insights=[{"text": "v3 differs by city", "chart": chart}]))
    result = service.get(profile, use_llm=True)
    assert _chart_by(result, "bar", x="city", y="value3") is None  # still capped out
    (insight,) = result["insights"]
    assert insight["chart_priority"] is None
    assert insight["supported"] == "weak"  # evidence was checked regardless
    assert insight["text"] == "v3 differs by city"


def test_malformed_insights_do_not_invalidate_response() -> None:
    # per-item tolerance carried over from stage 5, now shape-aware
    assert _parse_insights("single insight") == [("single insight", None)]
    assert _parse_insights(["ok", 42, None, "  "]) == [("ok", None)]
    assert _parse_insights({"not": "a list"}) == []
    assert _parse_insights([{"text": "paired", "chart": {"type": "bar"}}]) == [
        ("paired", {"type": "bar"})
    ]
    assert _parse_insights([{"text": "  ", "chart": None}, {"no_text": 1}]) == []


# --- API integration --------------------------------------------------------

CSV = (
    "date,city,value,value2,value3\n"
    + "\n".join(
        f"2024-01-{i + 1:02d},{'a b c'.split()[i % 3]},{i}.0,{2 * i}.0,{i % 2}.0" for i in range(12)
    )
    + "\n"
)


def _llm_client(tmp_path, response: LLMResponse) -> TestClient:
    settings = Settings(data_dir=tmp_path / "data", llm_enabled=True, llm_model="fake")
    app = create_app(settings)
    app.state.recommendations = RecommendationService(FakeProvider(response))
    return TestClient(app)


def test_endpoint_with_llm_and_llm_false_switch(tmp_path) -> None:
    client = _llm_client(
        tmp_path,
        LLMResponse(
            insights=[{"text": "value2 doubles value.", "chart": dict(NEW_CHART)}],
            charts=[],
        ),
    )
    dataset_id = client.post("/api/datasets", files={"file": ("d.csv", CSV.encode())}).json()["dataset_id"]

    body = client.get(f"/api/datasets/{dataset_id}/recommendations").json()
    (insight,) = body["insights"]
    assert insight["text"] == "value2 doubles value."
    assert insight["supported"] in ("strong", "weak", "unverified")
    assert isinstance(insight["chart_priority"], int)
    assert any(c["source"] == "llm" for c in body["charts"])

    rules_only = client.get(f"/api/datasets/{dataset_id}/recommendations", params={"llm": "false"}).json()
    assert rules_only["insights"] == []
    assert all(c["source"] == "rules" for c in rules_only["charts"])


def test_endpoint_llm_disabled_ignores_llm_param(client: TestClient) -> None:
    dataset_id = client.post("/api/datasets", files={"file": ("d.csv", CSV.encode())}).json()["dataset_id"]
    default = client.get(f"/api/datasets/{dataset_id}/recommendations").json()
    forced = client.get(f"/api/datasets/{dataset_id}/recommendations", params={"llm": "true"}).json()
    assert default == forced
    assert default["insights"] == []
