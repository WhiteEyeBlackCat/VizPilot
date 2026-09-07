from datetime import datetime, timedelta

import polars as pl
import pytest
from fastapi.testclient import TestClient

from app.charts.rules import recommend_charts
from app.config import Settings
from app.llm.provider import LLMError
from app.llm.schemas import LLMResponse
from app.llm.service import RecommendationService
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
    df = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(12)],
            "city": ["a", "b", "c"] * 4,
            "value": [float(i) for i in range(12)],
            "value2": [2.0 * i for i in range(12)],
            "value3": [float(i % 2) for i in range(12)],  # uncorrelated
        }
    )
    return profile_dataset(df, "0" * 32, BIG)


def _service(response=None, error=None) -> tuple[RecommendationService, FakeProvider]:
    provider = FakeProvider(response, error)
    return RecommendationService(provider), provider


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


def test_new_llm_chart_merged_ahead_of_rules(profile) -> None:
    service, _ = _service(LLMResponse(insights=["a insight"], charts=[NEW_CHART]))
    result = service.get(profile, use_llm=True)
    first = result["charts"][0]
    assert first["source"] == "llm" and first["score"] == 0.75
    assert first["spec"]["type"] == "scatter"
    assert (first["spec"]["x"], first["spec"]["y"]) == ("value", "value3")
    assert first["spec"]["priority"] == 1
    assert result["insights"] == ["a insight"]
    assert result["message"] is None
    # remaining charts keep score order with consecutive priorities
    priorities = [c["spec"]["priority"] for c in result["charts"]]
    assert priorities == list(range(1, len(priorities) + 1))


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
    assert len(insights) == 5
    assert all(len(i) == 300 for i in insights)


def test_cache_per_dataset_and_per_llm_flag(profile) -> None:
    service, provider = _service(LLMResponse(insights=["cached"], charts=[NEW_CHART]))
    first = service.get(profile, use_llm=True)
    assert service.get(profile, use_llm=True) is first  # cache hit, no second call
    assert provider.calls == 1
    rules_only = service.get(profile, use_llm=False)
    assert provider.calls == 1  # llm=false path never touches the provider
    assert rules_only["insights"] == []


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
            insights=["value2 doubles value."],
            charts=[dict(NEW_CHART, x="value", y="value3")],
        ),
    )
    dataset_id = client.post("/api/datasets", files={"file": ("d.csv", CSV.encode())}).json()["dataset_id"]

    body = client.get(f"/api/datasets/{dataset_id}/recommendations").json()
    assert body["insights"] == ["value2 doubles value."]
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


def test_malformed_insights_do_not_invalidate_response() -> None:
    # a bare string, and a list with non-string junk, must both survive
    from app.llm.schemas import LLMResponse

    r1 = LLMResponse.model_validate({"insights": "single insight", "charts": []})
    r2 = LLMResponse.model_validate({"insights": ["ok", 42, None, "  "], "charts": []})
    from app.llm.service import _clean_insights

    assert _clean_insights(r1.insights) == ["single insight"]
    assert _clean_insights(r2.insights) == ["ok"]
    assert _clean_insights({"not": "a list"}) == []
