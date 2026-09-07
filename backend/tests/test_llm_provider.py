import json
from datetime import datetime, timedelta

import httpx
import polars as pl
import pytest

from app.charts.rules import recommend_charts
from app.llm.prompts import build_messages
from app.llm.provider import DisabledProvider, LLMError, OpenAICompatProvider
from app.profiling.profiler import profile_dataset

BIG = 10**6

GOOD_JSON = json.dumps(
    {
        "insights": ["Values rise over time."],
        "charts": [
            {"title": "V over time", "type": "line", "x": "ts", "y": "value", "aggregation": "mean", "reason": "trend", "priority": 1}
        ],
    }
)


@pytest.fixture(scope="module")
def profile():
    df = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(12)],
            "city": ["a", "b", "c"] * 4,
            "value": [float(i) for i in range(12)],
            "value2": [2.0 * i for i in range(12)],
        }
    )
    return profile_dataset(df, "0" * 32, BIG)


@pytest.fixture(scope="module")
def candidates(profile):
    return recommend_charts(profile)


def _completion(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def _provider(handler, model: str = "test-model", **kwargs) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        "http://llm.test/v1", model=model, transport=httpx.MockTransport(handler), **kwargs
    )


def test_request_shape(profile, candidates) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_completion(GOOD_JSON))

    _provider(handler).recommend_charts(profile, candidates)
    (request,) = seen
    assert request.url.path == "/v1/chat/completions"
    assert "authorization" not in request.headers
    body = json.loads(request.content)
    assert body["model"] == "test-model"
    assert body["response_format"] == {"type": "json_object"}
    assert body["temperature"] == 0.1 and body["max_tokens"] == 2000
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_api_key_header(profile, candidates) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_completion(GOOD_JSON))

    _provider(handler, api_key="sk-local").recommend_charts(profile, candidates)
    assert seen[0].headers["authorization"] == "Bearer sk-local"


@pytest.mark.parametrize(
    "content",
    [
        GOOD_JSON,
        f"```json\n{GOOD_JSON}\n```",
        f"```\n{GOOD_JSON}\n```",
        f"Here is my analysis:\n{GOOD_JSON}\nHope it helps!",
    ],
    ids=["plain", "json-fence", "bare-fence", "prose-wrapped"],
)
def test_content_parsing(profile, candidates, content: str) -> None:
    provider = _provider(lambda req: httpx.Response(200, json=_completion(content)))
    response = provider.recommend_charts(profile, candidates)
    assert response.insights == ["Values rise over time."]
    assert response.charts[0]["type"] == "line"


def test_retry_once_then_success(profile, candidates) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json=_completion(GOOD_JSON))

    response = _provider(handler).recommend_charts(profile, candidates)
    assert len(calls) == 2 and response.charts


@pytest.mark.parametrize(
    "make_response",
    [
        lambda: httpx.Response(500, text="boom"),
        lambda: httpx.Response(200, json=_completion("this is not json at all")),
        lambda: httpx.Response(200, json={"unexpected": "envelope"}),
    ],
    ids=["http-500", "non-json-content", "malformed-envelope"],
)
def test_persistent_failures_retry_then_raise(profile, candidates, make_response) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return make_response()

    with pytest.raises(LLMError) as exc_info:
        _provider(handler).recommend_charts(profile, candidates)
    assert len(calls) == 2  # exactly one retry
    assert "retry" in exc_info.value.category


def test_timeout_does_not_retry(profile, candidates) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.ConnectTimeout("too slow")

    with pytest.raises(LLMError) as exc_info:
        _provider(handler).recommend_charts(profile, candidates)
    assert len(calls) == 1
    assert exc_info.value.category == "timeout"


def test_missing_model_fails_before_any_request(profile, candidates) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_completion(GOOD_JSON))

    with pytest.raises(LLMError) as exc_info:
        _provider(handler, model="").recommend_charts(profile, candidates)
    assert calls == []
    assert "misconfigured" in exc_info.value.category


def test_disabled_provider_returns_empty(profile, candidates) -> None:
    response = DisabledProvider().recommend_charts(profile, candidates)
    assert response.insights == [] and response.charts == []


# --- prompt assembly --------------------------------------------------------


def test_prompt_content(profile, candidates) -> None:
    system, user = build_messages(profile, candidates)
    assert "pie" in system["content"]  # explicit ban
    assert "aggregation" in system["content"]
    content = user["content"]
    for name in ("ts", "city", "value", "value2"):
        assert name in content
    assert "Strongest correlations" in content  # value vs value2 = 1.0
    assert "Sample rows:" in content
    assert "Rule-generated chart candidates:" in content


def test_prompt_evidence_summary_and_insight_contract(profile, candidates) -> None:
    system, user = build_messages(profile, candidates)
    # stage8: every insight must ship a supporting chart, from evidence
    assert "supporting" in system["content"]
    assert '"chart"' in system["content"]
    assert "close to zero" in system["content"]
    content = user["content"]
    assert "Measured evidence" in content
    assert "Group effects (eta" in content
    assert "eta=" in content  # top eta pairs listed (city -> value etc.)
    assert "spearman=" in content  # correlations carry both coefficients


def test_prompt_without_sample_rows(profile, candidates) -> None:
    _, user = build_messages(profile, candidates, include_sample_rows=False)
    assert "Sample rows:" not in user["content"]


def test_prompt_truncates_long_cells_and_column_list() -> None:
    data = {f"c{i:02d}": [float(i), float(i + 1)] for i in range(45)}
    data["long_text"] = ["x" * 500, "y" * 500]
    profile = profile_dataset(pl.DataFrame(data), "0" * 32, BIG)
    _, user = build_messages(profile, [])
    content = user["content"]
    assert "only the first 40 of 46 columns" in content
    assert "x" * 500 not in content and "x" * 100 + "…" in content


def test_fenced_json_with_trailing_prose_parses(profile, candidates) -> None:
    content = '```json\n{"insights": ["a"], "charts": []}\n```\nHope this helps!'
    resp = _provider(lambda r: httpx.Response(200, json=_completion(content))).recommend_charts(
        profile, candidates
    )
    assert resp.insights == ["a"]
