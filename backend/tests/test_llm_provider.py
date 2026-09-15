import json
import math
import random
from datetime import datetime, timedelta

import httpx
import polars as pl
import pytest

from app.charts.rules import recommend_charts
from app.llm.coverage import ValidatedHypothesis
from app.llm.prompts import (
    FINAL_SYSTEM_PROMPT,
    HYPOTHESIS_SYSTEM_PROMPT,
    build_final_messages,
    build_hypothesis_messages,
    build_messages,
)
from app.llm.provider import DisabledProvider, LLMError, OpenAICompatProvider
from app.charts.spec import ChartSpec
from app.profiling.profiler import profile_dataset

BIG = 10**6

GOOD_JSON = json.dumps(
    {
        "hypotheses": [
            {
                "statement": "Values rise over time.",
                "variables": ["ts", "value"],
                "evidence_available": True,
                "test_needed": None,
                "columns": {"time": "ts", "target": "value"},
                "chart": {"title": "V over time", "type": "line", "x": "ts", "y": "value", "aggregation": "mean", "reason": "trend", "priority": 1},
                "reason": "a trend changes how the values should be read",
                "importance": 4,
            }
        ],
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


def _completion(content: str, usage: dict | None = None) -> dict:
    body = {"choices": [{"message": {"content": content}}]}
    if usage is not None:
        body["usage"] = usage
    return body


def _provider(handler, model: str = "test-model", **kwargs) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        "http://llm.test/v1", model=model, transport=httpx.MockTransport(handler), **kwargs
    )


def test_request_shape(profile, candidates) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_completion(GOOD_JSON))

    _provider(handler).generate_hypotheses(profile, candidates)
    (request,) = seen
    assert request.url.path == "/v1/chat/completions"
    assert "authorization" not in request.headers
    body = json.loads(request.content)
    assert body["model"] == "test-model"
    assert body["response_format"] == {"type": "json_object"}
    assert body["temperature"] == 0.1 and body["max_tokens"] == 2000
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][0]["content"] == HYPOTHESIS_SYSTEM_PROMPT


def test_api_key_header(profile, candidates) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_completion(GOOD_JSON))

    _provider(handler, api_key="sk-local").generate_hypotheses(profile, candidates)
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
    response, usage = provider.generate_hypotheses(profile, candidates)
    assert response.hypotheses[0]["statement"] == "Values rise over time."
    assert response.hypotheses[0]["columns"] == {"time": "ts", "target": "value"}
    assert response.charts[0]["type"] == "line"
    assert usage.prompt_tokens is None and usage.prompt_chars > 1000 and usage.latency_ms >= 0


def test_usage_block_is_recorded(profile, candidates) -> None:
    handler = lambda req: httpx.Response(  # noqa: E731
        200, json=_completion(GOOD_JSON, usage={"prompt_tokens": 1234, "completion_tokens": 56})
    )
    _, usage = _provider(handler).generate_hypotheses(profile, candidates)
    assert (usage.prompt_tokens, usage.completion_tokens) == (1234, 56)
    assert usage.completion_chars == len(GOOD_JSON)


def test_retry_once_then_success(profile, candidates) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json=_completion(GOOD_JSON))

    response, _ = _provider(handler).generate_hypotheses(profile, candidates)
    assert len(calls) == 2 and response.hypotheses


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
        _provider(handler).generate_hypotheses(profile, candidates)
    assert len(calls) == 2  # exactly one retry
    assert "retry" in exc_info.value.category


def test_timeout_does_not_retry(profile, candidates) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.ConnectTimeout("too slow")

    with pytest.raises(LLMError) as exc_info:
        _provider(handler).generate_hypotheses(profile, candidates)
    assert len(calls) == 1
    assert exc_info.value.category == "timeout"


def test_missing_model_fails_before_any_request(profile, candidates) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_completion(GOOD_JSON))

    with pytest.raises(LLMError) as exc_info:
        _provider(handler, model="").generate_hypotheses(profile, candidates)
    assert calls == []
    assert "misconfigured" in exc_info.value.category


def test_disabled_provider_returns_empty(profile, candidates) -> None:
    response, usage = DisabledProvider().generate_hypotheses(profile, candidates)
    assert response.hypotheses == [] and response.charts == []
    final, _ = DisabledProvider().finalize_insights(profile, [])
    assert final.insights == [] and usage.prompt_chars == 0


# --- prompt assembly (LLM #1) -----------------------------------------------


def test_prompt_content(profile, candidates) -> None:
    system, user = build_messages(profile, candidates)
    assert "pie" in system["content"]  # explicit ban
    assert "aggregation" in system["content"]
    content = user["content"]
    for name in ("ts", "city", "value", "value2"):
        assert name in content
    assert "Strongest correlations" in content  # value vs value2 = 1.0
    assert "Sample rows" in content
    assert "Rule-generated chart candidates:" in content


def test_hypothesis_prompt_contract(profile, candidates) -> None:
    system, user = build_hypothesis_messages(profile, candidates)
    text = system["content"]
    # the user's rules, verbatim
    assert "Do not report a relationship merely because its effect size or correlation is large" in text
    assert "analytically useful" in text
    assert "0 to 5 hypotheses" in text and "empty list is a correct answer" in text
    # the closed probe vocabulary and its role constraints
    for probe in (
        "group_difference",
        "distribution_difference",
        "grouped_relationship",
        "slope_difference",
        "nonlinear_relationship",
        "time_pattern",
        "interaction",
    ):
        assert probe in text
    assert "At most 5 probes" in text
    assert "2-20 categories" in text and "datetime" in text
    # what is never a discovery
    for phrase in ("definitional", "near-duplicate", "identifiers", "close to zero"):
        assert phrase in text
    # the JSON contract
    for key in ('"statement"', '"test_needed"', '"columns"', '"chart"', '"importance"'):
        assert key in text
    content = user["content"]
    assert "Measured evidence" in content
    assert "Group effects (eta" in content
    assert "eta=" in content  # top eta pairs listed (city -> value etc.)
    assert "spearman=" in content  # correlations carry both coefficients
    assert "0-5 hypotheses" in content


def test_prompt_without_sample_rows(profile, candidates) -> None:
    _, user = build_hypothesis_messages(profile, candidates, include_sample_rows=False)
    assert "Sample rows" not in user["content"]


def test_prompt_truncates_long_cells_and_column_list() -> None:
    data = {f"c{i:02d}": [float(i), float(i + 1)] for i in range(45)}
    data["long_text"] = ["x" * 500, "y" * 500]
    profile = profile_dataset(pl.DataFrame(data), "0" * 32, BIG)
    _, user = build_hypothesis_messages(profile, [])
    content = user["content"]
    assert "only the first 40 of 46 columns" in content
    assert "x" * 500 not in content and "x" * 100 + "…" in content


def test_fenced_json_with_trailing_prose_parses(profile, candidates) -> None:
    content = '```json\n{"hypotheses": [{"statement": "a"}], "charts": []}\n```\nHope this helps!'
    resp, _ = _provider(lambda r: httpx.Response(200, json=_completion(content))).generate_hypotheses(
        profile, candidates
    )
    assert resp.hypotheses == [{"statement": "a"}]


def test_prompt_marks_nominal_and_id_columns() -> None:
    # stage 9 #4: the LLM must not treat a numeric-looking code as a quantity
    # nor propose an identifier as an axis; only metadata is added, no values
    df = pl.DataFrame(
        {
            "user_id": [f"u{i:03d}" for i in range(30)],
            "code": [101, 102, 103] * 10,
            "amount": [float(i) for i in range(30)],
        }
    )
    profile = profile_dataset(df, "0" * 32, BIG)
    code = next(c for c in profile.columns if c.name == "code")
    code.semantic_type, code.nominal, code.n_categories = "categorical", True, 3
    _, user = build_hypothesis_messages(profile, [], include_sample_rows=False)
    content = user["content"]
    code_line = next(line for line in content.splitlines() if line.startswith("- code"))
    assert "nominal code" in code_line and "not a quantity" in code_line
    id_line = next(line for line in content.splitlines() if line.startswith("- user_id"))
    assert "identifier" in id_line and "not usable as a chart axis" in id_line
    amount_line = next(line for line in content.splitlines() if line.startswith("- amount"))
    assert "nominal" not in amount_line and "mean=" in amount_line


def test_prompt_notes_suspected_sentinels_and_extremes() -> None:
    from app.profiling.models import SentinelCandidate

    df = pl.DataFrame({"t": [float(i) for i in range(30)], "clean": [float(i) for i in range(30)]})
    profile = profile_dataset(df, "0" * 32, BIG)
    t = next(c for c in profile.columns if c.name == "t")
    t.quality.suspected_sentinels = [
        SentinelCandidate(value=9999.0, count=3, signals=["extreme"]),
        SentinelCandidate(value=-999.0, count=2, signals=["extreme"]),
    ]
    t.quality.extreme_value_count = 4
    _, user = build_hypothesis_messages(profile, [], include_sample_rows=False)
    lines = user["content"].splitlines()
    t_line = next(line for line in lines if line.startswith("- t "))
    assert "suspected sentinel values: -999, 9999" in t_line and "4 extreme values" in t_line
    clean_line = next(line for line in lines if line.startswith("- clean"))
    assert "sentinel" not in clean_line and "extreme" not in clean_line


def test_prompt_lists_derived_columns_as_definitional() -> None:
    rng = random.Random(8)
    n = 60
    a = [rng.uniform(1, 100) for _ in range(n)]
    b = [rng.uniform(1, 100) for _ in range(n)]
    profile = profile_dataset(
        pl.DataFrame({"a": a, "b": b, "total": [x + y for x, y in zip(a, b)], "twice": [2 * x for x in a]}),
        "0" * 32,
        BIG,
    )
    system, user = build_hypothesis_messages(profile, recommend_charts(profile))
    assert "derived / definitional" in system["content"]
    content = user["content"]
    assert "Derived columns (definitional" in content and "NOT discoveries" in content
    assert "- total = a + b" in content
    assert "Near-duplicate columns (NOT discoveries" in content


def _structured_profile():
    """Planted layer-2 structure: a U-shaped pair, a subgroup that stands
    out, a right-skewed column and a group-dependent relationship."""
    rng = random.Random(21)
    n = 600
    x = [rng.uniform(0, 10) for _ in range(n)]
    seg = ["a", "b", "c", "d", "e"] * (n // 5)
    z = [rng.uniform(-1, 1) for _ in range(n)]
    return profile_dataset(
        pl.DataFrame(
            {
                "ts": [datetime(2024, 1, 1) + timedelta(days=i % 200) for i in range(n)],
                "x": x,
                "u": [(v - 5) ** 2 + rng.gauss(0, 1) for v in x],
                "segment": seg,
                "sales": [rng.gauss(50 if s != "e" else 80, 5) for s in seg],
                "skewed": [math.exp(rng.gauss(0, 1)) for _ in range(n)],
                "z": z,
                "w": [(zz if s in ("a", "b") else -zz) * 3 + rng.gauss(0, 0.5) for zz, s in zip(z, seg)],
            }
        ),
        "0" * 32,
        BIG,
    )


def test_prompt_layer2_section_selects_structure_not_noise() -> None:
    profile = _structured_profile()
    l2 = profile.evidence.layer2
    assert any(s.shape == "u_shape" for s in l2.nonlinear)
    _, user = build_hypothesis_messages(profile, [], include_sample_rows=False)
    content = user["content"]
    assert "Non-linear relationships" in content
    u_line = next(line for line in content.splitlines() if line.startswith("- x -> u"))
    assert "shape=u_shape" in u_line and "bin means=" in u_line and "n=" in u_line
    assert "Subgroups that stand out" in content and "segment=e" in content
    assert "Distribution shapes worth knowing" in content and "- skewed: right_skewed" in content
    assert "Relationships that differ by group" in content and "z vs w by segment" in content or "w vs z by segment" in content
    # every layer-2 category is capped, and series are compressed to shapes
    assert content.count("shape=") <= 3
    assert "TimeSeriesPoint" not in content and "[{" not in content
    # the whole evidence section stays bounded
    assert len(content) < 12_000


# --- LLM #2 ------------------------------------------------------------------


def _validated(profile) -> list[ValidatedHypothesis]:
    return [
        ValidatedHypothesis(
            id=1,
            statement="Values rise over time.",
            reason="a trend",
            importance=4,
            test_type="time_pattern",
            columns={"time": "ts", "target": "value"},
            validation="existing_evidence",
            verdict="pass",
            effect_label="adjusted eta-squared of target across time buckets",
            effect_value=0.42,
            n=12,
            evidence_lines=["adjusted eta-squared 0.420 across day buckets"],
            evidence={},
            chart_candidates=[("h1c1", ChartSpec(title="value over ts", type="line", x="ts", y="value", aggregation="mean"))],
        )
    ]


def test_final_prompt_carries_only_validated_facts(profile) -> None:
    system, user = build_final_messages(profile, _validated(profile))
    assert system["content"] == FINAL_SYSTEM_PROMPT
    assert "Do NOT add any factual claim" in system["content"]
    assert '"hypothesis_id"' in system["content"] and "No strong non-definitional" in system["content"]
    content = user["content"]
    assert "[1] Values rise over time." in content
    assert "effect: adjusted eta-squared of target across time buckets = 0.420 (n=12)" in content
    assert "chart h1c1: line, x=ts, y=value" in content
    assert "Columns involved:" in content and "- ts (datetime" in content and "- value (numeric" in content
    assert "city" not in content  # only the columns the hypotheses use
    assert "Sample rows" not in content and "Rule-generated" not in content


def test_finalize_insights_round_trip(profile) -> None:
    final_json = json.dumps({"insights": [{"hypothesis_id": 1, "text": "Values climb steadily.", "why_it_matters": "trend", "priority": 4, "chart_id": "h1c1"}], "message": None})
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_completion(final_json))

    final, usage = _provider(handler).finalize_insights(profile, _validated(profile))
    assert final.insights[0]["text"] == "Values climb steadily."
    assert seen[0]["messages"][0]["content"] == FINAL_SYSTEM_PROMPT
    assert usage.completion_chars == len(final_json)
