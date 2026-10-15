"""Audit A6: the recommendation cache never keeps an LLM outage for the
process lifetime, and its key covers what changes the LLM's answer (model,
endpoint, prompt version, profile version)."""

import pytest
from fastapi.testclient import TestClient

import app.llm.service as service_module
from app.config import Settings
from app.llm.provider import LLMError, OpenAICompatProvider
from app.llm.schemas import FinalResponse, HypothesisResponse, LLMUsage
from app.llm.service import FAILURE_TTL_SECONDS, RecommendationService
from app.main import create_app
from app.profiling.profiler import profile_dataset
from tests.test_llm_workflow import BIG, _planted, _two_covered


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Scripted:
    """LLM #1 / LLM #2 outcomes by call number: an LLMError instance raises."""

    def __init__(self, llm1=(), llm2=(), final=None, identity=("fake", "m1")) -> None:
        self.llm1 = list(llm1)
        self.llm2 = list(llm2)
        self.final = final
        self.calls = 0
        self.final_calls = 0
        self.cache_identity = identity

    def generate_hypotheses(self, profile, rules):
        self.calls += 1
        outcome = self.llm1.pop(0) if self.llm1 else None
        if isinstance(outcome, LLMError):
            raise outcome
        return HypothesisResponse(hypotheses=_two_covered()), LLMUsage()

    def finalize_insights(self, profile, validated):
        self.final_calls += 1
        outcome = self.llm2.pop(0) if self.llm2 else None
        if isinstance(outcome, LLMError):
            raise outcome
        return FinalResponse.model_validate(self.final or {"insights": []}), LLMUsage()


@pytest.fixture(scope="module")
def df():
    return _planted()


@pytest.fixture(scope="module")
def profile(df):
    return profile_dataset(df, "4" * 32, BIG)


def _service(provider):
    clock = Clock()
    return RecommendationService(provider, clock=clock), clock


def test_llm1_failure_is_cached_briefly_and_recovery_is_served(profile, df) -> None:
    provider = Scripted(llm1=[LLMError("timeout", "slow")])
    service, clock = _service(provider)
    first = service.get(profile, True, df=df)
    assert first["insights"] == [] and "timeout" in first["message"] and "rule-based" in first["message"]
    # within the TTL the fallback is served without hammering a dead LLM
    clock.now += FAILURE_TTL_SECONDS - 1
    assert service.get(profile, True, df=df) == first and provider.calls == 1
    # after it, the recovered LLM answers and that answer is what is served
    clock.now += 2
    recovered = service.get(profile, True, df=df)
    assert provider.calls == 2 and recovered["message"] is None
    assert len(recovered["insights"]) == 2 and recovered["insights"][0]["text"].startswith("Mean sales is highest for segment e")
    # a success never expires
    clock.now += 10 * FAILURE_TTL_SECONDS
    assert service.get(profile, True, df=df) == recovered and provider.calls == 2


def test_llm2_failure_is_transient_too(profile, df) -> None:
    good = {"insights": [{"hypothesis_id": 1, "text": "Segment e averages far higher sales.", "why_it_matters": "pricing", "priority": 5}]}
    provider = Scripted(llm2=[LLMError("request failed after retry", "502")], final=good)
    service, clock = _service(provider)
    first = service.get(profile, True, df=df, include_debug=True)
    assert "llm2: request failed after retry" in first["debug"]["errors"]
    assert all(w["source"] != "llm2" for w in first["debug"]["wording"])  # template wording meanwhile
    clock.now += FAILURE_TTL_SECONDS + 1
    second = service.get(profile, True, df=df, include_debug=True)
    assert provider.calls == 2 and provider.final_calls == 2
    assert any(i["text"] == "Segment e averages far higher sales." for i in second["insights"])


def test_success_is_cached_and_refresh_rebuilds(profile, df) -> None:
    provider = Scripted()
    service, _ = _service(provider)
    first = service.get(profile, True, df=df)
    assert service.get(profile, True, df=df) == first and provider.calls == 1
    refreshed = service.get(profile, True, df=df, refresh=True)
    assert provider.calls == 2 and refreshed == first
    assert service.get(profile, True, df=df) == refreshed and provider.calls == 2  # the refreshed entry is cached


def test_gate_rejections_are_deterministic_and_cached(profile, df) -> None:
    bad = {"insights": [{"hypothesis_id": 1, "text": "Segment a outsells segment e.", "why_it_matters": "?", "priority": 5}]}
    provider = Scripted(final=bad)
    service, clock = _service(provider)
    first = service.get(profile, True, df=df, include_debug=True)
    assert any(e.startswith("llm2 insight 1") for e in first["debug"]["errors"])
    clock.now += 10 * FAILURE_TTL_SECONDS
    service.get(profile, True, df=df)
    assert provider.calls == 1 and provider.final_calls == 1


def test_model_endpoint_and_prompt_version_are_part_of_the_key(profile, df, monkeypatch) -> None:
    provider = Scripted()
    service, _ = _service(provider)
    service.get(profile, True, df=df)
    provider.cache_identity = ("fake", "m2")  # another model answers now
    service.get(profile, True, df=df)
    assert provider.calls == 2
    monkeypatch.setattr(service_module, "PROMPT_VERSION", "next")
    service.get(profile, True, df=df)
    assert provider.calls == 3
    # rules-only results do not depend on the model and never call it
    rules = service.get(profile, False)
    provider.cache_identity = ("fake", "m3")
    assert service.get(profile, False) == rules and provider.calls == 3
    # the real provider names its model, endpoint and prompt options
    a = OpenAICompatProvider("http://localhost:1/v1", "qwen")
    b = OpenAICompatProvider("http://localhost:2/v1", "qwen")
    assert a.cache_identity != b.cache_identity
    assert a.cache_identity != OpenAICompatProvider("http://localhost:1/v1", "llama").cache_identity


def test_key_lock_table_is_bounded(profile, df, monkeypatch) -> None:
    monkeypatch.setattr(service_module, "MAX_KEY_LOCKS", 3)
    provider = Scripted()
    service, _ = _service(provider)
    for i in range(6):
        provider.cache_identity = ("fake", f"m{i}")
        service.get(profile, True, df=df)
    assert len(service._key_locks) <= 3 and provider.calls == 6


def test_refresh_query_parameter_rebuilds(tmp_path) -> None:
    app = create_app(Settings(data_dir=tmp_path / "data", llm_enabled=True, llm_model="fake"))
    provider = Scripted()
    app.state.recommendations = RecommendationService(provider)
    client = TestClient(app)
    csv = "segment,sales\n" + "\n".join(f"{'ab'[i % 2]},{i % 7 + (40 if i % 2 else 0)}" for i in range(80)) + "\n"
    dataset_id = client.post("/api/datasets", files={"file": ("s.csv", csv.encode())}).json()["dataset_id"]
    url = f"/api/datasets/{dataset_id}/recommendations"
    first = client.get(url)
    assert first.status_code == 200 and provider.calls == 1
    assert client.get(url).json() == first.json() and provider.calls == 1
    assert client.get(url + "?refresh=1").status_code == 200 and provider.calls == 2
    assert client.get(url + "?llm=false&refresh=1").status_code == 200 and provider.calls == 2
