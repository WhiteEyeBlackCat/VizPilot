"""The two-stage LLM workflow (stage 17.3): gate -> coverage -> probes ->
conditional LLM #2. Everything runs on a planted dataset with known
structure so every verdict can be predicted."""

import random
from datetime import datetime, timedelta

import polars as pl
import pytest

from app.charts.rules import recommend_charts
from app.llm.schemas import FinalResponse, HypothesisResponse, LLMUsage
from app.llm.service import NO_PATTERNS_MESSAGE, RecommendationService
from app.probes import ProbeCache
from app.profiling.profiler import profile_dataset

BIG = 10**6


class TwoStageFake:
    def __init__(self, hypotheses: list[dict] | None = None, charts: list[dict] | None = None, final: dict | None = None) -> None:
        self.hypotheses = hypotheses or []
        self.charts = charts or []
        self.final = final
        self.calls = 0
        self.final_calls = 0
        self.seen_validated = None

    def generate_hypotheses(self, profile, rule_candidates):
        self.calls += 1
        return HypothesisResponse(hypotheses=self.hypotheses, charts=self.charts), LLMUsage()

    def finalize_insights(self, profile, validated):
        self.final_calls += 1
        self.seen_validated = validated
        final = FinalResponse.model_validate(self.final) if self.final else FinalResponse()
        return final, LLMUsage()


def _planted(n: int = 600) -> pl.DataFrame:
    """segment shifts sales (group difference in the tables); x -> u is a U
    shape (layer 2); z -> w flips sign by segment (conditional); eight noise
    columns for probe-cap tests; ts spans 600 days."""
    rng = random.Random(21)
    seg = ["a", "b", "c", "d", "e"] * (n // 5)
    x = [rng.uniform(0, 10) for _ in range(n)]
    z = [rng.uniform(-1, 1) for _ in range(n)]
    data = {
        "ts": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n)],
        "segment": seg,
        "flag": [i % 2 == 0 for i in range(n)],
        "sales": [rng.gauss(50 if s != "e" else 80, 5) for s in seg],
        "x": x,
        "u": [(v - 5) ** 2 + rng.gauss(0, 1) for v in x],
        "z": z,
        "w": [(zz if s in ("a", "b") else -zz) * 3 + rng.gauss(0, 0.5) for zz, s in zip(z, seg)],
        "growth": [i * 0.1 + rng.gauss(0, 3) for i in range(n)],
    }
    for k in range(8):
        data[f"n{k}"] = [rng.gauss(0, 1) for _ in range(n)]
    return pl.DataFrame(data)


@pytest.fixture(scope="module")
def df():
    return _planted()


@pytest.fixture(scope="module")
def profile(df):
    return profile_dataset(df, "2" * 32, BIG)


def _run(profile, df, hypotheses=None, charts=None, final=None, cache=None):
    provider = TwoStageFake(hypotheses, charts, final)
    service = RecommendationService(provider, probe_cache=cache)
    return service.get(profile, use_llm=True, df=df, include_debug=True), provider


def _bar(group: str, target: str) -> dict:
    return {"title": f"Mean {target} by {group}", "type": "bar", "x": group, "y": target, "aggregation": "mean", "priority": 1}


def _h(statement: str, test: str | None, columns: dict | None, chart: dict | None, importance: int = 3, **extra) -> dict:
    return {
        "statement": statement,
        "variables": list((columns or {}).values()),
        "evidence_available": test is None,
        "test_needed": test,
        "columns": columns,
        "chart": chart,
        "reason": "matters",
        "importance": importance,
        **extra,
    }


# --- coverage: the tables already answer ------------------------------------------


def test_covered_hypothesis_needs_no_probe_and_no_second_call(profile, df) -> None:
    assert any(e.cat == "segment" and e.num == "sales" and e.eta_squared >= 0.1 for e in profile.evidence.cat_num)
    result, provider = _run(profile, df, [_h("Segment e sells far more.", "group_difference", {"group": "segment", "target": "sales"}, _bar("segment", "sales"), 5)])
    (insight,) = result["insights"]
    assert insight["validation"] == "existing_evidence" and insight["supported"] == "strong"
    assert insight["effect"]["label"].startswith("adjusted eta-squared") and insight["effect"]["value"] >= 0.1
    assert insight["priority"] == 5
    assert insight["text"].startswith("Segment e sells far more (adjusted eta-squared")
    assert result["debug"]["probes"] == [] and provider.final_calls == 0
    assert result["debug"]["covered_by_existing_evidence"][0]["verdict"] == "pass"
    assert result["message"] is None
    chart = result["charts"][insight["chart_priority"] - 1]
    assert (chart["spec"]["type"], chart["spec"]["x"], chart["spec"]["y"]) == ("bar", "segment", "sales")


def test_covered_but_failing_hypothesis_is_dropped(profile, df) -> None:
    # segment has no effect on n0: the table says so -> dropped, no probe
    result, provider = _run(profile, df, [_h("Segments differ in n0.", "group_difference", {"group": "segment", "target": "n0"}, _bar("segment", "n0"))])
    assert result["insights"] == [] and result["message"] == NO_PATTERNS_MESSAGE
    assert "existing evidence fails" in result["debug"]["dropped"][0]["reason"]
    assert result["debug"]["probes"] == [] and provider.final_calls == 0


def test_nonlinear_hypothesis_covered_by_layer2(profile, df) -> None:
    assert any({s.x, s.y} == {"x", "u"} and s.shape == "u_shape" for s in profile.evidence.layer2.nonlinear)
    result, _ = _run(profile, df, [_h("u is U-shaped in x.", "nonlinear_relationship", {"x": "x", "y": "u"}, None, 4)])
    (insight,) = result["insights"]
    assert insight["validation"] == "existing_evidence" and insight["supported"] == "strong"
    assert insight["effect"]["label"].startswith("nonlinear_gap")
    # no LLM chart: the backend supplied the scatter
    chart = result["charts"][insight["chart_priority"] - 1]
    assert chart["spec"]["type"] == "scatter" and {chart["spec"]["x"], chart["spec"]["y"]} == {"x", "u"}


# --- probes: the tables do not answer ---------------------------------------------


def test_uncovered_hypothesis_runs_a_probe_then_llm2(profile, df) -> None:
    final = {"insights": [{"hypothesis_id": 1, "text": "Sales in segment e are distributed far above the rest.", "why_it_matters": "e is a different market", "priority": 5, "chart_id": "h1c1"}]}
    result, provider = _run(
        profile,
        df,
        [_h("The sales distribution differs by segment.", "distribution_difference", {"group": "segment", "target": "sales"}, None, 4)],
        final=final,
    )
    (probe,) = result["debug"]["probes"]
    assert probe["type"] == "distribution_difference" and probe["verdict"] == "pass"
    assert provider.final_calls == 1 and result["debug"]["final_call"] is True
    (insight,) = result["insights"]
    assert insight["validation"] == "probe" and insight["supported"] == "strong"
    assert insight["text"] == "Sales in segment e are distributed far above the rest."
    assert insight["why_it_matters"] == "e is a different market" and insight["priority"] == 5
    assert insight["effect"]["label"].startswith("max two-sample KS")
    # LLM #2 only ever saw validated hypotheses with backend numbers
    (seen,) = provider.seen_validated
    assert seen.validation == "probe" and seen.effect_value == probe["effect"]


def test_failed_probe_drops_the_hypothesis(profile, df) -> None:
    # two independent noise columns: the binned dependence is nil -> the probe fails
    result, provider = _run(profile, df, [_h("n1 depends on n0 in a curved way.", "nonlinear_relationship", {"x": "n0", "y": "n1"}, None)])
    assert result["insights"] == [] and result["message"] == NO_PATTERNS_MESSAGE
    (probe,) = result["debug"]["probes"]
    assert probe["verdict"] == "fail"
    assert "probe failed" in result["debug"]["dropped"][0]["reason"]
    assert provider.final_calls == 0


def test_probe_queue_is_capped(profile, df) -> None:
    hypotheses = [
        _h(f"n{k} differs by segment", "distribution_difference", {"group": "segment", "target": f"n{k}"}, None, 5)
        for k in range(8)
    ]
    result, _ = _run(profile, df, hypotheses)
    debug = result["debug"]
    assert debug["hypotheses_proposed"] == 8
    assert sum(1 for d in debug["dropped"] if d["reason"] == "beyond the hypothesis cap") == 3
    assert len(debug["probes"]) == 5 and all("rejected" not in p for p in debug["probes"])


def test_probe_cache_is_shared_between_builds(profile, df) -> None:
    cache = ProbeCache()
    hyp = [_h("sales distribution differs by segment", "distribution_difference", {"group": "segment", "target": "sales"}, None)]
    first, _ = _run(profile, df, hyp, cache=cache)
    second, _ = _run(profile, df, hyp, cache=cache)  # a fresh service, same probe cache
    assert first["debug"]["probes"][0]["cached"] is False
    assert second["debug"]["probes"][0]["cached"] is True


# --- the gate: hallucinations never reach validation ------------------------------


@pytest.mark.parametrize(
    "hypothesis, needle",
    [
        (_h("ghost differs", "group_difference", {"group": "segment", "target": "ghost"}, None), "not a column"),
        (_h("bad variable", "group_difference", {"group": "segment", "target": "sales"}, None, variables=["nope"]), "not a column"),
        (_h("custom code", "custom_sql", {"group": "segment", "target": "sales"}, None), "unknown probe type"),
        (_h("wrong role types", "group_difference", {"group": "sales", "target": "segment"}, None), "probe rejected"),
        (_h("extra role", "group_difference", {"group": "segment", "target": "sales", "expression": "x"}, None), "does not accept roles"),
        (_h("same column twice", "slope_difference", {"x": "x", "y": "x", "group": "segment"}, None), "same column"),
        (_h("missing roles", "interaction", {"factor1": "segment"}, None), "missing"),
        (_h("no chart no test", None, None, None), "nothing to verify"),
    ],
    ids=["ghost-column", "ghost-variable", "unknown-probe", "wrong-types", "extra-role", "duplicate-role", "missing-roles", "empty"],
)
def test_gate_drops_hallucinated_hypotheses(profile, df, hypothesis, needle) -> None:
    result, provider = _run(profile, df, [hypothesis])
    assert result["insights"] == []
    (dropped,) = result["debug"]["dropped"]
    assert needle in dropped["reason"]
    assert provider.final_calls == 0


def test_llm_numbers_are_ignored_backend_numbers_win(profile, df) -> None:
    hyp = _h("Segment e sells far more (eta 0.99).", "group_difference", {"group": "segment", "target": "sales"}, None, effect_size=0.99)
    result, _ = _run(profile, df, [hyp])
    (insight,) = result["insights"]
    assert insight["effect"]["value"] < 0.9  # the table's value, not the LLM's
    assert insight["effect"]["value"] == pytest.approx(
        next(e.eta_squared for e in profile.evidence.cat_num if e.cat == "segment" and e.num == "sales")
    )


def test_empty_hypothesis_list_is_a_valid_answer(profile, df) -> None:
    result, provider = _run(profile, df, [])
    assert result["insights"] == [] and result["message"] == NO_PATTERNS_MESSAGE
    assert provider.calls == 1 and provider.final_calls == 0
    assert result["charts"] == [rec.model_dump() for rec in recommend_charts(profile)]


# --- LLM #2 gate and trigger -------------------------------------------------------


def _two_covered() -> list[dict]:
    return [
        _h("Segment e sells far more.", "group_difference", {"group": "segment", "target": "sales"}, None, 5),
        _h("u is U-shaped in x.", "nonlinear_relationship", {"x": "x", "y": "u"}, None, 4),
    ]


def test_two_validated_findings_trigger_llm2_and_wording_is_gated(profile, df) -> None:
    final = {
        "insights": [
            {"hypothesis_id": 1, "text": "Segment e averages far higher sales.", "why_it_matters": "pricing", "priority": 5, "chart_id": "h1c1"},
            {"hypothesis_id": 2, "text": "u follows a U shape driven by secret_col.", "why_it_matters": "?", "priority": 2},
            {"hypothesis_id": 9, "text": "an invented finding", "why_it_matters": "?", "priority": 5},
        ]
    }
    result, provider = _run(profile, df, _two_covered(), final=final)
    assert provider.final_calls == 1
    texts = {i["priority"]: i["text"] for i in result["insights"]}
    assert texts[5] == "Segment e averages far higher sales."  # LLM #2 wording accepted
    # the second wording named a column that does not exist -> template fallback
    assert texts[4].startswith("u is U-shaped in x (nonlinear_gap")
    assert len(result["insights"]) == 2  # the invented hypothesis_id never appears
    assert any("unknown tokens" in e for e in result["debug"]["errors"])


def test_llm2_failure_falls_back_to_template(profile, df) -> None:
    from app.llm.provider import LLMError

    class Failing(TwoStageFake):
        def finalize_insights(self, profile, validated):
            self.final_calls += 1
            raise LLMError("timeout", "slow")

    provider = Failing(_two_covered())
    result = RecommendationService(provider).get(profile, use_llm=True, df=df, include_debug=True)
    assert provider.final_calls == 1
    assert len(result["insights"]) == 2
    assert all("(" in i["text"] and "n=" in i["text"] for i in result["insights"])
    assert "llm2: timeout" in result["debug"]["errors"]


def test_single_probe_validated_finding_still_calls_llm2(profile, df) -> None:
    hyp = [_h("sales distribution differs by segment", "distribution_difference", {"group": "segment", "target": "sales"}, None)]
    _, provider = _run(profile, df, hyp)
    assert provider.final_calls == 1  # something new was measured


def test_insights_are_ordered_by_priority_and_capped(profile, df) -> None:
    hypotheses = _two_covered() + [
        _h("w's relation to z flips by segment.", "slope_difference", {"x": "z", "y": "w", "group": "segment"}, None, 3),
        _h("growth trends up over time.", "time_pattern", {"time": "ts", "target": "growth"}, None, 2),
    ]
    result, provider = _run(profile, df, hypotheses)
    priorities = [i["priority"] for i in result["insights"]]
    assert priorities == sorted(priorities, reverse=True)
    assert len(result["insights"]) == 4 and provider.final_calls == 1
    assert {i["validation"] for i in result["insights"]} <= {"existing_evidence", "probe"}


# --- rules path untouched -------------------------------------------------------------


def test_rules_only_output_carries_no_workflow_fields(profile, df) -> None:
    provider = TwoStageFake(_two_covered())
    result = RecommendationService(provider).get(profile, use_llm=False, df=df, include_debug=True)
    assert result == {
        "charts": [rec.model_dump() for rec in recommend_charts(profile)],
        "insights": [],
        "message": None,
        "warnings": [],
    }
    assert provider.calls == 0
