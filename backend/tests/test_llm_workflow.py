"""The two-stage LLM workflow (stage 17.3): gate -> coverage -> probes ->
conditional LLM #2. Everything runs on a planted dataset with known
structure so every verdict can be predicted."""

import random
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from app.charts.rules import recommend_charts
from app.llm.schemas import FinalResponse, HypothesisResponse, LLMUsage
from app.llm.service import NO_PATTERNS_MESSAGE, RecommendationService
from app.probes import ProbeCache
from app.datasets.loader import load_dataframe
from app.profiling.profiler import profile_dataset

BIG = 10**6
DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"


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


def test_slope_difference_needs_a_real_relationship_somewhere(profile, df) -> None:
    # z -> w flips sign by segment with |corr| ~ 0.9 inside each group: kept
    result, _ = _run(profile, df, [_h("w's relation to z flips by segment.", "slope_difference", {"x": "z", "y": "w", "group": "segment"}, None, 4)])
    (insight,) = result["insights"]
    assert insight["supported"] == "strong"


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


# --- stage 17.3 round 2: the gates B asked for ------------------------------------------


def _scatter(x: str, y: str, group: str | None = None) -> dict:
    return {"title": f"{y} vs {x}", "type": "scatter", "x": x, "y": y, "group_by": group, "priority": 1}


def test_chart_only_neutral_charts_are_not_testable(profile, df) -> None:
    # a histogram / heatmap / count bar implies no probe: the claim is dropped
    hypotheses = [
        _h("n0 is strongly bimodal.", None, None, {"title": "n0", "type": "histogram", "x": "n0", "priority": 1}),
        _h("everything correlates.", None, None, {"title": "hm", "type": "heatmap", "priority": 2}),
        _h("segments are unbalanced.", None, None, {"title": "count", "type": "bar", "x": "segment", "aggregation": "count", "priority": 3}),
    ]
    result, _ = _run(profile, df, hypotheses)
    assert result["insights"] == []
    assert all("nothing to verify" in d["reason"] for d in result["debug"]["dropped"])
    assert result["debug"]["probes"] == []


def test_chart_only_and_probe_paths_share_thresholds(profile, df) -> None:
    # the same claim through a plain scatter (chart-only) and through an
    # explicit nonlinear probe request must end with the same verdict/effect
    chart_only, _ = _run(profile, df, [_h("u depends on x in a curve.", None, None, _scatter("x", "u"))])
    explicit, _ = _run(profile, df, [_h("u depends on x in a curve.", "nonlinear_relationship", {"x": "x", "y": "u"}, None)])
    a, b = chart_only["insights"][0], explicit["insights"][0]
    assert (a["supported"], a["validation"], a["effect"]) == (b["supported"], b["validation"], b["effect"])
    # ...and a plain linear correlation is no finding on either path
    linear_chart, _ = _run(profile, df, [_h("growth relates to ts index.", None, None, _scatter("n0", "n1"))])
    assert linear_chart["insights"] == []


def test_llm_numbers_are_stripped_from_the_statement(profile, df) -> None:
    hyp = _h("Segment e sells far more (eta² = 0.99, n = 9999), about 80% more.", "group_difference", {"group": "segment", "target": "sales"}, None, 4)
    result, _ = _run(profile, df, [hyp])
    (insight,) = result["insights"]
    assert "0.99" not in insight["text"] and "9999" not in insight["text"]
    # the labelled numbers are stripped; the bare "80%" is an LLM number the
    # backend never measured, so the neutral wording replaces the statement
    assert insight["text"].startswith("Sales varies by segment (adjusted eta-squared")
    (record,) = result["debug"]["wording"]
    assert record["statement"] == "Segment e sells far more, about 80% more." and record["source"] == "neutral"


def test_statement_with_unknown_column_token_is_dropped(profile, df) -> None:
    hyp = _h("Sales rise with customer_tier in segment e.", "group_difference", {"group": "segment", "target": "sales"}, None)
    result, _ = _run(profile, df, [hyp])
    assert result["insights"] == []
    assert "unknown columns ['customer_tier']" in result["debug"]["dropped"][0]["reason"]


@pytest.mark.parametrize(
    "text, problem",
    [
        ("Segment e averages 9999 in sales.", "number 9999"),
        ("Segment e sells more; growth explains it.", "names columns outside"),
        ("Segment e sells more than segment_x.", "unknown tokens"),
    ],
    ids=["invented-number", "foreign-column", "foreign-token"],
)
def test_llm2_wording_gate_falls_back_to_template(profile, df, text, problem) -> None:
    final = {
        "insights": [
            {"hypothesis_id": 1, "text": text, "why_it_matters": "?", "priority": 5},
            {"hypothesis_id": 2, "text": "u follows a U shape in x.", "why_it_matters": "?", "priority": 3},
        ]
    }
    result, _ = _run(profile, df, _two_covered(), final=final)
    texts = {i["priority"]: i["text"] for i in result["insights"]}
    assert texts[5].startswith("Segment e sells far more (adjusted eta-squared")  # template fallback
    assert texts[3] == "u follows a U shape in x."  # clean wording accepted
    assert any(problem in e for e in result["debug"]["errors"])
    sources = {w["id"]: w["source"] for w in result["debug"]["wording"]}
    assert sources == {1: "template", 2: "llm2"}


def test_llm2_may_quote_backend_numbers(profile, df) -> None:
    eta = next(e.eta_squared for e in profile.evidence.cat_num if e.cat == "segment" and e.num == "sales")
    text = f"Segment e sells far more (eta-squared {eta:.2f}, n=600)."
    final = {"insights": [{"hypothesis_id": 1, "text": text, "why_it_matters": "pricing", "priority": 5}]}
    result, _ = _run(profile, df, _two_covered(), final=final)
    assert any(i["text"] == text for i in result["insights"])


def test_below_min_rows_nothing_is_a_finding() -> None:
    rng = random.Random(3)
    small = pl.DataFrame(
        {
            "g": ["a", "b"] * 10,
            "v": [rng.gauss(0 if i % 2 == 0 else 5, 0.3) for i in range(20)],
        }
    )
    profile = profile_dataset(small, "3" * 32, BIG)
    assert any(e.cat == "g" and e.num == "v" and e.eta_squared > 0.9 for e in profile.evidence.cat_num)
    result, provider = _run(profile, small, [_h("g splits v.", "group_difference", {"group": "g", "target": "v"}, None, 5)])
    assert result["insights"] == [] and result["message"] == NO_PATTERNS_MESSAGE
    assert "fewer than 30 rows" in result["debug"]["dropped"][0]["reason"]
    assert provider.final_calls == 0


def test_priority_is_clamped(profile, df) -> None:
    final = {"insights": [{"hypothesis_id": 1, "text": "Segment e sells far more.", "why_it_matters": "?", "priority": 42}, {"hypothesis_id": 2, "text": "u follows a U shape in x.", "why_it_matters": "?", "priority": -3}]}
    result, _ = _run(profile, df, _two_covered(), final=final)
    assert sorted(i["priority"] for i in result["insights"]) == [1, 5]
    hyp = [_h("Segment e sells far more.", "group_difference", {"group": "segment", "target": "sales"}, None, importance=99)]
    result, _ = _run(profile, df, hyp)
    assert result["insights"][0]["priority"] == 5


def test_grouped_time_pattern_is_not_verified_by_ungrouped_eta(profile, df) -> None:
    hyp = [_h("growth trends up differently per segment.", "time_pattern", {"time": "ts", "target": "growth", "group": "segment"}, None)]
    result, _ = _run(profile, df, hyp)
    assert result["insights"] == []
    (dropped,) = result["debug"]["dropped"]
    # either the layer-1 time x group interaction answers it (and says no) or
    # nothing can test it; the ungrouped time eta-squared is never used
    assert "time x group interaction" in dropped["reason"] or "grouped time pattern" in dropped["reason"]
    assert result["debug"]["probes"] == []


# --- noise benchmark: zero insights ---------------------------------------------------------


def _noise_frame() -> pl.DataFrame:
    path = DATASET_DIR / "noise.csv"
    if not path.exists():
        subprocess.run([sys.executable, str(DATASET_DIR / "syn" / "noise.py")], check=True, capture_output=True)
    return load_dataframe(path.read_bytes(), "noise.csv")


def test_noise_dataset_yields_zero_insights() -> None:
    df = _noise_frame()
    profile = profile_dataset(df, "4" * 32, BIG)
    hypotheses = [
        _h("n0 differs by segment.", "group_difference", {"group": "segment", "target": "n0"}, None, 5),
        _h("n1's distribution differs by segment.", "distribution_difference", {"group": "segment", "target": "n1"}, None, 5),
        _h("n2 and n3 relate differently per segment.", "slope_difference", {"x": "n2", "y": "n3", "group": "segment"}, None, 4),
        _h("n4 depends on n5 in a curve.", "nonlinear_relationship", {"x": "n4", "y": "n5"}, None, 4),
        _h("n6 changes over time.", "time_pattern", {"time": "ts", "target": "n6"}, None, 3),
    ]
    result, provider = _run(profile, df, hypotheses)
    assert result["insights"] == [] and result["message"] == NO_PATTERNS_MESSAGE
    assert provider.final_calls == 0
    more = [
        _h("segment and region interact on n7.", "interaction", {"factor1": "segment", "factor2": "region", "target": "n7"}, None, 3),
        _h("u0 and u1 move together in every flag group.", "grouped_relationship", {"x": "u0", "y": "u1", "group": "flag"}, None, 3),
        _h("u1's distribution differs by region.", "distribution_difference", {"group": "region", "target": "u1"}, None, 3),
        _h("n0 vs n1 differs by flag.", "slope_difference", {"x": "n0", "y": "n1", "group": "flag"}, None, 2),
    ]
    result, provider = _run(profile, df, more)
    assert result["insights"] == []
    reasons = " | ".join(d["reason"] for d in result["debug"]["dropped"])
    assert "fail" in reasons or "without" in reasons
    # the selection effect: layer 2 lists the largest correlation spreads
    # over many triples; proposing exactly those must still yield nothing
    picks = profile.evidence.layer2.conditional_relationships
    assert picks, "the noise table has conditional entries to tempt the LLM with"
    for i in range(0, len(picks), 5):
        batch = [
            _h(f"{c.x} vs {c.y} differs by {c.group}", "slope_difference", {"x": c.x, "y": c.y, "group": c.group}, None, 5)
            for c in picks[i : i + 5]
        ]
        result, _ = _run(profile, df, batch)
        assert result["insights"] == [], [d["reason"] for d in result["debug"]["dropped"]]
        assert all("without a real relationship" in d["reason"] or "fail" in d["reason"] for d in result["debug"]["dropped"])


# --- stage 17.3 round 3 -------------------------------------------------------------------


def test_statement_bare_numbers_fall_back_to_neutral_wording(profile, df) -> None:
    hyp = _h("Segment e sells about 60% more (η² = 0.99), 2.5 times the rest.", "group_difference", {"group": "segment", "target": "sales"}, None, 5)
    result, _ = _run(profile, df, [hyp])
    (insight,) = result["insights"]
    assert "60%" not in insight["text"] and "2.5" not in insight["text"] and "0.99" not in insight["text"]
    assert insight["text"].startswith("Sales varies by segment (adjusted eta-squared")
    (record,) = result["debug"]["wording"]
    assert record["source"] == "neutral"
    assert any("statement 1: number" in e for e in result["debug"]["errors"])


def test_statement_with_backend_number_keeps_template(profile, df) -> None:
    hyp = _h("Segment e sells far more across 600 rows.", "group_difference", {"group": "segment", "target": "sales"}, None, 5)
    result, _ = _run(profile, df, [hyp])
    (insight,) = result["insights"]
    assert insight["text"].startswith("Segment e sells far more across 600 rows (")
    assert result["debug"]["wording"][0]["source"] == "template"


def test_nonlinear_coverage_is_directed_and_matches_the_probe(profile, df) -> None:
    from app.probes import ProbeRequest, run_probes

    stored = next(s for s in profile.evidence.layer2.nonlinear if {s.x, s.y} == {"x", "u"})
    forward = {"x": stored.x, "y": stored.y}
    reverse = {"x": stored.y, "y": stored.x}
    cov, _ = _run(profile, df, [_h("forward", "nonlinear_relationship", forward, None)])
    assert cov["debug"]["probes"] == [] and cov["debug"]["covered_by_existing_evidence"]
    (probe_fwd,) = run_probes(df, profile, [ProbeRequest(type="nonlinear_relationship", columns=forward)])
    assert cov["debug"]["covered_by_existing_evidence"][0]["verdict"] == probe_fwd.verdict
    assert cov["insights"][0]["effect"]["value"] == pytest.approx(probe_fwd.effect_size, abs=1e-6)
    rev, _ = _run(profile, df, [_h("reverse", "nonlinear_relationship", reverse, None)])
    assert rev["debug"]["covered_by_existing_evidence"] == []
    (probe_rev,) = run_probes(df, profile, [ProbeRequest(type="nonlinear_relationship", columns=reverse)])
    (ran,) = rev["debug"]["probes"]
    assert ran["verdict"] == probe_rev.verdict


def test_llm2_may_quote_the_change_point_date(profile, df) -> None:
    change = next((c for c in profile.evidence.layer2.change_points if c.num == "growth" and c.strength != "none"), None)
    if change is None:
        pytest.skip("no graded change point on growth in this fixture")
    day = change.change_at[:10]
    hyps = _two_covered() + [_h("growth shifts over time.", "time_pattern", {"time": "ts", "target": "growth"}, None, 3)]
    final = {"insights": [{"hypothesis_id": 3, "text": f"Growth shifts level around {day}.", "why_it_matters": "?", "priority": 3}]}
    result, _ = _run(profile, df, hyps, final=final)
    assert any(i["text"] == f"Growth shifts level around {day}." for i in result["insights"])
    wrong = {"insights": [{"hypothesis_id": 3, "text": "Growth shifts level around 2025-04-19.", "why_it_matters": "?", "priority": 3}]}
    result, _ = _run(profile, df, hyps, final=wrong)
    assert not any("2025-04-19" in i["text"] for i in result["insights"])


def test_duplicate_claims_are_collapsed_after_the_gate(profile, df) -> None:
    hyps = [
        _h("Segment e sells far more.", "group_difference", {"group": "segment", "target": "sales"}, None, 5),
        _h("Sales differ across segments.", None, None, _bar("segment", "sales"), 4),
    ]
    result, _ = _run(profile, df, hyps)
    assert len(result["insights"]) == 1
    (dropped,) = result["debug"]["dropped"]
    assert "duplicate of hypothesis 1" in dropped["reason"]


def test_grouped_bar_implies_an_interaction_probe(profile, df) -> None:
    chart = {"title": "sales by segment and flag", "type": "bar", "x": "segment", "y": "sales", "group_by": "flag", "aggregation": "mean", "priority": 1}
    result, _ = _run(profile, df, [_h("flag changes how segment affects sales.", None, None, chart)])
    # the claim is judged as an interaction (here: layer 1 has segment x flag
    # -> sales at ~0 and says no), never as a plain group difference on x
    assert result["insights"] == []
    (dropped,) = result["debug"]["dropped"]
    assert "interaction share of variance" in dropped["reason"]


# --- round 4: LLM #1 reason and full-date literals go through the same gate ---------------


def test_llm1_reason_is_gated_before_it_captions_anything(profile, df) -> None:
    # LLM #2 not called (single covered finding) -> the hypothesis' own reason is the fallback
    bad = _h("Segment e sells far more.", "group_difference", {"group": "segment", "target": "sales"}, None, 5)
    bad["reason"] = "Segment e drove 1,284 orders in Q3 2019, a 37% lift; churn_rate fell to 4.2%."
    result, provider = _run(profile, df, [bad])
    assert provider.final_calls == 0
    (insight,) = result["insights"]
    assert insight["why_it_matters"] is None
    assert any(e.startswith("reason 1:") for e in result["debug"]["errors"])
    for chart in result["charts"]:
        assert "1,284" not in (chart["spec"]["reason"] or "") and "churn_rate" not in (chart["spec"]["reason"] or "")
    # LLM #2 called but silent on one hypothesis -> same fallback, same gate
    hyps = _two_covered()
    hyps[1]["reason"] = "Because revenue_total jumped 1,284% in 2019."
    final = {"insights": [{"hypothesis_id": 1, "text": "Segment e averages far higher sales.", "why_it_matters": "pricing", "priority": 5}]}
    result, provider = _run(profile, df, hyps, final=final)
    assert provider.final_calls == 1
    by_text = {i["text"][:20]: i for i in result["insights"]}
    assert by_text["Segment e averages f"]["why_it_matters"] == "pricing"
    assert by_text["u is U-shaped in x ("]["why_it_matters"] is None
    # a clean reason still passes through
    good = _h("Segment e sells far more.", "group_difference", {"group": "segment", "target": "sales"}, None, 5)
    good["reason"] = "segment e behaves like a separate market"
    result, _ = _run(profile, df, [good])
    assert result["insights"][0]["why_it_matters"] == "segment e behaves like a separate market"


def test_full_iso_date_must_match_the_change_point_literally(profile, df) -> None:
    change = next((c for c in profile.evidence.layer2.change_points if c.num == "growth" and c.strength != "none"), None)
    if change is None:
        pytest.skip("no graded change point on growth in this fixture")
    day = change.change_at[:10]
    year, month, dom = day.split("-")
    if int(dom) <= 12 and dom != month:
        recombined = f"{year}-{dom}-{month}"  # month and day swapped
    elif month != dom:
        recombined = f"{year}-{month}-{month}"  # day borrowed from the month
    else:
        pytest.skip("change point date has no distinct recombination")
    hyps = _two_covered() + [_h("growth shifts over time.", "time_pattern", {"time": "ts", "target": "growth"}, None, 3)]
    final = {"insights": [{"hypothesis_id": 3, "text": f"Growth shifts level around {recombined}.", "why_it_matters": "?", "priority": 3}]}
    result, _ = _run(profile, df, hyps, final=final)
    assert not any(recombined in i["text"] for i in result["insights"])
    assert any(f"date {recombined} is not in the validated evidence" in e for e in result["debug"]["errors"])
    # partial dates still lean on the parts: the month and year alone are quotable
    partial = {"insights": [{"hypothesis_id": 3, "text": f"Growth shifts level in month {int(month)} of {year}.", "why_it_matters": "?", "priority": 3}]}
    result, _ = _run(profile, df, hyps, final=partial)
    assert any(i["text"] == f"Growth shifts level in month {int(month)} of {year}." for i in result["insights"])


def test_new_llm_chart_caption_is_the_gated_reason(profile, df, monkeypatch) -> None:
    # no backend chart -> the LLM's own chart is the only candidate; its raw
    # reason carries fabricated numbers and must not become the caption
    monkeypatch.setattr("app.llm.service.suggest_chart", lambda *a, **k: None)
    chart = _scatter("x", "u")
    chart["reason"] = "u drops 1,284 units at x = 4.95 then recovers; churn_rate flat."
    h = _h("u is U-shaped in x.", "nonlinear_relationship", {"x": "x", "y": "u"}, chart, 4)
    h["reason"] = "the curve says 37% of the effect is nonlinear"
    result, _ = _run(profile, df, [h])
    (insight,) = result["insights"]
    llm_charts = [c for c in result["charts"] if c["source"] == "llm"]
    assert llm_charts and all("1,284" not in c["spec"]["reason"] and "37%" not in c["spec"]["reason"] for c in llm_charts)
    assert insight["why_it_matters"] is None
