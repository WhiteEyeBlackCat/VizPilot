"""Audit A1 / A2: the wording of an insight must say what the evidence says.

Every scenario runs on synthetic data with a planted truth; the assertions
check that the final insight states that truth (direction, order, numbers,
scope), not only that nothing crashed.

A1: backend-built claims, the claim template, the semantic gate on LLM #2
    text / headline / why_it_matters, LLM #1 reason and chart captions.
A2: grouped relationships — consistent positive / negative, sign flip,
    near-zero group, too-small group, Simpson-style pooled reversal.
"""

import random
from types import SimpleNamespace

import polars as pl
import pytest

from app.llm.claims import build_claim, render_claim, semantic_problem
from app.llm.prompts import FINAL_SYSTEM_PROMPT, build_final_messages
from app.llm.service import RecommendationService
from app.profiling.profiler import profile_dataset
from tests.test_llm_workflow import BIG, TwoStageFake, _bar, _h, _planted, _two_covered

GD = {"group": "segment", "target": "sales"}
REL = {"x": "z", "y": "w", "group": "region"}


@pytest.fixture(scope="module")
def df():
    return _planted()


@pytest.fixture(scope="module")
def profile(df):
    return profile_dataset(df, "2" * 32, BIG)


@pytest.fixture(scope="module")
def means(df):
    """The planted truth: segment e ~80, the other segments ~50."""
    table = df.group_by("segment").agg(pl.col("sales").mean()).iter_rows()
    return {seg: f"{m:.1f}" for seg, m in table}


def _run(profile, df, hypotheses, final=None):
    provider = TwoStageFake(hypotheses, None, final)
    result = RecommendationService(provider).get(profile, use_llm=True, df=df, include_debug=True)
    return result, provider


def _wording(result, hid=1):
    return next(w for w in result["debug"]["wording"] if w["id"] == hid)


def _final(text, why="pricing", headline=None, hid=1):
    item = {"hypothesis_id": hid, "text": text, "why_it_matters": why, "priority": 5}
    if headline is not None:
        item["headline"] = headline
    return {"insights": [item, {"hypothesis_id": 2, "text": "u follows a U shape in x.", "why_it_matters": "?", "priority": 3}]}


# --- A1: group difference on the planted dataset ------------------------------------


def test_claim_is_built_from_the_evidence_not_the_statement(profile, df) -> None:
    wrong = _h("Segment a has much higher sales than segment e.", "group_difference", GD, _bar("segment", "sales"), 5)
    result, _ = _run(profile, df, [wrong])
    claim = _wording(result)["claim"]
    assert claim["kind"] == "group_difference" and claim["top"] == "e" and claim["bottom"] == "b"
    assert claim["checks"] == {"comparisons": True, "signs": False, "universal": False}


def test_misleading_llm1_statement_is_never_shipped_by_the_template(profile, df, means) -> None:
    # repro case 1: the effect is real (eta² 0.84) but the statement inverts
    # it, adds causation and generalises to a column that does not exist
    statement = "Segment a has much higher sales than segment e; segment membership causes sales in every region."
    result, provider = _run(profile, df, [_h(statement, None, GD, _bar("segment", "sales"), 5)])
    assert provider.final_calls == 0
    (insight,) = result["insights"]
    assert insight["supported"] == "strong"
    text = insight["text"]
    assert text.startswith(f"Mean sales is highest for segment e ({means['e']}) and lowest for segment b ({means['b']})")
    for bad in ("Segment a has", "causes", "region"):
        assert bad not in text
    assert _wording(result)["source"] == "template"
    assert any(e.startswith("statement 1: causal language") for e in result["debug"]["errors"])
    # nor does the statement caption the chart
    assert all("Segment a has much higher" not in (c["spec"]["reason"] or "") for c in result["charts"])


def test_correct_comparison_wording_is_kept(profile, df, means) -> None:
    text = f"Segment e has higher sales than segment a ({means['e']} vs {means['a']})."
    result, _ = _run(profile, df, _two_covered(), final=_final(text, headline="Segment e outsells the rest"))
    insight = next(i for i in result["insights"] if i["priority"] == 5)
    assert insight["text"] == text and insight["headline"] == "Segment e outsells the rest"
    assert _wording(result)["source"] == "llm2"


@pytest.mark.parametrize(
    "text, problem",
    [
        ("Segment a has higher sales than segment e.", "contradicts the mean order"),
        ("Compared with segment e, segment a sells more.", "contradicts the mean order"),
        ("Segment b outsells every other segment.", "not above every other group"),
        ("Sales are highest in segment b.", "the mean top is e"),
    ],
    ids=["than", "compared-with", "every-other", "superlative"],
)
def test_reversed_direction_falls_back_to_the_claim_template(profile, df, means, text, problem) -> None:
    result, _ = _run(profile, df, _two_covered(), final=_final(text))
    insight = next(i for i in result["insights"] if i["priority"] == 5)
    assert insight["text"].startswith(f"Mean sales is highest for segment e ({means['e']})")
    assert _wording(result)["source"] == "template"
    assert any(problem in e for e in result["debug"]["errors"] if e.startswith("llm2 insight 1"))


def test_causal_language_is_rejected(profile, df) -> None:
    result, _ = _run(profile, df, _two_covered(), final=_final("Being in segment e drives sales up."))
    insight = next(i for i in result["insights"] if i["priority"] == 5)
    assert "drives" not in insight["text"] and insight["text"].startswith("Mean sales is highest for segment e")
    assert any("causal language 'drives'" in e for e in result["debug"]["errors"])


def test_correct_numbers_bound_to_the_wrong_groups_are_rejected(profile, df, means) -> None:
    # every number is a real backend number, but each sits next to the other group
    swapped = f"Segment e averages {means['a']} while segment a averages {means['e']}."
    result, _ = _run(profile, df, _two_covered(), final=_final(swapped))
    insight = next(i for i in result["insights"] if i["priority"] == 5)
    assert insight["text"] != swapped
    assert any(f"number {means['a']} belongs to another group, not e" in e for e in result["debug"]["errors"])
    # the same numbers on the right groups pass
    right = f"Segment e averages {means['e']} while segment a averages {means['a']}."
    result, _ = _run(profile, df, _two_covered(), final=_final(right))
    assert any(i["text"] == right for i in result["insights"])


def test_unsupported_subgroup_generalisation_is_rejected(profile, df) -> None:
    result, _ = _run(profile, df, _two_covered(), final=_final("Segment e sells more in every region."))
    insight = next(i for i in result["insights"] if i["priority"] == 5)
    assert "region" not in insight["text"]
    assert any("unsupported generalisation 'every region'" in e for e in result["debug"]["errors"])


def test_headline_and_why_it_matters_are_checked_against_the_claim(profile, df) -> None:
    good = "Segment e averages far higher sales."
    # a reversed headline costs the headline only
    result, _ = _run(profile, df, _two_covered(), final=_final(good, headline="Segment a outsells segment e"))
    insight = next(i for i in result["insights"] if i["priority"] == 5)
    assert insight["text"] == good and insight["headline"] is None
    assert any(e.startswith("llm2 headline 1") for e in result["debug"]["errors"])
    # a causal why_it_matters costs the why only (LLM #1's gated reason is shown instead)
    result, _ = _run(profile, df, _two_covered(), final=_final(good, why="Segment e causes higher revenue."))
    insight = next(i for i in result["insights"] if i["priority"] == 5)
    assert insight["text"] == good and "causes" not in (insight["why_it_matters"] or "")
    assert any(e.startswith("llm2 why 1: causal language") for e in result["debug"]["errors"])


def test_llm1_reason_and_chart_caption_are_checked_against_the_claim(profile, df) -> None:
    chart = _bar("segment", "sales")
    chart["reason"] = "segment a sells more than segment e"
    h = _h("e sells more.", "group_difference", GD, chart, 5)
    h["reason"] = "segment a outperforms segment e, which drives revenue"
    result, provider = _run(profile, df, [h])
    assert provider.final_calls == 0
    (insight,) = result["insights"]
    assert insight["why_it_matters"] is None  # LLM #1 reason contradicts the evidence
    reasons = [c["spec"]["reason"] or "" for c in result["charts"]]
    assert not any("segment a sells more" in r or "outperforms" in r or "e sells more." == r for r in reasons)
    # a reason that agrees with the evidence still captions and explains
    h["reason"] = "segment e outsells every other segment"
    chart["reason"] = "segment e stands out"
    result, _ = _run(profile, df, [h])
    assert result["insights"][0]["why_it_matters"] == "segment e outsells every other segment"


def test_bare_chart_caption_and_title_cannot_assert(profile, df) -> None:
    bare = {"title": "Higher x always means higher u", "type": "scatter", "x": "x", "y": "u", "priority": 2,
            "reason": "x drives u up in every region"}
    result, _ = _run_bare(profile, df, bare)
    assert all("drives" not in (c["spec"]["reason"] or "") for c in result["charts"])
    assert all("always" not in c["spec"]["title"] for c in result["charts"])
    assert any("caption dropped" in e for e in result["debug"]["errors"])
    assert any("neutral title used" in e for e in result["debug"]["errors"])


def test_llm_chart_title_is_checked_against_the_claim(profile, df, monkeypatch) -> None:
    # no backend chart: the LLM's own chart (and its title) is the only candidate
    monkeypatch.setattr("app.llm.service.suggest_chart", lambda *a, **k: None)
    chart = _bar("segment", "sales")
    chart["title"] = "Segment a outsells segment e"
    result, _ = _run(profile, df, [_h("e sells more.", "group_difference", GD, chart, 5)])
    assert result["insights"] and all("outsells" not in c["spec"]["title"] for c in result["charts"])
    assert any(e.startswith("chart title 1") for e in result["debug"]["errors"])


def _run_bare(profile, df, chart):
    provider = TwoStageFake([], [chart], None)
    return RecommendationService(provider).get(profile, use_llm=True, df=df, include_debug=True), provider


def test_templates_pass_their_own_check_and_reach_the_prompt(profile, df) -> None:
    hyps = _two_covered() + [_h("growth shifts over time.", "time_pattern", {"time": "ts", "target": "growth"}, None, 3)]
    result, provider = _run(profile, df, hyps, final={"insights": []})
    names = {c.name for c in profile.columns}
    for v in provider.seen_validated:
        sentence = render_claim(v.claim)
        assert sentence and semantic_problem(sentence, v.claim, names) is None, (v.test_type, sentence)
    growth = next(v for v in provider.seen_validated if v.test_type == "time_pattern")
    if growth.claim.change:  # growth is planted rising: the claim says so
        assert growth.claim.direction == "positive" and render_claim(growth.claim).startswith("Mean growth rises from")
        assert semantic_problem("growth falls over ts.", growth.claim) is not None
        assert semantic_problem("growth rises over ts.", growth.claim) is None
    content = build_final_messages(profile, provider.seen_validated)[1]["content"]
    assert "verified finding: Mean sales is highest for segment e" in content
    assert "No causal language" in FINAL_SYSTEM_PROMPT


# --- A1: label safety and evidence normalisation --------------------------------------


def _v(test_type, columns, evidence):
    return SimpleNamespace(test_type=test_type, columns=columns, evidence=evidence)


def test_numeric_labels_never_misfire() -> None:
    groups = [{"group": "1", "n": 100, "mean": 10.0}, {"group": "2", "n": 100, "mean": 30.0}, {"group": "3", "n": 100, "mean": 20.0}]
    claim = build_claim(_v("group_difference", {"group": "grp", "target": "y"}, {"groups": groups}))
    # bare numbers are not taken for labels: "3 groups", "n=100" do not bind
    assert semantic_problem("The 3 groups differ (n=100 each).", claim) is None
    # "Group" is not the column, so "2" is not found as a label: a superlative
    # bound to nothing cannot be verified and fails closed
    assert "not bound to a group" in semantic_problem("Group 2 has the highest mean y.", claim)
    # with the column name the label is found and checked
    assert semantic_problem("grp 2 has the highest mean y.", claim) is None
    assert "the mean top is 2" in semantic_problem("grp 3 has the highest mean y.", claim)
    assert "contradicts" in semantic_problem("grp 1 has higher y than grp 3.", claim)
    assert "belongs to another group" in semantic_problem("grp 1 averages 30.0.", claim)
    # an ambiguous label ("a") is only matched in context, never as the article
    groups = [{"group": "a", "n": 50, "mean": 1.0}, {"group": "b", "n": 50, "mean": 2.0}]
    claim = build_claim(_v("group_difference", {"group": "seg", "target": "y"}, {"groups": groups}))
    assert semantic_problem("b is a group with a higher mean.", claim) is None
    assert semantic_problem("seg a has a higher mean than seg b.", claim) is not None


def test_coverage_and_probe_evidence_shapes_give_the_same_claim() -> None:
    rows = [{"group": "x", "n": 40, "mean": 5.0, "median": 5.0}, {"group": "y", "n": 40, "mean": 9.0, "median": 8.0}]
    coverage = build_claim(_v("group_difference", {"group": "g", "target": "t"}, {"groups": rows, "eta_squared": 0.3}))
    probe = build_claim(
        _v("group_difference", {"group": "g", "target": "t"}, {"groups": rows, "top_group": "y", "bottom_group": "x"})
    )
    assert coverage.as_dict() == probe.as_dict() and coverage.as_dict()["top"] == "y"
    corr_rows = [{"group": "p", "n": 80, "corr": 0.8, "slope": 1.0}, {"group": "q", "n": 80, "corr": -0.7, "slope": -1.0}]
    as_list = build_claim(_v("slope_difference", REL, {"groups": corr_rows}))
    as_dict = build_claim(_v("slope_difference", REL, {"groups": {"p": 0.8, "q": -0.7}}))
    assert as_list.direction == as_dict.direction == "mixed"
    assert {g.label: g.sign() for g in as_list.groups} == {g.label: g.sign() for g in as_dict.groups}
    # coverage without per-group means: nothing to bind an order to -> restricted
    bare = build_claim(_v("group_difference", {"group": "g", "target": "t"}, {"eta_squared": 0.3}))
    assert not bare.comparisons_ok and render_claim(bare) is None
    assert semantic_problem("t differs by g.", bare) is None
    assert semantic_problem("t is higher in some g.", bare) is not None


def test_unverifiable_claim_types_accept_only_plain_wording() -> None:
    claim = build_claim(_v("interaction", {"factor1": "f", "factor2": "h", "target": "t"}, {"strength": 0.2}))
    assert semantic_problem("How t varies with f differs between levels of h.", claim) is None
    # "effect of" / "depends on" are causal wording when the LLM writes them
    # (the backend's own neutral interaction sentence is not LLM text)
    for text in ("t is highest when f is high.", "t rises with f.", "This holds in every h.", "f drives t.",
                 "The effect of f on t depends on h."):
        assert semantic_problem(text, claim) is not None, text


# --- A2: grouped relationships ------------------------------------------------------------


def _relationship_frame(slopes, n=150, small=None, offsets=None, shift=3.0, seed=5):
    """region -> slope of w on z; `offsets` shift z and w per group (Simpson);
    `small` gives a group fewer rows. A second column `level` differs by
    region so two findings validate (LLM #2 is then called)."""
    rng = random.Random(seed)
    data = {"region": [], "z": [], "w": [], "level": []}
    for k, (g, slope) in enumerate(slopes.items()):
        off = (offsets or {}).get(g, 0.0)
        for _ in range((small or {}).get(g, n)):
            z = rng.uniform(-1, 1)
            data["region"].append(g)
            data["z"].append(z + off)
            data["w"].append(slope * z + shift * off + rng.gauss(0, 0.3))
            data["level"].append(10 * k + rng.gauss(0, 1))
    return pl.DataFrame(data)


def _rel_run(frame, text=None, test="grouped_relationship"):
    profile = profile_dataset(frame, "9" * 32, BIG)
    hyps = [
        _h("w and z move together in each region.", test, REL, None, 5),
        _h("level differs by region.", "group_difference", {"group": "region", "target": "level"}, None, 3),
    ]
    final = {"insights": [{"hypothesis_id": 1, "text": text, "why_it_matters": "?", "priority": 5}]} if text else {"insights": []}
    return _run(profile, frame, hyps, final)


def _rel_insight(result):
    wording = _wording(result)
    insight = next(i for i in result["insights"] if i["text"] == wording["final_text"][:300])
    return insight, wording


@pytest.mark.parametrize("sign", [1, -1], ids=["positive", "negative"])
def test_consistent_relationship(sign) -> None:
    frame = _relationship_frame({"north": 2.0 * sign, "south": 1.5 * sign, "east": 2.5 * sign, "west": 1.8 * sign})
    verb, wrong = ("rises", "falls") if sign > 0 else ("falls", "rises")
    result, _ = _rel_run(frame)
    insight, wording = _rel_insight(result)
    claim = wording["claim"]
    assert claim["kind"] == "grouped_relationship" and claim["direction"] == ("positive" if sign > 0 else "negative")
    assert claim["checks"]["universal"] is True
    assert insight["supported"] == "strong"
    assert insight["text"].startswith(f"w {verb} with z within each of the 4 region groups")
    # the right direction, even universally quantified, is kept
    ok = f"w {verb} with z in every region."
    result, _ = _rel_run(frame, ok)
    assert _rel_insight(result)[0]["text"] == ok
    as_driver = "w falls as z rises." if sign < 0 else "w rises as z rises."
    result, _ = _rel_run(frame, as_driver)
    assert _rel_insight(result)[0]["text"] == as_driver
    # the wrong direction is not
    result, _ = _rel_run(frame, f"w {wrong} with z in every region.")
    insight, wording = _rel_insight(result)
    assert wording["source"] == "template" and insight["text"].startswith(f"w {verb} with z")


@pytest.mark.parametrize("flip", [False, True], ids=["north-south-positive", "north-south-negative"])
def test_sign_flip_is_reported_as_a_direction_difference(flip) -> None:
    s = -1.0 if flip else 1.0
    frame = _relationship_frame({"north": 2.0 * s, "south": 2.0 * s, "east": -2.0 * s, "west": -2.0 * s})
    pos, neg = (("east", "west"), ("north", "south")) if flip else (("north", "south"), ("east", "west"))
    result, _ = _rel_run(frame)
    insight, wording = _rel_insight(result)
    claim = wording["claim"]
    assert claim["kind"] == "sign_flip" and claim["direction"] == "mixed" and claim["checks"]["universal"] is False
    assert insight["text"].startswith("z and w move together in region")
    head, tail = insight["text"].split(" but in opposite directions in ")
    assert all(f"region {g}" in head for g in pos) and all(f"region {g}" in tail for g in neg)
    # "consistent positive" wording is rejected
    result, _ = _rel_run(frame, "w rises with z consistently in every region: a positive relationship everywhere.")
    assert _wording(result)["source"] == "template"
    # one direction without saying where is rejected
    result, _ = _rel_run(frame, "w is positively related to z.")
    assert _wording(result)["source"] == "template"
    # the correct per-group statement is kept; the reversed one is not
    right = f"w rises with z in {pos[0]} and {pos[1]}, but falls with z in {neg[0]} and {neg[1]}."
    result, _ = _rel_run(frame, right)
    assert _rel_insight(result)[0]["text"] == right
    reversed_ = f"w falls with z in {pos[0]} and {pos[1]}, but rises with z in {neg[0]} and {neg[1]}."
    result, _ = _rel_run(frame, reversed_)
    assert _wording(result)["source"] == "template"


def test_planted_sign_flip_from_the_audit_repro(profile, df) -> None:
    hyp = _h("w rises with z consistently in every segment: a positive relationship everywhere.", "grouped_relationship",
             {"x": "z", "y": "w", "group": "segment"}, None, 5)
    result, _ = _run(profile, df, [hyp])
    (insight,) = result["insights"]
    assert "consistently" not in insight["text"] and "everywhere" not in insight["text"]
    head, tail = insight["text"].split(" but in opposite directions in ")
    assert "segment a" in head and "segment b" in head
    assert all(f"segment {g}" in tail for g in "cde")


def test_near_zero_group_breaks_every_group_and_has_no_direction() -> None:
    frame = _relationship_frame({"north": 2.0, "south": 2.0, "east": 2.0, "west": 0.0}, small={"west": 800})
    # "the relationship holds in every region" fails: west has none
    result, _ = _rel_run(frame)
    assert all(w["id"] != 1 for w in result["debug"]["wording"])
    assert any(d["id"] == 1 and "fail" in d["reason"] for d in result["debug"]["dropped"])
    # as a slope difference it validates; west counts for no direction
    result, _ = _rel_run(frame, test="slope_difference")
    insight, wording = _rel_insight(result)
    assert wording["claim"]["direction"] == "positive" and wording["claim"]["signs"]["west"] == 0
    assert "no clear relationship in region west" in insight["text"]
    for text in ("w rises with z in west.", "w rises with z in every region."):
        result, _ = _rel_run(frame, text, test="slope_difference")
        assert _wording(result)["source"] == "template", text


def test_too_small_group_is_excluded_explicitly_and_caps_the_verdict() -> None:
    frame = _relationship_frame({"north": 2.0, "south": 2.0, "east": 2.0, "west": 2.0}, small={"west": 20})
    result, _ = _rel_run(frame)
    insight, wording = _rel_insight(result)
    claim = wording["claim"]
    assert claim["unmeasured"] == ["west"] and claim["checks"]["universal"] is False
    assert insight["supported"] == "weak"  # "every group" cannot be strong with an unestimated group
    assert "region west had too few rows to estimate" in insight["text"]
    assert "each of the" not in insight["text"]
    result, _ = _rel_run(frame, "w rises with z in every region.")
    assert _wording(result)["source"] == "template"
    result, _ = _rel_run(frame, "w rises with z in north, south and east.")
    assert _wording(result)["source"] == "llm2"


def test_simpson_pooled_direction_is_never_claimed_from_group_evidence() -> None:
    # inside every region w falls with z; the regions sit on a rising line
    frame = _relationship_frame(
        {"north": -1.5, "south": -1.5, "east": -1.5, "west": -1.5},
        offsets={"north": 0.0, "south": 0.7, "east": 1.4, "west": 2.1},
        shift=6.0,
    )
    assert frame.select(pl.corr("z", "w")).item() > 0.5
    result, _ = _rel_run(frame)
    insight, wording = _rel_insight(result)
    claim = wording["claim"]
    assert claim["direction"] == "negative" and claim["simpson"] is True
    assert insight["text"].startswith("w falls with z within each of the 4 region groups")
    assert "pooled over all rows the correlation is" in insight["text"] and "opposite direction" in insight["text"]
    for text in ("Overall, w rises with z.", "Pooled over all rows, w falls with z.", "w rises with z."):
        result, _ = _rel_run(frame, text)
        assert _wording(result)["source"] == "template", text
    result, _ = _rel_run(frame, "Within each region w falls with z.")
    assert _wording(result)["source"] == "llm2"
