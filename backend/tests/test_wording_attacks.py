"""Adversarial regression set for the LLM wording gate (independent review of
audit A1/A2). Every attack string is wrong about planted data and must never
ship; every correct string should keep the LLM's wording.

Planted truth (group_difference, region -> revenue):
South 200 > North East ~150 > West ~120 > North ~100.
"""

import random
import threading
import time as _time
from datetime import datetime, timedelta

import polars as pl
import pytest

from app.llm.provider import LLMError
from app.llm.service import RecommendationService
from app.profiling.profiler import profile_dataset
from tests.test_llm_workflow import TwoStageFake, _bar, _h
from tests.test_semantic_claims import _rel_run, _relationship_frame, _wording

GD = {"group": "region", "target": "revenue"}
GD2 = {"group": "channel", "target": "units"}


def _frame() -> pl.DataFrame:
    rng = random.Random(3)
    rows = {"region": [], "revenue": [], "units": [], "channel": []}
    for i in range(600):
        r = ["North", "South", "North East", "West"][i % 4]
        mu = {"North": 100, "South": 200, "North East": 150, "West": 120}[r]
        rows["region"].append(r)
        rows["revenue"].append(rng.gauss(mu, 15))
        ch = rng.choice(["web", "store"])
        rows["channel"].append(ch)
        rows["units"].append(rng.gauss(30 if ch == "web" else 20, 3))
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def data():
    df = _frame()
    return df, profile_dataset(df, "7" * 32, 10**6)


def _run(data, text="South has the highest revenue.", why="matters", headline=None, reason="matters", chart=None, final=True, hyps=None):
    df, prof = data
    h = _h("Revenue differs by region.", "group_difference", GD, chart or _bar("region", "revenue"), 5)
    h["reason"] = reason
    hs = hyps or [h, _h("units differ by channel.", "group_difference", GD2, None, 3)]
    item = {"hypothesis_id": 1, "text": text, "why_it_matters": why, "priority": 5}
    if headline is not None:
        item["headline"] = headline
    provider = TwoStageFake(hs, None, {"insights": [item]} if final else None)
    result = RecommendationService(provider).get(prof, True, df=df, include_debug=True)
    w = next((x for x in result["debug"]["wording"] if x["id"] == 1), None)
    ins = next((i for i in result["insights"] if w and i["text"] == w["final_text"][:300]), None)
    return result, w, ins


ATTACKS = [
    ("text", "North tops South in revenue."),
    ("text", "North edges out South on revenue."),
    ("text", "North beats South on revenue."),
    ("text", "South trails North in revenue."),
    ("text", "South lags behind North."),
    ("text", "South falls short of North in revenue."),
    ("text", "North sells more than double what South sells."),
    ("text", "North sells twice as much as South."),
    ("text", "North dominates revenue."),
    ("text", "North is the leader in revenue."),
    ("text", "North leads all regions in revenue."),
    ("text", "North comes out on top for revenue."),
    ("text", "South's revenue is lower than North's."),
    ("text", "Revenue in South is not higher than in North."),
    ("text", "South is not lower than North East... wait North is not lower than South."),
    ("text", "North is no lower than South in revenue."),
    ("text", "North ranks first in revenue; South ranks last."),
    ("text", "North has the top revenue."),
    ("text", "Revenue peaks in North and bottoms out in South."),
    ("text", "North East is the lowest-revenue region."),
    ("text", "Revenue is lowest in North East."),
    ("text", "North East has the highest revenue."),
    ("text", "Being in South is responsible for higher revenue."),
    ("text", "Region explains revenue."),
    ("text", "Higher revenue in South is owing to its location."),
    ("text", "South's higher revenue comes from its region."),
    ("text", "Region shapes revenue."),
    ("text", "South's location increases revenue."),
    ("text", "South generates higher revenue as a consequence of region."),
    ("text", "The South effect on revenue is large."),
    ("text", "Region makes revenue higher in South."),
    ("text", "South has higher revenue, universally."),
    ("text", "South has higher revenue no matter which channel."),
    ("text", "South has higher revenue in every channel."),
    ("text", "South has higher revenue across all channels."),
    ("text", "South has higher revenue for all customers."),
    ("text", "South earns more than North in all months."),
    ("text", "NORTH has HIGHER revenue than SOUTH."),
    ("text", "“North” has higher revenue than “South”."),
    ("text", "Ｎorth has higher revenue than South."),
    ("text", "North has higher revenue than South."),
    ("text", "North averages 200.0 and South averages 99.1."),
    ("text", "North averages 200.0, South 99.1."),
    ("text", "North (200.0) and South (99.1)."),
    ("text", "The mean for North is 200.0."),
    ("text", "South: 99.1; North: 200.0."),
    ("text", "Revenue is 200.0 in North East and 99.1 in South."),
    ("text", "南區營收高於北區。"),
    ("text", "北區的營收高於南區。"),
    ("text", "北區營收最高。"),
    ("text", "地區導致營收差異。"),
    ("text", "因為地區不同，造成營收不同。"),
    ("text", "所有通路中北區都較高。"),
    ("headline", "North outsells South"),
    ("headline", "North tops the revenue chart"),
    ("headline", "Region drives revenue"),
    ("why", "Because North sells more than South, invest there."),
    ("why", "North has higher revenue than South."),
    ("why", "North's higher revenue means it should get budget."),
    ("why", "Region boosts revenue, so reorganise."),
    ("text", "Revenue is higher in North relative to South."),
    ("text", "Revenue is higher in North, compared with South."),
    ("text", "Compared with North, South sells less."),
    ("text", "Revenue in South exceeds none of the others."),
    ("text", "North East is above South."),
    ("text", "South is below North East in revenue."),
    ("text", "North is above-average in revenue."),
    ("text", "North outperforms."),
    ("text", "North sells most."),
    ("text", "North is the strongest region by revenue."),
    ("text", "Revenue in North is greatest among regions."),
    ("text", "Revenue rises from South to North."),
    ("text", "North has high revenue while South has low revenue."),
    ("text", "North has more revenue. South has less."),
    # review additions: unknown entities, symbols, derived names
    ("text", "Northeast has the highest revenue."),
    ("text", "The southern region trails the northern one."),
    ("text", "North > South in revenue."),
    ("text", "Revenue: North ≥ South."),
    ("text", "North = top region."),
]


@pytest.mark.parametrize("field, text", ATTACKS, ids=[f"{i:02d}" for i in range(len(ATTACKS))])
def test_wrong_group_wording_never_ships(data, field, text) -> None:
    kw = {"text": "South has the highest revenue."}
    kw[{"text": "text", "headline": "headline", "why": "why"}[field]] = text
    _, w, ins = _run(data, **kw)
    assert ins is not None  # the finding itself survives, worded by the template
    shipped = {"text": ins["text"], "headline": ins.get("headline"), "why": ins.get("why_it_matters")}[field]
    assert shipped != text


CORRECT = [
    "South has the highest revenue.",
    "South has higher revenue than North.",
    "North's revenue is lower than South's.",
    "South outsells North.",
    "North trails South in revenue.",
    "South leads, while North has the lowest revenue.",
    "South has the highest mean revenue (200.0) and North the lowest (99.1).",
    "Revenue is highest in South (200.0) and lowest in North (99.1).",
    "South averages 200.0 versus 99.1 in North.",
    "South averages 200.0 compared with 99.1 for North.",
    "Mean revenue in South (200.0) is roughly double that of North (99.1).",
    "Revenue differs strongly by region, with South on top.",
    "North East sits between South and West.",
    "South sells more than every other region.",
    "South sells more than any other region.",
    "North sells less than every other region.",
    "Revenue is higher in South than in North and West.",
    "North and West have lower revenue than South.",
    "South is the top region, North the bottom.",
    "South leads on revenue, and North East is second.",
    "Revenue is not the same across regions; South is highest.",
    "South has the highest revenue, followed by North East, West and North.",
    "South and North East have higher revenue than West and North.",
    "The gap between South and North is large.",
    "South earns more; North earns the least.",
    "In South, revenue is higher than in any other region.",
    "Region is associated with revenue: South is highest.",
    "North East revenue (149.5) is above West (118.3).",
    "West (118.3) is above North (99.1) but below North East (149.5).",
]
# not kept by design: "South's revenue exceeds North's by about 100." — 100 is
# a difference the backend never measured, so the number gate rejects it


@pytest.fixture(scope="module")
def planted_means(data):
    df, _ = data
    return {r: f"{m:.1f}" for r, m in df.group_by("region").agg(pl.col("revenue").mean()).iter_rows()}


@pytest.mark.parametrize("text", CORRECT, ids=[f"{i:02d}" for i in range(len(CORRECT))])
def test_correct_group_wording_is_kept(data, planted_means, text) -> None:
    # the strings quote the reviewer's rounded means; re-quote the fixture's own
    for label, quoted in {"South": "200.0", "North": "99.1", "North East": "149.5", "West": "118.3"}.items():
        text = text.replace(f"({quoted})", f"({planted_means[label]})").replace(f" {quoted} ", f" {planted_means[label]} ")
    result, w, ins = _run(data, text=text)
    assert ins is not None and ins["text"] == text, [e for e in result["debug"]["errors"] if "insight 1" in e]


def test_llm1_reason_caption_and_title_attacks(data, monkeypatch) -> None:
    for t in ["North tops the regions on revenue.", "North dominates revenue.", "Region explains revenue.",
              "北區營收最高，因為地區導致差異。", "North has higher revenue than South."]:
        h = _h("Revenue differs by region.", "group_difference", GD, _bar("region", "revenue"), 5)
        h["reason"] = t
        result, _, ins = _run(data, hyps=[h], final=False)
        assert ins is not None and ins.get("why_it_matters") != t
        assert all(c["spec"]["reason"] != t for c in result["charts"])
    for t in ["North tops South in revenue.", "North dominates revenue.", "区域决定营收", "北區營收高於南區"]:
        c = _bar("region", "revenue")
        c["reason"] = t
        h = _h("Revenue differs by region.", "group_difference", GD, c, 5)
        h["reason"] = "ok"
        result, _, _ = _run(data, hyps=[h], final=False)
        assert all(x["spec"]["reason"] != t for x in result["charts"])
    monkeypatch.setattr("app.llm.service.suggest_chart", lambda *a, **k: None)
    for t in ["North tops the revenue ranking", "North leads revenue", "北區營收最高"]:
        c = _bar("region", "revenue")
        c["title"] = t
        result, _, _ = _run(data, hyps=[_h("Revenue differs by region.", "group_difference", GD, c, 5)], final=False)
        assert all(x["spec"]["title"] != t for x in result["charts"])


def test_bare_chart_attacks(data) -> None:
    df, prof = data
    for title, reason in [("North dominates revenue", "North tops the chart"), ("Region explains revenue", "区域导致营收差异"),
                          ("Revenue peaks in North", "x")]:
        bare = {"title": title, "type": "bar", "x": "region", "y": "revenue", "aggregation": "mean", "priority": 2, "reason": reason}
        result = RecommendationService(TwoStageFake([], [bare], None)).get(prof, True, df=df, include_debug=True)
        assert all(c["spec"]["title"] != title for c in result["charts"])
        assert all(c["spec"]["reason"] != reason or reason == "x" for c in result["charts"])


def test_tied_groups_are_not_called_highest_and_lowest() -> None:
    from types import SimpleNamespace

    from app.llm.claims import build_claim, render_claim

    rows = [{"group": "p", "n": 40, "mean": 5.0}, {"group": "q", "n": 40, "mean": 5.0}]
    claim = build_claim(SimpleNamespace(test_type="group_difference", columns={"group": "g", "target": "t"}, evidence={"groups": rows}))
    assert render_claim(claim) == "Mean t is equal across the 2 g groups (5)"


# --- relationships --------------------------------------------------------------------------


def _shipped(frame, text, test="grouped_relationship") -> bool:
    result, _ = _rel_run(frame, text, test)
    return _wording(result)["source"] == "llm2"


FLIP_WRONG = [
    "w and z are linked the same way in all four regions.",
    "w tracks z in every region.",
    "w goes hand in hand with z.",
    "Higher z means higher w.",
    "As z grows, w grows too.",
    "w and z are positively correlated.",
    "z and w share a strong link that holds across regions.",
    "w increases with z in north and east.",
    "w climbs with z in east.",
    "The relationship between z and w is the same in each region.",
    "w and z correlate in a similar way in every region.",
    "w and z are strongly related, irrespective of region.",
    "In all regions, w rises with z.",
    "w rises with z for north, south, east and west.",
    "z 與 w 在各地區都呈正相關。",
    "w and z move together across regions.",
    "w and z are related universally.",
]
POS_WRONG = [
    "w declines as z goes up.",
    "Higher z means lower w.",
    "w moves opposite to z.",
    "w and z are anti-correlated.",
    "w and z are inversely related.",
    "w decreases with z in north.",
    "z raises w.",
    "z pushes w up.",
    "w depends on z.",
    "z 越高 w 越低。",
    "w is driven by z in every region.",
    "w rises with z in every region and every channel.",
    "w rises with z in all countries.",
    "z > w in every region.",
]
POS_OK = [
    "w rises with z in every region.",
    "w increases with z within each region.",
    "Higher z goes with higher w in each region.",
    "w and z are positively correlated in all four regions.",
    "Across regions, w tends to be higher when z is higher.",
    "The positive link between z and w is strongest in east.",
]
FLIP_OK = [
    "w rises with z in north and south but falls with z in east and west.",
    "The direction of the z-w relationship differs by region.",
    "In north and south w increases with z, whereas in east and west it decreases.",
]


@pytest.fixture(scope="module")
def flip():
    return _relationship_frame({"north": 2.0, "south": 2.0, "east": -2.0, "west": -2.0})


@pytest.fixture(scope="module")
def pos():
    return _relationship_frame({"north": 2.0, "south": 1.5, "east": 2.5, "west": 1.8})


@pytest.mark.parametrize("text", FLIP_WRONG)
def test_sign_flip_attacks(flip, text) -> None:
    assert not _shipped(flip, text)


@pytest.mark.parametrize("text", POS_WRONG)
def test_consistent_positive_attacks(pos, text) -> None:
    assert not _shipped(pos, text)


@pytest.mark.parametrize("text", POS_OK)
def test_consistent_positive_correct_wording(pos, text) -> None:
    assert _shipped(pos, text)


@pytest.mark.parametrize("text", FLIP_OK)
def test_sign_flip_correct_wording(flip, text) -> None:
    assert _shipped(flip, text)


# --- time pattern and a U-shaped curve ------------------------------------------------------


@pytest.fixture(scope="module")
def time_data():
    rng = random.Random(1)
    n = 600
    df = pl.DataFrame(
        {
            "ts": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n)],
            "load": [rng.gauss(100 if i < 300 else 50, 5) for i in range(n)],
            "x": [rng.uniform(0, 10) for _ in range(n)],
        }
    )
    df = df.with_columns(u=((pl.col("x") - 5) ** 2) + pl.Series([rng.gauss(0, 1) for _ in range(n)]))
    return df, profile_dataset(df, "8" * 32, 10**6)


def _time_source(time_data, hid, text) -> str:
    df, prof = time_data
    hyps = [_h("load shifts over time.", "time_pattern", {"time": "ts", "target": "load"}, None, 5),
            _h("u is U-shaped in x.", "nonlinear_relationship", {"x": "x", "y": "u"}, None, 4)]
    provider = TwoStageFake(hyps, None, {"insights": [{"hypothesis_id": hid, "text": text, "why_it_matters": "?", "priority": 5}]})
    result = RecommendationService(provider).get(prof, True, df=df, include_debug=True)
    return next(x for x in result["debug"]["wording"] if x["id"] == hid)["source"]


@pytest.mark.parametrize("text", [
    "load rises after mid-2023.", "load increased over time.", "load went up in the second half.",
    "load doubled after the change.", "load is higher after the change than before.", "load is lower before the change.",
    "load surged after the change.", "load jumped after the change.", "load spiked later.", "load picked up later.",
    "load soared after the change.", "load climbed after the change.",
])
def test_time_attacks(time_data, text) -> None:
    assert _time_source(time_data, 1, text) != "llm2"


@pytest.mark.parametrize("text", [
    "load halved after the change.", "load dropped after the change.", "load is lower in the later period.",
    "load trends downward over time.", "load was higher earlier.",
])
def test_time_correct_wording(time_data, text) -> None:
    assert _time_source(time_data, 1, text) == "llm2"


@pytest.mark.parametrize("text", [
    "u rises with x.", "u increases steadily with x.", "u peaks at mid-range x.", "u is lowest at mid x and higher at both ends.",
    "higher x means higher u.", "u falls as x rises.", "u bottoms out in the middle of x.", "u is minimised at moderate x.",
    "u keeps climbing with x.", "u is a monotone function of x.", "u and x are positively related.",
])
def test_u_shape_accepts_only_its_shape_name(time_data, text) -> None:
    assert _time_source(time_data, 2, text) != "llm2"


def test_u_shape_name_is_accepted(time_data) -> None:
    assert _time_source(time_data, 2, "u is U-shaped in x.") == "llm2"


# --- A6: concurrent refreshes coalesce ---------------------------------------------------------


def test_concurrent_refreshes_build_once(data) -> None:
    df, prof = data

    class Slow(TwoStageFake):
        def generate_hypotheses(self, profile, rules):
            _time.sleep(0.2)
            return super().generate_hypotheses(profile, rules)

    provider = Slow([_h("Revenue differs by region.", "group_difference", GD, None, 5)], None, None)
    service = RecommendationService(provider)
    service.get(prof, True, df=df)
    provider.calls = 0
    threads = [threading.Thread(target=lambda: service.get(prof, True, df=df, refresh=True)) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert provider.calls == 1


def test_failure_then_concurrent_refresh(data) -> None:
    df, prof = data

    class Flaky(TwoStageFake):
        fail = True

        def generate_hypotheses(self, profile, rules):
            if self.fail:
                self.calls += 1
                raise LLMError("timeout", "t")
            return super().generate_hypotheses(profile, rules)

    provider = Flaky([_h("Revenue differs by region.", "group_difference", GD, None, 5)], None, None)
    service = RecommendationService(provider)
    assert "timeout" in service.get(prof, True, df=df)["message"]
    provider.fail = False
    assert service.get(prof, True, df=df, refresh=True)["insights"]
