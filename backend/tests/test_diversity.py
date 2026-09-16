"""Stage 19.1: recommendation diversity — diminishing returns per analytical
question (intent, x, group_by) and per raw chart type beyond two free slots,
on synthetic candidates and on the bike-sharing hours (hour_like.csv)."""

from collections import Counter

from app.charts.rules import (
    MAX_CHARTS,
    MAX_PER_TYPE,
    MAX_PER_X,
    MAX_PER_Y,
    QUESTION_DECAY,
    TYPE_DECAY,
    TYPE_FREE_SLOTS,
    Recommendation,
    apply_diversity_caps,
    recommend_charts,
)
from app.charts.spec import ChartSpec
from app.profiling.profiler import profile_dataset
from tests.test_recommendation_quality import _real_dataset

BIG = 10**6


def _rec(type_: str, intent: str, x: str | None, y: str | None, score: float, group: str | None = None) -> Recommendation:
    agg = "mean" if type_ in ("bar", "line") else None
    spec = ChartSpec(title=f"{type_} {x} {y}", type=type_, x=x, y=y, group_by=group, aggregation=agg, reason="r")
    return Recommendation(spec=spec, score=score, intent=intent)


def _pool() -> list[Recommendation]:
    """Six trend lines over one date column, seven cyclic lines over two
    cycles, five bars over one category, three scatters and four histograms:
    plenty of strong lines that all answer two or three questions."""
    recs = []
    recs += [_rec("line", "trend", "ts", f"m{i}", 0.90 - 0.01 * i) for i in range(6)]
    recs += [_rec("line", "cyclic_pattern", "hr", f"m{i}", 0.84 - 0.01 * i) for i in range(4)]
    recs += [_rec("line", "cyclic_pattern", "mnth", f"m{i}", 0.80 - 0.01 * i) for i in range(3)]
    recs += [_rec("bar", "group_comparison", "region", f"r{i}", 0.70 - 0.01 * i) for i in range(5)]
    recs += [_rec("scatter", "numeric_relationship", f"a{i}", f"b{i}", 0.65 - 0.01 * i) for i in range(3)]
    recs += [_rec("histogram", "distribution", f"m{i}", None, 0.60 - 0.01 * i) for i in range(4)]
    return recs


def _types(recs: list[Recommendation]) -> Counter:
    return Counter(r.spec.type for r in recs)


def test_same_question_answers_are_discounted_not_stacked() -> None:
    kept = apply_diversity_caps(_pool())
    assert len(kept) == MAX_CHARTS
    # the old caps alone (family 3, per-x 2) let six lines through; now the
    # third answer to "trend over ts" competes at 0.8² against a bar / scatter
    assert _types(kept)["line"] <= 4
    assert _types(kept)["bar"] >= 1 and _types(kept)["scatter"] >= 1 and _types(kept)["histogram"] >= 1
    # the best answer to every question survives: nothing of value is sacrificed
    questions = {(r.intent, r.spec.x, r.spec.group_by): r.score for r in kept}
    for rec in _pool():
        key = (rec.intent, rec.spec.x, rec.spec.group_by)
        if key in questions:
            assert questions[key] >= rec.score or rec in kept
    for x in ("ts", "hr", "mnth"):
        assert any(r.spec.x == x and r.spec.type == "line" for r in kept)


def test_different_intents_are_not_eliminated_for_sharing_a_renderer() -> None:
    # a cyclic line and a trend line are different findings: both keep a slot
    # even though the raw type is the same; a bare "line <= 2" cap would drop one
    kept = apply_diversity_caps(_pool())
    intents_of_lines = {r.intent for r in kept if r.spec.type == "line"}
    assert {"trend", "cyclic_pattern"} <= intents_of_lines
    assert TYPE_FREE_SLOTS == 2 and 0 < TYPE_DECAY < 1 and 0 < QUESTION_DECAY < 1


def test_hard_caps_and_score_order_still_hold() -> None:
    kept = apply_diversity_caps(_pool())
    assert all(a.score >= b.score for a, b in zip(kept, kept[1:]))
    assert max(Counter(r.intent for r in kept).values()) <= MAX_PER_TYPE
    cyclic_x = Counter(r.spec.x for r in kept if r.intent == "cyclic_pattern")
    assert max(cyclic_x.values()) <= MAX_PER_X
    per_y = Counter(r.spec.y for r in kept if r.spec.type in ("bar", "line") and r.spec.y)
    assert max(per_y.values()) <= MAX_PER_Y


def test_selection_is_deterministic_and_idempotent() -> None:
    first = apply_diversity_caps(_pool())
    second = apply_diversity_caps(_pool())
    assert [r.spec for r in first] == [r.spec for r in second]
    # the LLM merge re-applies the caps to an already capped list: a no-op
    again = apply_diversity_caps(list(first))
    assert [r.spec for r in again] == [r.spec for r in first]


def test_few_candidates_are_all_kept() -> None:
    small = [_rec("line", "trend", "ts", f"m{i}", 0.9 - 0.01 * i) for i in range(3)]
    small += [_rec("histogram", "distribution", "m0", None, 0.5)]
    kept = apply_diversity_caps(small)
    assert len(kept) == 4  # discounts only matter when slots are contested


def test_hour_like_line_duplication_is_reduced() -> None:
    profile = profile_dataset(_real_dataset("hour_like.csv"), "0" * 32, BIG)
    recs = recommend_charts(profile)
    assert len(recs) == MAX_CHARTS
    lines = [r for r in recs if r.spec.type == "line"]
    # stage 19: 5 of 12 charts were lines (3 cyclic + 2 trend); the third
    # answer to the same cycle / trend question no longer wins a slot on raw score
    assert len(lines) <= 4
    assert {r.intent for r in lines} == {"cyclic_pattern", "trend"}
    assert {r.spec.x for r in lines if r.intent == "cyclic_pattern"} >= {"hr", "mnth"}
    assert len({r.spec.type for r in recs}) >= 5
