"""Stage 19: intent classification and chart selection are deterministic
functions of roles + context, with a named guard behind every chart."""

import pytest

from app.charts.intent import (
    DENSITY_MIN_N,
    HEXBIN_MIN_N,
    ChartChoice,
    IntentContext,
    classify_intent,
    eligible_charts,
    select_chart,
)
from app.charts.roles import ColumnRole


def _role(name: str, role: str, **kw) -> ColumnRole:
    defaults = dict(ordered=role in ("numeric", "datetime", "ordinal", "cyclic"), cyclic=role == "cyclic")
    return ColumnRole(name=name, role=role, **{**defaults, **kw})


NUM = _role("y", "numeric")
DT = _role("ts", "datetime")
HOUR = _role("hr", "cyclic", period=24)
MONTH = _role("mnth", "cyclic", period=12)
REGION = _role("region", "nominal", n_categories=4)
LEVEL = _role("level", "ordinal", order=["low", "medium", "high"], n_categories=3)
FLAG = _role("flag", "boolean", two_state=True, n_categories=2)
X = _role("x", "numeric")


# --- intent -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x, y, group, ctx, expected",
    [
        (DT, NUM, None, IntentContext(n_rows=600), "trend"),
        (X, NUM, None, IntentContext(n_rows=600), "numeric_relationship"),
        (X, NUM, None, IntentContext(n_rows=60000), "numeric_relationship"),
        (None, NUM, None, IntentContext(n_rows=600), "distribution"),
        (HOUR, NUM, None, IntentContext(n_rows=600, n_groups=24), "cyclic_pattern"),
        (MONTH, NUM, None, IntentContext(n_rows=600, n_groups=12), "cyclic_pattern"),
        (REGION, NUM, None, IntentContext(n_rows=600, n_groups=4), "group_comparison"),
        (REGION, NUM, None, IntentContext(n_rows=600, n_groups=4, distribution_evidence=True), "distribution_by_group"),
        (LEVEL, NUM, None, IntentContext(n_rows=600, n_groups=3, distribution_evidence=True), "distribution_by_group"),
        (_role("city", "nominal", n_categories=15), NUM, None, IntentContext(n_rows=600, n_groups=15), "ranking"),
        (REGION, None, None, IntentContext(n_rows=600, n_groups=4, count_measure=True), None),
        (REGION, NUM, FLAG, IntentContext(n_rows=600, n_groups=4), "two_state_comparison"),
        (MONTH, NUM, HOUR, IntentContext(n_rows=6000, n_groups=12), "matrix_pattern"),
        (LEVEL, NUM, MONTH, IntentContext(n_rows=6000, n_groups=3), "matrix_pattern"),
        (REGION, NUM, None, IntentContext(n_rows=600, n_groups=4, ranking_requested=True), "ranking"),
    ],
)
def test_classify_intent(x, y, group, ctx, expected) -> None:
    assert classify_intent(x, y, group, ctx) == expected
    assert classify_intent(x, y, group, ctx) == expected  # deterministic


def test_region_sales_is_group_comparison_never_cyclic() -> None:
    # "region → sales" is nominal: rose / polar must never be eligible
    ctx = IntentContext(n_rows=1000, n_groups=4, n_min_group=250, magnitude_measure=True)
    intent = classify_intent(REGION, NUM, None, ctx)
    assert intent == "group_comparison"
    assert "rose" not in eligible_charts(intent, REGION, None, ctx)
    assert "polar_line" not in eligible_charts(intent, REGION, None, ctx)


# --- eligibility guards -------------------------------------------------------------------


def test_cyclic_guards() -> None:
    ctx = IntentContext(n_rows=2160, n_groups=24, n_min_group=90)
    charts = eligible_charts("cyclic_pattern", HOUR, None, ctx)
    assert charts[0] == "line" and "polar_line" in charts and "rose" not in charts  # mean is not a magnitude
    rose_ctx = IntentContext(n_rows=2160, n_groups=24, n_min_group=90, magnitude_measure=True)
    assert "rose" in eligible_charts("cyclic_pattern", HOUR, None, rose_ctx)
    # a 366-day cycle is too fine for a polar axis
    doy = _role("doy", "cyclic", period=366)
    assert "polar_line" not in eligible_charts("cyclic_pattern", doy, None, IntentContext(n_rows=5000, n_groups=366))
    # three months only: no polar (too few categories)
    assert eligible_charts("cyclic_pattern", MONTH, None, IntentContext(n_rows=600, n_groups=3)) == ["line"]


def test_density_guards() -> None:
    dense = IntentContext(n_rows=600, n_groups=4, n_min_group=150, distribution_evidence=True)
    charts = eligible_charts("distribution_by_group", REGION, None, dense)
    assert charts[0] == "raincloud" and "box" in charts and "ridgeline" not in charts  # nominal: no ridge
    ordered = eligible_charts("distribution_by_group", LEVEL, None, IntentContext(n_rows=600, n_groups=3, n_min_group=150))
    assert ordered[0] == "raincloud" and "ridgeline" in ordered
    # small groups: only a box (no density smoothing on a handful of rows)
    tiny = IntentContext(n_rows=40, n_groups=4, n_min_group=DENSITY_MIN_N - 1)
    assert eligible_charts("distribution_by_group", REGION, None, tiny) == ["box"]
    # many groups: raincloud unreadable, ridgeline only when ordered and ≤ 12
    many = IntentContext(n_rows=6000, n_groups=10, n_min_group=500)
    assert eligible_charts("distribution_by_group", REGION, None, many)[0] == "box"
    ten_levels = _role("grade", "ordinal", n_categories=10)
    assert eligible_charts("distribution_by_group", ten_levels, None, many)[0] == "ridgeline"
    assert eligible_charts("distribution", None, None, IntentContext(n_rows=10)) == ["histogram", "box"]
    assert "violin" in eligible_charts("distribution", None, None, IntentContext(n_rows=600))


def test_numeric_relationship_guards() -> None:
    small = eligible_charts("numeric_relationship", X, None, IntentContext(n_rows=600))
    assert small == ["scatter", "hexbin"]  # scatter first: readable at this size
    assert eligible_charts("numeric_relationship", X, None, IntentContext(n_rows=100)) == ["scatter"]
    large = eligible_charts("numeric_relationship", X, None, IntentContext(n_rows=HEXBIN_MIN_N + 1))
    assert large[0] == "hexbin" and "scatter" in large


def test_two_state_and_ranking_guards() -> None:
    two = eligible_charts("two_state_comparison", REGION, FLAG, IntentContext(n_rows=600, n_groups=4))
    assert two[0] == "dumbbell" and "slopegraph" in two
    many = eligible_charts("two_state_comparison", _role("city", "nominal"), FLAG, IntentContext(n_rows=6000, n_groups=30))
    assert many[0] == "dumbbell" and "slopegraph" not in many
    assert eligible_charts("ranking", REGION, None, IntentContext(n_groups=4)) == ["bar", "lollipop"]
    assert eligible_charts("ranking", REGION, None, IntentContext(n_groups=20))[0] == "lollipop"
    assert eligible_charts("group_comparison", REGION, None, IntentContext(n_groups=4, n_min_group=100)) == ["bar", "dot_interval", "box"]
    assert eligible_charts("group_comparison", REGION, None, IntentContext(n_groups=4, n_min_group=2)) == ["bar", "box"]
    assert eligible_charts("group_comparison", REGION, None, IntentContext(n_groups=4, count_measure=True)) == ["bar"]
    assert eligible_charts("composition", REGION, None, IntentContext()) == []
    assert eligible_charts("flow", REGION, None, IntentContext()) == []


# --- selection ----------------------------------------------------------------------------


def test_select_chart_prefers_first_eligible_and_orders_categories() -> None:
    ctx = IntentContext(n_rows=2160, n_groups=24, n_min_group=90)
    choice = select_chart("cyclic_pattern", HOUR, None, ctx)
    assert choice == ChartChoice(intent="cyclic_pattern", preferred="line", alternatives=("polar_line",), sort="natural")
    ranking = select_chart("ranking", REGION, None, IntentContext(n_groups=4))
    assert ranking.preferred == "bar" and ranking.sort == "value"
    trend = select_chart("trend", DT, None, IntentContext(n_rows=600))
    assert trend.preferred == "line" and trend.sort is None and trend.alternatives == ()


def test_hint_only_promotes_an_eligible_chart() -> None:
    ctx = IntentContext(n_rows=600, n_groups=4, n_min_group=150, distribution_evidence=True)
    boxed = select_chart("distribution_by_group", REGION, None, ctx, hint="box")
    assert boxed.preferred == "box" and "raincloud" in boxed.alternatives
    rose = select_chart("group_comparison", REGION, None, IntentContext(n_groups=4, n_min_group=150), hint="rose")
    assert rose.preferred == "bar" and "rose" not in rose.alternatives
    assert any("not eligible" in n for n in rose.notes)
    assert select_chart("composition", REGION, None, IntentContext()) is None
