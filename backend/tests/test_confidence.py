"""Confidence layer (stage 9): sample size, missingness, warnings, caps and
the exact/estimated count paths. All frames are synthetic and seeded; no
dataset file is read and nothing is keyed on a dataset name."""

import math
import random
from datetime import datetime, timedelta

import polars as pl
import pytest
from fastapi.testclient import TestClient

from app.charts.confidence import (
    MISSING_PENALTY_WEIGHT,
    N_FULL,
    N_GROUP_FULL,
    TOP_CONFIDENCE_FLOOR,
    Warning,
    assess,
    excluded_column_warnings,
    group_confidence,
    size_confidence,
    with_caution,
)
from app.charts.rules import TOP_SCORE_FLOOR, Recommendation, recommend_charts
from app.charts.spec import ChartSpec
from app.llm.schemas import FinalResponse, HypothesisResponse, LLMUsage
from app.llm.service import RecommendationService
from app.profiling.profiler import profile_dataset

BIG = 10**6


def _profile(df: pl.DataFrame):
    return profile_dataset(df, "0" * 32, BIG)


def _spec(**kwargs) -> ChartSpec:
    return ChartSpec(title="t", **kwargs)


def _find(recs, type_, x=None, y=None):
    for rec in recs:
        if rec.spec.type == type_ and (x is None or rec.spec.x == x) and (y is None or rec.spec.y == y):
            return rec
    return None


# --- the curves ---------------------------------------------------------------


def test_size_confidence_curve() -> None:
    assert size_confidence(0, N_FULL) == 0.0
    assert size_confidence(2, 30) == pytest.approx(math.sqrt(2 / 30))
    assert size_confidence(30, 30) == 1.0
    assert size_confidence(45, 30) == 1.0  # exactly 1 above n_full: healthy data untouched
    # monotone
    values = [size_confidence(n, N_FULL) for n in range(0, 60)]
    assert values == sorted(values)


def test_group_confidence_is_row_weighted() -> None:
    assert group_confidence({}) == 1.0
    assert group_confidence({"a": 2, "b": 2}) == pytest.approx(math.sqrt(2 / N_GROUP_FULL))
    # one small tail group among large ones barely moves it; min-based would give 0.87
    weighted = group_confidence({"M": 229, "F": 227, "Other": 15})
    assert 0.99 < weighted < 1.0
    assert group_confidence({"a": 40, "b": 40}) == 1.0


# --- tiny sample: never top, every chart type discounted ----------------------


def _tiny_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "dept": ["Sales", "Eng", "Sales", "Eng"],
            "score": [88.5, 72.0, 91.3, 65.8],
            "age": [29.0, 41.0, 35.0, 52.0],
            "extra": [1.5, 2.5, 3.5, 4.5],
        }
    )


def test_tiny_sample_never_reaches_top_and_says_so() -> None:
    recs = recommend_charts(_profile(_tiny_df()))
    assert recs
    for rec in recs:
        assert rec.tier != "top"
        assert rec.score < TOP_SCORE_FLOOR
        assert rec.confidence is not None
        assert rec.confidence.sample_size < TOP_CONFIDENCE_FLOOR
        assert rec.confidence.n_total == 4 and rec.confidence.n_effective == 4
        severe = [w for w in rec.warnings if w.code == "small_sample"]
        assert severe and severe[0].severity == "severe"
        assert "Caution: Only 4 rows" in rec.spec.reason
    bar = _find(recs, "bar", x="dept", y="score")
    assert bar is not None
    assert bar.confidence.min_group_n == 2
    assert bar.confidence.n_source == "exact"
    assert bar.warnings[0].meta["group_sizes"] == {"Eng": 2, "Sales": 2}
    assert bar.confidence.sample_size == pytest.approx(math.sqrt(2 / N_GROUP_FULL))
    # score == base x overall, base recomputed from the evidence table
    eta2 = next(e.eta_squared for e in bar_profile_evidence(recs, bar))
    assert bar.score == pytest.approx((0.45 + 0.45 * math.sqrt(eta2)) * bar.confidence.overall)


def bar_profile_evidence(recs, bar):
    profile = _profile(_tiny_df())
    return [e for e in profile.evidence.cat_num if (e.cat, e.num) == (bar.spec.x, bar.spec.y)]


def test_fixed_score_charts_go_through_the_same_chain() -> None:
    # Agent C's flag: heatmap/histogram/count bar must not become "best" on a
    # 4-row table merely because the evidence charts were penalised
    recs = recommend_charts(_profile(_tiny_df()))
    fixed = [r for r in recs if r.spec.type in ("heatmap", "histogram") or r.spec.aggregation == "count"]
    assert fixed
    for rec in fixed:
        assert rec.confidence.overall < TOP_CONFIDENCE_FLOOR
        assert rec.tier_cap == "secondary"
    assert max(r.score for r in recs) < TOP_SCORE_FLOOR


# --- missingness is separate from sample size ---------------------------------


def _missing_df(n: int, missing: float, seed: int) -> pl.DataFrame:
    rng = random.Random(seed)
    groups = ["a", "b", "c", "d"]
    return pl.DataFrame(
        {
            "g": [groups[i % 4] for i in range(n)],
            "v": [None if i < int(n * missing) else rng.gauss(0, 1) for i in range(n)],
        }
    )


def test_same_missing_ratio_different_sample_size() -> None:
    big = assess(_spec(type="bar", x="g", y="v", aggregation="mean"), _profile(_missing_df(1000, 0.3, 1)))
    small = assess(_spec(type="bar", x="g", y="v", aggregation="mean"), _profile(_missing_df(10, 0.3, 2)))
    assert big.confidence.missingness == pytest.approx(1 - MISSING_PENALTY_WEIGHT * 0.3)
    assert small.confidence.missingness == pytest.approx(big.confidence.missingness)
    assert big.confidence.sample_size == 1.0 and big.tier_cap is None
    assert small.confidence.sample_size < TOP_CONFIDENCE_FLOOR and small.tier_cap == "secondary"
    assert big.confidence.n_effective == 700 and small.confidence.n_effective == 7
    assert {w.code for w in big.warnings} == {"missing_data"}
    assert {w.code for w in small.warnings} == {"missing_data", "small_sample"}
    missing = next(w for w in big.warnings if w.code == "missing_data")
    assert missing.severity == "warning"
    assert missing.meta["n_effective"] == 700 and missing.meta["n_total"] == 1000
    assert "700 of 1000 rows" in missing.message and "30%" in missing.message


def test_missing_severity_thresholds() -> None:
    spec = _spec(type="histogram", x="v")
    assert assess(spec, _profile(_missing_df(500, 0.1, 3))).warnings == []
    assert assess(spec, _profile(_missing_df(500, 0.25, 4))).warnings[0].severity == "warning"
    assert assess(spec, _profile(_missing_df(500, 0.45, 5))).warnings[0].severity == "severe"


# --- healthy data is numerically untouched -----------------------------------


def _healthy_df(n: int = 600) -> pl.DataFrame:
    rng = random.Random(7)
    offsets = {"North": -1.5, "Central": 0.0, "South": 1.5}
    rows = []
    for i in range(n):
        region = list(offsets)[i % 3]
        x = rng.uniform(0, 10)
        rows.append(
            {
                "ts": datetime(2024, 1, 1) + timedelta(days=i % 90),
                "region": region,
                "x": x,
                "y": 2 * x + rng.gauss(0, 1),
                "sales": offsets[region] + rng.gauss(0, 1),
            }
        )
    return pl.DataFrame(rows)


def test_healthy_dataset_has_all_factors_one_and_no_warnings() -> None:
    profile = _profile(_healthy_df())
    recs = recommend_charts(profile)
    assert {r.spec.type for r in recs} >= {"bar", "box", "scatter", "line", "histogram", "heatmap"}
    for rec in recs:
        assert rec.confidence.overall == 1.0
        assert rec.confidence.n_source == "exact"
        assert rec.warnings == []
        assert "Caution" not in rec.spec.reason
    bar = _find(recs, "bar", x="region", y="sales")
    eta2 = next(e.eta_squared for e in profile.evidence.cat_num if (e.cat, e.num) == ("region", "sales"))
    assert bar.score == pytest.approx(0.45 + 0.45 * math.sqrt(eta2))
    assert bar.tier == "top"


# --- per-type effective rows --------------------------------------------------


def test_effective_rows_per_chart_type() -> None:
    n = 200
    df = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 1) + timedelta(days=i % 60) for i in range(n)],
            "g": [None if i % 10 == 0 else ["a", "b"][i % 2] for i in range(n)],  # 10% missing
            "a": [None if i % 4 == 0 else float(i) for i in range(n)],  # 25% missing
            "b": [None if i % 5 == 0 else float(i) * 2 for i in range(n)],  # 20% missing
            "c": [float(i % 7) for i in range(n)],
        }
    )
    profile = _profile(df)
    pair_ab = 200 - len({i for i in range(n) if i % 4 == 0 or i % 5 == 0})  # 120
    assert assess(_spec(type="histogram", x="a"), profile).confidence.n_effective == 150
    assert assess(_spec(type="bar", x="g", aggregation="count"), profile).confidence.n_effective == 180
    scatter = assess(_spec(type="scatter", x="a", y="b"), profile)
    assert scatter.confidence.n_effective == pair_ab and scatter.confidence.n_source == "exact"
    bar = assess(_spec(type="bar", x="g", y="a", aggregation="mean"), profile)
    assert bar.confidence.n_effective == 200 - len({i for i in range(n) if i % 10 == 0 or i % 4 == 0})
    assert bar.confidence.n_source == "exact" and bar.confidence.min_group_n is not None
    line = assess(_spec(type="line", x="ts", y="a", aggregation="mean", time_granularity="day"), profile)
    assert line.confidence.n_effective == 150 and line.confidence.n_source == "exact"
    heat = assess(_spec(type="heatmap"), profile)
    assert heat.confidence.n_effective == pair_ab  # min off-diagonal pair count
    assert heat.confidence.n_source == "exact"


# --- exact vs estimated counts ------------------------------------------------


def test_estimate_fallback_is_flagged_and_close() -> None:
    rng = random.Random(11)
    n = 1000
    df = pl.DataFrame(
        {
            "g": [None if rng.random() < 0.3 else ["p", "q", "r"][i % 3] for i in range(n)],
            "v": [None if rng.random() < 0.3 else rng.gauss(0, 1) for _ in range(n)],
            "w": [None if rng.random() < 0.2 else rng.gauss(0, 1) for _ in range(n)],
        }
    )
    profile = _profile(df)
    spec = _spec(type="bar", x="g", y="v", aggregation="mean")
    exact = assess(spec, profile)
    assert exact.confidence.n_source == "exact"

    stripped = profile.model_copy(deep=True)
    for effect in stripped.evidence.cat_num:
        effect.n_total, effect.n_min, effect.group_counts = 0, 0, {}
    stripped.correlations.pair_counts = None
    estimated = assess(spec, stripped)
    assert estimated.confidence.n_source == "estimated"
    assert estimated.confidence.n_effective == pytest.approx(exact.confidence.n_effective, rel=0.05)
    assert all(w.meta["n_source"] == "estimated" for w in estimated.warnings)

    scatter = assess(_spec(type="scatter", x="v", y="w"), stripped)
    assert scatter.confidence.n_source == "estimated"
    assert scatter.confidence.n_effective == pytest.approx(
        assess(_spec(type="scatter", x="v", y="w"), profile).confidence.n_effective, rel=0.05
    )


def test_small_tail_group_warns_without_sinking_the_chart() -> None:
    rng = random.Random(13)
    n = 500
    df = pl.DataFrame(
        {
            "g": ["M" if i % 2 == 0 else ("F" if i % 33 else "Other") for i in range(n)],
            "v": [rng.gauss(0, 1) for _ in range(n)],
        }
    )
    result = assess(_spec(type="bar", x="g", y="v", aggregation="mean"), _profile(df))
    assert result.confidence.sample_size > 0.98
    assert result.tier_cap is None
    (warning,) = [w for w in result.warnings if w.code == "small_groups"]
    assert warning.severity == "warning" and "Other" in warning.message
    assert warning.meta["min_group_n"] == result.confidence.min_group_n < N_GROUP_FULL


# --- reason suffix and dataset-level warnings ---------------------------------


def test_with_caution_is_idempotent_and_skips_info() -> None:
    info = Warning(code="small_sample", severity="info", message="Only 25 rows.")
    warn = Warning(code="missing_data", severity="warning", message="30% of rows lack v.")
    assert with_caution("Shows v.", [info]) == "Shows v."
    once = with_caution("Shows v.", [warn])
    assert once == "Shows v. Caution: 30% of rows lack v."
    assert with_caution(once, [warn]) == once
    assert with_caution(once, []) == "Shows v."
    assert with_caution("", [warn]) == "Caution: 30% of rows lack v."


def test_excluded_column_warnings_only_for_missingness() -> None:
    df = pl.DataFrame(
        {
            "keep": [float(i) for i in range(40)],
            "gone": [None if i < 24 else float(i) for i in range(40)],  # 60% missing
            "user_id": [f"u{i:03d}" for i in range(40)],
            "note": [f"quite a long unique free text comment number {i}" for i in range(40)],
        }
    )
    warnings = excluded_column_warnings(_profile(df))
    assert [w.code for w in warnings] == ["column_excluded"]
    assert warnings[0].meta == {"column": "gone", "missing_ratio": 0.6}
    assert "60%" in warnings[0].message


# --- LLM path ----------------------------------------------------------------


class _FakeProvider:
    def __init__(self, response: HypothesisResponse) -> None:
        self.response = response

    def generate_hypotheses(self, profile, rule_candidates):
        return self.response, LLMUsage()

    def finalize_insights(self, profile, validated):
        return FinalResponse(), LLMUsage()


def test_llm_chart_on_tiny_data_is_weak_and_keeps_caution() -> None:
    profile = _profile(_tiny_df())
    rules = recommend_charts(profile)
    bar = _find(rules, "bar", x="dept", y="score")
    dedup = {
        "title": "dept effect",
        "type": "bar",
        "x": "dept",
        "y": "score",
        "aggregation": "mean",
        "reason": "the LLM's reason",
        "priority": 1,
    }
    new = {"title": "age vs score", "type": "scatter", "x": "age", "y": "score", "reason": "guess", "priority": 2}
    service = RecommendationService(
        _FakeProvider(HypothesisResponse(hypotheses=[{"statement": "dept matters", "chart": dedup}], charts=[dedup, new]))
    )
    result = service.get(profile, use_llm=True, include_debug=True)
    # stage 17.3: on 4 rows no claim becomes an insight, not even a weak one
    assert result["insights"] == []
    assert "fewer than 30 rows" in result["debug"]["dropped"][0]["reason"]
    merged = next(c for c in result["charts"] if c["spec"]["x"] == "dept" and c["spec"]["type"] == "bar")
    assert merged["source"] == "rules"
    assert merged["spec"]["reason"].startswith("the LLM's reason Caution: Only 4 rows")
    assert merged["tier"] != "top" and merged["warnings"] == [w.model_dump() for w in bar.warnings]
    llm = next(c for c in result["charts"] if c["source"] == "llm")
    assert llm["confidence"]["overall"] < TOP_CONFIDENCE_FLOOR
    assert llm["tier"] != "top" and llm["warnings"][0]["code"] == "small_sample"
    assert result["warnings"] == []


# --- serialization and API ---------------------------------------------------


def test_recommendation_roundtrip_keeps_new_fields_and_hides_tier_cap() -> None:
    rec = next(r for r in recommend_charts(_profile(_tiny_df())) if r.spec.type == "histogram")
    payload = rec.model_dump()
    assert "tier_cap" not in payload
    assert set(payload["confidence"]) == {
        "sample_size", "missingness", "robustness", "overall", "n_total", "n_effective",
        "missing_ratio", "min_group_n", "n_source",
    }
    assert Recommendation.model_validate_json(rec.model_dump_json()).confidence == rec.confidence


def test_endpoint_exposes_chart_and_dataset_warnings(client: TestClient) -> None:
    csv = "g,v,gone\n" + "\n".join(f"{'ab'[i % 2]},{i}.0,{'' if i < 15 else i}" for i in range(20)) + "\n"
    dataset_id = client.post("/api/datasets", files={"file": ("w.csv", csv.encode())}).json()["dataset_id"]
    body = client.get(f"/api/datasets/{dataset_id}/recommendations").json()
    assert [w["code"] for w in body["warnings"]] == ["column_excluded"]
    assert body["warnings"][0]["meta"]["column"] == "gone"
    for chart in body["charts"]:
        assert chart["confidence"]["n_source"] in ("exact", "estimated")
        assert chart["confidence"]["n_total"] == 20
        assert any(w["code"] == "small_sample" for w in chart["warnings"])  # 20 < N_FULL
        assert chart["tier"] != "top"


# --- stage 2 verification follow-ups: D1 (group rescale) and J1 (majority guard)


def test_grouped_counts_not_rescaled_by_group_rate() -> None:
    # D1: (g, v)-complete group counts from evidence already exclude null-g
    # rows; multiplying them by g's own valid rate again understated them
    n = 600
    df = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 1) + timedelta(days=i % 60) for i in range(n)],
            "g": [None if i % 5 == 0 else ["A", "B"][i % 2] for i in range(n)],  # 20% null
            "v": [float(i % 13) for i in range(n)],
        }
    )
    profile = _profile(df)
    true_sizes = df.drop_nulls(["g", "v"])["g"].value_counts()
    expected = dict(zip(true_sizes["g"].to_list(), true_sizes["count"].to_list()))
    for spec in (
        _spec(type="line", x="ts", y="v", group_by="g", aggregation="mean", time_granularity="day"),
        _spec(type="scatter", x="v", y="v", group_by="g"),
    ):
        result = assess(spec, profile)
        assert result.confidence.min_group_n == min(expected.values()) == 240
        assert result.confidence.n_effective == 480  # rows with a non-null group


def _shares_df(sizes: dict[str, int]) -> pl.DataFrame:
    rng = random.Random(17)
    rows = [{"g": name, "v": rng.gauss(0, 1)} for name, count in sizes.items() for _ in range(count)]
    return pl.DataFrame(rows)


def test_majority_tiny_groups_cap_tier_but_keep_weighted_factor() -> None:
    # J1: 2/2/996 -> row-weighted factor stays ~0.997, yet two of the three
    # bars rest on 2 rows each: cap at secondary and say so
    result = assess(_spec(type="bar", x="g", y="v", aggregation="mean"), _profile(_shares_df({"a": 2, "b": 2, "c": 996})))
    assert result.confidence.sample_size > 0.99
    assert result.tier_cap == "secondary"
    (warning,) = [w for w in result.warnings if w.code == "small_groups"]
    assert warning.severity == "warning"  # 2-row groups are tiny, not degenerate
    assert "2 of 3 g groups have fewer than 5 rows" in warning.message
    assert warning.meta["small_groups"] == 2 and warning.meta["plotted_groups"] == 3
    # degenerate majority (1-row groups) escalates to severe
    severe = assess(_spec(type="bar", x="g", y="v", aggregation="mean"), _profile(_shares_df({"a": 1, "b": 1, "c": 996})))
    assert severe.tier_cap == "secondary"
    assert next(w for w in severe.warnings if w.code == "small_groups").severity == "severe"


def test_single_small_tail_group_warns_without_cap() -> None:
    result = assess(_spec(type="bar", x="g", y="v", aggregation="mean"), _profile(_shares_df({"a": 997, "b": 3})))
    assert result.tier_cap is None
    assert result.confidence.sample_size > 0.99
    (warning,) = [w for w in result.warnings if w.code == "small_groups"]
    assert warning.severity == "warning" and "b (3 rows)" in warning.message


def test_healthy_groups_untouched_by_guard() -> None:
    result = assess(_spec(type="bar", x="g", y="v", aggregation="mean"), _profile(_shares_df({"a": 20, "b": 20, "c": 20})))
    assert result.tier_cap is None
    assert result.confidence.sample_size == 1.0 and result.warnings == []


# --- stage 5: robustness (suspected sentinels / extreme values) ---------------
# Quality flags are hand-built so these tests do not depend on the detector.

from app.charts.confidence import (  # noqa: E402
    EXTREME_REF_RATIO,
    EXTREME_WEIGHT,
    HEATMAP_SENTINEL_CAP_SHARE,
    SENTINEL_FACTOR,
    SENTINEL_FACTOR_ROBUST_DISPLAY,
    column_robustness,
)
from app.profiling.models import ColumnQuality, RobustRange, SentinelCandidate  # noqa: E402


def _flag(profile, name, *, sentinels=None, extreme_ratio=0.0, robust_range=None):
    col = next(c for c in profile.columns if c.name == name)
    q = col.quality
    col.quality = ColumnQuality(
        profiled_rows=q.profiled_rows,
        missing_count=q.missing_count,
        missing_token_count=q.missing_token_count,
        invalid_count=q.invalid_count,
        valid_count=q.valid_count,
        valid_ratio=q.valid_ratio,
        suspected_sentinels=[SentinelCandidate(value=v, count=c, signals=["extreme", "repeated", "pattern"]) for v, c in (sentinels or [])],
        sentinel_row_count=sum(c for _, c in (sentinels or [])),
        extreme_value_count=int(round(extreme_ratio * q.valid_count)),
        extreme_value_ratio=extreme_ratio,
        robust_range=RobustRange(lo=robust_range[0], hi=robust_range[1]) if robust_range else None,
    )
    return profile


def _robust_df(n: int = 600) -> pl.DataFrame:
    rng = random.Random(23)
    return pl.DataFrame(
        {
            "g": [["a", "b", "c"][i % 3] for i in range(n)],
            "t": [rng.gauss(25, 2) for _ in range(n)],
            "v": [rng.gauss(50, 5) for _ in range(n)],
            "w": [rng.gauss(0, 1) for _ in range(n)],
        }
    )


SENTINELS = [(9999.0, 11), (-999.0, 8)]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(type="histogram", x="t"),
        dict(type="scatter", x="t", y="v"),
        dict(type="bar", x="g", y="t", aggregation="mean"),
        dict(type="bar", x="g", y="t", aggregation="max"),
        dict(type="line", x="v", y="t", aggregation="sum"),
    ],
    ids=["histogram", "scatter", "bar-mean", "bar-max", "line-sum"],
)
def test_sentinel_on_sensitive_axis_discounts_caps_and_warns(kwargs) -> None:
    profile = _flag(_profile(_robust_df()), "t", sentinels=SENTINELS, robust_range=(18.0, 32.0))
    result = assess(_spec(**kwargs), profile)
    assert result.confidence.robustness == SENTINEL_FACTOR
    assert result.confidence.overall == pytest.approx(SENTINEL_FACTOR)  # n=600, no missing
    assert result.tier_cap == "secondary"
    (warning,) = [w for w in result.warnings if w.code == "suspected_sentinel"]
    assert warning.severity == "severe"
    assert "t contains suspected sentinel values (9999 x11, -999 x8)" in warning.message
    assert "invalid" not in warning.message and "removed" not in warning.message
    assert warning.meta["sentinel_rows"] == 19 and warning.meta["sensitive"] is True


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(type="box", x="g", y="t"),
        dict(type="bar", x="g", y="t", aggregation="median"),
        dict(type="line", x="v", y="t", aggregation="count"),
    ],
    ids=["box", "bar-median", "line-count"],
)
def test_sentinel_on_robust_display_warns_without_cap(kwargs) -> None:
    profile = _flag(_profile(_robust_df()), "t", sentinels=SENTINELS, robust_range=(18.0, 32.0))
    result = assess(_spec(**kwargs), profile)
    assert result.confidence.robustness == SENTINEL_FACTOR_ROBUST_DISPLAY
    assert result.tier_cap is None
    (warning,) = [w for w in result.warnings if w.code == "suspected_sentinel"]
    assert warning.severity == "warning" and "robust" in warning.message


@pytest.mark.parametrize(
    "ratio,severity",
    [(0.005, "info"), (0.02, "warning"), (0.06, "warning"), (0.15, "warning")],
)
def test_extreme_values_gentle_factor_never_caps(ratio, severity) -> None:
    # factor follows the constants (EXTREME_REF_RATIO calibration is a
    # coordinator decision); it saturates at 1 - EXTREME_WEIGHT and a clean
    # heavy-tailed column at ~6% keeps >= 0.88
    factor = 1 - EXTREME_WEIGHT * min(1.0, ratio / EXTREME_REF_RATIO)
    profile = _flag(_profile(_robust_df()), "v", extreme_ratio=ratio)
    result = assess(_spec(type="histogram", x="v"), profile)
    assert result.confidence.robustness == pytest.approx(factor)
    assert result.confidence.robustness >= 1 - EXTREME_WEIGHT
    if ratio <= 0.06:
        assert result.confidence.robustness >= 0.88
    assert result.tier_cap is None
    (warning,) = [w for w in result.warnings if w.code == "extreme_value"]
    assert warning.severity == severity and "extreme values" in warning.message
    assert "invalid" not in warning.message
    # box charts show outliers by design: no discount, no note
    box = assess(_spec(type="box", x="g", y="v"), profile)
    assert box.confidence.robustness == 1.0 and box.warnings == []


def test_heatmap_prorated_by_affected_columns() -> None:
    one = _flag(_profile(_robust_df()), "t", sentinels=SENTINELS, robust_range=(18.0, 32.0))
    result = assess(_spec(type="heatmap"), one)
    assert result.confidence.robustness == pytest.approx(1 - (1 - SENTINEL_FACTOR) / 3)
    assert result.tier_cap is None
    (warning,) = result.warnings
    assert warning.severity == "warning" and "1 of 3 numeric columns (t)" in warning.message
    two = _flag(one, "v", sentinels=[(-1.0, 40)], robust_range=(30.0, 70.0))
    result = assess(_spec(type="heatmap"), two)
    assert result.confidence.robustness == pytest.approx(1 - (1 - SENTINEL_FACTOR) * 2 / 3)
    assert 2 / 3 >= HEATMAP_SENTINEL_CAP_SHARE and result.tier_cap == "secondary"
    assert result.warnings[0].severity == "severe"


def test_clean_columns_and_missing_quality_are_neutral() -> None:
    profile = _profile(_robust_df())
    for kwargs in (dict(type="histogram", x="t"), dict(type="scatter", x="t", y="v"), dict(type="heatmap")):
        result = assess(_spec(**kwargs), profile)
        assert result.confidence.robustness == 1.0 and result.warnings == []
    col = next(c for c in profile.columns if c.name == "t")
    col.quality = None  # pre-stage-5 payload
    assert column_robustness(col, True) == (1.0, None)
    assert assess(_spec(type="histogram", x="t"), profile).confidence.robustness == 1.0


def test_twin_frames_sentinel_flag_lowers_score_and_tier() -> None:
    # identical evidence; the only difference is the quality flag on t
    clean = recommend_charts(_profile(_robust_df()))
    flagged = recommend_charts(_flag(_profile(_robust_df()), "t", sentinels=SENTINELS, robust_range=(18.0, 32.0)))
    for chart in ("histogram",):
        before = _find(clean, chart, x="t")
        after = _find(flagged, chart, x="t")
        assert before is not None and after is not None
        assert after.score < before.score
        assert after.score == pytest.approx(before.score * SENTINEL_FACTOR)
        assert after.tier != "top"
        assert "Caution: t contains suspected sentinel values" in after.spec.reason
    untouched = [r for r in flagged if "t" not in (r.spec.x, r.spec.y) and r.spec.type != "heatmap"]
    assert untouched and all(r.confidence.robustness == 1.0 for r in untouched)
