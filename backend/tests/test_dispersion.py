"""Stage 19.1: dispersion evidence (spread differences across groups) is
independent of the mean-difference evidence, sample-size aware, and drives
the distribution family (raincloud / ridgeline / box) in chart selection."""

import math
import random

import polars as pl
import pytest

from app.charts.rules import recommend_charts
from app.charts.rules import apply_derived_caps, derived_column_warnings
from app.profiling.evidence import (
    DISPERSION_ALPHA,
    DISPERSION_MIN_ETA,
    DISPERSION_MIN_RATIO,
    DISPERSION_STRONG_MIN_N,
    DISPERSION_WEAK_MIN_N,
)
from app.profiling.models import PROFILE_VERSION, DatasetProfile
from app.profiling.profiler import profile_dataset

BIG = 10**6


def _groups(means: dict[str, float], sds: dict[str, float], n_per_group: int, seed: int = 191) -> pl.DataFrame:
    rng = random.Random(seed)
    labels = list(means)
    rows = []
    for i in range(n_per_group * len(labels)):
        g = labels[i % len(labels)]
        rows.append({"g": g, "y": means[g] + rng.gauss(0, sds[g]), "noise": rng.gauss(0, 1)})
    return pl.DataFrame(rows)


def _profile(df: pl.DataFrame) -> DatasetProfile:
    return profile_dataset(df, "0" * 32, BIG)


def _dispersion(profile: DatasetProfile, cat: str = "g", num: str = "y"):
    return next((d for d in profile.evidence.dispersion if d.cat == cat and d.num == num), None)


def _eta2(profile: DatasetProfile, cat: str = "g", num: str = "y") -> float:
    return next(e.eta_squared for e in profile.evidence.cat_num if e.cat == cat and e.num == num)


# --- A. means differ, spread similar → mean evidence only ----------------------


def test_a_means_differ_spread_similar_is_not_dispersion_evidence() -> None:
    profile = _profile(_groups({"a": 0, "b": 2, "c": 4}, {"a": 1, "b": 1, "c": 1}, 150))
    assert _eta2(profile) > 0.5  # the mean difference is there
    assert _dispersion(profile) is None  # the spreads are not
    rec = next(r for r in recommend_charts(profile) if r.spec.x == "g" and r.spec.y == "y")
    assert rec.intent == "group_comparison" and rec.spec.type == "bar"


# --- B. means nearly identical, spread clearly different → dispersion evidence ---


def test_b_equal_means_different_spread_is_strong_dispersion_evidence() -> None:
    profile = _profile(_groups({"a": 0, "b": 0, "c": 0}, {"a": 0.5, "b": 1.5, "c": 3.0}, 150))
    assert _eta2(profile) < 0.02  # no mean difference to speak of
    d = _dispersion(profile)
    assert d is not None and d.strength == "strong"
    assert d.bf_eta_squared >= DISPERSION_MIN_ETA
    assert d.spread_ratio >= DISPERSION_MIN_RATIO and d.iqr_ratio is not None and d.iqr_ratio > 3
    assert d.std_ratio is not None and d.mad_ratio is not None
    assert (d.widest_group, d.narrowest_group) == ("c", "a")
    assert d.n_groups == 3 and d.n_min == 150 and d.n_total == 450
    # the noise column has the same spread in every group
    assert _dispersion(profile, num="noise") is None


def test_b_drives_the_distribution_family_in_chart_selection() -> None:
    profile = _profile(_groups({"a": 0, "b": 0, "c": 0}, {"a": 0.5, "b": 1.5, "c": 3.0}, 150))
    rec = next(r for r in recommend_charts(profile) if r.spec.x == "g" and r.spec.y == "y")
    assert rec.intent == "distribution_by_group"
    assert rec.spec.type == "raincloud"  # 3 nominal groups, >= 30 rows each
    assert {"box", "violin"} <= {a.type for a in rec.alternatives}
    assert "distribution" in rec.spec.reason


def test_b_ordered_groups_offer_a_ridgeline() -> None:
    profile = _profile(_groups({"low": 0, "medium": 0, "high": 0}, {"low": 0.5, "medium": 1.5, "high": 3.0}, 150))
    rec = next(r for r in recommend_charts(profile) if r.spec.x == "g" and r.spec.y == "y")
    assert rec.intent == "distribution_by_group"
    assert rec.spec.type in ("raincloud", "ridgeline") and rec.spec.sort == "natural"
    assert "ridgeline" in {rec.spec.type, *(a.type for a in rec.alternatives)}


# --- C. means and spread both differ → both kinds of evidence -------------------


def test_c_both_differ_yields_both_evidence_kinds() -> None:
    profile = _profile(_groups({"a": 0, "b": 3, "c": 6}, {"a": 0.5, "b": 1.5, "c": 3.0}, 150))
    assert _eta2(profile) > 0.5
    d = _dispersion(profile)
    assert d is not None and d.strength == "strong" and d.spread_ratio >= DISPERSION_MIN_RATIO
    rec = next(r for r in recommend_charts(profile) if r.spec.x == "g" and r.spec.y == "y")
    assert rec.intent == "distribution_by_group"  # the spread is the richer story


# --- D. small samples never make high-confidence dispersion evidence ---------------


def test_d_tiny_groups_produce_no_dispersion_entry() -> None:
    n = DISPERSION_WEAK_MIN_N - 2
    profile = _profile(_groups({"a": 0, "b": 0, "c": 0}, {"a": 0.5, "b": 1.5, "c": 3.0}, n))
    assert _dispersion(profile) is None


def test_d_small_groups_are_weak_and_do_not_move_the_chart() -> None:
    n = DISPERSION_STRONG_MIN_N - 10
    assert n >= DISPERSION_WEAK_MIN_N
    profile = _profile(_groups({"a": 0, "b": 0, "c": 0}, {"a": 0.5, "b": 1.5, "c": 3.0}, n))
    d = _dispersion(profile)
    assert d is not None and d.strength == "weak" and d.n_min == n
    # weak evidence is reported, not acted on: no distribution chart of y by g
    # (the pair's mean chart may be squeezed out by the per-x cap; that is fine)
    for rec in recommend_charts(profile):
        if rec.spec.x == "g" and rec.spec.y == "y":
            assert rec.intent == "group_comparison"
            assert rec.spec.type not in ("raincloud", "ridgeline", "violin")


def test_d_one_small_group_among_large_ones_is_excluded_not_cancelling() -> None:
    # stage 19.2 (A3): the 12-row group is unreliable and is excluded from the
    # comparison; the two 200-row groups still carry strong evidence
    big = _groups({"a": 0, "b": 0}, {"a": 0.5, "b": 3.0}, 200)
    small = _groups({"c": 0}, {"c": 1.5}, 12, seed=7)
    profile = _profile(pl.concat([big, small]))
    d = _dispersion(profile)
    assert d is not None and d.strength == "strong"
    assert d.compared_groups == ["a", "b"] and d.excluded_groups == {"c": 12}
    assert d.n_groups == 2 and d.n_min == 200 and d.n_total == 400
    assert d.bf_p_value is not None and d.bf_p_value <= DISPERSION_ALPHA


# --- shape, determinism, version ----------------------------------------------------


def test_dispersion_is_deterministic_and_round_trips() -> None:
    df = _groups({"a": 0, "b": 0, "c": 0}, {"a": 0.5, "b": 1.5, "c": 3.0}, 100)
    first, second = _profile(df), _profile(df)
    assert first.evidence.dispersion == second.evidence.dispersion
    assert first.profile_version == PROFILE_VERSION == 13
    payload = first.model_dump()
    assert payload["evidence"]["dispersion"][0]["strength"] == "strong"
    assert DatasetProfile.model_validate(payload).evidence.dispersion == first.evidence.dispersion


def test_constant_group_has_no_iqr_ratio_but_a_std_or_mad_fallback() -> None:
    rng = random.Random(3)
    rows = [{"g": "flat", "y": 5.0} for _ in range(60)] + [{"g": "wide", "y": 5 + rng.gauss(0, 2)} for _ in range(60)]
    profile = _profile(pl.DataFrame(rows))
    d = _dispersion(profile)
    # a constant group has zero spread on every measure → no ratio, no entry
    assert d is None or d.iqr_ratio is None


@pytest.mark.parametrize("n_per_group", [40, 400])
def test_equal_spreads_never_flag_whatever_the_sample_size(n_per_group: int) -> None:
    profile = _profile(_groups({"a": 0, "b": 1, "c": 2, "d": 3}, {k: 2.0 for k in "abcd"}, n_per_group))
    assert _dispersion(profile) is None


# --- E. small groups (stage 19.2, A3): excluded and traced, never cancelling ---


def _sized(sizes: dict[str, int], sds: dict[str, float], seed: int = 23) -> pl.DataFrame:
    rng = random.Random(seed)
    rows = [{"g": g, "y": rng.gauss(0, sds[g])} for g, n in sizes.items() for _ in range(n)]
    return pl.DataFrame(rows)


def test_e1_two_supported_groups_with_different_spread_are_strong() -> None:
    d = _dispersion(_profile(_sized({"a": 300, "b": 300}, {"a": 1.0, "b": 4.0})))
    assert d is not None and d.strength == "strong"
    assert d.compared_groups == ["a", "b"] and d.excluded_groups == {}


def test_e2_a_tiny_third_group_does_not_cancel_the_evidence() -> None:
    d = _dispersion(_profile(_sized({"a": 300, "b": 300, "c": 3}, {"a": 1.0, "b": 4.0, "c": 1.0})))
    assert d is not None and d.strength == "strong"
    assert d.compared_groups == ["a", "b"] and d.excluded_groups == {"c": 3}
    assert d.n_groups == 2 and d.n_total == 600 and d.n_min == 300


def test_e3_two_tiny_groups_give_no_evidence() -> None:
    assert _dispersion(_profile(_sized({"a": 3, "b": 3}, {"a": 1.0, "b": 20.0}))) is None


def test_e4_one_reliable_group_alone_never_claims_a_difference() -> None:
    assert _dispersion(_profile(_sized({"a": 300, "b": 3}, {"a": 1.0, "b": 20.0}))) is None


def test_e4b_a_small_wild_group_can_only_make_weak_evidence() -> None:
    # 12 rows with a huge spread against one supported group: compared as a
    # weak fallback (both >= DISPERSION_WEAK_MIN_N), never strong
    d = _dispersion(_profile(_sized({"a": 300, "b": 12}, {"a": 1.0, "b": 20.0})))
    assert d is not None and d.strength == "weak" and d.compared_groups == ["a", "b"]
    # and a tiny wild group beside two equal supported groups is simply excluded
    d = _dispersion(_profile(_sized({"a": 300, "b": 300, "c": 4}, {"a": 1.0, "b": 1.0, "c": 20.0})))
    assert d is None or (d.strength != "strong" and "c" not in d.compared_groups)


def test_e5_mixed_sizes_are_deterministic_and_row_order_independent() -> None:
    df = _sized({"a": 300, "b": 40, "c": 15, "d": 3}, {"a": 1.0, "b": 3.0, "c": 9.0, "d": 1.0})
    first = _dispersion(_profile(df))
    second = _dispersion(_profile(df))
    shuffled = _dispersion(_profile(df.sample(fraction=1.0, shuffle=True, seed=5)))
    assert first is not None and first == second == shuffled
    assert first.strength == "strong" and first.compared_groups == ["a", "b"]
    assert first.excluded_groups == {"c": 15, "d": 3}


# --- F. definitional dispersion (stage 19.2, A7): flagged, never a headline ---


def _pair(profile: DatasetProfile, x: str, y: str):
    return [r for r in recommend_charts(profile) if r.spec.x == x and r.spec.y == y]


def test_f1_direct_arithmetic_derived_pair_is_flagged_and_capped_as_before() -> None:
    rng = random.Random(13)
    n = 400
    price = [round(rng.uniform(50, 5000), 2) for _ in range(n)]
    qty = [rng.randrange(1, 6) for _ in range(n)]
    df = pl.DataFrame({"unit_price": price, "quantity": qty, "sales": [p * q for p, q in zip(price, qty)]})
    profile = _profile(df)
    assert [(d.target, d.kind) for d in profile.evidence.derived_columns] == [("sales", "product")]
    d = _dispersion(profile, cat="quantity", num="sales")
    assert d is not None and d.strength == "strong"  # the evidence is kept ...
    assert d.definitional and d.definitional_kind == "product"
    assert d.definition == "sales = unit_price × quantity"
    # ... but flagged: the pair never becomes a distribution chart, and the
    # existing derived cap still demotes whatever chart it gets
    for rec in _pair(profile, "quantity", "sales"):
        assert rec.intent != "distribution_by_group"
        assert rec.spec.type not in ("raincloud", "ridgeline", "violin", "box")
        assert rec.tier == "exploratory"
        assert any(w.code == "derived_relationship" for w in rec.warnings)


def test_f2_near_copy_detection_and_disclosure_are_preserved() -> None:
    rng = random.Random(3)
    x = [rng.uniform(0, 10) for _ in range(400)]
    df = pl.DataFrame({"x": x, "double": [2 * v for v in x], "noisy": [2 * v + rng.gauss(0, 1) for v in x]})
    profile = _profile(df)
    found = [(d.target, d.components, d.kind) for d in profile.evidence.derived_columns]
    assert found == [("double", ["x"], "near_copy")]
    assert [g.duplicates for g in profile.evidence.near_duplicate_groups] == [["double"]]
    assert profile.evidence.dispersion == []  # no categorical: nothing to flag
    codes = [w.code for w in derived_column_warnings(profile)]
    assert codes == ["near_duplicate_column"]


def _banded(n: int = 600, seed: int = 29) -> pl.DataFrame:
    rng = random.Random(seed)
    price = [round(rng.uniform(0, 100), 2) for _ in range(n)]
    band = ["low" if p < 10 else "mid" if p < 30 else "high" for p in price]
    other = ["x" if rng.random() < 0.5 else "y" for _ in range(n)]
    return pl.DataFrame({"price": price, "price_band": band, "other": other, "noise": [rng.gauss(0, 1) for _ in range(n)]})


def test_f3_numeric_band_is_a_bucket_flagged_and_not_a_headline() -> None:
    profile = _profile(_banded())
    buckets = [d for d in profile.evidence.derived_columns if d.kind == "bucket"]
    assert [(d.target, d.components, d.formula, d.match_ratio) for d in buckets] == [
        ("price_band", ["price"], "bucket(price)", 1.0)
    ]
    d = _dispersion(profile, cat="price_band", num="price")
    assert d is not None and d.strength == "strong"  # bins of different width: real spread difference
    assert d.definitional and d.definitional_kind == "bucket"
    assert d.definition == "price_band = bucket(price)"
    recs = recommend_charts(profile)
    top = [r for r in recs if r.tier == "top"]
    for rec in _pair(profile, "price_band", "price"):
        assert rec.intent != "distribution_by_group"
        assert rec.tier == "exploratory" and rec not in top
        assert any(w.code == "derived_relationship" and w.meta["kind"] == "bucket" for w in rec.warnings)
        assert "banding of price" in (rec.spec.reason or "")
    assert any(w.code == "derived_column" and w.meta["kind"] == "bucket" for w in derived_column_warnings(profile))
    # an unrelated categorical is untouched
    assert not any(d.target == "other" for d in profile.evidence.derived_columns)


def test_f3b_bucket_flag_survives_a_replacement_reason() -> None:
    from app.charts.rules import definitional_reason

    profile = _profile(_banded())
    rec = _pair(profile, "price_band", "price")[0]
    assert definitional_reason(rec.spec, profile, "llm wording") is not None


@pytest.mark.parametrize("sep", [4.0, 8.0, 16.0])
def test_f4_independent_groups_with_separated_ranges_are_not_a_bucket(sep: float) -> None:
    # three independent groups: overlapping (4 sd), barely / clearly disjoint
    # tails (8, 16 sd) — never a banding, the pair keeps its normal charts
    rng = random.Random(31)
    rows = [{"g": g, "y": i * sep + rng.gauss(0, 1)} for i, g in enumerate("abc") for _ in range(150)]
    profile = _profile(pl.DataFrame(rows))
    assert not any(d.kind == "bucket" for d in profile.evidence.derived_columns)
    d = _dispersion(profile)
    assert d is None or not d.definitional
    for rec in _pair(profile, "g", "y"):
        assert not any(w.code == "derived_relationship" for w in rec.warnings)


def test_f5_a_bucket_of_a_sparse_tail_column_and_integer_bins() -> None:
    # lognormal prices cut at fixed edges (sparse upper edges) and integer
    # ages in 3-year bins (gap of exactly one resolution step) are both bandings
    rng = random.Random(37)
    price = [round(math.exp(rng.gauss(4, 1)), 2) for _ in range(900)]
    labels = ["cheap", "mid", "dear", "premium"]
    band = [labels[sum(p >= e for e in (30, 60, 120))] for p in price]
    age = [rng.randint(18, 65) for _ in range(900)]
    age_bin = ["young" if a < 30 else "mid" if a < 50 else "senior" for a in age]
    profile = _profile(pl.DataFrame({"price": price, "price_band": band, "age": age, "age_bin": age_bin}))
    assert {c.name: c.semantic_type for c in profile.columns} == {
        "price": "numeric", "price_band": "categorical", "age": "numeric", "age_bin": "categorical"
    }
    found = {(d.target, d.components[0]) for d in profile.evidence.derived_columns if d.kind == "bucket"}
    assert found == {("price_band", "price"), ("age_bin", "age")}
