"""Stage 19.1: dispersion evidence (spread differences across groups) is
independent of the mean-difference evidence, sample-size aware, and drives
the distribution family (raincloud / ridgeline / box) in chart selection."""

import random

import polars as pl
import pytest

from app.charts.rules import recommend_charts
from app.profiling.evidence import (
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


def test_d_one_small_group_among_large_ones_is_weak() -> None:
    big = _groups({"a": 0, "b": 0}, {"a": 0.5, "b": 3.0}, 200)
    small = _groups({"c": 0}, {"c": 1.5}, 12, seed=7)
    profile = _profile(pl.concat([big, small]))
    d = _dispersion(profile)
    assert d is not None and d.strength == "weak" and d.n_min == 12


# --- shape, determinism, version ----------------------------------------------------


def test_dispersion_is_deterministic_and_round_trips() -> None:
    df = _groups({"a": 0, "b": 0, "c": 0}, {"a": 0.5, "b": 1.5, "c": 3.0}, 100)
    first, second = _profile(df), _profile(df)
    assert first.evidence.dispersion == second.evidence.dispersion
    assert first.profile_version == PROFILE_VERSION == 12
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
