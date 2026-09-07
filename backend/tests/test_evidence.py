import random
from datetime import datetime, timedelta

import polars as pl
import pytest

from app.profiling.evidence import (
    _eligible_cats,
    _interaction_strength,
    _spearman_matrix,
    adjusted_eta_squared,
    compute_evidence,
    span_bucket,
)
from app.profiling.models import Correlations
from app.profiling.profiler import profile_dataset

BIG = 10**6


def _profile(df: pl.DataFrame):
    return profile_dataset(df, "0" * 32, BIG)


# --- adjusted eta-squared ---------------------------------------------------


def test_adjusted_eta_squared_hand_calculated() -> None:
    # groups [1,1,2] vs [5,5,6]: SS_within=4/3, SS_total=76/3, n=6, k=2
    # adjusted = 1 - (SS_w/(n-k)) / (SS_t/(n-1)) = 1 - (1/3)/(76/15) = 0.93421...
    df = pl.DataFrame({"g": ["a", "a", "a", "b", "b", "b"], "v": [1.0, 1.0, 2.0, 5.0, 5.0, 6.0]})
    eta2, n_groups = adjusted_eta_squared(df, "g", "v")
    assert n_groups == 2
    assert eta2 == pytest.approx(1 - (4 / 3 / 4) / (76 / 3 / 5))


def test_adjusted_eta_squared_skips_constant_numeric() -> None:
    df = pl.DataFrame({"g": ["a", "a", "b", "b"], "v": [3.0, 3.0, 3.0, 3.0]})
    assert adjusted_eta_squared(df, "g", "v") is None


def test_adjusted_eta_squared_skips_without_within_df() -> None:
    # one row per group: n - k = 0 -> undefined, must not divide by zero
    df = pl.DataFrame({"g": ["a", "b", "c"], "v": [1.0, 2.0, 3.0]})
    assert adjusted_eta_squared(df, "g", "v") is None


def test_high_cardinality_noise_stays_near_zero() -> None:
    # blocking #1: raw eta2 for 20 categories at n=100 measures ~0.24 on pure
    # noise; the adjusted form must stay near zero
    rng = random.Random(7)
    df = pl.DataFrame(
        {"g": [f"c{i % 20:02d}" for i in range(100)], "v": [rng.gauss(0, 1) for _ in range(100)]}
    )
    eta2, _ = adjusted_eta_squared(df, "g", "v")
    assert eta2 < 0.05


# --- scope caps -------------------------------------------------------------


def test_eligible_cats_capped_at_15_lowest_cardinality() -> None:
    n = 400
    data = {f"c{i:02d}": [f"v{j % (i + 2)}" for j in range(n)] for i in range(18)}
    profile = _profile(pl.DataFrame(data))
    eligible = _eligible_cats(profile.columns)
    assert len(eligible) == 15
    assert max(c.n_categories for c in eligible) <= 16  # the 3 widest were dropped


# --- Spearman ---------------------------------------------------------------


def test_spearman_detects_monotonic_nonlinear() -> None:
    x = [float(i) for i in range(1, 40)]
    df = pl.DataFrame({"x": x, "y": [v**3 for v in x]})
    corr = Correlations(columns=["x", "y"], matrix=[[1.0, 0.0], [0.0, 1.0]])
    matrix = _spearman_matrix(df, corr).matrix
    assert matrix[0][1] == pytest.approx(1.0)
    assert matrix[0][0] == 1.0


def test_spearman_in_profile_aligns_with_pearson_columns() -> None:
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 8.0, 27.0, 64.0], "c": [4.0, 3.0, 2.0, 1.0]})
    profile = _profile(df)
    spearman = profile.evidence.num_num_spearman
    assert spearman is not None
    assert spearman.columns == profile.correlations.columns
    i, j = spearman.columns.index("a"), spearman.columns.index("b")
    assert spearman.matrix[i][j] == pytest.approx(1.0)  # cubic is rank-perfect


# --- interaction strength ---------------------------------------------------


def _factorial_df(effect, n_per_cell: int = 50) -> pl.DataFrame:
    rng = random.Random(11)
    rows = []
    for c1 in ("a", "b"):
        for c2 in ("x", "y"):
            for _ in range(n_per_cell):
                rows.append({"c1": c1, "c2": c2, "v": effect(c1, c2) + rng.gauss(0, 1)})
    return pl.DataFrame(rows)


def test_interaction_additive_is_near_zero() -> None:
    df = _factorial_df(lambda c1, c2: (2.0 if c1 == "a" else 0.0) + (1.0 if c2 == "x" else 0.0))
    assert _interaction_strength(df, "c1", "c2", "v") < 0.02


def test_interaction_crossed_effect_is_detected() -> None:
    df = _factorial_df(lambda c1, c2: 2.0 if (c1 == "a") == (c2 == "x") else -2.0)
    assert _interaction_strength(df, "c1", "c2", "v") > 0.2


def test_interaction_unbalanced_but_additive_stays_low() -> None:
    # critique #6: unbalanced cells must not fake an interaction
    rng = random.Random(13)
    rows = []
    for c1, c2, count in (("a", "x", 200), ("a", "y", 40), ("b", "x", 40), ("b", "y", 200)):
        for _ in range(count):
            value = (2.0 if c1 == "a" else 0.0) + (1.0 if c2 == "x" else 0.0) + rng.gauss(0, 1)
            rows.append({"c1": c1, "c2": c2, "v": value})
    assert _interaction_strength(pl.DataFrame(rows), "c1", "c2", "v") < 0.05


def test_interaction_sparse_cells_skipped() -> None:
    df = pl.DataFrame({"c1": ["a", "a", "b", "b"], "c2": ["x", "y", "x", "y"], "v": [1.0, 2.0, 3.0, 4.0]})
    assert _interaction_strength(df, "c1", "c2", "v") is None  # every cell < 5 rows


# --- slope heterogeneity ----------------------------------------------------


def test_slope_heterogeneity_only_scans_correlated_pairs() -> None:
    # perfectly opposite slopes cancel to |corr| ~0 overall, so the pair never
    # enters the scan scope (it mirrors what scatter would consider)
    rng = random.Random(17)
    rows = []
    for i in range(600):
        x = rng.uniform(0, 10)
        g = "A" if i % 2 == 0 else "B"
        rows.append({"x": x, "y": (x if g == "A" else -x) + rng.gauss(0, 0.5), "g": g})
    profile = _profile(pl.DataFrame(rows))
    assert profile.evidence.slope_heterogeneity == []


def test_slope_heterogeneity_flat_group_detected() -> None:
    rng = random.Random(19)
    rows = []
    for i in range(600):
        x = rng.uniform(0, 10)
        g = "A" if i % 2 == 0 else "B"
        y = x + rng.gauss(0, 0.3) if g == "A" else rng.gauss(5, 3)
        rows.append({"x": x, "y": y, "g": g})
    profile = _profile(pl.DataFrame(rows))
    (het,) = [h for h in profile.evidence.slope_heterogeneity if h.group == "g"]
    assert set(het.corrs) == {"A", "B"}  # stringified keys
    assert het.corrs["A"] > 0.9 and abs(het.corrs["B"]) < 0.3
    assert het.spread > 0.7
    assert het.n_min >= 290


def test_slope_heterogeneity_small_groups_excluded() -> None:
    rng = random.Random(23)
    n = 600
    rows = []
    for i in range(n):
        x = rng.uniform(0, 10)
        # 25 tiny groups of ~24 rows each -> below the 30-row floor
        rows.append({"x": x, "y": 2 * x + rng.gauss(0, 1), "g": f"g{i % 25}"})
    profile = _profile(pl.DataFrame(rows))
    assert profile.evidence.slope_heterogeneity == []


# --- time effects -----------------------------------------------------------


@pytest.mark.parametrize(
    "span_days,expected",
    [(60, "day"), (90, "day"), (400, "month"), (3 * 365, "month"), (5 * 365, "year")],
)
def test_span_bucket(span_days: float, expected: str) -> None:
    assert span_bucket(span_days) == expected


def test_time_effect_buckets_and_strength() -> None:
    rng = random.Random(29)
    n = 600
    days = [i % 60 for i in range(n)]  # 10 observations per day over 60 days
    df = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 1) + timedelta(days=d) for d in days],
            "trend": [d * 0.05 + rng.gauss(0, 1) for d in days],
            "noise": [rng.gauss(0, 1) for _ in range(n)],
        }
    )
    effects = {t.num: t for t in _profile(df).evidence.time_effects}
    assert effects["trend"].bucket == "day"  # 60-day span stays fine-grained
    assert effects["trend"].eta_squared > 0.2
    assert effects["noise"].eta_squared < 0.05


def test_time_effect_coarsens_when_daily_has_no_df() -> None:
    # one observation per day: the day bucket has no within-group df, so the
    # computation falls back to month (critique #5's degenerate edge)
    df = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(60)],
            "v": [float(i) for i in range(60)],
        }
    )
    (effect,) = _profile(df).evidence.time_effects
    assert effect.bucket == "month"
    assert effect.eta_squared > 0.5  # strong trend across the two months


def test_evidence_survives_json_roundtrip() -> None:
    rng = random.Random(31)
    df = pl.DataFrame(
        {
            "g": ["a", "b"] * 100,
            "x": [float(i) for i in range(200)],
            "y": [2.0 * i + rng.gauss(0, 1) for i in range(200)],
        }
    )
    profile = _profile(df)
    from app.profiling.models import DatasetProfile

    reloaded = DatasetProfile.model_validate_json(profile.model_dump_json())
    assert reloaded.evidence == profile.evidence
