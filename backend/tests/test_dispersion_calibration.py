"""Stage 19.2 (A4): seeded false-positive / power calibration of the
dispersion evidence. Replicates are packed as numeric columns of one frame
(the evidence pass batches every num per cat), so a whole grid runs in
seconds. The strong entry needs the Brown–Forsythe F test at
DISPERSION_ALPHA on top of the effect-size gates; the bounds below have a
little slack over the measured rates (stage 19.1 gates alone: null strong
rate 7.5 / 12.5 / 10.7 % at N=30 for k = 2 / 3 / 6, outlier case 18 %)."""

import random

import polars as pl
import pytest

from app.profiling.evidence import DISPERSION_ALPHA, _dispersion_effects, f_survival
from app.profiling.models import ColumnProfile

REPS = 200


def _col(name: str, kind: str, n_categories: int | None = None) -> ColumnProfile:
    return ColumnProfile(
        name=name, original_dtype="Float64", semantic_type=kind, missing_count=0, missing_ratio=0.0,
        unique_count=10, n_categories=n_categories,
    )


def _frame(n: int, sds: list[float], seed: int, mean_shift: float = 0.0, outliers: int = 0) -> pl.DataFrame:
    rng = random.Random(seed)
    data: dict[str, list] = {"g": [f"g{i}" for i in range(len(sds)) for _ in range(n)]}
    for r in range(REPS):
        values: list[float] = []
        for i, sd in enumerate(sds):
            values += [i * mean_shift + rng.gauss(0, sd) for _ in range(n)]
        for j in range(outliers):  # a few extreme values in the first group, same core spread
            values[j] = 50.0 if j % 2 else -50.0
        data[f"y{r}"] = values
    return pl.DataFrame(data)


def _strong_rate(n: int, sds: list[float], seed: int, **kw) -> float:
    df = _frame(n, sds, seed, **kw)
    effects, _ = _dispersion_effects(df, [_col("g", "categorical", len(sds))], [_col(f"y{r}", "numeric") for r in range(REPS)], [])
    return sum(1 for e in effects if e.strength == "strong") / REPS


# --- the F distribution (no scipy): tabulated 5 % critical values --------------


@pytest.mark.parametrize(
    "f, d1, d2", [(4.001, 1, 60), (3.150, 2, 60), (2.290, 5, 120), (4.965, 1, 10), (2.603, 5, 25), (3.101, 2, 87)]
)
def test_f_survival_matches_tabulated_critical_values(f: float, d1: int, d2: int) -> None:
    assert abs(f_survival(f, d1, d2) - 0.05) < 5e-4


def test_f_survival_edges() -> None:
    assert f_survival(0.0, 2, 50) == 1.0 and f_survival(-1.0, 2, 50) == 1.0
    assert f_survival(float("inf"), 2, 50) == 0.0
    assert f_survival(1000.0, 2, 500) < 1e-12
    assert 0.39 < f_survival(1.0, 3, 50) < 0.41


# --- null: identical distributions ------------------------------------------------


@pytest.mark.parametrize("k", [2, 3, 6])
@pytest.mark.parametrize("n", [30, 50, 100])
def test_null_strong_rate_is_bounded_by_alpha_with_slack(n: int, k: int) -> None:
    rate = _strong_rate(n, [1.0] * k, seed=1000 + 10 * k + n)
    assert rate <= DISPERSION_ALPHA + 0.025, f"N={n} k={k}: {rate}"  # measured 0.5-4.8 %


@pytest.mark.parametrize("k", [2, 3, 6])
def test_null_below_the_supported_size_is_never_strong(k: int) -> None:
    assert _strong_rate(20, [1.0] * k, seed=1020 + k) == 0.0


# --- power: same mean, different sd ------------------------------------------------


@pytest.mark.parametrize(
    "n, ratio, floor", [(30, 2.0, 0.6), (100, 2.0, 0.85), (30, 5.0, 0.97), (100, 5.0, 0.97), (30, 20.0, 0.97), (100, 20.0, 0.97)]
)
def test_power_true_spread_differences_are_found(n: int, ratio: float, floor: float) -> None:
    rate = _strong_rate(n, [1.0, ratio], seed=2000 + n + int(ratio))
    assert rate >= floor, f"N={n} sd 1 vs {ratio}: {rate}"  # measured 0.78 / 0.955 / 1.0 ...


# --- robustness ------------------------------------------------------------------------


@pytest.mark.parametrize("n", [30, 100])
def test_mean_shift_with_equal_spread_is_not_a_spread_difference(n: int) -> None:
    rate = _strong_rate(n, [1.0, 1.0, 1.0], seed=3000 + n, mean_shift=3.0)
    assert rate <= DISPERSION_ALPHA + 0.025, f"N={n}: {rate}"  # the median centring holds


@pytest.mark.parametrize("n", [30, 100])
def test_a_few_extreme_outliers_do_not_fake_a_spread_difference(n: int) -> None:
    # three values at ±50 in one group of unit-sd data: the robust ratio (IQR)
    # is unmoved and the median-centred F test is not fooled (was 18 % at N=30)
    rate = _strong_rate(n, [1.0, 1.0], seed=4000 + n, outliers=3)
    assert rate <= 0.02, f"N={n}: {rate}"
