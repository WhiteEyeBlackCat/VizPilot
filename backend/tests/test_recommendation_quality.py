"""Planted-signal evals (stage 7): "is this recommendation sensible" turned
into assertions. All datasets are synthetic with n >= 500 and fixed seeds;
planted effects are stated in sigma units with the implied eta noted.
"""

import math
import random
from datetime import datetime, timedelta

import polars as pl

from app.charts.rules import Recommendation, recommend_charts
from app.profiling.profiler import profile_dataset

BIG = 10**6


def _recs(df: pl.DataFrame) -> list[Recommendation]:
    return recommend_charts(profile_dataset(df, "0" * 32, BIG))


def _find(recs, type_, x=None, y=None):
    for rec in recs:
        spec = rec.spec
        if spec.type == type_ and (x is None or spec.x == x) and (y is None or spec.y == y):
            return rec
    return None


# --- case 1: strong main effect vs pure-noise categorical -------------------


def _main_effect_df() -> pl.DataFrame:
    # region offsets {-1.5, 0, +1.5} sigma -> between-var 1.5, within-var 1
    # -> eta2 ~= 0.6, eta ~= 0.77, bar score ~= 0.45 + 0.45*0.77 = 0.80
    rng = random.Random(101)
    offsets = {"North": -1.5, "Central": 0.0, "South": 1.5}
    rows = []
    for i in range(600):
        region = ["North", "Central", "South"][i % 3]
        rows.append(
            {
                "region": region,
                "noise_cat": ["p", "q", "r"][rng.randrange(3)],
                "sales": offsets[region] + rng.gauss(0, 1),
            }
        )
    return pl.DataFrame(rows)


def test_strong_main_effect_reaches_top_and_noise_does_not() -> None:
    recs = _recs(_main_effect_df())
    signal = _find(recs, "bar", x="region", y="sales")
    assert signal is not None and signal.tier == "top"
    for rec in recs:
        if "noise_cat" in (rec.spec.x, rec.spec.y, rec.spec.group_by):
            assert rec.tier != "top"


def test_pure_noise_dataset_mints_no_top_charts() -> None:
    # critique #4's false-positive guard: nothing here deserves "top"
    rng = random.Random(103)
    df = pl.DataFrame(
        {
            "cat": [f"c{rng.randrange(4)}" for _ in range(600)],
            "value": [rng.gauss(0, 1) for _ in range(600)],
        }
    )
    assert all(rec.tier != "top" for rec in _recs(df))


def test_pure_noise_multi_numeric_mints_no_top_charts() -> None:
    # heatmap (fixed 0.65) and weak scatters must not clear the top floor
    # on a dataset that is all noise (stage7 review blocking #2)
    rng = random.Random(104)
    df = pl.DataFrame(
        {f"v{k}": [rng.gauss(0, 1) for _ in range(600)] for k in range(4)}
        | {"cat": [f"c{rng.randrange(4)}" for _ in range(600)]}
    )
    recs = _recs(df)
    assert any(rec.spec.type == "heatmap" for rec in recs)
    assert all(rec.tier != "top" for rec in recs)


def test_one_variable_cannot_monopolize_a_type() -> None:
    # sales_basic regression (stage7 review blocking #1): quantity-like column
    # with the strongest effect must not fill every bar slot; the next-best
    # distinct x column gets the remaining one
    rng = random.Random(105)
    rows = 800
    quantity = [rng.randrange(1, 11) for _ in range(rows)]
    region = [rng.choice(["North", "South", "East", "West"]) for _ in range(rows)]
    region_bump = {"North": 0.6, "South": 0.0, "East": -0.3, "West": 0.2}
    df = pl.DataFrame(
        {
            "quantity": quantity,
            "region": region,
            "sales": [q * rng.uniform(50, 150) for q in quantity],
            "profit": [q * rng.uniform(5, 15) + region_bump[r] * 40 for q, r in zip(quantity, region)],
        }
    )
    recs = _recs(df)
    bars = [rec for rec in recs if rec.spec.type == "bar"]
    xs = [rec.spec.x for rec in bars]
    assert len(bars) >= 3
    assert any(x == "region" for x in xs), xs  # second variable keeps a slot
    assert max(xs.count(x) for x in set(xs)) <= 2


# --- case 2: stronger effect ranks first ------------------------------------


def test_stronger_effect_outranks_weaker() -> None:
    # strong: +-1.5 sigma (eta ~0.77); weak: +-0.4 sigma (eta2 0.16/1.16 ~ eta 0.37)
    rng = random.Random(107)
    rows = []
    for i in range(600):
        strong = "s1" if i % 2 == 0 else "s2"
        weak = "w1" if rng.random() < 0.5 else "w2"
        value = (1.5 if strong == "s1" else -1.5) + (0.4 if weak == "w1" else -0.4) + rng.gauss(0, 1)
        rows.append({"cat_strong": strong, "cat_weak": weak, "value": value})
    recs = _recs(pl.DataFrame(rows))
    strong_bar = _find(recs, "bar", x="cat_strong", y="value")
    weak_bar = _find(recs, "bar", x="cat_weak", y="value")
    assert strong_bar is not None and weak_bar is not None
    assert strong_bar.spec.priority < weak_bar.spec.priority


# --- case 3: interaction earns group_by, its absence removes it -------------


def _time_group_df(interaction: bool) -> pl.DataFrame:
    # main effect of g is cancelled; only the (late-half x g=A) cell moves.
    # interaction share ~ 0.25n / 1.5n ~= 0.17 >> 0.05 threshold
    rng = random.Random(109)
    rows = []
    for i in range(720):
        day = i % 360
        g = "A" if i % 2 == 0 else "B"
        late = day >= 180
        if interaction:
            effect = (2.0 if (late and g == "A") else 0.0) - (1.0 if g == "A" else 0.0)
        else:
            effect = 2.0 if late else 0.0  # same trend for both groups
        rows.append(
            {
                "ts": datetime(2024, 1, 1) + timedelta(days=day),
                "g": g,
                "y": effect + rng.gauss(0, 1),
            }
        )
    return pl.DataFrame(rows)


def test_interaction_mounts_line_group_by() -> None:
    recs = _recs(_time_group_df(interaction=True))
    line = _find(recs, "line", x="ts", y="y")
    assert line is not None and line.spec.group_by == "g"


def test_no_interaction_no_group_by() -> None:
    recs = _recs(_time_group_df(interaction=False))
    line = _find(recs, "line", x="ts", y="y")
    assert line is not None and line.spec.group_by is None


def _slope_df(heterogeneous: bool) -> pl.DataFrame:
    rng = random.Random(113)
    rows = []
    for i in range(600):
        x = rng.uniform(0, 10)
        g = "A" if i % 2 == 0 else "B"
        if heterogeneous:
            y = x + rng.gauss(0, 0.3) if g == "A" else rng.gauss(5, 3)  # corr 1 vs ~0
        else:
            y = x + rng.gauss(0, 1)  # identical relationship in both groups
        rows.append({"x": x, "y": y, "g": g})
    return pl.DataFrame(rows)


def test_slope_heterogeneity_mounts_scatter_group_by() -> None:
    recs = _recs(_slope_df(heterogeneous=True))
    scatter = _find(recs, "scatter")
    assert scatter is not None and scatter.spec.group_by == "g"


def test_homogeneous_slopes_no_group_by() -> None:
    # critique #4: near-identical group correlations must not fake a split
    recs = _recs(_slope_df(heterogeneous=False))
    scatter = _find(recs, "scatter")
    assert scatter is not None and scatter.spec.group_by is None


# --- case 4: monotonic non-linear relationships -----------------------------


def test_monotonic_quadratic_still_recommended() -> None:
    # y = x^2 on positive x: Pearson ~0.97, Spearman 1.0 (blocking #3);
    # non-monotonic U-shapes are explicitly out of scope
    rng = random.Random(127)
    x = [rng.uniform(0.5, 3.0) for _ in range(600)]
    df = pl.DataFrame({"x": x, "y": [v**2 for v in x]})
    scatter = _find(_recs(df), "scatter")
    assert scatter is not None
    assert {scatter.spec.x, scatter.spec.y} == {"x", "y"}
    assert scatter.score > 0.85


def test_exponential_gets_nonlinear_note() -> None:
    # exp(x) drags Pearson well below Spearman (gap > 0.15) -> reason notes it
    rng = random.Random(131)
    x = [rng.uniform(0.0, 6.0) for _ in range(600)]
    df = pl.DataFrame({"x": x, "y": [math.exp(v) for v in x]})
    scatter = _find(_recs(df), "scatter")
    assert scatter is not None
    assert "non-linear" in scatter.spec.reason
    assert scatter.score > 0.85  # scored on the Spearman channel


# --- case 5: the sales_basic lexicographic-bias regression ------------------


def test_lexicographic_bias_regression() -> None:
    # region/category share cardinality; region moves sales (offsets spanning
    # 2.4 sigma -> eta2 ~0.44, eta ~0.67, bar ~0.75), discount is pure noise
    rng = random.Random(137)
    offsets = {"North": -1.2, "East": -0.4, "South": 0.4, "West": 1.2}
    rows = []
    for i in range(600):
        region = list(offsets)[i % 4]
        rows.append(
            {
                "region": region,
                "category": ["Electronics", "Clothing", "Food", "Home"][rng.randrange(4)],
                "sales": offsets[region] + rng.gauss(0, 1),
                "discount": rng.gauss(0, 1),
            }
        )
    recs = _recs(pl.DataFrame(rows))

    region_sales = _find(recs, "bar", x="region", y="sales")
    assert region_sales is not None and region_sales.tier == "top"
    for rec in recs:
        if "discount" in (rec.spec.x, rec.spec.y):
            assert region_sales.spec.priority < rec.spec.priority
    bars = [rec for rec in recs if rec.spec.type == "bar"]
    assert any(rec.spec.x != "category" for rec in bars)  # category no longer monopolizes


# --- critique #5: short-span daily series must not sink ---------------------


def test_short_span_daily_trend_ranks_top() -> None:
    # 60-day span -> "day" bucket, 10 rows per day; trend spans 3 sigma
    # (between-day var ~0.75 -> eta2 ~0.43 -> line ~0.5+0.3*0.65+0.05 = 0.75)
    rng = random.Random(139)
    days = [i % 60 for i in range(600)]
    df = pl.DataFrame(
        {
            "ts": [datetime(2024, 3, 1) + timedelta(days=d) for d in days],
            "kpi": [d * 0.05 + rng.gauss(0, 1) for d in days],
            "noise": [rng.gauss(0, 1) for _ in range(600)],
        }
    )
    recs = _recs(df)
    line = _find(recs, "line", x="ts", y="kpi")
    assert line is not None and line.tier == "top"
    assert line.spec.priority == 1
    noise_line = _find(recs, "line", x="ts", y="noise")
    assert noise_line is not None and noise_line.tier != "top"


# --- determinism ------------------------------------------------------------


def test_recommendations_deterministic() -> None:
    df = _main_effect_df()
    first = [rec.model_dump() for rec in _recs(df)]
    second = [rec.model_dump() for rec in _recs(df)]
    assert first == second
