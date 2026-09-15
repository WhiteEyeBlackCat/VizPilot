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


# --- stage 9: batched effect sizes ------------------------------------------


def _reference_eta(df: pl.DataFrame, cat: str, num: str):
    """Pre-stage-9 single-pair adjusted eta-squared, kept as the oracle."""
    data = df.select(cat, num).drop_nulls()
    if data.schema[num].is_float():
        data = data.filter(pl.col(num).is_finite())
    n, k = data.height, data[cat].n_unique()
    if k < 2 or n - k < 2:
        return None
    grand = data[num].mean()
    ss_total = ((data[num] - grand) ** 2).sum()
    if ss_total is None or ss_total <= 0:
        return None
    ss_within = (
        data.group_by(cat, maintain_order=True)
        .agg(((pl.col(num) - pl.col(num).mean()) ** 2).sum().alias("ss"))["ss"]
        .sum()
    )
    return max(float(1 - (ss_within / (n - k)) / (ss_total / (n - 1))), 0.0), k


def _dirty_group_df() -> pl.DataFrame:
    rng = random.Random(41)
    n = 400
    cat = [rng.choice(["a", "b", "c", "d", None]) if i % 50 else None for i in range(n)]
    cat = [c if i % 97 else None for i, c in enumerate(cat)]

    def noisy(base):
        out = []
        for i in range(n):
            r = rng.random()
            if r < 0.05:
                out.append(None)
            elif r < 0.08:
                out.append(float("nan"))
            elif r < 0.1:
                out.append(float("-inf"))
            else:
                out.append(base(i) + rng.gauss(0, 1))
        return out

    return pl.DataFrame(
        {
            "g": cat,
            "v": noisy(lambda i: {"a": 0, "b": 3, "c": -2, "d": 1, None: 0}[cat[i]]),
            "noise": noisy(lambda i: 0.0),
            "ints": [i % 13 if i % 11 else None for i in range(n)],
            "const": [7.0] * n,
            "thin": [float(i) if i < 3 else None for i in range(n)],  # n - k < 2
        }
    )


def test_adjusted_eta_squared_many_matches_single_pair() -> None:
    from app.profiling.evidence import adjusted_eta_squared_many

    df = _dirty_group_df()
    nums = ["v", "noise", "ints", "const", "thin"]
    batched = adjusted_eta_squared_many(df, "g", nums)
    for num in nums:
        expected = _reference_eta(df, "g", num)
        wrapper = adjusted_eta_squared(df, "g", num)
        if expected is None:
            assert batched[num] is None and wrapper is None
        else:
            assert batched[num][0] == pytest.approx(expected[0], abs=1e-12)
            assert batched[num][1] == expected[1]
            assert wrapper == (batched[num][0], batched[num][1])
    assert batched["const"] is None and batched["thin"] is None


def test_adjusted_eta_squared_many_reports_sample_sizes() -> None:
    from app.profiling.evidence import adjusted_eta_squared_many

    df = pl.DataFrame(
        {
            "g": ["a", "a", "a", "b", "b", None, "c", "c"],
            "v": [1.0, 2.0, None, 5.0, 6.0, 9.0, float("nan"), 3.0],
        }
    )
    (result,) = adjusted_eta_squared_many(df, "g", ["v"]).values()
    assert result.n_groups == 3 and result.n_total == 5 and result.n_min == 1  # c keeps one finite row
    assert result.group_counts == {"a": 2, "b": 2, "c": 1}
    assert result.eta_squared == pytest.approx(_reference_eta(df, "g", "v")[0])


def test_cat_num_and_time_effects_carry_counts() -> None:
    rng = random.Random(43)
    n = 300
    df = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 1) + timedelta(days=i % 30) for i in range(n)],
            "g": ["x", "y", "z"][0:1] * 0 + [["x", "y", "z"][i % 3] if i % 10 else None for i in range(n)],
            "v": [rng.gauss(0, 1) if i % 17 else None for i in range(n)],
        }
    )
    profile = _profile(df)
    (effect,) = [e for e in profile.evidence.cat_num if e.cat == "g" and e.num == "v"]
    valid = df.filter(pl.col("g").is_not_null() & pl.col("v").is_not_null())
    sizes = valid.group_by("g").len()["len"]
    assert effect.n_total == valid.height and effect.n_min == sizes.min() and effect.n_groups == 3
    (time_effect,) = profile.evidence.time_effects
    buckets = df.filter(pl.col("v").is_not_null()).group_by("ts").len()["len"]
    assert time_effect.bucket == "day"
    assert time_effect.n_total == df["v"].drop_nulls().len() and time_effect.n_min == buckets.min()
    assert profile.profiled_rows == n


def test_time_effect_coarsening_keeps_num_order_and_counts() -> None:
    # "daily" has one row per day (no within-df) while "dense" has ten: the
    # first must fall back to month, the second stay at day, and the output
    # order must follow the numeric column order regardless of bucket
    rng = random.Random(47)
    days = [i % 60 for i in range(600)]
    df = pl.DataFrame(
        {
            "ts": [datetime(2024, 1, 1) + timedelta(days=d) for d in days],
            "daily": [float(d) if i < 60 else None for i, d in enumerate(days)],
            "dense": [d * 0.05 + rng.gauss(0, 1) for d in days],
        }
    )
    effects = _profile(df).evidence.time_effects
    assert [e.num for e in effects] == ["daily", "dense"]
    assert effects[0].bucket == "month" and effects[0].n_total == 60
    assert effects[1].bucket == "day" and effects[1].n_total == 600 and effects[1].n_min == 10


def test_interaction_effects_carry_row_counts() -> None:
    df = _factorial_df(lambda c1, c2: 2.0 if (c1 == "a") == (c2 == "x") else -2.0)
    profile = _profile(df)
    (effect,) = [e for e in profile.evidence.interactions if {e.cat1, e.cat2} == {"c1", "c2"}]
    assert effect.n_total == 200 and effect.strength > 0.2


def test_spearman_matrix_carries_pair_counts() -> None:
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, None], "b": [1.0, 8.0, 27.0, 64.0], "c": [4.0, 3.0, 2.0, 1.0]})
    spearman = _profile(df).evidence.num_num_spearman
    i, j = spearman.columns.index("a"), spearman.columns.index("b")
    assert spearman.pair_counts[i][j] == 3 and spearman.pair_counts[j][j] == 4


def test_evidence_uses_batched_executions(monkeypatch: pytest.MonkeyPatch) -> None:
    # 2 cats x 12 nums used to be 24 per-pair executions for cat_num alone;
    # now each cat costs a fixed 3 (filter/select, grand means, group_by)
    from app.profiling.evidence import _cat_num_effects

    rng = random.Random(53)
    n = 200
    data = {f"n{k}": [rng.gauss(0, 1) for _ in range(n)] for k in range(12)}
    data["c1"] = [["a", "b", "c"][i % 3] for i in range(n)]
    data["c2"] = [["p", "q"][i % 2] for i in range(n)]
    profile = _profile(pl.DataFrame(data))
    cats = [c for c in profile.columns if c.semantic_type == "categorical"]
    nums = [c for c in profile.columns if c.semantic_type == "numeric"]
    calls = {"n": 0}
    real = pl.DataFrame.group_by

    def counting(self, *args, **kwargs):
        calls["n"] += 1
        return real(self, *args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "group_by", counting)
    effects = _cat_num_effects(pl.DataFrame(data), cats, nums)
    assert len(effects) == 24
    assert calls["n"] == len(cats)


# --- stage 9 (stage 2): per-group counts ------------------------------------


@pytest.mark.parametrize(
    "keys,expected",
    [
        ([1, 2, 3], {"1", "2", "3"}),
        ([True, False, True], {"True", "False"}),
        (
            [datetime(2024, 1, 1).date(), datetime(2024, 2, 1).date(), datetime(2024, 1, 1).date()],
            {"2024-01-01", "2024-02-01"},
        ),
    ],
    ids=["int", "bool", "date"],
)
def test_group_counts_keys_are_stringified_like_render_labels(keys, expected) -> None:
    from app.profiling.evidence import adjusted_eta_squared_many

    rng = random.Random(59)
    n = 90
    df = pl.DataFrame(
        {
            "g": [keys[i % len(keys)] for i in range(n)],
            "v": [keys.index(keys[i % len(keys)]) * 3.0 + rng.gauss(0, 1) for i in range(n)],
        }
    )
    result = adjusted_eta_squared_many(df, "g", ["v"])["v"]
    assert set(result.group_counts) == expected
    assert sum(result.group_counts.values()) == result.n_total == n
    assert min(result.group_counts.values()) == result.n_min
    assert len(result.group_counts) == result.n_groups


def test_group_counts_exclude_groups_without_valid_rows() -> None:
    from app.profiling.evidence import adjusted_eta_squared_many

    df = pl.DataFrame(
        {
            "g": ["a", "a", "a", "b", "b", "c", "c", None, "d"],
            "v": [1.0, 2.0, 4.0, 5.0, None, float("nan"), float("inf"), 9.0, 7.0],
        }
    )
    result = adjusted_eta_squared_many(df, "g", ["v"])["v"]
    # c has no finite row and the null-keyed row is dropped: both absent
    assert result.group_counts == {"a": 3, "b": 1, "d": 1}
    assert result.n_groups == 3 and result.n_total == 5 and result.n_min == 1


def test_profile_cat_num_group_counts_and_json_roundtrip() -> None:
    from app.profiling.models import DatasetProfile

    rng = random.Random(61)
    n = 300
    df = pl.DataFrame(
        {
            "flag": [i % 2 == 0 for i in range(n)],
            "rank": [i % 4 for i in range(n)],  # Int64 categorical
            "v": [rng.gauss(0, 1) + (i % 2) if i % 9 else None for i in range(n)],
        }
    )
    profile = _profile(df)
    by_cat = {e.cat: e for e in profile.evidence.cat_num if e.num == "v"}
    valid = df.filter(pl.col("v").is_not_null())
    assert by_cat["flag"].group_counts == {
        str(k): c for k, c in valid.group_by("flag").len().iter_rows()
    }
    assert set(by_cat["rank"].group_counts) == {"0", "1", "2", "3"}
    for effect in by_cat.values():
        assert sum(effect.group_counts.values()) == effect.n_total
        assert min(effect.group_counts.values()) == effect.n_min
    reloaded = DatasetProfile.model_validate_json(profile.model_dump_json())
    assert reloaded.evidence == profile.evidence


# --- derived columns (stage 13) ---------------------------------------------

from app.profiling.evidence import (  # noqa: E402
    DERIVED_MAX_COLUMNS,
    derived_candidates,
    derived_columns,
)


def _derived(df: pl.DataFrame):
    return _profile(df).evidence.derived_columns


def _by_target(found):
    return {d.target: d for d in found}


def _sales_like(n: int = 400, seed: int = 13) -> tuple[random.Random, dict]:
    rng = random.Random(seed)
    price = [round(rng.uniform(50, 5000), 2) for _ in range(n)]
    qty = [rng.randrange(1, 11) for _ in range(n)]  # 10 codes -> numeric-backed categorical
    disc = [rng.choice([0.0, 0.05, 0.1, 0.2]) for _ in range(n)]
    return rng, {"unit_price": price, "quantity": qty, "discount": disc}


def test_derived_product_discount_identity_with_categorical_quantity() -> None:
    rng, cols = _sales_like()
    sales = [round(p * q * (1 - d), 2) for p, q, d in zip(cols["unit_price"], cols["quantity"], cols["discount"])]
    df = pl.DataFrame({**cols, "sales": sales, "noise": [rng.gauss(0, 1) for _ in range(400)]})
    profile = _profile(df)
    assert next(c for c in profile.columns if c.name == "quantity").semantic_type == "categorical"
    found = profile.evidence.derived_columns
    assert [(d.target, d.formula, d.kind) for d in found] == [
        ("sales", "unit_price × quantity × (1 − discount)", "product_discount")
    ]
    assert found[0].components == ["unit_price", "quantity", "discount"]
    assert found[0].match_ratio == 1.0 and found[0].n == 400


def test_derived_plain_product_preferred_over_zero_discount_form() -> None:
    # z is all zeros (inside [0, 1]) so a × b × (1 − z) matches too; the
    # fewer-components form wins the tie
    rng, cols = _sales_like()
    total = [p * q for p, q in zip(cols["unit_price"], cols["quantity"])]
    df = pl.DataFrame({"a": cols["unit_price"], "b": [float(q) for q in cols["quantity"]], "z": [0.0] * 400, "t": total})
    found = _by_target(_derived(df))
    assert found["t"].formula == "a × b" and found["t"].kind == "product"


def test_derived_sum_difference_ratio_forms_and_rearrangement_dedup() -> None:
    rng = random.Random(5)
    a = [rng.uniform(1, 100) for _ in range(300)]
    b = [rng.uniform(1, 100) for _ in range(300)]
    df = pl.DataFrame(
        {
            "a": a,
            "b": b,
            "total": [x + y for x, y in zip(a, b)],
            "gap": [x - y for x, y in zip(a, b)],
            "share": [x / y for x, y in zip(a, b)],
        }
    )
    found = _derived(df)
    by_target = _by_target(found)
    assert by_target["total"].formula == "a + b" and by_target["total"].kind == "sum"
    assert by_target["gap"].formula == "a − b" and by_target["gap"].kind == "difference"
    assert by_target["share"].formula == "a / b" and by_target["share"].kind == "ratio"
    # each identity is reported once: a = total − b, b = total − a etc. collapse
    # onto the last-in-column-order target (total / gap / share, not a or b)
    assert {d.target for d in found} == {"total", "gap", "share"}
    assert len(found) == 3


def test_derived_requires_99_percent_and_rejects_noisy_relationships() -> None:
    rng, cols = _sales_like()
    exact = [round(p * q * (1 - d), 2) for p, q, d in zip(cols["unit_price"], cols["quantity"], cols["discount"])]
    # 5% of rows perturbed -> 95% match, below the 99% floor
    perturbed = [v * 1.5 if i % 20 == 0 else v for i, v in enumerate(exact)]
    df = pl.DataFrame({**cols, "sales": perturbed})
    assert _derived(df) == []
    # a genuinely strong but noisy relationship (r ~ 0.7) is a finding, not a formula
    x = [rng.uniform(0, 10) for _ in range(400)]
    y = [v + rng.gauss(0, 3) for v in x]
    assert _derived(pl.DataFrame({"x": x, "y": y})) == []


def test_derived_tolerates_rounding_nulls_and_non_finite_rows() -> None:
    rng, cols = _sales_like(n=300)
    sales = [round(p * q, 2) for p, q in zip(cols["unit_price"], cols["quantity"])]
    price = list(cols["unit_price"])
    price[0], price[1], price[2] = None, float("nan"), float("inf")  # skipped, not failures
    sales[3] = None
    df = pl.DataFrame({"unit_price": price, "quantity": cols["quantity"], "sales": sales})
    found = _by_target(_derived(df))
    assert found["sales"].formula == "unit_price × quantity"
    assert found["sales"].match_ratio == 1.0
    assert found["sales"].n == 300 - 4  # only fully finite rows count


def test_derived_integer_identity_and_candidate_filtering() -> None:
    rng = random.Random(9)
    n = 300
    a = [rng.randrange(1, 1000) for _ in range(n)]
    b = [rng.randrange(1, 1000) for _ in range(n)]
    df = pl.DataFrame(
        {
            "order_id": list(range(1, n + 1)),  # identifier: never a candidate
            "a": a,
            "b": b,
            "total": [x + y for x, y in zip(a, b)],
            "flag": [i % 2 == 0 for i in range(n)],  # boolean: never a candidate
            "constant": [7.0] * n,  # constant: never a candidate
            "label": [f"L{i}" for i in range(n)],
        }
    )
    profile = _profile(df)
    assert derived_candidates(profile.columns) == ["a", "b", "total"]
    found = _by_target(profile.evidence.derived_columns)
    assert found["total"].formula == "a + b" and found["total"].match_ratio == 1.0


def test_near_copy_flagged_only_for_transformed_duplicates() -> None:
    rng = random.Random(3)
    x = [rng.uniform(0, 10) for _ in range(400)]
    df = pl.DataFrame({"x": x, "double": [2 * v for v in x], "noisy": [2 * v + rng.gauss(0, 1) for v in x]})
    found = _derived(df)
    copies = [(d.target, d.components, d.kind) for d in found if d.kind == "near_copy"]
    # double ranks identically to x (rho = 1); noisy (r ~ 0.985) does not qualify
    assert copies == [("double", ["x"], "near_copy")]
    assert all("noisy" not in (d.target, *d.components) for d in found)


def test_near_copy_not_repeated_for_an_explained_identity() -> None:
    rng = random.Random(4)
    a = [rng.uniform(100, 200) for _ in range(300)]
    b = [rng.uniform(0, 1) for _ in range(300)]  # tiny next to a: total ~ a (rho ~ 1)
    df = pl.DataFrame({"a": a, "b": b, "total": [x + y for x, y in zip(a, b)]})
    found = _derived(df)
    assert [(d.target, d.kind) for d in found] == [("total", "sum")]


def _wide_with_formula(n_noise: int, n: int = 2000, seed: int = 11) -> pl.DataFrame:
    """sales = unit_price × quantity × (1 − discount) planted among n_noise
    Gaussian sensors; quantity is an integer 1..10 (numeric-backed
    categorical, outside the profiler's correlation table)."""
    rng = random.Random(seed)
    cols: dict[str, list] = {}
    cols["unit_price"] = [round(rng.uniform(50, 5000), 2) for _ in range(n)]
    cols["quantity"] = [rng.randrange(1, 11) for _ in range(n)]
    cols["discount"] = [rng.choice([0.0, 0.05, 0.1, 0.2]) for _ in range(n)]
    cols["sales"] = [
        round(p * q * (1 - d), 2) for p, q, d in zip(cols["unit_price"], cols["quantity"], cols["discount"])
    ]
    for k in range(n_noise):
        cols[f"s{k:02d}"] = [rng.gauss(0, 1) for _ in range(n)]
    return pl.DataFrame(cols)


def _timed_derived(df: pl.DataFrame):
    import time

    from app.profiling.types import apply_casts

    profile = _profile(df)
    casted = apply_casts(df, {c.name: c.cast_params for c in profile.columns if c.cast_params})
    started = time.perf_counter()
    found = derived_columns(casted, profile.columns, profile.correlations, profile.evidence.num_num_spearman)
    return profile, found, time.perf_counter() - started


def test_derived_search_is_bounded_on_wide_tables() -> None:
    # 63 columns: capped at DERIVED_MAX_COLUMNS candidates, the planted
    # identity (inside the cap, with an integer quantity factor) still found,
    # and the whole search inside the 1 s budget from the stage brief
    profile, found, elapsed = _timed_derived(_wide_with_formula(n_noise=59))
    assert len(derived_candidates(profile.columns)) == DERIVED_MAX_COLUMNS
    assert [(d.target, d.formula) for d in found] == [("sales", "unit_price × quantity × (1 − discount)")]
    assert found[0].match_ratio == 1.0
    assert elapsed < 1.0, f"derived search took {elapsed:.2f}s"


def test_derived_found_on_40_column_table_with_categorical_factor() -> None:
    # verifier finding: above DERIVED_FULL_POOL_COLUMNS the peer pool is
    # ranked by correlation, and quantity (numeric-backed categorical) is
    # not in the profile's correlation table — it must still make the pool
    profile, found, _ = _timed_derived(_wide_with_formula(n_noise=36, seed=12))
    assert next(c for c in profile.columns if c.name == "quantity").semantic_type == "categorical"
    assert len(derived_candidates(profile.columns)) == DERIVED_MAX_COLUMNS  # 40 > cap, formula inside it
    assert [(d.target, d.formula, d.match_ratio) for d in found] == [
        ("sales", "unit_price × quantity × (1 − discount)", 1.0)
    ]
