"""Targeted validation probes (stage 17.2): planted pass / fail per probe
type, request validation, the sample-size cap, de-duplication, the run cap,
caching, determinism, bounded cost and chart validity."""

import random
import time
from datetime import datetime, timedelta

import polars as pl
import pytest

from app.charts.spec import validate_spec
from app.probes import (
    MAX_PROBES,
    PROBE_ROLES,
    ProbeCache,
    ProbeRejected,
    ProbeRequest,
    ProbeResult,
    run_probes,
    validate_request,
)
from app.probes import engine
from app.profiling.profiler import profile_dataset

BIG = 10**6


def _profile(df: pl.DataFrame, threshold: int = BIG):
    return profile_dataset(df, "0" * 32, threshold)


def _planted(n: int = 3000, seed: int = 7) -> pl.DataFrame:
    rng = random.Random(seed)
    g = [["a", "b", "c"][i % 3] for i in range(n)]
    h = [["p", "q"][i % 2] for i in range(n)]
    x = [rng.uniform(0, 10) for _ in range(n)]
    return pl.DataFrame(
        {
            "g": g,
            "h": h,
            "x": x,
            "y_lin": [3 * v + rng.gauss(0, 1) for v in x],
            "y_u": [(v - 5) ** 2 + rng.gauss(0, 1) for v in x],
            "grp_shift": [rng.gauss(0, 1) + (2.0 if gg == "c" else 0.0) for gg in g],
            "noise": [rng.gauss(0, 1) for _ in range(n)],
            "slope": [(v if gg == "a" else -v if gg == "b" else 0) + rng.gauss(0, 1) for v, gg in zip(x, g)],
            "inter": [(1.0 if (gg == "a") == (hh == "p") else -1.0) + rng.gauss(0, 1) for gg, hh in zip(g, h)],
            "ts": [datetime(2024, 1, 1) + timedelta(hours=i * 4) for i in range(n)],
            "step": [(1.0 if i > n // 2 else 0.0) + rng.gauss(0, 1) for i in range(n)],
        }
    )


@pytest.fixture(scope="module")
def planted():
    df = _planted()
    return df, _profile(df)


def _one(df, profile, req: ProbeRequest) -> ProbeResult:
    out = run_probes(df, profile, [req], cache=ProbeCache())
    assert len(out) == 1
    assert isinstance(out[0], ProbeResult), out[0]
    return out[0]


def _req(type_: str, **columns: str) -> ProbeRequest:
    return ProbeRequest(type=type_, columns=columns)


# --- planted pass / fail per type ---------------------------------------------------


def test_group_difference_pass_and_fail(planted) -> None:
    df, profile = planted
    hit = _one(df, profile, _req("group_difference", group="g", target="grp_shift"))
    assert hit.verdict == "pass"
    assert hit.effect_label.startswith("adjusted eta-squared")
    assert hit.effect_size > 0.3
    assert hit.n == 3000 and hit.n_min_group == 1000
    assert hit.evidence["top_group"] == "c" and hit.evidence["max_diff_sd"] > 1.5
    assert {g["group"] for g in hit.evidence["groups"]} == {"a", "b", "c"}
    assert hit.chart is not None and hit.chart.type == "bar" and hit.chart.aggregation == "mean"

    miss = _one(df, profile, _req("group_difference", group="g", target="noise"))
    assert miss.verdict == "fail"
    assert miss.effect_size < 0.03


def test_nonlinear_pass_on_u_shape_fail_on_line(planted) -> None:
    df, profile = planted
    u = _one(df, profile, _req("nonlinear_relationship", x="x", y="y_u"))
    assert u.verdict == "pass"
    assert u.evidence["shape"] == "u_shape"
    assert u.evidence["binned_eta2"] > 0.9 and u.evidence["r2_pearson"] < 0.05
    assert u.effect_size == pytest.approx(u.evidence["nonlinear_gap"])
    assert len(u.evidence["bins"]) == 10 and all(b["n"] >= 250 for b in u.evidence["bins"])
    assert u.chart is not None and u.chart.type == "scatter"

    line = _one(df, profile, _req("nonlinear_relationship", x="x", y="y_lin"))
    assert line.verdict == "fail"
    assert line.evidence["shape"] == "linear"
    assert line.effect_size < 0.05


def test_slope_difference_pass_and_fail(planted) -> None:
    df, profile = planted
    hit = _one(df, profile, _req("slope_difference", x="x", y="slope", group="g"))
    assert hit.verdict == "pass"
    corrs = {e["group"]: e["corr"] for e in hit.evidence["groups"]}
    assert corrs["a"] > 0.9 and corrs["b"] < -0.9 and abs(corrs["c"]) < 0.1
    assert hit.effect_size > 1.8
    assert hit.thresholds["pass"] == pytest.approx(engine.slope_spread_threshold(1000))
    assert hit.chart is not None and hit.chart.group_by == "g"

    # grouped_relationship is the same estimator: identical y_lin slope in every group -> no spread
    same = _one(df, profile, _req("grouped_relationship", x="x", y="y_lin", group="g"))
    assert same.verdict == "fail"
    assert same.effect_size < 0.05
    assert all(e["corr"] > 0.99 for e in same.evidence["groups"])


def test_distribution_difference_pass_and_fail(planted) -> None:
    df, profile = planted
    hit = _one(df, profile, _req("distribution_difference", group="g", target="grp_shift"))
    assert hit.verdict == "pass"
    assert hit.effect_size > 0.6
    assert "c" in hit.evidence["ks_pair"]
    assert len(hit.evidence["pairs"]) == 3
    assert hit.chart is not None and hit.chart.type == "box"

    miss = _one(df, profile, _req("distribution_difference", group="g", target="noise"))
    assert miss.verdict == "fail"
    assert miss.effect_size < 0.1


def test_time_pattern_pass_on_step_fail_on_noise(planted) -> None:
    df, profile = planted
    hit = _one(df, profile, _req("time_pattern", time="ts", target="step"))
    assert hit.verdict == "pass"
    change = hit.evidence["change_point"]
    assert change is not None and change["flagged"] is True
    assert change["after_mean"] - change["before_mean"] > 0.8
    assert len(hit.evidence["series"]) <= 24
    assert hit.chart is not None and hit.chart.type == "line" and hit.chart.aggregation == "mean"

    miss = _one(df, profile, _req("time_pattern", time="ts", target="noise", group="g"))
    assert miss.verdict == "fail"
    assert set(miss.evidence["group_series"]) == {"a", "b", "c"}
    assert miss.chart is not None and miss.chart.group_by == "g"


def test_interaction_pass_and_fail(planted) -> None:
    df, profile = planted
    hit = _one(df, profile, _req("interaction", factor1="g", factor2="h", target="inter"))
    assert hit.verdict == "pass"
    assert hit.effect_size > 0.3
    assert len(hit.evidence["cells"]) == 6
    assert hit.chart is not None and hit.chart.type == "bar" and hit.chart.group_by == "h"

    miss = _one(df, profile, _req("interaction", factor1="g", factor2="h", target="noise"))
    assert miss.verdict == "fail"


# --- request validation ---------------------------------------------------------------


def test_every_probe_type_has_roles_and_a_runner() -> None:
    assert set(PROBE_ROLES) == set(engine._RUNNERS)


@pytest.mark.parametrize(
    "req, fragment",
    [
        (_req("group_difference", group="nope", target="noise"), "does not exist"),
        (_req("group_difference", group="x", target="noise"), "needs a categorical"),
        (_req("nonlinear_relationship", x="g", y="noise"), "needs a numeric"),
        (_req("time_pattern", time="x", target="noise"), "needs a datetime"),
        (_req("group_difference", group="g"), "missing"),
        (_req("group_difference", group="g", target="noise", extra="x"), "does not accept"),
        (_req("nonlinear_relationship", x="x", y="x"), "different column"),
        (_req("interaction", factor1="g", factor2="g", target="noise"), "different column"),
    ],
)
def test_invalid_requests_are_rejected_without_running(planted, req, fragment) -> None:
    df, profile = planted
    reason = validate_request(req, profile)
    assert reason is not None and fragment in reason
    out = run_probes(df, profile, [req], cache=ProbeCache())
    assert isinstance(out[0], ProbeRejected) and fragment in out[0].reason


def test_unknown_probe_type_is_rejected_by_the_schema() -> None:
    with pytest.raises(ValueError):
        ProbeRequest(type="run_python", columns={"code": "1+1"})


def test_derived_pair_is_rejected_as_definitional() -> None:
    rng = random.Random(3)
    n = 400
    price = [round(rng.uniform(10, 100), 2) for _ in range(n)]
    qty = [rng.randint(1, 9) for _ in range(n)]
    df = pl.DataFrame(
        {
            "price": price,
            "qty": qty,
            "sales": [round(p * q, 2) for p, q in zip(price, qty)],
            "region": [["n", "s", "e", "w"][i % 4] for i in range(n)],
        }
    )
    profile = _profile(df)
    assert any(d.target == "sales" for d in profile.evidence.derived_columns)
    out = run_probes(
        df,
        profile,
        [
            _req("nonlinear_relationship", x="price", y="sales"),
            _req("group_difference", group="qty", target="sales"),  # qty is a numeric-backed categorical
            _req("group_difference", group="region", target="sales"),
        ],
        cache=ProbeCache(),
    )
    assert isinstance(out[0], ProbeRejected) and "definitional" in out[0].reason
    assert isinstance(out[1], ProbeRejected) and "definitional" in out[1].reason
    assert isinstance(out[2], ProbeResult)


def test_near_duplicate_column_is_rejected_in_favour_of_representative() -> None:
    rng = random.Random(5)
    n = 600
    temp = [rng.uniform(0, 30) for _ in range(n)]
    df = pl.DataFrame(
        {
            "temp": temp,
            "atemp": [0.9 * t + rng.gauss(0, 0.3) for t in temp],
            "hum": [rng.uniform(20, 90) for _ in range(n)],
            "season": [["w", "sp", "su", "f"][i % 4] for i in range(n)],
        }
    )
    profile = _profile(df)
    assert profile.evidence.near_duplicate_groups
    reason = validate_request(_req("nonlinear_relationship", x="atemp", y="hum"), profile)
    assert reason is not None and "near-duplicate" in reason and "'temp'" in reason
    assert validate_request(_req("nonlinear_relationship", x="temp", y="hum"), profile) is None


def test_mostly_missing_column_is_rejected() -> None:
    n = 300
    df = pl.DataFrame(
        {
            "g": [["a", "b"][i % 2] for i in range(n)],
            "v": [float(i) if i % 4 == 0 else None for i in range(n)],  # 75% missing
            "w": [float(i) for i in range(n)],
        }
    )
    profile = _profile(df)
    assert "missing" in (validate_request(_req("group_difference", group="g", target="v"), profile) or "")
    assert validate_request(_req("group_difference", group="g", target="w"), profile) is None


# --- verdict caps, run caps, de-duplication, cache -------------------------------------


def test_tiny_sample_cannot_pass() -> None:
    rng = random.Random(11)
    n = 12
    g = [["a", "b", "c"][i % 3] for i in range(n)]
    df = pl.DataFrame(
        {
            "g": g,
            "v": [rng.gauss(0, 0.1) + (5.0 if gg == "c" else 0.0) for gg in g],  # huge planted effect
            "x": [float(i) for i in range(n)],
        }
    )
    profile = _profile(df)
    out = run_probes(
        df,
        profile,
        [_req("group_difference", group="g", target="v"), _req("distribution_difference", group="g", target="v")],
        cache=ProbeCache(),
    )
    for r in out:
        assert isinstance(r, ProbeResult)
        assert r.verdict in ("weak", "fail")
        assert r.confidence.overall < 1.0
    assert out[0].effect_size > 0.9  # the effect is real; the verdict is capped, not the estimate
    assert any("capped" in note for note in out[0].notes)


def test_run_cap_rejects_requests_beyond_max_probes(planted) -> None:
    df, profile = planted
    targets = ["grp_shift", "noise", "y_lin", "y_u", "slope", "inter"]
    reqs = [_req("group_difference", group="g", target=t) for t in targets]
    out = run_probes(df, profile, reqs, cache=ProbeCache())
    assert len(out) == 6
    assert all(isinstance(r, ProbeResult) for r in out[:MAX_PROBES])
    assert isinstance(out[MAX_PROBES], ProbeRejected) and out[MAX_PROBES].reason.startswith("cap")
    # invalid requests do not consume a slot
    reqs = [_req("group_difference", group="nope", target="noise")] + [
        _req("group_difference", group="g", target=t) for t in targets[:5]
    ]
    out = run_probes(df, profile, reqs, cache=ProbeCache())
    assert isinstance(out[0], ProbeRejected)
    assert all(isinstance(r, ProbeResult) for r in out[1:])


def test_duplicate_requests_collapse_and_role_order_is_irrelevant(planted) -> None:
    df, profile = planted
    a = ProbeRequest(type="group_difference", columns={"group": "g", "target": "grp_shift"})
    b = ProbeRequest(type="group_difference", columns={"target": "grp_shift", "group": "g"})
    out = run_probes(df, profile, [a, b, a], cache=ProbeCache())
    assert isinstance(out[0], ProbeResult)
    assert isinstance(out[1], ProbeRejected) and out[1].reason == "duplicate"
    assert isinstance(out[2], ProbeRejected) and out[2].reason == "duplicate"


def test_cache_hit_does_not_recompute(planted, monkeypatch) -> None:
    df, profile = planted
    calls = {"n": 0}
    original = engine._run_one

    def counting(frame, prof, req):
        calls["n"] += 1
        return original(frame, prof, req)

    monkeypatch.setattr(engine, "_run_one", counting)
    cache = ProbeCache()
    req = _req("group_difference", group="g", target="grp_shift")
    first = run_probes(df, profile, [req], cache=cache)[0]
    second = run_probes(df, profile, [req], cache=cache)[0]
    assert calls["n"] == 1
    assert isinstance(first, ProbeResult) and first.cached is False
    assert isinstance(second, ProbeResult) and second.cached is True
    assert second.model_dump(exclude={"cached"}) == first.model_dump(exclude={"cached"})
    # a fresh cache computes again
    run_probes(df, profile, [req], cache=ProbeCache())
    assert calls["n"] == 2


def _assert_close(a, b, tol: float = 1e-6) -> None:
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys()
        for key in a:
            _assert_close(a[key], b[key], tol)
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b)
        for x, y in zip(a, b):
            _assert_close(x, y, tol)
    elif isinstance(a, float) and isinstance(b, float):
        assert a == pytest.approx(b, rel=tol, abs=tol)
    else:
        assert a == b


def test_results_are_deterministic_and_carry_no_raw_rows(planted) -> None:
    df, profile = planted
    reqs = [
        _req("group_difference", group="g", target="grp_shift"),
        _req("nonlinear_relationship", x="x", y="y_u"),
        _req("time_pattern", time="ts", target="step", group="g"),
        _req("distribution_difference", group="g", target="grp_shift"),
        _req("interaction", factor1="g", factor2="h", target="inter"),
    ]
    first = run_probes(df, profile, reqs, cache=ProbeCache())
    again = run_probes(df, profile, reqs, cache=ProbeCache())
    assert [r.model_dump_json() for r in first] == [r.model_dump_json() for r in again]
    # row order only changes floating-point summation order: every number
    # agrees to 1e-6 and every verdict / label / structure is identical
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=99)
    for a, b in zip(first, run_probes(shuffled, profile, reqs, cache=ProbeCache())):
        _assert_close(a.model_dump(), b.model_dump())
    for r in first:
        assert isinstance(r, ProbeResult)
        text = r.model_dump_json()
        assert len(text) < 20_000  # summaries only
        for key in ("groups", "bins", "cells", "series"):
            block = r.evidence.get(key)
            if isinstance(block, list):
                assert len(block) <= 40


def test_suggested_charts_validate_and_confidence_is_the_stage9_chain(planted) -> None:
    df, profile = planted
    reqs = [
        _req("group_difference", group="g", target="grp_shift"),
        _req("slope_difference", x="x", y="slope", group="g"),
        _req("nonlinear_relationship", x="x", y="y_u"),
        _req("distribution_difference", group="g", target="grp_shift"),
        _req("time_pattern", time="ts", target="step"),
    ]
    for r in run_probes(df, profile, reqs, cache=ProbeCache()):
        assert isinstance(r, ProbeResult)
        assert r.chart is not None
        assert validate_spec(r.chart, profile) == []
        assert r.confidence.n_source in ("exact", "estimated")
        assert 0 < r.confidence.overall <= 1


def test_sampled_profile_uses_the_profiled_sample() -> None:
    df = _planted(n=4000, seed=21)
    profile = _profile(df, threshold=500)
    assert profile.sampled and profile.profiled_rows == 500
    r = _one(df, profile, _req("group_difference", group="g", target="grp_shift"))
    assert r.n == 500
    assert r.verdict == "pass"


def test_each_probe_stays_under_half_a_second_on_100k_rows() -> None:
    df = _planted(n=100_000, seed=42)
    profile = _profile(df)
    reqs = [
        _req("group_difference", group="g", target="grp_shift"),
        _req("slope_difference", x="x", y="slope", group="g"),
        _req("nonlinear_relationship", x="x", y="y_u"),
        _req("distribution_difference", group="g", target="grp_shift"),
        _req("time_pattern", time="ts", target="step", group="g"),
        _req("interaction", factor1="g", factor2="h", target="inter"),
    ]
    frame = engine.prepare_frame(df, profile)
    for req in reqs:
        start = time.perf_counter()
        result = engine._run_one(frame, profile, req)
        elapsed = time.perf_counter() - start
        assert isinstance(result, ProbeResult)
        assert elapsed < 0.5, f"{req.type} took {elapsed:.2f}s"
