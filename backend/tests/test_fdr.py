"""Stage 21: per-test p-values, Bartlett effective n, Benjamini–Hochberg
discovery flags and the top-tier gate in the rules."""

import random
from datetime import datetime, timedelta

import polars as pl

from app.charts.rules import _EvidenceIndex, fdr_unsupported, recommend_charts
from app.profiling.evidence import FDR_Q, benjamini_hochberg
from app.profiling.profiler import profile_dataset

T0 = datetime(2022, 1, 1)


def _noise(n: int, seed: int, n_num: int = 12) -> pl.DataFrame:
    rng = random.Random(seed)
    data = {f"x{i}": [rng.gauss(0, 1) for _ in range(n)] for i in range(1, n_num + 1)}
    for name, k in (("c1", 2), ("c2", 3), ("c3", 4), ("c4", 6)):
        data[name] = [f"g{rng.randrange(k)}" for _ in range(n)]
    data["d1"] = [T0 + timedelta(seconds=rng.uniform(0, 730 * 86400)) for _ in range(n)]
    return pl.DataFrame(data)


def _profile(df: pl.DataFrame, name: str):
    return profile_dataset(df, name, 100_000)


def test_benjamini_hochberg_step_up():
    # m = 5, q = 0.05: thresholds 0.01 0.02 0.03 0.04 0.05; the largest rank
    # under its threshold is 3, so the first three are rejected (step-up:
    # p = 0.025 > 0.02 at rank 2 is still rejected)
    p = [0.001, 0.025, 0.03, 0.2, 0.9]
    assert benjamini_hochberg(p, 0.05) == [True, True, True, False, False]
    assert benjamini_hochberg([0.5, 0.6], 0.05) == [False, False]
    assert benjamini_hochberg([], 0.05) == []


def test_every_l1_test_has_a_p_value_and_the_family_is_summarised():
    prof = _profile(_noise(200, 1), "fdr-summary")
    ev = prof.evidence
    assert ev.fdr is not None and ev.fdr.q == FDR_Q
    tested = (
        ev.cat_num + ev.time_effects + ev.interactions + ev.correlation_tests
    )
    assert all(t.p_value is not None and 0.0 <= t.p_value <= 1.0 for t in tested)
    assert all(t.discovery is not None for t in tested)
    assert ev.fdr.m >= len(tested)
    assert ev.fdr.discoveries == sum(1 for t in tested + ev.dispersion if t.discovery)


def test_planted_effects_are_discoveries():
    rng = random.Random(1007)
    n = 400
    df = _noise(n, 7)
    x1 = df["x1"].to_list()
    c1 = df["c1"].to_list()
    df = df.with_columns(
        pl.Series("x2", [0.5 * a + 0.87 * rng.gauss(0, 1) for a in x1]),
        pl.Series("x3", [rng.gauss(1.0 if c == "g1" else 0.0, 1) for c in c1]),
    )
    ev = _profile(df, "fdr-planted").evidence
    corr = next(t for t in ev.correlation_tests if {t.a, t.b} == {"x1", "x2"})
    assert corr.discovery is True and corr.p_value < 1e-6
    eta = next(e for e in ev.cat_num if (e.cat, e.num) == ("c1", "x3"))
    assert eta.discovery is True and eta.p_value < 1e-6


def test_effective_n_shrinks_for_independent_random_walks():
    rng = random.Random(11)
    n = 500
    a = b = 0.0
    xs, ys = [], []
    for _ in range(n):
        a += rng.gauss(0, 1)
        b += rng.gauss(0, 1)
        xs.append(a)
        ys.append(b)
    df = pl.DataFrame({"t": [T0 + timedelta(days=i) for i in range(n)], "walk_a": xs, "walk_b": ys})
    test = _profile(df, "fdr-rw").evidence.correlation_tests[0]
    assert test.autocorr_a > 0.9 and test.autocorr_b > 0.9
    assert test.n_eff < 0.1 * n
    assert test.p_value > 0.01 or abs(test.r) > 0.9


def test_effective_n_is_full_n_for_rows_without_time_order():
    df = _noise(300, 3).drop("d1")
    for t in _profile(df, "fdr-iid").evidence.correlation_tests:
        assert t.n_eff == t.n and t.autocorr_a is None


def test_interaction_p_value_detects_a_planted_crossover():
    rng = random.Random(1005)
    n = 2000
    df = _noise(n, 5)
    s1 = {"g0": -1, "g1": 1}
    s2 = {"g0": -1, "g1": 0, "g2": 1}
    y = [rng.gauss(0, 1) + 0.8 * s1[a] * s2[b] for a, b in zip(df["c1"], df["c2"])]
    df = df.with_columns(pl.Series("x1", y))
    from app.profiling.evidence import _interaction_test

    strength, rows, p_value = _interaction_test(df, "c1", "c2", "x1")
    assert rows == n and strength > 0.1 and p_value < 1e-6
    _, _, p_noise = _interaction_test(df, "c3", "c4", "x5")
    assert p_noise > 1e-3


def test_noise_never_gets_a_top_chart_without_a_discovery():
    """The audit's worst cell: n = 30 with many columns, where multiplicity
    alone used to mint top charts in most seeds."""
    tops = 0
    for seed in range(40):
        prof = _profile(_noise(30, 1000 + seed), f"fdr-null-{seed}")
        index = _EvidenceIndex(prof.evidence)
        for rec in recommend_charts(prof):
            if rec.tier == "top":
                tops += 1
                assert not fdr_unsupported(rec.spec, prof, index)
    assert tops <= 4  # BH keeps a few chance discoveries, never the old ~1.5 per dataset


def test_a_strong_planted_correlation_stays_top():
    rng = random.Random(1009)
    df = _noise(300, 9)
    x1 = df["x1"].to_list()
    df = df.with_columns(pl.Series("x2", [0.7 * a + 0.71 * rng.gauss(0, 1) for a in x1]))
    prof = _profile(df, "fdr-top")
    top = [r for r in recommend_charts(prof) if r.tier == "top"]
    assert any(r.spec.type == "scatter" and {r.spec.x, r.spec.y} == {"x1", "x2"} for r in top)


def _additive_grid(seed: int, missing: set, sizes: list[int] | None = None) -> pl.DataFrame:
    rng = random.Random(seed)
    rows = []
    for i, r_eff in enumerate((0, 3, 6, 9)):
        for j, c_eff in enumerate((0, 2, 4, 8)):
            if (i, j) in missing:
                continue
            n = sizes[(4 * i + j) % len(sizes)] if sizes else 10
            rows += [(f"a{i}", f"b{j}", r_eff + c_eff + rng.gauss(0, 1)) for _ in range(n)]
    return pl.DataFrame(rows, schema=["f1", "f2", "y"], orient="row")


def test_interaction_f_is_calibrated_on_incomplete_and_unbalanced_grids():
    """Review finding: the unweighted-means SS made 400/400 additive seeds
    'significant' with two empty cells; the LS additive fit fixes it."""
    from app.profiling.evidence import _interaction_test

    for missing, sizes in (({(0, 0), (3, 3)}, None), (set(), [8, 8, 10, 30, 40])):
        hits = sum(_interaction_test(_additive_grid(s, missing, sizes), "f1", "f2", "y")[2] < 0.05 for s in range(150))
        assert hits / 150 <= 0.11, (missing, sizes, hits)


def test_an_untested_short_series_is_not_capped():
    """Review finding: 12 monthly points have no within-bucket df, so no time
    test exists; that must not count as a failed test."""
    rng = random.Random(1012)
    df = pl.DataFrame({
        "month": [datetime(2023, m, 1) for m in range(1, 13)],
        "sales": [100 + 10 * m + rng.gauss(0, 2) for m in range(12)],
    })
    prof = _profile(df, "fdr-short")
    index = _EvidenceIndex(prof.evidence)
    lines = [r for r in recommend_charts(prof) if r.spec.type == "line" and r.spec.x == "month"]
    assert lines and all(not fdr_unsupported(r.spec, prof, index) for r in lines)


def test_spearman_is_tested_and_can_carry_a_scatter():
    prof = _profile(_noise(200, 4), "fdr-spearman")
    t = prof.evidence.correlation_tests[0]
    assert t.rho is not None and t.p_value_spearman is not None and t.discovery_spearman is not None


def test_llm_claim_on_a_non_discovery_fails_coverage():
    """Stage 21: the LLM path cannot bypass the BH gate through existing evidence."""
    from app.llm.coverage import check_coverage

    for seed in range(30):
        prof = _profile(_noise(30, 2000 + seed), f"fdr-cov-{seed}")
        for e in prof.evidence.cat_num:
            hit = check_coverage("group_difference", {"group": e.cat, "target": e.num}, prof)
            if hit is not None and e.discovery is False:
                assert hit.verdict == "fail"


# --- stage 21 B: structural tests, probes ------------------------------------


def test_chi2_and_ks_tail_functions_match_tabulated_values():
    from app.profiling.evidence import chi2_survival, ks_p

    for x, df in ((3.841, 1), (5.991, 2), (11.070, 5), (18.307, 10)):
        assert abs(chi2_survival(x, df) - 0.05) < 5e-4
    assert chi2_survival(0.0, 3) == 1.0
    # 5 % critical value of the two-sample KS: D ≈ 1.358 / sqrt(n_e) (large n)
    n = 2000
    assert abs(ks_p(1.358 / (n / 2) ** 0.5, n, n) - 0.05) < 0.005


def test_holm_step_down():
    from app.profiling.evidence import holm

    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    assert holm([]) == []


def test_structural_tests_hold_their_level_under_the_null():
    from app.profiling.evidence import fisher_z_homogeneity_p, lack_of_fit_p

    rng = random.Random(21)
    hits_z = hits_lof = 0
    reps = 300
    for _ in range(reps):
        groups = []
        for _ in range(3):
            xs = [rng.gauss(0, 1) for _ in range(60)]
            ys = [0.4 * x + rng.gauss(0, 1) for x in xs]
            df = pl.DataFrame({"x": xs, "y": ys})
            groups.append((df.select(pl.corr("x", "y")).item(), 60))
        hits_z += fisher_z_homogeneity_p(groups) < 0.05
        # a straight line: binned means only add noise
        xs = sorted(rng.gauss(0, 1) for _ in range(200))
        ys = [0.5 * x + rng.gauss(0, 1) for x in xs]
        k = 10
        bins = [ys[i * 20:(i + 1) * 20] for i in range(k)]
        grand = sum(ys) / len(ys)
        ssw = sum(sum((v - sum(b) / len(b)) ** 2 for v in b) for b in bins)
        sst = sum((v - grand) ** 2 for v in ys)
        adj = 1 - (ssw / (200 - k)) / (sst / 199)
        r = pl.DataFrame({"x": xs, "y": ys}).select(pl.corr("x", "y")).item()
        hits_lof += lack_of_fit_p(max(adj, 0.0), r * r, 200, k) < 0.05
    assert hits_z / reps <= 0.09 and hits_lof / reps <= 0.09


def test_structural_entries_join_the_bh_family():
    prof = _profile(_noise(100, 31), "fdr-l2")
    ev = prof.evidence
    l2_tests = [c for c in ev.layer2.conditional_relationships if c.p_value is not None]
    l2_tests += [s for s in ev.layer2.nonlinear if s.p_value is not None]
    assert l2_tests, "the 20-column noise layout always lists some structure"
    assert all(t.discovery is not None for t in l2_tests)
    counted = (
        len(ev.cat_num) + len(ev.time_effects) + len(ev.interactions) + len(ev.slope_heterogeneity)
        + 2 * len(ev.correlation_tests) + len(ev.dispersion) + len(l2_tests)
    )
    assert ev.fdr.m <= counted + len(ev.layer2.conditional_relationships)


def test_llm_nonlinear_claim_on_noise_needs_a_discovery():
    from app.llm.coverage import check_coverage

    for seed in range(20):
        prof = _profile(_noise(100, 3000 + seed), f"fdr-nl-{seed}")
        for s in prof.evidence.layer2.nonlinear:
            hit = check_coverage("nonlinear_relationship", {"x": s.x, "y": s.y}, prof)
            if hit is not None and s.discovery is False:
                assert hit.verdict == "fail"
