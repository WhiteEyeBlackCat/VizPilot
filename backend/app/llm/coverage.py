"""Evidence coverage check for LLM hypotheses (stage 17.3).

A hypothesis names a probe type and its columns. Before the backend computes
anything new, this module asks whether the evidence tables (layer 1 effect
sizes, layer 2 structural summaries) already answer the question:

- covered and at/above the probe thresholds  -> validated by existing evidence
- covered but below the "weak" threshold     -> the hypothesis fails (dropped)
- not covered                                -> a targeted probe runs

The thresholds are the probe engine's own numbers, so "validated by existing
evidence" and "validated by probe" mean the same thing. Layer-1 tables that
keep only the top-k entries (interactions, slope heterogeneity) and every
layer-2 table are partial: an absent entry is "not covered", never "fail".
"""

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from ..charts.confidence import assess
from ..charts.rules import _EvidenceIndex, slope_spread_threshold
from ..charts.spec import ChartSpec, validate_spec
from ..probes.engine import MIN_ROWS, THRESHOLDS, TOP_CONFIDENCE_FLOOR
from ..probes.schemas import PROBE_ROLES
from ..profiling.models import DatasetProfile

Verdict = Literal["pass", "weak", "fail"]
Validation = Literal["existing_evidence", "probe"]

# grouped_relationship: the x-y relationship holds inside every group — the
# effect is the weakest per-group |corr|. The engine owns the numbers once it
# ships the probe (stage 17.2 revision); until then the same constants live here.
GROUPED_THRESHOLDS = THRESHOLDS.get("grouped_relationship", {"pass": 0.3, "weak": 0.15})


@dataclass
class CoverageHit:
    verdict: Verdict
    effect_label: str
    effect_value: float
    n: int
    evidence_lines: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    n_min_group: int | None = None


@dataclass
class ValidatedHypothesis:
    """A hypothesis that survived the gate and the evidence check; what LLM #2
    (or the deterministic template) is allowed to talk about."""

    id: int
    statement: str
    reason: str
    importance: int
    test_type: str
    columns: dict[str, str]
    validation: Validation
    verdict: Literal["pass", "weak"]
    effect_label: str
    effect_value: float
    n: int
    evidence_lines: list[str]
    evidence: dict[str, Any]
    chart_candidates: list[tuple[str, ChartSpec]]  # (id, spec), backend chart first


def _verdict(value: float, thresholds: dict[str, float]) -> Verdict:
    if value >= thresholds["pass"]:
        return "pass"
    if value >= thresholds["weak"]:
        return "weak"
    return "fail"


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    rounded = round(value, digits)
    return str(int(rounded)) if float(rounded).is_integer() else f"{rounded:g}"


# --- coverage --------------------------------------------------------------------


def check_coverage(test_type: str, columns: dict[str, str], profile: DatasetProfile) -> CoverageHit | None:
    """None when the evidence tables do not answer the question (run a probe)."""
    evidence = profile.evidence
    l2 = evidence.layer2
    if test_type == "group_difference":
        group, target = columns["group"], columns["target"]
        effect = next((e for e in evidence.cat_num if e.cat == group and e.num == target), None)
        if effect is None:
            return None
        lines: list[str] = []
        summary = next((g for g in l2.group_summaries if g.cat == group and g.num == target), None)
        ev: dict[str, Any] = {"eta_squared": effect.eta_squared, "n_groups": effect.n_groups}
        if summary is not None:
            listed = sorted(summary.groups, key=lambda s: -s.mean)
            lines.append(
                "mean per group: " + ", ".join(f"{s.group}={_fmt(s.mean)} (n={s.n})" for s in listed[:8])
            )
            if summary.max_diff_sd is not None:
                lines.append(
                    f"largest gap {summary.max_diff_sd:.2f} pooled SD ({summary.top_group} vs {summary.bottom_group})"
                )
            ev["groups"] = [s.model_dump() for s in summary.groups]
            ev["max_diff_sd"] = summary.max_diff_sd
        lines.insert(0, f"adjusted eta-squared {effect.eta_squared:.3f} over {effect.n_groups} groups")
        return CoverageHit(
            verdict=_verdict(effect.eta_squared, THRESHOLDS["group_difference"]),
            effect_label="adjusted eta-squared of target across groups",
            effect_value=effect.eta_squared,
            n=effect.n_total,
            evidence_lines=lines,
            evidence=ev,
            n_min_group=effect.n_min or None,
        )

    if test_type == "distribution_difference":
        return None  # no KS statistic in the evidence tables: always a probe

    if test_type in ("grouped_relationship", "slope_difference"):
        x, y, group = columns["x"], columns["y"], columns["group"]
        cond = next(
            (
                c
                for c in l2.conditional_relationships
                if c.group == group and {c.x, c.y} == {x, y}
            ),
            None,
        )
        if test_type == "slope_difference":
            het = next(
                (s for s in evidence.slope_heterogeneity if s.group == group and {s.x, s.y} == {x, y}),
                None,
            )
            if het is not None:
                threshold = slope_spread_threshold(het.n_min)
                corrs = {k: v for k, v in het.corrs.items() if v is not None}
                lines = [
                    "correlation per group: "
                    + ", ".join(f"{k}={v:.2f}" for k, v in sorted(corrs.items(), key=lambda kv: -abs(kv[1]))),
                    f"spread {het.spread:.2f} vs threshold {threshold:.2f} (smallest group n={het.n_min})",
                ]
                return CoverageHit(
                    verdict=_verdict(het.spread, {"pass": threshold, "weak": threshold / 2}),
                    effect_label="spread of per-group correlations (max - min)",
                    effect_value=het.spread,
                    n=profile.profiled_rows or profile.n_rows,
                    evidence_lines=lines,
                    evidence={"groups": het.corrs, "spread": het.spread, "threshold": threshold},
                    n_min_group=het.n_min,
                )
            if cond is None:
                return None
            threshold = slope_spread_threshold(cond.n_min)
            return CoverageHit(
                verdict=_verdict(cond.corr_spread, {"pass": threshold, "weak": threshold / 2}),
                effect_label="spread of per-group correlations (max - min)",
                effect_value=cond.corr_spread,
                n=sum(g.n for g in cond.groups),
                evidence_lines=_conditional_lines(cond) + [f"spread {cond.corr_spread:.2f} vs threshold {threshold:.2f}"],
                evidence={"groups": [g.model_dump() for g in cond.groups], "spread": cond.corr_spread},
                n_min_group=cond.n_min,
            )
        # grouped_relationship: weakest |corr| across groups with a correlation
        if cond is None:
            return None
        corrs = [g.corr for g in cond.groups if g.corr is not None]
        if len(corrs) < 2:
            return None
        weakest = min(abs(c) for c in corrs)
        consistent = all(c > 0 for c in corrs) or all(c < 0 for c in corrs)
        return CoverageHit(
            verdict=_verdict(weakest, GROUPED_THRESHOLDS),
            effect_label="minimum |correlation| across groups",
            effect_value=weakest,
            n=sum(g.n for g in cond.groups),
            evidence_lines=_conditional_lines(cond)
            + [f"weakest |corr| {weakest:.2f}" + ("" if consistent else "; the sign differs between groups")],
            evidence={"groups": [g.model_dump() for g in cond.groups], "min_abs_corr": weakest, "same_sign": consistent},
            n_min_group=cond.n_min,
        )

    if test_type == "nonlinear_relationship":
        x, y = columns["x"], columns["y"]
        signal = next((s for s in l2.nonlinear if {s.x, s.y} == {x, y}), None)
        if signal is None:
            return None
        thresholds = THRESHOLDS["nonlinear_relationship"]
        verdict = _verdict(signal.nonlinear_gap, thresholds)
        means = ", ".join(_fmt(b.y_mean) for b in signal.bins)
        r2 = "n/a" if signal.r2_pearson is None else f"{signal.r2_pearson:.2f}"
        lines = [
            f"shape {signal.shape}; binned eta-squared {signal.binned_eta2:.2f} vs linear r2 {r2}",
            f"{signal.y} mean per {signal.x} bin (low to high): {means}",
        ]
        return CoverageHit(
            verdict=verdict,
            effect_label="nonlinear_gap = binned eta-squared - Pearson r-squared",
            effect_value=signal.nonlinear_gap,
            n=signal.n,
            evidence_lines=lines,
            evidence={
                "shape": signal.shape,
                "binned_eta2": signal.binned_eta2,
                "r2_pearson": signal.r2_pearson,
                "bins": [b.model_dump() for b in signal.bins],
            },
        )

    if test_type == "time_pattern":
        time, target = columns["time"], columns["target"]
        effect = next((t for t in evidence.time_effects if t.datetime_col == time and t.num == target), None)
        if effect is None:
            return None
        change = next(
            (c for c in l2.change_points if c.datetime_col == time and c.num == target), None
        )
        thresholds = THRESHOLDS["time_pattern"]
        verdict = _verdict(effect.eta_squared, thresholds)
        lines = [f"adjusted eta-squared {effect.eta_squared:.3f} across {effect.bucket} buckets"]
        ev: dict[str, Any] = {"eta_squared": effect.eta_squared, "bucket": effect.bucket}
        if change is not None and change.strength != "none":
            lines.append(
                f"level shift at {change.change_at[:10]}: mean {_fmt(change.before_mean)} before, "
                f"{_fmt(change.after_mean)} after ({change.strength}, {change.diff_sd:.2f} column SD)"
            )
            ev["change_point"] = change.model_dump()
            if change.strength == "strong" and verdict != "pass":
                verdict = "pass"
        return CoverageHit(
            verdict=verdict,
            effect_label="adjusted eta-squared of target across time buckets",
            effect_value=effect.eta_squared,
            n=effect.n_total,
            evidence_lines=lines,
            evidence=ev,
            n_min_group=effect.n_min or None,
        )

    if test_type == "interaction":
        f1, f2, target = columns["factor1"], columns["factor2"], columns["target"]
        hit = next(
            (e for e in evidence.interactions if e.num == target and {e.cat1, e.cat2} == {f1, f2}),
            None,
        )
        if hit is None:
            return None  # layer 1 keeps only the top pairs: absent is "not covered"
        return CoverageHit(
            verdict=_verdict(hit.strength, THRESHOLDS["interaction"]),
            effect_label="interaction share of variance (unweighted-means ANOVA)",
            effect_value=hit.strength,
            n=hit.n_total,
            evidence_lines=[f"interaction share of variance {hit.strength:.3f} (n={hit.n_total})"],
            evidence={"strength": hit.strength},
        )
    return None


def _conditional_lines(cond) -> list[str]:
    parts = []
    for g in cond.groups:
        corr = "n/a" if g.corr is None else f"{g.corr:.2f}"
        parts.append(f"{g.group}={corr} (n={g.n})")
    return [f"correlation of {cond.x} and {cond.y} per {cond.group}: " + ", ".join(parts)]


def apply_confidence_cap(hit: CoverageHit, chart: ChartSpec | None, profile: DatasetProfile) -> CoverageHit:
    """The probe engine's sample-size rule for evidence-table hits: a pass on
    a tiny sample, on a low-confidence chart, or on tiny groups is at most weak."""
    if hit.verdict != "pass":
        return hit
    if hit.n < MIN_ROWS:
        hit.verdict = "weak"
        hit.evidence_lines.append(f"capped at weak: only {hit.n} rows")
        return hit
    if hit.n_min_group is not None and hit.n_min_group < MIN_ROWS:
        hit.verdict = "weak"
        hit.evidence_lines.append(f"capped at weak: smallest group has {hit.n_min_group} rows")
        return hit
    if chart is not None:
        confidence = assess(chart, profile).confidence
        if confidence.overall < TOP_CONFIDENCE_FLOOR:
            hit.verdict = "weak"
            hit.evidence_lines.append(f"capped at weak: confidence {confidence.overall:.2f}")
    return hit


# --- probe inference from a chart (legacy / chart-only hypotheses) ----------------


def infer_probe(spec: ChartSpec, profile: DatasetProfile) -> tuple[str, dict[str, str]] | None:
    """The probe a chart-only hypothesis implicitly asks for: bar -> group
    difference, box -> distribution difference, grouped scatter -> slope
    difference, plain scatter -> non-linear dependence, line -> time pattern.
    Heatmaps and histograms carry no testable claim."""
    columns = {c.name: c for c in profile.columns}
    x = columns.get(spec.x or "")
    y = columns.get(spec.y or "")
    if spec.type in ("bar", "box") and x is not None and y is not None:
        probe = "group_difference" if spec.type == "bar" else "distribution_difference"
        return probe, {"group": x.name, "target": y.name}
    if spec.type == "scatter" and x is not None and y is not None:
        if spec.group_by:
            return "slope_difference", {"x": x.name, "y": y.name, "group": spec.group_by}
        return "nonlinear_relationship", {"x": x.name, "y": y.name}
    if spec.type == "line" and x is not None and y is not None and x.semantic_type == "datetime":
        cols = {"time": x.name, "target": y.name}
        if spec.group_by:
            cols["group"] = spec.group_by
        return "time_pattern", cols
    return None


def suggest_chart(test_type: str, columns: dict[str, str], profile: DatasetProfile) -> ChartSpec | None:
    """The backend's own chart for a validated hypothesis (mirrors the probe
    engine's suggestions), or None when the combination cannot be drawn."""
    c = columns
    try:
        if test_type == "group_difference":
            spec = ChartSpec(title=f"Mean {c['target']} by {c['group']}", type="bar", x=c["group"], y=c["target"], aggregation="mean")
        elif test_type == "distribution_difference":
            spec = ChartSpec(title=f"{c['target']} distribution by {c['group']}", type="box", x=c["group"], y=c["target"])
        elif test_type in ("grouped_relationship", "slope_difference"):
            spec = ChartSpec(title=f"{c['y']} vs {c['x']} by {c['group']}", type="scatter", x=c["x"], y=c["y"], group_by=c["group"])
        elif test_type == "nonlinear_relationship":
            spec = ChartSpec(title=f"{c['y']} vs {c['x']}", type="scatter", x=c["x"], y=c["y"])
        elif test_type == "time_pattern":
            title = f"{c['target']} over {c['time']}" + (f" by {c['group']}" if c.get("group") else "")
            spec = ChartSpec(title=title, type="line", x=c["time"], y=c["target"], group_by=c.get("group"), aggregation="mean")
        elif test_type == "interaction":
            spec = ChartSpec(
                title=f"Mean {c['target']} by {c['factor1']} and {c['factor2']}",
                type="bar",
                x=c["factor1"],
                y=c["target"],
                group_by=c["factor2"],
                aggregation="mean",
            )
        else:
            return None
    except (KeyError, ValueError):
        return None
    return spec if not validate_spec(spec, profile) else None


def required_roles(test_type: str) -> tuple[str, ...]:
    roles = PROBE_ROLES.get(test_type)
    return roles[0] if roles else ()


def definitional_conflict(names: list[str], profile: DatasetProfile) -> str | None:
    """A derived-formula pair or a near-duplicate pair among the named
    columns: the claim restates a definition (stage 13/14) and is not a
    finding the data can support."""
    index = _EvidenceIndex(profile.evidence)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            derived = index.derived_pairs.get(frozenset((a, b)))
            if derived is not None:
                return f"definitional: {derived.target} = {derived.formula}"
            if index.suppressed.get(a) == b or index.suppressed.get(b) == a:
                return f"near-duplicate pair: {a} / {b}"
    return None


def effect_from_chart(spec: ChartSpec, profile: DatasetProfile, score: float) -> tuple[str, float, int]:
    """Effect summary for a chart-only hypothesis verified by
    evaluate_llm_spec: the statistic the score was derived from."""
    index = _EvidenceIndex(profile.evidence)
    n = profile.profiled_rows or profile.n_rows
    if spec.type in ("bar", "box") and spec.x and spec.y:
        eta2 = index.eta2.get((spec.x, spec.y))
        if eta2 is not None:
            return "adjusted eta-squared of y across x groups", eta2, n
    if spec.type == "scatter" and spec.x and spec.y:
        corr = profile.correlations
        value = None
        if corr is not None and spec.x in corr.columns and spec.y in corr.columns:
            value = corr.matrix[corr.columns.index(spec.x)][corr.columns.index(spec.y)]
        spearman = index.spearman.get((spec.x, spec.y))
        strength = max(abs(value or 0.0), abs(spearman or 0.0))
        return "|correlation| (max of Pearson, Spearman)", strength, n
    if spec.type == "line" and spec.x and spec.y:
        eta2 = index.time_eta2.get((spec.x, spec.y))
        if eta2 is not None:
            return "adjusted eta-squared of y across time buckets", eta2, n
    return "evidence score", score, n


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)
