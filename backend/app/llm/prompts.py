"""Prompt assembly for the two-stage workflow (stage 17.3).

Both user messages are built exclusively from the profile — raw dataset rows
never leave the machine beyond the profile's 5 sample rows (semantics only,
never evidence), and even those can be disabled via settings. Evidence is
selected by importance and diversity (flagged / above-threshold entries,
a few per category, series compressed to shapes), not dumped.
"""

import json
from typing import Any

from ..charts.rules import (
    LINE_GROUP_INTERACTION_MIN,
    Recommendation,
    slope_spread_threshold,
)
from ..probes.engine import THRESHOLDS
from ..profiling.models import (
    ChangePoint,
    ColumnProfile,
    ConditionalRelationship,
    DatasetProfile,
    DistributionSummary,
    GroupSummary,
    GroupTimePattern,
    NonlinearSignal,
    SubgroupAnomaly,
)
from .coverage import ValidatedHypothesis

MAX_COLUMNS = 40
MAX_CORR_PAIRS = 8
MAX_ETA_PAIRS = 8
MAX_CELL_CHARS = 100
MAX_SAMPLE_ROWS = 5
# layer-2 selection caps: a few entries per category, strongest first
L2_PER_CATEGORY = 3
L2_GROUP_MIN_DIFF_SD = 0.3  # group summaries below this mean gap are not worth a line
L2_MIN_SKEW = 1.0  # distributions: only clearly skewed or multimodal ones
L2_MIN_GROUP_CORR = 0.30  # conditional relationships: some group must show a real correlation

PROBE_TYPE_GUIDE = """Probe types (test_needed) and their column roles — the backend computes the statistic, you \
only choose type and columns:
- group_difference {group, target}: mean of target differs across the groups.
- distribution_difference {group, target}: spread/shape of target differs across the groups.
- grouped_relationship {x, y, group}: the x-y relationship holds inside EVERY group.
- slope_difference {x, y, group}: the x-y relationship DIFFERS between groups (moderation).
- nonlinear_relationship {x, y}: y depends on x non-linearly (U shape, saturation, peaks).
- time_pattern {time, target}: target changes over time (trend, seasonality, level shift). A time \
pattern that differs BY GROUP can only be proposed when the evidence lists that time x group \
interaction (it cannot be probed).
- interaction {factor1, factor2, target}: the effect of one factor on target depends on the other.
Role constraints (violations are rejected, not run): group / factor1 / factor2 = categorical or \
boolean with 2-20 categories (distribution_difference: up to 50); x / y / target = numeric (a \
categorical column can never be x or y); time = datetime. Also rejected: identifiers, text, \
constant columns, >50% missing, the same column in two roles, derived-formula pairs, \
near-duplicate pairs. Use distribution_difference when group means look alike but the shapes may \
differ, interaction when you suspect two factors act together. At most 5 probes per dataset: \
request one only when the evidence section does not already answer the question; otherwise \
test_needed = null, evidence_available = true."""

HYPOTHESIS_SYSTEM_PROMPT = """You are the hypothesis generator of a data exploration assistant. You read a \
dataset profile with measured evidence (effect sizes, correlations, distribution / group / \
conditional / non-linear / change summaries) and propose the few patterns worth checking. You do \
NOT write final insights and you do NOT compute statistics: the backend verifies every hypothesis \
and drops what fails.

Respond with a single JSON object, no prose, matching exactly:
{
  "hypotheses": [
    {
      "statement": "one testable claim, in plain words",
      "variables": ["column", "column"],
      "evidence_available": true,
      "test_needed": "<probe type> or null",
      "columns": {"role": "column name"},
      "chart": {"title": "string", "type": "line|bar|scatter|histogram|box|heatmap", "x": "column or null", \
"y": "column or null", "group_by": "column or null", "aggregation": "mean|sum|count|median|min|max or null", \
"reason": "why this chart shows it", "priority": 1},
      "reason": "why this changes how a user understands the dataset",
      "importance": 3
    }
  ],
  "charts": []
}

Rules:
- 0 to 5 hypotheses. Returning an empty list is a correct answer when nothing meets all of: \
statistically supported by the evidence, analytically meaningful, non-definitional, non-redundant.
- Do not report a relationship merely because its effect size or correlation is large. An insight \
should be analytically useful: it should meaningfully change how a user understands the dataset.
- Never treat these as findings: derived / definitional relationships (a column computed from \
others by a formula), near-duplicate variables, identifiers, measurements that obviously repeat \
each other, effects close to zero, the same phenomenon stated twice, or a bare "X correlates with Y" \
that adds no analytical value.
- Prefer: meaningful group differences, temporal or seasonal patterns, interactions, heterogeneous \
relationships (a relationship that differs between groups), non-linear relationships, \
subgroup-specific behaviour, distribution differences, unusual but well-supported patterns, and \
anything that changes how the data should be interpreted.
- No strong hypotheses about effects the evidence shows as close to zero; no probes just to \
probe. Use only the listed columns, spelled exactly (case-sensitive).
- importance: 5 = changes the whole reading of the dataset, 1 = minor. A chart is required. \
Chart types: line, bar, scatter, histogram, box, heatmap (never pie); aggregations: mean, sum, \
count, median, min, max; every line chart needs an aggregation.
- "charts" (optional, may be empty): extra charts or re-ranking of the rule candidates, same \
chart shape, lower priority number = more important.

""" + PROBE_TYPE_GUIDE + """

Example: {"hypotheses": [{"statement": "Casual riders are far more numerous on weekends.", \
"variables": ["weekday", "casual"], "evidence_available": true, "test_needed": null, "columns": \
{"group": "weekday", "target": "casual"}, "chart": {"title": "Mean casual by weekday", "type": "bar", \
"x": "weekday", "y": "casual", "group_by": null, "aggregation": "mean", "reason": "weekend jump", \
"priority": 1}, "reason": "leisure vs commuting demand", "importance": 4}], "charts": []}"""

FINAL_SYSTEM_PROMPT = """You are the final analyst of a data exploration assistant. You receive \
hypotheses the backend has already validated, each with its measured evidence (effect size, sample \
size, per-group or per-bin numbers) and candidate charts. Your job is communication only: write a \
concise, precise insight for each validated hypothesis and say why it matters.

Hard rules:
- Do NOT add any factual claim that is not in the supplied evidence, and do NOT change, round \
differently, or invent numbers. Quote numbers from the evidence as given (you may omit them).
- Do NOT mention columns that are not listed. Do NOT merge two hypotheses into one claim.
- One insight per hypothesis_id at most; you may leave a hypothesis out if it adds nothing new.
- text: 1-2 sentences, specific, no hedging words like "may" for validated findings. \
why_it_matters: one sentence. priority: 5 = most important.
- chart_id must be one of the listed candidate ids for that hypothesis, or null.

Respond with a single JSON object, no prose, matching exactly:
{
  "insights": [
    {"hypothesis_id": 1, "text": "...", "why_it_matters": "...", "priority": 4, "chart_id": "h1c1"}
  ],
  "message": null
}
If no hypothesis is worth reporting, respond with {"insights": [], "message": "No strong \
non-definitional and analytically useful patterns were found."}"""


# --- LLM #1 -------------------------------------------------------------------


def build_hypothesis_messages(
    profile: DatasetProfile,
    rule_candidates: list[Recommendation],
    include_sample_rows: bool = True,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": HYPOTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": _user_content(profile, rule_candidates, include_sample_rows)},
    ]


# the single-call name kept for callers/tests that only care about "the prompt"
build_messages = build_hypothesis_messages


def _user_content(
    profile: DatasetProfile, rule_candidates: list[Recommendation], include_sample_rows: bool
) -> str:
    lines = [f"Dataset: {profile.n_rows} rows, {profile.n_cols} columns.", "", "Columns:"]
    columns = profile.columns[:MAX_COLUMNS]
    for col in columns:
        lines.append(f"- {_describe_column(col)}")
    if len(profile.columns) > MAX_COLUMNS:
        lines.append(f"(only the first {MAX_COLUMNS} of {len(profile.columns)} columns are listed)")

    lines += _evidence_section(profile)

    if include_sample_rows and profile.sample_rows:
        lines += ["", "Sample rows (for column semantics only — never statistical evidence):"]
        for row in profile.sample_rows[:MAX_SAMPLE_ROWS]:
            lines.append(json.dumps({k: _truncate_cell(v) for k, v in row.items()}, ensure_ascii=False))

    lines += ["", "Rule-generated chart candidates:"]
    for rec in rule_candidates:
        s = rec.spec
        lines.append(f"- {s.type}: x={s.x}, y={s.y}, group_by={s.group_by}, aggregation={s.aggregation}")

    lines += ["", "Propose 0-5 hypotheses worth verifying, each with its supporting chart."]
    return "\n".join(lines)


def _evidence_section(profile: DatasetProfile) -> list[str]:
    """Measured-evidence summary so LLM hypotheses start from data, not
    guesses. Layer 1 caps (critique #5): top 8 eta pairs, top 8 correlation
    pairs (with spearman), only above-threshold interaction/slope hits.
    Layer 2 (stage 17.3): only flagged / above-threshold entries, a few per
    category, series compressed to shapes."""
    lines: list[str] = ["", "Measured evidence (effect sizes; ~0.00 means no effect):"]
    evidence = profile.evidence

    etas = sorted(evidence.cat_num, key=lambda e: (-e.eta_squared, e.cat, e.num))[:MAX_ETA_PAIRS]
    if etas:
        lines.append("Group effects (eta, 0..1):")
        lines += [f"- {e.cat} -> {e.num}: eta={e.eta_squared ** 0.5:.2f}" for e in etas]

    pairs = _top_correlations(profile)
    if pairs:
        spearman = evidence.num_num_spearman
        lines.append("Strongest correlations:")
        for a, b, value in pairs:
            entry = f"- {a} vs {b}: pearson={value:.2f}"
            if spearman is not None and a in spearman.columns and b in spearman.columns:
                s = spearman.matrix[spearman.columns.index(a)][spearman.columns.index(b)]
                if s is not None:
                    entry += f", spearman={s:.2f}"
            lines.append(entry)

    trends = sorted(evidence.time_effects, key=lambda t: (-t.eta_squared, t.num))[:MAX_ETA_PAIRS]
    if trends:
        lines.append("Time effects (eta over time buckets):")
        lines += [
            f"- {t.datetime_col} -> {t.num}: eta={t.eta_squared ** 0.5:.2f} (per {t.bucket})"
            for t in trends
        ]

    interactions = [e for e in evidence.interactions if e.strength >= LINE_GROUP_INTERACTION_MIN]
    if interactions:
        lines.append("Interaction hits (trend differs across groups):")
        lines += [
            f"- {_factor_label(e.cat1)} x {_factor_label(e.cat2)} -> {e.num}: strength={e.strength:.2f}"
            for e in interactions
        ]

    slopes = [
        s
        for s in evidence.slope_heterogeneity
        if s.spread >= slope_spread_threshold(s.n_min)
    ]
    if slopes:
        lines.append("Slope heterogeneity hits (relationship differs across groups):")
        lines += [f"- {s.x} vs {s.y} by {s.group}: spread={s.spread:.2f}" for s in slopes]

    lines += _layer2_section(profile)

    derived = getattr(evidence, "derived_columns", [])
    identities = [d for d in derived if d.kind != "near_copy"]
    if identities:
        lines.append(
            "Derived columns (definitional — NOT discoveries; never present these, or a chart of the "
            "column against its inputs, as an insight):"
        )
        lines += [f"- {d.target} = {d.formula}" for d in identities]
    groups = getattr(evidence, "near_duplicate_groups", [])
    if groups:
        lines.append(
            "Near-duplicate columns (NOT discoveries; use the representative only; never chart a "
            "duplicate or the pair — it only shows the duplication):"
        )
        for g in groups:
            dups = ", ".join(f"{d} (rank correlation {g.rho[d]:.2f})" for d in g.duplicates)
            lines.append(f"- {dups} ≈ {g.representative} — use {g.representative}")

    return lines if len(lines) > 2 else []


def _layer2_section(profile: DatasetProfile) -> list[str]:
    l2 = getattr(profile.evidence, "layer2", None)
    if l2 is None:
        return []
    lines: list[str] = []

    nonlinear = sorted(
        (
            s
            for s in l2.nonlinear
            if s.shape not in ("flat", "linear") and s.nonlinear_gap >= THRESHOLDS["nonlinear_relationship"]["weak"]
        ),
        key=lambda s: (-s.nonlinear_gap, s.x, s.y),
    )[:L2_PER_CATEGORY]
    if nonlinear:
        lines.append("Non-linear relationships (y averaged in 10 equal-frequency bins of x, low to high x):")
        lines += [_nonlinear_line(s) for s in nonlinear]

    conditional = sorted(
        (
            c
            for c in l2.conditional_relationships
            if c.corr_spread >= slope_spread_threshold(c.n_min) / 2
            # a spread only matters when some group shows a real relationship
            # (the same corroboration the workflow applies); this keeps the
            # largest-of-many noise spreads out of the LLM's view
            and max((abs(g.corr) for g in c.groups if g.corr is not None), default=0.0) >= L2_MIN_GROUP_CORR
        ),
        key=lambda c: (-c.corr_spread, c.x, c.y, c.group),
    )[:L2_PER_CATEGORY]
    if conditional:
        lines.append("Relationships that differ by group (correlation per group):")
        lines += [_conditional_line(c) for c in conditional]

    anomalies = sorted(l2.subgroup_anomalies, key=lambda a: (-abs(a.robust_z), a.cat, a.num, a.group))[
        :L2_PER_CATEGORY
    ]
    if anomalies:
        lines.append("Subgroups that stand out (group mean vs the other groups):")
        lines += [_anomaly_line(a) for a in anomalies]

    changes = sorted(
        (c for c in l2.change_points if c.strength in ("strong", "weak")),
        key=lambda c: (0 if c.strength == "strong" else 1, -c.effect_size, c.num),
    )[:L2_PER_CATEGORY]
    if changes:
        lines.append("Level shifts over time (single best split of the bucket means):")
        lines += [_change_line(c) for c in changes]

    groups = sorted(
        (g for g in l2.group_summaries if g.max_diff_sd is not None and g.max_diff_sd >= L2_GROUP_MIN_DIFF_SD),
        key=lambda g: (-(g.max_diff_sd or 0.0), g.cat, g.num),
    )[:L2_PER_CATEGORY]
    if groups:
        lines.append("Group means (largest gaps, in pooled within-group SDs):")
        lines += [_group_line(g) for g in groups]

    dists = sorted(
        (
            d
            for d in l2.distributions
            if d.multimodal_signal or (d.skewness is not None and abs(d.skewness) >= L2_MIN_SKEW)
        ),
        key=lambda d: (0 if d.multimodal_signal else 1, -abs(d.skewness or 0.0), d.column),
    )[:L2_PER_CATEGORY]
    if dists:
        lines.append("Distribution shapes worth knowing:")
        lines += [_distribution_line(d) for d in dists]

    # group_time_patterns are deliberately not listed: long series, noisy on
    # small groups (verifier note) — time_pattern probes answer those questions
    return lines


def _nonlinear_line(s: NonlinearSignal) -> str:
    means = ", ".join(f"{b.y_mean:.3g}" for b in s.bins)
    r2 = "n/a" if s.r2_pearson is None else f"{s.r2_pearson:.2f}"
    return (
        f"- {s.x} -> {s.y}: shape={s.shape}, binned eta2={s.binned_eta2:.2f}, linear r2={r2}, "
        f"bin means={means} (n={s.n})"
    )


def _conditional_line(c: ConditionalRelationship) -> str:
    parts = []
    for g in c.groups:
        corr = "n/a" if g.corr is None else f"{g.corr:.2f}"
        parts.append(f"{g.group}={corr} (n={g.n})")
    return f"- {c.x} vs {c.y} by {c.group}: {', '.join(parts)}; spread={c.corr_spread:.2f}"


def _anomaly_line(a: SubgroupAnomaly) -> str:
    return (
        f"- {a.cat}={a.group}: mean {a.num} {_format_number(round(a.mean, 3))} vs "
        f"{_format_number(round(a.others_median, 3))} in the other groups (z={a.robust_z:.1f}, "
        f"d={a.diff_sd:.2f}, n={a.n})"
    )


def _change_line(c: ChangePoint) -> str:
    return (
        f"- {c.num} over {c.datetime_col}: level shifts at {c.change_at[:10]} from "
        f"{_format_number(round(c.before_mean, 3))} to {_format_number(round(c.after_mean, 3))} "
        f"(strength {c.strength}, {c.diff_sd:.2f} SD)"
    )


def _group_line(g: GroupSummary) -> str:
    listed = sorted(g.groups, key=lambda s: -s.mean)[:6]
    means = ", ".join(f"{s.group}={_format_number(round(s.mean, 3))} (n={s.n})" for s in listed)
    more = "" if len(g.groups) <= 6 else f", … {len(g.groups)} groups"
    return f"- {g.cat} -> {g.num}: {means}{more}; max gap={g.max_diff_sd:.2f} SD, eta2={g.eta_squared:.2f}"


def _distribution_line(d: DistributionSummary) -> str:
    skew = "n/a" if d.skewness is None else f"{d.skewness:.2f}"
    modes = f", {d.modes} separated peaks" if d.multimodal_signal else ""
    return (
        f"- {d.column}: {d.shape} (skew {skew}){modes}; p05={_format_number(round(d.p05, 3))}, "
        f"median={_format_number(round(d.p50, 3))}, p95={_format_number(round(d.p95, 3))} (n={d.n})"
    )


def _pattern_line(p: GroupTimePattern) -> str:
    parts = []
    for name, points in p.series.items():
        if not points:
            continue
        peak = max(points, key=lambda pt: pt.mean)
        lo = min(pt.mean for pt in points)
        parts.append(
            f"{name}: peak {peak.bucket[:10]} ({_format_number(round(peak.mean, 3))}), "
            f"min {_format_number(round(lo, 3))}"
        )
    return f"- {p.num} over {p.datetime_col} by {p.group} (per {p.bucket}): " + "; ".join(parts)


# --- LLM #2 -------------------------------------------------------------------


def build_final_messages(profile: DatasetProfile, validated: list[ValidatedHypothesis]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": FINAL_SYSTEM_PROMPT},
        {"role": "user", "content": _final_user_content(profile, validated)},
    ]


def _final_user_content(profile: DatasetProfile, validated: list[ValidatedHypothesis]) -> str:
    names = {c.name for c in profile.columns}
    used = sorted({name for v in validated for name in v.columns.values()} & names)
    lines = [f"Dataset: {profile.n_rows} rows, {profile.n_cols} columns.", "", "Columns involved:"]
    by_name = {c.name: c for c in profile.columns}
    for name in used:
        lines.append(f"- {_describe_column_brief(by_name[name])}")
    lines += ["", "Validated hypotheses (backend-verified; numbers are authoritative):"]
    for v in validated:
        lines.append(
            f"[{v.id}] {v.statement}\n"
            f"    test: {v.test_type}, columns: {json.dumps(v.columns)}, verdict: {v.verdict}, "
            f"validated by: {v.validation}\n"
            f"    effect: {v.effect_label} = {v.effect_value:.3f} (n={v.n})"
        )
        for line in v.evidence_lines[:8]:
            lines.append(f"    - {line}")
        for cid, spec in v.chart_candidates:
            lines.append(
                f"    chart {cid}: {spec.type}, x={spec.x}, y={spec.y}, group_by={spec.group_by}, "
                f"aggregation={spec.aggregation}"
            )
    lines += [
        "",
        "Write the final insights. Use only the facts above; leave out any hypothesis that adds "
        "nothing analytically useful.",
    ]
    return "\n".join(lines)


# --- shared -------------------------------------------------------------------


def _describe_column(col: ColumnProfile) -> str:
    base = f"{col.name} ({col.semantic_type}"
    if col.missing_ratio > 0:
        base += f", {col.missing_ratio:.0%} missing"
    base += ")"
    if col.semantic_type == "id":
        # stage 9 #4: identifiers are metadata only — never an axis
        return f"{base}: identifier, not usable as a chart axis"
    if col.semantic_type == "numeric":
        line = f"{base}: mean={col.mean}, std={col.std}, min={col.min}, max={col.max}"
        # stage 9 #6 (decision B.7): quality flags, summary-level only — the
        # distinct suspected values and a count, never rows
        quality = getattr(col, "quality", None)
        sentinels = getattr(quality, "suspected_sentinels", None) or []
        if sentinels:
            values = ", ".join(_format_number(s.value) for s in sorted(sentinels, key=lambda s: s.value))
            line += f"; suspected sentinel values: {values} (mean/min/max are distorted)"
        extreme = int(getattr(quality, "extreme_value_count", 0) or 0)
        if extreme:
            line += f"; {extreme} extreme values"
        return line
    if col.semantic_type in ("categorical", "boolean"):
        top = ", ".join(str(t.value) for t in (col.top_values or [])[:3])
        summary = f"{base}: {col.n_categories} categories, top: {top}"
        if getattr(col, "nominal", False):
            # numeric-looking code: a label, never a quantity to average
            summary += " (nominal code - not a quantity; never use as y or aggregate it)"
        return summary
    if col.semantic_type == "datetime":
        return f"{base}: {col.min} to {col.max}, frequency={col.inferred_frequency}"
    return base


def _describe_column_brief(col: ColumnProfile) -> str:
    """LLM #2 column line: type and a rounded range only (no quality flags,
    no extreme-value counts — nothing the analyst could mistake for a fact
    to report)."""
    base = f"{col.name} ({col.semantic_type})"
    if col.semantic_type == "numeric":
        return f"{base}: mean={_round3(col.mean)}, min={_round3(col.min)}, max={_round3(col.max)}"
    if col.semantic_type in ("categorical", "boolean"):
        top = ", ".join(str(t.value) for t in (col.top_values or [])[:3])
        return f"{base}: {col.n_categories} categories, e.g. {top}"
    if col.semantic_type == "datetime":
        return f"{base}: {str(col.min)[:10]} to {str(col.max)[:10]}"
    return base


def _round3(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    rounded = round(float(value), 3)
    return int(rounded) if float(rounded).is_integer() else rounded


def _top_correlations(profile: DatasetProfile) -> list[tuple[str, str, float]]:
    """Strongest pairs, leaving out the ones that only restate a definition
    (derived-formula pairs) or a duplication (near-duplicate pairs)."""
    corr = profile.correlations
    if corr is None:
        return []
    evidence = profile.evidence
    skip: set[frozenset[str]] = set()
    for d in getattr(evidence, "derived_columns", []):
        for component in d.components:
            skip.add(frozenset((d.target, component)))
    suppressed = {dup for g in getattr(evidence, "near_duplicate_groups", []) for dup in g.duplicates}
    pairs = []
    for i, a in enumerate(corr.columns):
        for j in range(i + 1, len(corr.columns)):
            b = corr.columns[j]
            value = corr.matrix[i][j]
            if value is None or frozenset((a, b)) in skip or a in suppressed or b in suppressed:
                continue
            pairs.append((a, b, value))
    pairs.sort(key=lambda p: -abs(p[2]))
    return pairs[:MAX_CORR_PAIRS]


def _factor_label(name: str) -> str:
    """A layer-1 time marker `col@bucket` is not a column: spell it out so the
    LLM names the datetime column, not the marker."""
    if "@" in name:
        col, bucket = name.split("@", 1)
        return f"{col} (per {bucket})"
    return name


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _truncate_cell(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_CELL_CHARS:
        return value[:MAX_CELL_CHARS] + "…"
    return value
