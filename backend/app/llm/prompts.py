"""Prompt assembly. The user message is built exclusively from the profile —
raw dataset rows never leave the machine beyond the profile's 5 sample rows,
and even those can be disabled via settings.
"""

import json
from typing import Any

from ..charts.rules import (
    LINE_GROUP_INTERACTION_MIN,
    Recommendation,
    slope_spread_threshold,
)
from ..profiling.models import ColumnProfile, DatasetProfile

MAX_COLUMNS = 40
MAX_CORR_PAIRS = 8
MAX_ETA_PAIRS = 8
MAX_CELL_CHARS = 100

SYSTEM_PROMPT = """You are a data visualization assistant. Given a dataset profile with measured \
statistical evidence and a list of rule-generated chart candidates, you produce short analytical \
insights and chart suggestions. Your hypotheses will be verified against the evidence afterwards.

Respond with a single JSON object, no prose, matching exactly:
{
  "insights": [
    {
      "text": "one short observation about the data",
      "chart": {
        "title": "string",
        "type": "line|bar|scatter|histogram|box|heatmap",
        "x": "column name or null",
        "y": "column name or null",
        "group_by": "column name or null",
        "aggregation": "mean|sum|count|median|min|max or null",
        "reason": "why this chart supports the insight",
        "priority": 1
      }
    }
  ],
  "charts": [ { ...same chart shape... } ]
}

Rules:
- Every insight MUST include a supporting "chart" (same shape as a chart suggestion).
- Base insights on the evidence section. If the evidence shows a column's effect is close to \
zero, do NOT draw conclusions about that column.
- Use only the listed columns, with exact column names (case-sensitive).
- Allowed chart types: line, bar, scatter, histogram, box, heatmap. Never suggest pie charts.
- Allowed aggregations: mean, sum, count, median, min, max.
- Every line chart MUST include an aggregation (e.g. "mean").
- 2 to 5 insights; at most 6 charts in total. Lower priority number = more important.
- "charts" is for extra suggestions or re-ranking existing candidates; it may be empty.

Example response:
{"insights": [{"text": "Sales rise steadily over the year, and the trend differs by region.",
               "chart": {"title": "Sales over time", "type": "line", "x": "date", "y": "sales",
                         "group_by": "region", "aggregation": "mean",
                         "reason": "Shows the seasonal trend per region.", "priority": 1}}],
 "charts": []}"""


def build_messages(
    profile: DatasetProfile,
    rule_candidates: list[Recommendation],
    include_sample_rows: bool = True,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_content(profile, rule_candidates, include_sample_rows)},
    ]


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
        lines += ["", "Sample rows:"]
        for row in profile.sample_rows:
            lines.append(json.dumps({k: _truncate_cell(v) for k, v in row.items()}, ensure_ascii=False))

    lines += ["", "Rule-generated chart candidates:"]
    for rec in rule_candidates:
        s = rec.spec
        lines.append(f"- {s.type}: x={s.x}, y={s.y}, group_by={s.group_by}, aggregation={s.aggregation}")

    lines += [
        "",
        "Interpret the column semantics and give 2-5 insights, each with its supporting chart. "
        "Build on the evidence above — do not conclude anything about columns whose measured "
        "effects are close to zero. Add useful charts the rules missed and rank the most "
        "important charts with priority numbers.",
    ]
    return "\n".join(lines)


def _evidence_section(profile: DatasetProfile) -> list[str]:
    """Measured-evidence summary so LLM hypotheses start from data, not
    guesses. Caps (critique #5): top 8 eta pairs, top 8 correlation pairs
    (with spearman), only above-threshold interaction/slope hits, 2 decimals."""
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
            f"- {e.cat1} x {e.cat2} -> {e.num}: strength={e.strength:.2f}" for e in interactions
        ]

    slopes = [
        s
        for s in evidence.slope_heterogeneity
        if s.spread >= slope_spread_threshold(s.n_min)
    ]
    if slopes:
        lines.append("Slope heterogeneity hits (relationship differs across groups):")
        lines += [f"- {s.x} vs {s.y} by {s.group}: spread={s.spread:.2f}" for s in slopes]

    return lines if len(lines) > 2 else []


def _describe_column(col: ColumnProfile) -> str:
    base = f"{col.name} ({col.semantic_type}"
    if col.missing_ratio > 0:
        base += f", {col.missing_ratio:.0%} missing"
    base += ")"
    if col.semantic_type == "numeric":
        return f"{base}: mean={col.mean}, std={col.std}, min={col.min}, max={col.max}"
    if col.semantic_type in ("categorical", "boolean"):
        top = ", ".join(str(t.value) for t in (col.top_values or [])[:3])
        return f"{base}: {col.n_categories} categories, top: {top}"
    if col.semantic_type == "datetime":
        return f"{base}: {col.min} to {col.max}, frequency={col.inferred_frequency}"
    return base


def _top_correlations(profile: DatasetProfile) -> list[tuple[str, str, float]]:
    corr = profile.correlations
    if corr is None:
        return []
    pairs = []
    for i, a in enumerate(corr.columns):
        for j in range(i + 1, len(corr.columns)):
            value = corr.matrix[i][j]
            if value is not None:
                pairs.append((a, corr.columns[j], value))
    pairs.sort(key=lambda p: -abs(p[2]))
    return pairs[:MAX_CORR_PAIRS]


def _truncate_cell(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_CELL_CHARS:
        return value[:MAX_CELL_CHARS] + "…"
    return value
