"""Prompt assembly. The user message is built exclusively from the profile —
raw dataset rows never leave the machine beyond the profile's 5 sample rows,
and even those can be disabled via settings.
"""

import json
from typing import Any

from ..charts.rules import Recommendation
from ..profiling.models import ColumnProfile, DatasetProfile

MAX_COLUMNS = 40
MAX_CORR_PAIRS = 10
MAX_CELL_CHARS = 100

SYSTEM_PROMPT = """You are a data visualization assistant. Given a dataset profile and a list of \
rule-generated chart candidates, you suggest insightful charts and short analytical insights.

Respond with a single JSON object, no prose, matching exactly:
{
  "insights": ["2 to 5 short observations about the data"],
  "charts": [
    {
      "title": "string",
      "type": "line|bar|scatter|histogram|box|heatmap",
      "x": "column name or null",
      "y": "column name or null",
      "group_by": "column name or null",
      "aggregation": "mean|sum|count|median|min|max or null",
      "reason": "why this chart is useful",
      "priority": 1
    }
  ]
}

Rules:
- Use only the listed columns, with exact column names (case-sensitive).
- Allowed chart types: line, bar, scatter, histogram, box, heatmap. Never suggest pie charts.
- Allowed aggregations: mean, sum, count, median, min, max.
- Every line chart MUST include an aggregation (e.g. "mean").
- Suggest at most 6 charts. Lower priority number = more important.
- You may re-rank the existing candidates by including them with your own priority and reason.

Example response:
{"insights": ["Sales grow steadily over the year.", "The North region outperforms the others."],
 "charts": [{"title": "Sales over time", "type": "line", "x": "date", "y": "sales",
             "group_by": "region", "aggregation": "mean",
             "reason": "Shows the seasonal trend per region.", "priority": 1}]}"""


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

    pairs = _top_correlations(profile)
    if pairs:
        lines += ["", "Strongest correlations:"]
        lines += [f"- {a} vs {b}: {v:.2f}" for a, b, v in pairs]

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
        "Interpret the column semantics, give 2-5 insights, add useful charts the rules missed, "
        "and rank the most important charts with priority numbers.",
    ]
    return "\n".join(lines)


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
