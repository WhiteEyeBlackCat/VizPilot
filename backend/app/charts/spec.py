"""ChartSpec model and the per-type validation matrix.

validate_spec collects every violation (never stops at the first) and is the
single gate shared by the rule engine, the LLM output (Stage 5) and the
manual builder (Stage 6).
"""

import re
from typing import Literal

from pydantic import BaseModel

from ..profiling.models import ColumnProfile, DatasetProfile

ChartType = Literal["line", "bar", "scatter", "histogram", "box", "heatmap"]
Aggregation = Literal["mean", "sum", "count", "median", "min", "max"]
TimeGranularity = Literal["raw", "day", "week", "month"]

BINS_RANGE = (5, 200)
TOP_N_RANGE = (3, 50)
GROUP_BY_MAX_CATEGORIES = 20
BOX_X_CATEGORIES_RANGE = (2, 50)

_NUMERIC_DTYPE_RE = re.compile(r"^(U?Int|Float)\d+$")
_AXIS_BANNED = ("id", "unknown", "text")


class ChartSpec(BaseModel):
    title: str
    type: ChartType
    x: str | None = None
    y: str | None = None
    group_by: str | None = None
    aggregation: Aggregation | None = None
    bins: int | None = None  # histogram only
    time_granularity: TimeGranularity | None = None  # line with datetime x only
    top_n: int | None = None  # bar only; render applies default 20
    reason: str = ""
    priority: int = 100  # smaller = shown first


def _is_numeric_y(col: ColumnProfile) -> bool:
    # rule 5: numeric-backed categorical (e.g. 1-5 rating) is an acceptable y
    return col.semantic_type == "numeric" or (
        col.semantic_type == "categorical" and bool(_NUMERIC_DTYPE_RE.match(col.original_dtype))
    )


def _is_groupable(col: ColumnProfile) -> bool:
    return col.semantic_type in ("categorical", "boolean")


def _categories(col: ColumnProfile) -> int:
    return col.n_categories if col.n_categories is not None else col.unique_count


def validate_spec(spec: ChartSpec, profile: DatasetProfile) -> list[str]:
    errors: list[str] = []
    cols = {c.name: c for c in profile.columns}

    def col(role: str) -> ColumnProfile | None:
        """Existence/axis-eligibility errors are reported once, here."""
        name = getattr(spec, role)
        if name is None:
            return None
        c = cols.get(name)
        if c is None:
            errors.append(f"{role} column '{name}' does not exist")
            return None
        if c.semantic_type in _AXIS_BANNED:
            errors.append(
                f"{role} column '{name}' has semantic type '{c.semantic_type}' "
                "and cannot be used as a chart axis"
            )
            return None
        return c

    x, y, group = col("x"), col("y"), col("group_by")

    # Axes must be pairwise distinct: x==y draws an identity chart, and
    # group_by==y groups by the very column being aggregated.
    named = [(role, getattr(spec, role)) for role in ("x", "y", "group_by")]
    named = [(role, name) for role, name in named if name is not None]
    for i, (role_a, name_a) in enumerate(named):
        for role_b, name_b in named[i + 1 :]:
            if name_a == name_b:
                errors.append(f"{role_a} and {role_b} must be different columns (both '{name_a}')")

    if group is not None:
        if not _is_groupable(group):
            errors.append(f"group_by column '{group.name}' must be categorical or boolean")
        elif _categories(group) > GROUP_BY_MAX_CATEGORIES:
            errors.append(
                f"group_by column '{group.name}' has {_categories(group)} categories "
                f"(maximum {GROUP_BY_MAX_CATEGORIES})"
            )

    if spec.bins is not None:
        if spec.type != "histogram":
            errors.append("bins is only valid for histogram")
        elif not BINS_RANGE[0] <= spec.bins <= BINS_RANGE[1]:
            errors.append(f"bins must be within {BINS_RANGE[0]}..{BINS_RANGE[1]}")
    if spec.top_n is not None:
        if spec.type != "bar":
            errors.append("top_n is only valid for bar")
        elif not TOP_N_RANGE[0] <= spec.top_n <= TOP_N_RANGE[1]:
            errors.append(f"top_n must be within {TOP_N_RANGE[0]}..{TOP_N_RANGE[1]}")
    if spec.time_granularity is not None and spec.type != "line":
        errors.append("time_granularity is only valid for line charts")

    check = _TYPE_CHECKS[spec.type]
    check(spec, profile, x, y, group, errors)
    return errors


def _require(role: str, col: ColumnProfile | None, spec: ChartSpec, errors: list[str]) -> bool:
    """True when the column is present and eligible; missing name is an error,
    a name that already failed existence/axis checks adds no duplicate error."""
    if getattr(spec, role) is None:
        errors.append(f"{spec.type} requires {role}")
        return False
    return col is not None


def _check_line(spec, profile, x, y, group, errors) -> None:
    if _require("x", x, spec, errors) and x.semantic_type not in ("datetime", "numeric"):
        errors.append(f"line x column '{x.name}' must be datetime or numeric")
    if _require("y", y, spec, errors) and not _is_numeric_y(y):
        errors.append(f"line y column '{y.name}' must be numeric")
    if spec.time_granularity is not None and x is not None and x.semantic_type != "datetime":
        errors.append("time_granularity requires a datetime x column")
    # granularity/aggregation closure (stage3 blocking #2)
    if spec.time_granularity not in (None, "raw"):
        if spec.aggregation is None:
            errors.append("aggregation is required when time_granularity is coarser than raw")
    elif x is not None and spec.aggregation is None:
        non_missing = profile.n_rows - x.missing_count
        if x.unique_count < non_missing:
            errors.append(
                f"aggregation is required: x column '{x.name}' has duplicate values "
                "and no coarser time_granularity"
            )


def _check_bar(spec, profile, x, y, group, errors) -> None:
    if _require("x", x, spec, errors) and not _is_groupable(x):
        errors.append(f"bar x column '{x.name}' must be categorical or boolean")
    if spec.y is None:
        if spec.aggregation != "count":
            errors.append("bar with y=None requires aggregation='count'")
    else:
        if y is not None and not _is_numeric_y(y):
            errors.append(f"bar y column '{y.name}' must be numeric")
        if spec.aggregation is None:
            errors.append("bar requires aggregation when y is given")


def _check_scatter(spec, profile, x, y, group, errors) -> None:
    if _require("x", x, spec, errors) and x.semantic_type != "numeric":
        errors.append(f"scatter x column '{x.name}' must be numeric")
    if _require("y", y, spec, errors) and y.semantic_type != "numeric":
        errors.append(f"scatter y column '{y.name}' must be numeric")
    if spec.aggregation is not None:
        errors.append("scatter must not have an aggregation")


def _check_histogram(spec, profile, x, y, group, errors) -> None:
    if _require("x", x, spec, errors) and x.semantic_type != "numeric":
        errors.append(f"histogram x column '{x.name}' must be numeric")
    if spec.y is not None:
        errors.append("histogram must not have y")
    if spec.aggregation is not None:
        errors.append("histogram must not have an aggregation")


def _check_box(spec, profile, x, y, group, errors) -> None:
    if x is not None:
        if not _is_groupable(x):
            errors.append(f"box x column '{x.name}' must be categorical or boolean")
        else:
            lo, hi = BOX_X_CATEGORIES_RANGE
            if not lo <= _categories(x) <= hi:
                errors.append(f"box x column '{x.name}' must have {lo}..{hi} categories")
    if _require("y", y, spec, errors) and not _is_numeric_y(y):
        errors.append(f"box y column '{y.name}' must be numeric")
    if spec.group_by is not None:
        errors.append("box must not have group_by (x is the grouping)")
    if spec.aggregation is not None:
        errors.append("box must not have an aggregation")


def _check_heatmap(spec, profile, x, y, group, errors) -> None:
    for role in ("x", "y", "group_by", "aggregation"):
        if getattr(spec, role) is not None:
            errors.append(f"heatmap must not have {role}")
    # dataset-level requirement (stage3 blocking #4)
    n_numeric = sum(1 for c in profile.columns if c.semantic_type == "numeric")
    if n_numeric < 2:
        errors.append("heatmap requires at least 2 numeric columns in the dataset")


_TYPE_CHECKS = {
    "line": _check_line,
    "bar": _check_bar,
    "scatter": _check_scatter,
    "histogram": _check_histogram,
    "box": _check_box,
    "heatmap": _check_heatmap,
}
