"""ChartSpec model and the per-type validation matrix.

validate_spec collects every violation (never stops at the first) and is the
single gate shared by the rule engine, the LLM output (Stage 5) and the
manual builder (Stage 6).
"""

import re
from typing import Literal

from pydantic import BaseModel

from ..profiling.models import ColumnProfile, DatasetProfile
from .roles import column_role

# the six original types plus the stage-19 family; the type says how the
# chart is drawn, the intent (charts/intent.py) says why it was chosen
ChartType = Literal[
    "line",
    "bar",
    "scatter",
    "histogram",
    "box",
    "heatmap",
    "violin",
    "raincloud",
    "ridgeline",
    "dot_interval",
    "lollipop",
    "dumbbell",
    "slopegraph",
    "hexbin",
    "polar_line",
    "rose",
    # stage 20b: composition (part-to-whole) and flow
    "stacked_bar",
    "sankey",
]
Aggregation = Literal["mean", "sum", "count", "median", "min", "max"]
TimeGranularity = Literal["raw", "day", "week", "month"]
# category order on a category axis (stage 19): "natural" = the column's own
# order (month 1..12, Mon..Sun, low..high), "value" = ranked by the aggregate;
# None keeps the pre-stage-19 behaviour (ordered roles natural, else ranked)
SortOrder = Literal["natural", "value"]

BINS_RANGE = (5, 200)
TOP_N_RANGE = (3, 50)
GROUP_BY_MAX_CATEGORIES = 20
BOX_X_CATEGORIES_RANGE = (2, 50)
RAINCLOUD_X_CATEGORIES_RANGE = (2, 8)
RIDGELINE_X_CATEGORIES_RANGE = (2, 12)
TWO_STATE_X_CATEGORIES_RANGE = (2, 50)
POLAR_X_CATEGORIES_MAX = 60
# stage 20b composition: the x axis is a bar axis, the parts are the stack —
# beyond a handful of parts a stack is unreadable and stops being a composition
STACKED_X_CATEGORIES_RANGE = (2, 20)
STACKED_PARTS_RANGE = (2, 8)
# stage 20b flow: both ends of a sankey are category axes of bounded size
SANKEY_CATEGORIES_RANGE = (2, 20)
# only sum / count compose: a mean of parts does not add up to the whole
COMPOSITION_AGGREGATIONS = ("sum", "count")

# types whose x is a category axis (sort applies)
CATEGORY_AXIS_TYPES = frozenset(
    {"bar", "lollipop", "box", "violin", "raincloud", "ridgeline", "dot_interval", "dumbbell", "slopegraph", "polar_line", "rose", "line", "stacked_bar"}
)
# same data as a bar: rendered by the bar aggregation
BAR_LIKE_TYPES = frozenset({"bar", "lollipop", "polar_line", "rose"})
DENSITY_TYPES = frozenset({"violin", "raincloud", "ridgeline"})
TWO_STATE_TYPES = frozenset({"dumbbell", "slopegraph"})

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
    top_n: int | None = None  # bar / lollipop only; render applies default 20
    sort: SortOrder | None = None  # category order (stage 19, additive)
    # stage 20b, additive: stacked_bar only — True normalises every bar to
    # 100 % so the chart reads as a share instead of a magnitude
    percent: bool | None = None
    reason: str = ""
    priority: int = 100  # smaller = shown first


def _is_numeric_y(col: ColumnProfile) -> bool:
    # rule 5: numeric-backed categorical (e.g. 1-5 rating) is an acceptable y —
    # unless its numbers are labels (nominal codes, stage 9 #4): averaging a
    # product code is meaningless
    return col.semantic_type == "numeric" or (
        col.semantic_type == "categorical"
        and not col.nominal
        and bool(_NUMERIC_DTYPE_RE.match(col.original_dtype))
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
        if spec.type not in ("bar", "lollipop"):
            errors.append("top_n is only valid for bar / lollipop")
        elif not TOP_N_RANGE[0] <= spec.top_n <= TOP_N_RANGE[1]:
            errors.append(f"top_n must be within {TOP_N_RANGE[0]}..{TOP_N_RANGE[1]}")
    if spec.time_granularity is not None and spec.type != "line":
        errors.append("time_granularity is only valid for line charts")
    if spec.sort is not None and spec.type not in CATEGORY_AXIS_TYPES:
        errors.append(f"sort is not valid for {spec.type} (no category axis)")
    if spec.percent is not None and spec.type != "stacked_bar":
        errors.append("percent is only valid for stacked_bar")

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


def _is_ordered_category(col: ColumnProfile) -> bool:
    """A categorical whose values have their own order (month, weekday,
    rating): may serve as a line's x (stage 19) — the line follows that order."""
    return col.semantic_type == "categorical" and column_role(col).ordered


def _check_line(spec, profile, x, y, group, errors) -> None:
    if _require("x", x, spec, errors) and x.semantic_type not in ("datetime", "numeric") and not _is_ordered_category(x):
        errors.append(f"line x column '{x.name}' must be datetime, numeric or an ordered category (month, weekday, level)")
    if _require("y", y, spec, errors) and not _is_numeric_y(y):
        errors.append(f"line y column '{y.name}' must be numeric")
    if spec.time_granularity is not None and x is not None and x.semantic_type != "datetime":
        errors.append("time_granularity requires a datetime x column")
    if x is not None and x.semantic_type == "categorical" and spec.aggregation is None:
        errors.append("aggregation is required when a line's x is a category")
    # granularity/aggregation closure (stage3 blocking #2)
    if spec.time_granularity not in (None, "raw"):
        if spec.aggregation is None:
            errors.append("aggregation is required when time_granularity is coarser than raw")
    elif x is not None and x.semantic_type != "categorical" and spec.aggregation is None:
        non_missing = profile.n_rows - x.missing_count
        if x.unique_count < non_missing:
            errors.append(
                f"aggregation is required: x column '{x.name}' has duplicate values "
                "and no coarser time_granularity"
            )


def _check_bar(spec, profile, x, y, group, errors) -> None:
    if _require("x", x, spec, errors) and not _is_groupable(x):
        errors.append(f"{spec.type} x column '{x.name}' must be categorical or boolean")
    if spec.y is None:
        if spec.aggregation != "count":
            errors.append(f"{spec.type} with y=None requires aggregation='count'")
    else:
        if y is not None and not _is_numeric_y(y):
            errors.append(f"{spec.type} y column '{y.name}' must be numeric")
        if spec.aggregation is None:
            errors.append(f"{spec.type} requires aggregation when y is given")


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
    given = [role for role in ("x", "y", "group_by", "aggregation") if getattr(spec, role) is not None]
    if not given:
        # correlation matrix: dataset-level requirement (stage3 blocking #4)
        n_numeric = sum(1 for c in profile.columns if c.semantic_type == "numeric")
        if n_numeric < 2:
            errors.append("heatmap requires at least 2 numeric columns in the dataset")
        return
    # stage 19 matrix pattern: x × group_by cells coloured by aggregate(y)
    if len(given) < 4:
        errors.append("heatmap is either a correlation matrix (no fields) or a matrix of x × group_by with y and aggregation")
    if x is not None and not (_is_groupable(x) or column_role(x).cyclic):
        errors.append(f"heatmap x column '{x.name}' must be categorical or cyclic")
    elif x is not None and _categories(x) > GROUP_BY_MAX_CATEGORIES and not column_role(x).cyclic:
        errors.append(f"heatmap x column '{x.name}' has {_categories(x)} categories (maximum {GROUP_BY_MAX_CATEGORIES})")
    if group is not None and not (_is_groupable(group) or column_role(group).cyclic):
        errors.append(f"heatmap group_by column '{group.name}' must be categorical or cyclic")
    if y is not None and not _is_numeric_y(y):
        errors.append(f"heatmap y column '{y.name}' must be numeric")


# --- stage 19 family --------------------------------------------------------


def _check_density(spec, profile, x, y, group, errors) -> None:
    """violin / raincloud / ridgeline: a distribution of numeric y per category x."""
    ranges = {"violin": BOX_X_CATEGORIES_RANGE, "raincloud": RAINCLOUD_X_CATEGORIES_RANGE, "ridgeline": RIDGELINE_X_CATEGORIES_RANGE}
    if spec.type != "violin":
        _require("x", x, spec, errors)
    if x is not None:
        if not _is_groupable(x):
            errors.append(f"{spec.type} x column '{x.name}' must be categorical or boolean")
        else:
            lo, hi = ranges[spec.type]
            if not lo <= _categories(x) <= hi:
                errors.append(f"{spec.type} x column '{x.name}' must have {lo}..{hi} categories")
    if _require("y", y, spec, errors) and not _is_numeric_y(y):
        errors.append(f"{spec.type} y column '{y.name}' must be numeric")
    if spec.group_by is not None:
        errors.append(f"{spec.type} must not have group_by (x is the grouping)")
    if spec.aggregation is not None:
        errors.append(f"{spec.type} must not have an aggregation")


def _check_dot_interval(spec, profile, x, y, group, errors) -> None:
    """Mean of y with an interval per category x (the mean is implied)."""
    if _require("x", x, spec, errors):
        if not _is_groupable(x):
            errors.append(f"dot_interval x column '{x.name}' must be categorical or boolean")
        else:
            lo, hi = BOX_X_CATEGORIES_RANGE
            if not lo <= _categories(x) <= hi:
                errors.append(f"dot_interval x column '{x.name}' must have {lo}..{hi} categories")
    if _require("y", y, spec, errors) and not _is_numeric_y(y):
        errors.append(f"dot_interval y column '{y.name}' must be numeric")
    if spec.group_by is not None:
        errors.append("dot_interval must not have group_by")
    if spec.aggregation not in (None, "mean"):
        errors.append("dot_interval shows the mean: aggregation must be omitted or 'mean'")


def _check_two_state(spec, profile, x, y, group, errors) -> None:
    """dumbbell / slopegraph: aggregate of y per category x for exactly the
    two states of group_by."""
    if _require("x", x, spec, errors):
        if not _is_groupable(x):
            errors.append(f"{spec.type} x column '{x.name}' must be categorical or boolean")
        else:
            lo, hi = TWO_STATE_X_CATEGORIES_RANGE
            if not lo <= _categories(x) <= hi:
                errors.append(f"{spec.type} x column '{x.name}' must have {lo}..{hi} categories")
    if _require("y", y, spec, errors) and not _is_numeric_y(y):
        errors.append(f"{spec.type} y column '{y.name}' must be numeric")
    if spec.group_by is None:
        errors.append(f"{spec.type} requires group_by: the column holding the two states")
    elif group is not None and _categories(group) != 2:
        errors.append(f"{spec.type} group_by column '{group.name}' must have exactly 2 categories (states), has {_categories(group)}")
    if spec.aggregation is None:
        errors.append(f"{spec.type} requires aggregation")


def _check_hexbin(spec, profile, x, y, group, errors) -> None:
    if _require("x", x, spec, errors) and x.semantic_type != "numeric":
        errors.append(f"hexbin x column '{x.name}' must be numeric")
    if _require("y", y, spec, errors) and y.semantic_type != "numeric":
        errors.append(f"hexbin y column '{y.name}' must be numeric")
    if spec.aggregation is not None:
        errors.append("hexbin must not have an aggregation")
    if spec.group_by is not None:
        errors.append("hexbin must not have group_by")


def _check_polar(spec, profile, x, y, group, errors) -> None:
    """polar_line / rose: an aggregate of y around a cyclic x (hour, month,
    weekday, direction). A nominal category is never a cycle."""
    if _require("x", x, spec, errors):
        role = column_role(x)
        if not role.cyclic:
            errors.append(f"{spec.type} x column '{x.name}' must be cyclic (hour, month, weekday, season, direction)")
        elif x.semantic_type == "categorical" and _categories(x) > POLAR_X_CATEGORIES_MAX:
            errors.append(f"{spec.type} x column '{x.name}' has too many categories")
    if spec.y is None:
        if spec.aggregation != "count":
            errors.append(f"{spec.type} with y=None requires aggregation='count'")
    else:
        if y is not None and not _is_numeric_y(y):
            errors.append(f"{spec.type} y column '{y.name}' must be numeric")
        if spec.aggregation is None:
            errors.append(f"{spec.type} requires aggregation when y is given")
        elif spec.type == "rose" and spec.aggregation not in ("sum", "count"):
            # a sector's area reads as a quantity: a mean is not one
            errors.append("rose encodes a magnitude: aggregation must be 'sum' or 'count' (use polar_line for a mean)")
    if spec.group_by is not None:
        errors.append(f"{spec.type} must not have group_by")


def _check_lollipop(spec, profile, x, y, group, errors) -> None:
    _check_bar(spec, profile, x, y, group, errors)


# --- stage 20b: composition / flow -------------------------------------------


def _check_composition_measure(spec, y, errors) -> None:
    """A part-to-whole chart adds its parts up: only sum (of a numeric y) and
    count compose. A mean of the parts is not the whole."""
    if spec.y is None:
        if spec.aggregation != "count":
            errors.append(f"{spec.type} with y=None requires aggregation='count'")
        return
    if y is not None and not _is_numeric_y(y):
        errors.append(f"{spec.type} y column '{y.name}' must be numeric")
    if spec.aggregation is None:
        errors.append(f"{spec.type} requires aggregation when y is given")
    elif spec.aggregation not in COMPOSITION_AGGREGATIONS:
        errors.append(
            f"{spec.type} shows parts of a whole: aggregation must be "
            f"{' or '.join(repr(a) for a in COMPOSITION_AGGREGATIONS)} (a mean does not compose)"
        )


def _check_stacked_bar(spec, profile, x, y, group, errors) -> None:
    """Parts of a whole per category: x is the bar axis, group_by is the part."""
    if _require("x", x, spec, errors):
        if not _is_groupable(x):
            errors.append(f"stacked_bar x column '{x.name}' must be categorical or boolean")
        else:
            lo, hi = STACKED_X_CATEGORIES_RANGE
            if not lo <= _categories(x) <= hi:
                errors.append(f"stacked_bar x column '{x.name}' must have {lo}..{hi} categories")
    if spec.group_by is None:
        errors.append("stacked_bar requires group_by: the column holding the parts")
    elif group is not None and _is_groupable(group):
        lo, hi = STACKED_PARTS_RANGE
        if not lo <= _categories(group) <= hi:
            errors.append(
                f"stacked_bar group_by column '{group.name}' must have {lo}..{hi} parts, "
                f"has {_categories(group)}"
            )
    _check_composition_measure(spec, y, errors)


def _check_sankey(spec, profile, x, y, group, errors) -> None:
    """Flow from one category column to another: x is the source, group_by the
    target, the weight is a count or a sum."""
    lo, hi = SANKEY_CATEGORIES_RANGE
    if _require("x", x, spec, errors):
        if not _is_groupable(x):
            errors.append(f"sankey x column '{x.name}' (the source) must be categorical or boolean")
        elif not lo <= _categories(x) <= hi:
            errors.append(f"sankey x column '{x.name}' must have {lo}..{hi} categories")
    if spec.group_by is None:
        errors.append("sankey requires group_by: the target column the flow goes to")
    elif group is not None and _is_groupable(group) and not lo <= _categories(group) <= hi:
        errors.append(f"sankey group_by column '{group.name}' must have {lo}..{hi} categories")
    _check_composition_measure(spec, y, errors)


_TYPE_CHECKS = {
    "line": _check_line,
    "bar": _check_bar,
    "scatter": _check_scatter,
    "histogram": _check_histogram,
    "box": _check_box,
    "heatmap": _check_heatmap,
    "violin": _check_density,
    "raincloud": _check_density,
    "ridgeline": _check_density,
    "dot_interval": _check_dot_interval,
    "lollipop": _check_lollipop,
    "dumbbell": _check_two_state,
    "slopegraph": _check_two_state,
    "hexbin": _check_hexbin,
    "polar_line": _check_polar,
    "rose": _check_polar,
    "stacked_bar": _check_stacked_bar,
    "sankey": _check_sankey,
}
