"""Semantic roles of columns for chart selection (stage 19).

The profiler's semantic type says what a column *is* (numeric / categorical /
datetime …); a chart needs to know how its values are *ordered*: a month is
cyclic, a rating is ordinal, a region is nominal. Everything here is a pure,
deterministic function of the ColumnProfile (name, dtype, category count and
top values / min / max) — the profiler is not touched and nothing is
re-computed from the data.
"""

import re
from typing import Literal

from pydantic import BaseModel

from ..profiling.models import ColumnProfile

Role = Literal["numeric", "datetime", "boolean", "nominal", "ordinal", "cyclic"]

# --- cyclic fields: name hint + a value range that fits the period ----------

# (name pattern, period, lowest value allowed, highest value allowed)
_CYCLIC_HINTS: list[tuple[re.Pattern[str], int, int, int]] = [
    (re.compile(r"^(hr|hour|hours|hour_of_day|hourofday|hod)$", re.I), 24, 0, 23),
    (re.compile(r"^(mnth|month|months|month_of_year|mon)$", re.I), 12, 1, 12),
    (re.compile(r"^(weekday|week_day|wday|dow|day_of_week|dayofweek|dayweek)$", re.I), 7, 0, 7),
    (re.compile(r"^(season|seasons)$", re.I), 4, 0, 4),
    (re.compile(r"^(quarter|qtr|q)$", re.I), 4, 1, 4),
    (re.compile(r"^(minute|min_of_hour|minute_of_hour)$", re.I), 60, 0, 59),
    (re.compile(r"^(day_of_year|dayofyear|doy|yday)$", re.I), 366, 1, 366),
    (re.compile(r"^(day_of_month|dayofmonth|dom|mday)$", re.I), 31, 1, 31),
    (re.compile(r"^(wind_dir|wind_direction|winddir|direction|bearing|heading|azimuth)$", re.I), 360, 0, 360),
]

_MONTH_NAMES = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]
_MONTH_ABBR = [m[:3] for m in _MONTH_NAMES]
_MONTH_ALT = {"sept": "sep"}  # other common spellings, lower-cased and dot-free
_WEEKDAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_WEEKDAY_ABBR = [d[:3] for d in _WEEKDAY_NAMES]
_WEEKDAY_ALT = {"tues": "tue", "weds": "wed", "thur": "thu", "thurs": "thu"}
_SEASON_NAMES = ["winter", "spring", "summer", "autumn"]
_SEASON_ALT = {"fall": "autumn"}

# --- ordinal label scales (lower-cased) -------------------------------------

_ORDINAL_SCALES: list[list[str]] = [
    ["very low", "low", "medium", "high", "very high"],
    ["low", "medium", "high"],
    ["low", "mid", "high"],
    ["small", "medium", "large"],
    ["xs", "s", "m", "l", "xl", "xxl"],
    ["bronze", "silver", "gold", "platinum", "diamond"],
    ["basic", "standard", "premium"],
    ["poor", "fair", "good", "very good", "excellent"],
    ["never", "rarely", "sometimes", "often", "always"],
    ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"],
    ["q1", "q2", "q3", "q4"],
    ["cold", "mild", "warm", "hot"],
    ["short", "medium", "long"],
    # two-state pairs: their order is the reading order of a dumbbell / slopegraph
    ["before", "after"],
    ["pre", "post"],
    ["baseline", "treatment"],
    ["control", "treatment"],
    ["weekday", "weekend"],
    ["old", "new"],
    ["false", "true"],
    ["no", "yes"],
    ["off", "on"],
]

_NUMERIC_DTYPE_RE = re.compile(r"^(U?Int|Float)\d+$")


class ColumnRole(BaseModel):
    """How a column's values are ordered, for chart selection."""

    name: str
    role: Role
    ordered: bool  # ordinal / cyclic / datetime / numeric
    cyclic: bool
    period: int | None = None  # cyclic only
    # display order of the categories when known (cyclic and label-scale
    # ordinals); None means "sort by value" (integer codes) or "no order"
    order: list[str] | None = None
    n_categories: int | None = None
    two_state: bool = False  # exactly two categories (boolean or 2-level categorical)


def _category_labels(col: ColumnProfile) -> list[str] | None:
    """Every category as a string, when the profile carries them all: the
    stage 19.2 `categories` field (complete up to CATEGORY_LABELS_MAX, so
    twelve month names qualify), else top_values when nothing is cut off
    (profiles from before PROFILE_VERSION 14, hand-built profiles)."""
    categories = getattr(col, "categories", None)
    if categories is not None and col.n_categories is not None and len(categories) >= col.n_categories:
        return list(categories)
    if col.top_values is None or col.n_categories is None:
        return None
    if len(col.top_values) < col.n_categories:
        return None
    return [str(tv.value) for tv in col.top_values]


def _label_key(value: str) -> str:
    """Case-insensitive, trailing-dot-free label ("Sept." -> "sept")."""
    return value.strip().lower().rstrip(".").strip()


def _int_bounds(col: ColumnProfile) -> tuple[int, int] | None:
    """Integer value range from the profile: numeric min/max, or every
    category label when the profile carries them all; None otherwise."""
    if isinstance(col.min, (int, float)) and isinstance(col.max, (int, float)):
        if float(col.min).is_integer() and float(col.max).is_integer():
            return int(col.min), int(col.max)
    labels = _category_labels(col)
    if labels and all(re.fullmatch(r"-?\d+", v.strip()) for v in labels):
        values = [int(v) for v in labels]
        return min(values), max(values)
    return None


def _cyclic_by_name(col: ColumnProfile) -> tuple[int, list[str] | None] | None:
    """(period, order) when the name is a known cyclic field and the values
    fit its range; None otherwise."""
    for pattern, period, lo, hi in _CYCLIC_HINTS:
        if not pattern.match(col.name.strip()):
            continue
        bounds = _int_bounds(col)
        if bounds is None:
            # integer-coded categorical with more categories than the
            # profile lists (12 months as Int): the profiler keeps no
            # min/max, so the name plus a plausible category count decides
            if (
                col.semantic_type == "categorical"
                and not col.nominal
                and _NUMERIC_DTYPE_RE.match(col.original_dtype or "")
                and col.n_categories is not None
                and 2 <= col.n_categories <= hi - lo + 1
            ):
                return period, None
            return None
        if bounds[0] < lo or bounds[1] > hi:
            return None
        labels = _category_labels(col)
        order = sorted(labels, key=lambda v: int(v)) if labels else None
        return period, order
    return None


def _cyclic_by_labels(col: ColumnProfile) -> tuple[int, list[str]] | None:
    labels = _category_labels(col)
    if not labels or len(labels) < 2:
        return None
    lowered = [_label_key(v) for v in labels]
    for names, abbr, alt, period in (
        (_MONTH_NAMES, _MONTH_ABBR, _MONTH_ALT, 12),
        (_WEEKDAY_NAMES, _WEEKDAY_ABBR, _WEEKDAY_ALT, 7),
    ):
        keys = [alt.get(v, v) for v in lowered]
        # every label a month / weekday, and no two labels the same one
        # (Jan + January would be two spellings, not a cycle of twelve)
        if all(v in names or v in abbr for v in keys):
            rank = {v: i for i, v in enumerate(names)}
            rank.update({v: i for i, v in enumerate(abbr)})
            ranks = [rank[v] for v in keys]
            if len(set(ranks)) != len(ranks):
                return None
            return period, [label for _, label in sorted(zip(ranks, labels))]
    normalised = [_SEASON_ALT.get(v, v) for v in lowered]
    if all(v in _SEASON_NAMES for v in normalised):
        rank = {v: i for i, v in enumerate(_SEASON_NAMES)}
        return 4, [label for _, label in sorted(zip((rank[v] for v in normalised), labels))]
    return None


def _ordinal_scale(col: ColumnProfile) -> list[str] | None:
    labels = _category_labels(col)
    if not labels or len(labels) < 2:
        return None
    lowered = {v.strip().lower(): v for v in labels}
    for scale in _ORDINAL_SCALES:
        if set(lowered) <= set(scale):
            return [lowered[v] for v in scale if v in lowered]
    return None


def column_role(col: ColumnProfile) -> ColumnRole:
    n_cat = col.n_categories if col.n_categories is not None else None
    base = dict(name=col.name, n_categories=n_cat)

    if col.semantic_type == "datetime":
        return ColumnRole(role="datetime", ordered=True, cyclic=False, **base)

    if col.semantic_type == "boolean":
        return ColumnRole(role="boolean", ordered=False, cyclic=False, two_state=True, **base)

    if col.semantic_type == "numeric":
        cyc = _cyclic_by_name(col)
        if cyc is not None:
            period, order = cyc
            return ColumnRole(role="cyclic", ordered=True, cyclic=True, period=period, order=order, **base)
        return ColumnRole(role="numeric", ordered=True, cyclic=False, **base)

    if col.semantic_type == "categorical":
        two_state = n_cat == 2
        # a nominal code (product ids that happen to be called "season") is
        # never a cycle by name; month / weekday labels still are
        cyc = (None if col.nominal else _cyclic_by_name(col)) or _cyclic_by_labels(col)
        if cyc is not None:
            period, order = cyc
            return ColumnRole(
                role="cyclic", ordered=True, cyclic=True, period=period, order=order, two_state=two_state, **base
            )
        scale = _ordinal_scale(col)
        if scale is not None:
            return ColumnRole(role="ordinal", ordered=True, cyclic=False, order=scale, two_state=two_state, **base)
        # integer codes (weathersit 1..3, rating 1..5): ordered by value —
        # unless the profiler marked them as nominal codes (product ids)
        if not col.nominal and _NUMERIC_DTYPE_RE.match(col.original_dtype or ""):
            labels = _category_labels(col)
            order = None
            if labels and all(re.fullmatch(r"-?\d+(\.0+)?", v.strip()) for v in labels):
                order = sorted(labels, key=lambda v: float(v))
            return ColumnRole(role="ordinal", ordered=True, cyclic=False, order=order, two_state=two_state, **base)
        return ColumnRole(role="nominal", ordered=False, cyclic=False, two_state=two_state, **base)

    # id / text / unknown never chart; report them as nominal, unordered
    return ColumnRole(role="nominal", ordered=False, cyclic=False, **base)


def roles_for(columns: list[ColumnProfile]) -> dict[str, ColumnRole]:
    return {c.name: column_role(c) for c in columns}


# --- category order at render time --------------------------------------------

_LABEL_RANKS: dict[str, int] = {}
for _i, _m in enumerate(_MONTH_NAMES):
    _LABEL_RANKS[_m] = _i
    _LABEL_RANKS[_m[:3]] = _i
for _i, _d in enumerate(_WEEKDAY_NAMES):
    _LABEL_RANKS[_d] = _i
    _LABEL_RANKS[_d[:3]] = _i
for _i, _s in enumerate(_SEASON_NAMES):
    _LABEL_RANKS[_s] = _i
_LABEL_RANKS["fall"] = _SEASON_NAMES.index("autumn")


def category_key(role: ColumnRole | None):
    """Sort key over category values (as rendered, any scalar) that gives
    the column's natural order: the role's explicit order, else known
    month / weekday / season / scale labels, else numeric value, else the
    label itself. Values outside every table sort last, alphabetically."""
    explicit = {v: i for i, v in enumerate(role.order)} if role is not None and role.order else {}

    def key(value) -> tuple:
        text = str(value)
        if text in explicit:
            return (0, explicit[text], "")
        lowered = text.strip().lower()
        if lowered in _LABEL_RANKS and role is not None and role.cyclic:
            return (1, _LABEL_RANKS[lowered], "")
        for table in _ORDINAL_SCALES:
            if lowered in table and role is not None and role.role == "ordinal":
                return (2, table.index(lowered), "")
        try:
            return (3, float(text), "")
        except ValueError:
            return (4, 0.0, text)

    return key
