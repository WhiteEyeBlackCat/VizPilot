"""Backend aggregation for chart rendering (D2): the browser only ever gets
small, aggregated chart_data plus an echoed spec that reflects what was
actually rendered (granularity downgrades included).
"""

import math
from typing import Any

import polars as pl

from ..profiling.models import DatasetProfile
from ..profiling.types import SAMPLE_SEED, apply_semantic_casts
from ..serialization import jsonify_scalar
from .confidence import has_suspected_sentinels, sentinel_summary
from .roles import ColumnRole, category_key, column_role
from .rules import choose_time_granularity
from .spec import ChartSpec

LINE_MAX_POINTS = 10_000
SCATTER_MAX_POINTS = 10_000
BOX_MAX_OUTLIERS = 100
DEFAULT_TOP_N = 20
POLAR_MAX_CATEGORIES = 60  # a polar axis shows the whole cycle (validate_spec caps it)
HARD_MAX_POINTS = 50_000

_TRUNCATE_UNIT = {"day": "1d", "week": "1w", "month": "1mo"}


class RenderError(ValueError):
    """Turned into the unified 422 {detail: {errors: [...]}} shape."""


class CastedFrameCache:
    """Single-slot dataset_id -> casted df cache: consecutive renders hit the
    same dataset (critique #9). Never invalidated — cast_params are immutable
    per dataset; a restart simply recomputes (D5)."""

    def __init__(self) -> None:
        # single tuple swapped atomically so concurrent renders of two
        # datasets can never pair an id with the other dataset's frame
        self._entry: tuple[str, pl.DataFrame] | None = None

    def get(self, dataset_id: str, df: pl.DataFrame, profile: DatasetProfile) -> pl.DataFrame:
        entry = self._entry
        if entry is not None and entry[0] == dataset_id:
            return entry[1]
        casted = apply_semantic_casts(df, profile)
        self._entry = (dataset_id, casted)
        return casted


def render_chart(df: pl.DataFrame, spec: ChartSpec, profile: DatasetProfile) -> dict[str, Any]:
    spec = spec.model_copy()
    renderer = _RENDERERS[spec.type]
    chart_data, sampled, n_points = renderer(df, spec, profile)
    if n_points > HARD_MAX_POINTS:
        raise RenderError(f"chart would contain {n_points} points (hard limit {HARD_MAX_POINTS})")
    # stage 9 #6: a histogram over a sentinel-laden column is binned over the
    # robust range; the rows left outside are counted and disclosed here,
    # never dropped silently (None for every other case)
    display_range = chart_data.pop("display_range", None)
    return {
        "spec": spec.model_dump(),
        "chart_data": chart_data,
        "sampled": sampled,
        "n_points": n_points,
        "display_range": display_range,
    }


def _y_label(spec: ChartSpec) -> str:
    # blocking #3: the frontend derives the y-axis title from this alone
    if spec.type == "histogram":
        return "count"
    if spec.type == "heatmap" and spec.aggregation is None:
        return "correlation"
    if spec.aggregation == "count":
        return "count"
    if spec.aggregation is not None:
        return f"{spec.aggregation}({spec.y})"
    return spec.y or ""


def _values(s: pl.Series) -> list[Any]:
    return [jsonify_scalar(v) for v in s.to_list()]


def _finite_only(data: pl.DataFrame, columns: list[str | None]) -> pl.DataFrame:
    for name in columns:
        if name is not None and data.schema[name].is_float():
            data = data.filter(pl.col(name).is_finite())
    return data


def _clean(df: pl.DataFrame, spec: ChartSpec, finite: list[str | None]) -> pl.DataFrame:
    cols = [c for c in (spec.x, spec.y, spec.group_by) if c is not None]
    return _finite_only(df.select(cols).drop_nulls(), finite)


def _agg_expr(spec: ChartSpec) -> pl.Expr:
    if spec.aggregation == "count":
        return pl.len().cast(pl.Int64)
    return getattr(pl.col(spec.y), spec.aggregation)()  # type: ignore[union-attr]


def _sorted_groups(data: pl.DataFrame, group: str) -> list[tuple[str, pl.DataFrame]]:
    parts = [(str(key[0]), part) for key, part in data.group_by(group)]
    return sorted(parts, key=lambda p: p[0])  # lexicographic (critique #5)


# --- category order (stage 19) ----------------------------------------------------
#
# A category axis follows the column's own order when it has one (month
# 1..12, Mon..Sun, low..high; ChartSpec.sort = "natural" or an ordered role)
# and is ranked by the aggregate only when the spec asks for it (sort =
# "value") or, for nominal categories on a bar, by the pre-stage-19 default.


def _role_of(profile: DatasetProfile, name: str | None) -> ColumnRole | None:
    col = next((c for c in profile.columns if c.name == name), None)
    return column_role(col) if col is not None else None


def _natural(spec: ChartSpec, profile: DatasetProfile, name: str | None) -> bool:
    """True when the categories of `name` must keep their natural order."""
    if spec.sort == "value":
        return False
    if spec.sort == "natural":
        return True
    role = _role_of(profile, name)
    return role is not None and role.ordered


def _order_values(values: list, name: str | None, profile: DatasetProfile, spec: ChartSpec) -> list:
    """Categories in natural order (role key) — the fallback for a column
    without one is the lexicographic order the box plot always used."""
    role = _role_of(profile, name)
    key = category_key(role) if role is not None and role.ordered else (lambda v: (0, 0.0, str(v)))
    return sorted(values, key=key)


def _ordered_groups(data: pl.DataFrame, col: str, profile: DatasetProfile, spec: ChartSpec) -> list[tuple[str, pl.DataFrame]]:
    parts = {str(key[0]): part for key, part in data.group_by(col)}
    names = _order_values(list(parts), col, profile, spec) if _natural(spec, profile, col) else sorted(parts)
    return [(name, parts[name]) for name in names]


def _order_group_dicts(groups: list[dict], col: str, profile: DatasetProfile, spec: ChartSpec) -> list[dict]:
    by_name = {g["name"]: g for g in groups}
    names = _order_values(list(by_name), col, profile, spec) if _natural(spec, profile, col) else sorted(by_name)
    return [by_name[n] for n in names]


# --- line -------------------------------------------------------------------


def _render_line(df: pl.DataFrame, spec: ChartSpec, profile: DatasetProfile):
    x, group = spec.x, spec.group_by
    data = _clean(df, spec, [spec.x, spec.y])
    temporal = data.schema[x].is_temporal() if data.height else False
    # stage 19: an ordered category (month, weekday, level) as x — the
    # points follow the column's natural order, not the label's spelling
    categorical_x = data.height > 0 and not temporal and not data.schema[x].is_numeric()
    order: dict[str, int] = {}
    if categorical_x:
        order = {v: i for i, v in enumerate(_order_values([str(v) for v in data[x].unique().to_list()], x, profile, spec))}

    if temporal and spec.time_granularity in (None, "raw") and data.height:
        estimate = data.height if spec.aggregation is None else data.n_unique(subset=[c for c in (x, group) if c])
        if estimate > LINE_MAX_POINTS:
            # blocking #2: auto-downgrade; the echoed spec reflects what rendered
            span = (data[x].max() - data[x].min()).total_seconds() / 86400
            spec.time_granularity = choose_time_granularity(data[x].n_unique(), span)
            spec.aggregation = spec.aggregation or "mean"

    if temporal and spec.time_granularity in _TRUNCATE_UNIT:
        data = data.with_columns(pl.col(x).dt.truncate(_TRUNCATE_UNIT[spec.time_granularity]))

    if spec.aggregation is not None:
        keys = [c for c in (x, group) if c is not None]
        data = data.group_by(keys).agg(_agg_expr(spec).alias("_y"))
        y_col = "_y"
    else:
        y_col = spec.y

    # numeric-x lines have no granularity to downgrade to: stride-sample
    # each series instead of tripping the hard point limit
    sampled = False

    def _line_part(part: pl.DataFrame) -> pl.DataFrame:
        nonlocal sampled
        if categorical_x:
            rank = pl.col(x).cast(pl.Utf8).replace_strict(order, default=len(order), return_dtype=pl.Int64)
            part = part.with_columns(rank.alias("_ord")).sort("_ord").drop("_ord")
        else:
            part = part.sort(x)
        if not temporal and part.height > LINE_MAX_POINTS:
            sampled = True
            part = part.gather_every(-(-part.height // LINE_MAX_POINTS))
        return part

    series = []
    if group is None:
        if data.height:
            part = _line_part(data)
            series.append({"name": spec.y, "x": _values(part[x]), "y": _values(part[y_col])})
    else:
        for name, part in _sorted_groups(data, group):
            part = _line_part(part)
            series.append({"name": name, "x": _values(part[x]), "y": _values(part[y_col])})

    n_points = sum(len(s["x"]) for s in series)
    return {"series": series, "y_label": _y_label(spec)}, sampled, n_points


# --- bar --------------------------------------------------------------------


def _render_bar(df: pl.DataFrame, spec: ChartSpec, profile: DatasetProfile):
    x, group = spec.x, spec.group_by
    data = _clean(df, spec, [spec.y])
    if not data.height:
        return {"categories": [], "series": [], "truncated": False, "y_label": _y_label(spec)}, False, 0

    # blocking #1: rank categories ignoring group (count: total rows; other
    # aggregations: |aggregate over all rows|), then align every series to them
    rank = data.group_by(x).agg(_agg_expr(spec).alias("_r"))
    rank_key = pl.col("_r") if spec.aggregation == "count" else pl.col("_r").abs()
    rank = rank.with_columns(rank_key.alias("_k")).sort(["_k", x], descending=[True, False])
    # a cycle is drawn whole (polar / rose never rank-truncate); bars keep top_n
    top_n = POLAR_MAX_CATEGORIES if spec.type in ("polar_line", "rose") else (spec.top_n or DEFAULT_TOP_N)
    categories = rank[x].head(top_n).to_list()
    truncated = rank.height > len(categories)
    if _natural(spec, profile, x):
        # stage 19: the kept categories in the column's own order
        categories = _order_values(categories, x, profile, spec)

    keys = [x] if group is None else [x, group]
    cells = data.group_by(keys).agg(_agg_expr(spec).alias("_v"))

    series = []
    if group is None:
        lookup = {row[0]: row[1] for row in cells.iter_rows()}
        name = spec.y if spec.y is not None else "count"
        series.append({"name": name, "values": [jsonify_scalar(lookup.get(c)) for c in categories]})
    else:
        lookup = {(row[0], row[1]): row[2] for row in cells.iter_rows()}
        group_names = sorted({str(v) for v in cells[group].to_list()})
        raw_by_name = {str(v): v for v in cells[group].to_list()}
        for name in group_names:
            g = raw_by_name[name]
            values = [jsonify_scalar(lookup.get((c, g))) for c in categories]  # missing cell -> null
            series.append({"name": name, "values": values})

    n_points = sum(sum(v is not None for v in s["values"]) for s in series)
    chart_data = {
        "categories": [jsonify_scalar(c) for c in categories],
        "series": series,
        "truncated": truncated,
        "y_label": _y_label(spec),
    }
    return chart_data, False, n_points


# --- scatter ----------------------------------------------------------------


def _render_scatter(df: pl.DataFrame, spec: ChartSpec, profile: DatasetProfile):
    x, y, group = spec.x, spec.y, spec.group_by
    data = _clean(df, spec, [x, y])
    sampled = data.height > SCATTER_MAX_POINTS
    if sampled:
        # sample the whole frame first so group proportions are preserved (#10)
        data = data.sample(SCATTER_MAX_POINTS, seed=SAMPLE_SEED)

    if group is None:
        parts = [(y, data)] if data.height else []
    else:
        parts = _sorted_groups(data, group)
    series = [{"name": name, "x": _values(p[x]), "y": _values(p[y])} for name, p in parts]
    n_points = sum(len(s["x"]) for s in series)
    return {"series": series, "y_label": _y_label(spec)}, sampled, n_points


# --- histogram --------------------------------------------------------------


def _bin_counts(values: pl.Series, edges: list[float], n_bins: int) -> list[int]:
    lo, hi = edges[0], edges[-1]
    width = (hi - lo) / n_bins
    idx = ((values - lo) / width).floor().clip(0, n_bins - 1).cast(pl.Int64)
    counts = [0] * n_bins
    for i, c in idx.value_counts().iter_rows():
        counts[i] = c
    return counts


def _display_range(values: pl.Series, column) -> tuple[float, float, int, int] | None:
    """Robust display window for a column with suspected sentinels: A's
    robust_range (far-out fences recomputed without the sentinel rows,
    intersected with the data range). None when the column is clean, when no
    range is available, or when the window is degenerate / empty — the full
    range is then shown as before."""
    if column is None or not has_suspected_sentinels(column):
        return None
    robust = getattr(column.quality, "robust_range", None)
    if robust is None or robust.lo >= robust.hi:
        return None
    lo, hi = float(robust.lo), float(robust.hi)
    below = int((values < lo).sum())
    above = int((values > hi).sum())
    if below + above == 0 or below + above >= len(values):
        return None
    return lo, hi, below, above


def _render_histogram(df: pl.DataFrame, spec: ChartSpec, profile: DatasetProfile):
    x, group = spec.x, spec.group_by
    data = _clean(df, spec, [x])  # nulls and NaN/inf excluded from binning
    if not data.height:
        return {"bins": {"edges": [], "counts": []}, "y_label": _y_label(spec)}, False, 0

    column = next((c for c in profile.columns if c.name == x), None)
    window = _display_range(data[x], column)
    display_range = None
    if window is not None:
        lo, hi, below, above = window
        # raw rows are untouched; only the binned window narrows, and the
        # excluded counts travel with the response
        data = data.filter((pl.col(x) >= lo) & (pl.col(x) <= hi))
        display_range = {
            "lo": lo,
            "hi": hi,
            "excluded_below": below,
            "excluded_above": above,
            "reason": (
                f"suspected sentinel values ({sentinel_summary(column)}) "
                "lie outside the display range"
            ),
        }
    values = data[x]
    if window is None:
        lo, hi = float(values.min()), float(values.max())
    if lo == hi:  # constant column: one bucket holding everything
        edges, n_bins = [lo - 0.5, hi + 0.5], 1
    else:
        n_bins = spec.bins or max(10, min(50, int(math.sqrt(len(values)))))
        step = (hi - lo) / n_bins
        edges = [lo + i * step for i in range(n_bins)] + [hi]

    chart_data: dict[str, Any] = {
        "bins": {"edges": edges, "counts": _bin_counts(values, edges, n_bins)},
        "y_label": _y_label(spec),
        "display_range": display_range,
    }
    n_points = n_bins
    if group is not None:
        series = []
        for name, part in _sorted_groups(data, group):  # shared edges per group
            series.append({"name": name, "edges": edges, "counts": _bin_counts(part[x], edges, n_bins)})
        chart_data["series"] = series
        n_points += n_bins * len(series)
    return chart_data, False, n_points


# --- box --------------------------------------------------------------------


def _box_group(name: str, values: pl.Series) -> tuple[dict[str, Any], bool]:
    q1 = values.quantile(0.25, interpolation="linear")
    q3 = values.quantile(0.75, interpolation="linear")
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = values.filter((values < lower) | (values > upper))
    capped = len(outliers) > BOX_MAX_OUTLIERS
    if capped:
        outliers = outliers.sample(BOX_MAX_OUTLIERS, seed=SAMPLE_SEED)
    group = {
        "name": name,
        "q1": jsonify_scalar(q1),
        "median": jsonify_scalar(values.quantile(0.5, interpolation="linear")),
        "q3": jsonify_scalar(q3),
        "lower_fence": jsonify_scalar(lower),
        "upper_fence": jsonify_scalar(upper),
        "outliers": _values(outliers.sort()),
        "mean": jsonify_scalar(values.mean()),
        "count": len(values),
    }
    return group, capped


def _render_box(df: pl.DataFrame, spec: ChartSpec, profile: DatasetProfile):
    x, y = spec.x, spec.y
    data = _clean(df, spec, [y])
    if not data.height:
        return {"groups": [], "y_label": _y_label(spec)}, False, 0

    parts = [("all", data)] if x is None else _ordered_groups(data, x, profile, spec)
    groups, sampled = [], False
    for name, part in parts:
        group, capped = _box_group(name, part[y])
        groups.append(group)
        sampled = sampled or capped
    n_points = sum(5 + len(g["outliers"]) for g in groups)
    return {"groups": groups, "y_label": _y_label(spec)}, sampled, n_points


# --- heatmap ----------------------------------------------------------------


def _render_heatmap(df: pl.DataFrame, spec: ChartSpec, profile: DatasetProfile):
    if spec.x is not None:  # stage 19 matrix pattern: x × group_by → aggregate(y)
        return _render_matrix(df, spec, profile)
    corr = profile.correlations
    if corr is None:  # validate_spec guarantees >=2 numeric, so this is defensive
        raise RenderError("no correlation matrix is available for this dataset")
    chart_data = {"columns": corr.columns, "matrix": corr.matrix, "y_label": _y_label(spec)}
    return chart_data, False, len(corr.columns) ** 2


# --- stage 19: density family (violin / raincloud / ridgeline) --------------
#
# Densities are binned kernel estimates: the finite values of a group are
# counted into DENSITY_BINS equal-width bins over the shared range, then the
# bin counts are smoothed with a Gaussian kernel whose bandwidth follows
# Silverman's rule (in bin units). Polars does the counting; the smoothing is
# a 64×k loop over bins — never over rows.

DENSITY_BINS = 64
DENSITY_MAX_POINTS = 200  # raw points per group carried for a raincloud's "rain"


def _density(values: pl.Series, lo: float, hi: float) -> tuple[list[float], list[float], float]:
    """(grid centres, density per centre normalised to peak 1, bandwidth) for
    the finite values of one group over the shared [lo, hi] range."""
    n = len(values)
    if n == 0 or hi <= lo:
        return [], [], 0.0
    width = (hi - lo) / DENSITY_BINS
    counts = _bin_counts(values, [lo + i * width for i in range(DENSITY_BINS)] + [hi], DENSITY_BINS)
    std = float(values.std()) if n > 1 else 0.0
    q1, q3 = values.quantile(0.25, interpolation="linear"), values.quantile(0.75, interpolation="linear")
    iqr = float(q3 - q1) if q1 is not None and q3 is not None else 0.0
    spread = min(std, iqr / 1.34) if iqr > 0 else std
    bandwidth = 0.9 * spread * n ** (-0.2) if spread > 0 else width
    h_bins = max(bandwidth / width, 0.75)
    radius = int(math.ceil(3 * h_bins))
    kernel = [math.exp(-0.5 * (k / h_bins) ** 2) for k in range(-radius, radius + 1)]
    smoothed = []
    for i in range(DENSITY_BINS):
        acc = 0.0
        for k, w in zip(range(-radius, radius + 1), kernel):
            j = i + k
            if 0 <= j < DENSITY_BINS:
                acc += counts[j] * w
        smoothed.append(acc)
    peak = max(smoothed) or 1.0
    centres = [lo + (i + 0.5) * width for i in range(DENSITY_BINS)]
    return centres, [v / peak for v in smoothed], bandwidth


def _density_groups(df: pl.DataFrame, spec: ChartSpec, profile: DatasetProfile, *, points: bool):
    """Shared payload of violin / raincloud / ridgeline: per group the
    density curve, the box statistics and (raincloud) a deterministic sample
    of raw points. Groups follow the natural order of x."""
    x, y = spec.x, spec.y
    data = _clean(df, spec, [y])
    if not data.height:
        return {"groups": [], "grid": [], "y_label": _y_label(spec)}, False, 0
    lo, hi = float(data[y].min()), float(data[y].max())
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    parts = [("all", data)] if x is None else _ordered_groups(data, x, profile, spec)
    groups, sampled, n_points = [], False, 0
    grid: list[float] = []
    for name, part in parts:
        values = part[y]
        centres, density, bandwidth = _density(values, lo, hi)
        grid = centres or grid
        box, capped = _box_group(name, values)
        group = {**box, "density": density, "bandwidth": bandwidth}
        if points:
            pts = values if len(values) <= DENSITY_MAX_POINTS else values.sample(DENSITY_MAX_POINTS, seed=SAMPLE_SEED)
            sampled = sampled or len(values) > DENSITY_MAX_POINTS
            group["points"] = _values(pts)
            n_points += len(pts)
        groups.append(group)
        n_points += len(density) + 5
    return {"groups": groups, "grid": grid, "y_label": _y_label(spec)}, sampled, n_points


def _render_violin(df, spec, profile):
    return _density_groups(df, spec, profile, points=False)


def _render_raincloud(df, spec, profile):
    return _density_groups(df, spec, profile, points=True)


def _render_ridgeline(df, spec, profile):
    return _density_groups(df, spec, profile, points=False)


# --- stage 19: dot_interval ---------------------------------------------------


def _render_dot_interval(df, spec, profile):
    """Mean of y per category with a 95 % normal interval (1.96 · sd / √n);
    categories in natural order. Groups with n < 2 carry a null interval."""
    x, y = spec.x, spec.y
    data = _clean(df, spec, [y])
    if not data.height:
        return {"groups": [], "y_label": _y_label(spec)}, False, 0
    stats = data.group_by(x).agg(
        pl.col(y).mean().alias("_mean"),
        pl.col(y).std().alias("_std"),
        pl.len().alias("_n"),
    )
    groups = []
    for row in stats.iter_rows(named=True):
        n = int(row["_n"])
        mean = row["_mean"]
        std = row["_std"]
        half = 1.96 * std / math.sqrt(n) if std is not None and n > 1 else None
        groups.append(
            {
                "name": str(row[x]),
                "mean": jsonify_scalar(mean),
                "lo": jsonify_scalar(mean - half) if half is not None else None,
                "hi": jsonify_scalar(mean + half) if half is not None else None,
                "n": n,
            }
        )
    groups = _order_group_dicts(groups, x, profile, spec)
    return {"groups": groups, "y_label": _y_label(spec)}, False, len(groups)


# --- stage 19: dumbbell / slopegraph ------------------------------------------


def _render_two_state(df, spec, profile):
    """Aggregate of y per (category x, state group_by) for exactly two
    states: {categories, states: [a, b], series: [{name: state, values}]}.
    Categories in natural order; a missing (category, state) cell is null."""
    x, group = spec.x, spec.group_by
    data = _clean(df, spec, [spec.y])
    if not data.height:
        return {"categories": [], "states": [], "series": [], "y_label": _y_label(spec)}, False, 0
    # the states in their own order (before → after, weekday → weekend); alphabetical otherwise
    states = _order_values([str(v) for v in data[group].unique().to_list()], group, profile, spec)
    if len(states) != 2:
        raise RenderError(f"{spec.type} needs exactly two states in '{group}', found {len(states)}")
    cells = data.group_by([x, group]).agg(_agg_expr(spec).alias("_v"))
    lookup = {(str(r[0]), str(r[1])): r[2] for r in cells.iter_rows()}
    categories = _order_values([str(v) for v in cells[x].unique().to_list()], x, profile, spec)
    series = [
        {"name": s, "values": [jsonify_scalar(lookup.get((c, s))) for c in categories]} for s in states
    ]
    n_points = sum(v is not None for s in series for v in s["values"])
    return {"categories": categories, "states": states, "series": series, "y_label": _y_label(spec)}, False, n_points


# --- stage 19: hexbin ---------------------------------------------------------

HEXBIN_COLUMNS = 40  # hexagons across the x range


def _render_hexbin(df, spec, profile):
    """Pointy-top hexagonal bins over the finite (x, y) pairs: cells as
    [column, row, count] in offset coordinates plus the geometry the adapter
    needs to draw them back in data space. Counting is a Polars group_by over
    the integer cell index — no per-row Python."""
    x, y = spec.x, spec.y
    data = _clean(df, spec, [x, y])
    n = data.height
    if n == 0:
        return {"cells": [], "x_range": [0, 0], "y_range": [0, 0], "columns": 0, "rows": 0, "y_label": _y_label(spec)}, False, 0
    x_lo, x_hi = float(data[x].min()), float(data[x].max())
    y_lo, y_hi = float(data[y].min()), float(data[y].max())
    if x_hi == x_lo:
        x_lo, x_hi = x_lo - 0.5, x_hi + 0.5
    if y_hi == y_lo:
        y_lo, y_hi = y_lo - 0.5, y_hi + 0.5
    dx = (x_hi - x_lo) / HEXBIN_COLUMNS  # hex width in x units
    dy = dx * (y_hi - y_lo) / (x_hi - x_lo) * (2 / math.sqrt(3))  # keep hexagons regular in plot units
    # offset coordinates: row from y, column from x shifted half a cell on odd rows
    row = ((pl.col(y) - y_lo) / (dy * 0.75)).floor().cast(pl.Int64)
    shifted = pl.col(x) - x_lo - pl.when(row % 2 == 1).then(dx / 2).otherwise(0.0)
    col = (shifted / dx).floor().cast(pl.Int64)
    binned = data.select(col.alias("_c"), row.alias("_r")).group_by(["_c", "_r"]).agg(pl.len().alias("_n"))
    cells = [[int(c), int(r), int(k)] for c, r, k in binned.iter_rows()]
    cells.sort()
    rows = int(math.floor((y_hi - y_lo) / (dy * 0.75))) + 1
    payload = {
        "cells": cells,
        "x_range": [x_lo, x_hi],
        "y_range": [y_lo, y_hi],
        "dx": dx,
        "dy": dy,
        "columns": HEXBIN_COLUMNS,
        "rows": rows,
        "n": n,
        "y_label": _y_label(spec),
    }
    return payload, False, len(cells)


# --- stage 19: matrix heatmap (x × group_by → aggregate of y) -----------------


def _render_matrix(df, spec, profile):
    x, group = spec.x, spec.group_by
    data = _clean(df, spec, [spec.y])
    if not data.height:
        return {"columns": [], "rows": [], "matrix": [], "y_label": _y_label(spec)}, False, 0
    cells = data.group_by([x, group]).agg(_agg_expr(spec).alias("_v"))
    columns = _order_values([str(v) for v in cells[x].unique().to_list()], x, profile, spec)
    rows = _order_values([str(v) for v in cells[group].unique().to_list()], group, profile, spec)
    lookup = {(str(r[0]), str(r[1])): r[2] for r in cells.iter_rows()}
    matrix = [[jsonify_scalar(lookup.get((c, r))) for c in columns] for r in rows]
    n_points = sum(v is not None for row_ in matrix for v in row_)
    return {"columns": columns, "rows": rows, "matrix": matrix, "y_label": _y_label(spec)}, False, n_points


# --- stage 20b: composition (stacked_bar) -------------------------------------


def _render_stacked_bar(df, spec, profile):
    """Parts of a whole per category: the bar aggregation (same ranking,
    same top_n truncation, same null-for-a-missing-cell rule) plus the column
    total of every bar. Parts whose own column has an order (low < high) are
    stacked in it; otherwise alphabetically, as the grouped bar does.
    `percent` normalises each bar to 100 % — the absolute totals travel in
    `totals` so the magnitude behind a share is never lost."""
    chart_data, sampled, n_points = _render_bar(df, spec, profile)
    series = chart_data["series"]
    role = _role_of(profile, spec.group_by)
    if role is not None and role.ordered:
        by_name = {s["name"]: s for s in series}
        series = [by_name[n] for n in _order_values(list(by_name), spec.group_by, profile, spec)]
    n_categories = len(chart_data["categories"])
    totals: list[Any] = []  # int for a count, float for a sum, None for an empty bar
    for i in range(n_categories):
        parts = [s["values"][i] for s in series if s["values"][i] is not None]
        totals.append(sum(parts) if parts else None)
    percent = bool(spec.percent)
    if percent:
        series = [
            {
                "name": s["name"],
                "values": [
                    None
                    if s["values"][i] is None or not totals[i]
                    else 100.0 * s["values"][i] / totals[i]
                    for i in range(n_categories)
                ],
            }
            for s in series
        ]
    chart_data.update(
        series=series,
        totals=totals,
        percent=percent,
        y_label=(f"share of {_y_label(spec)} (%)" if percent else _y_label(spec)),
    )
    return chart_data, sampled, n_points


# --- stage 20b: flow (sankey) --------------------------------------------------

SANKEY_MAX_LINKS = 60  # the strongest flows stay readable; the rest is disclosed


def _render_sankey(df, spec, profile):
    """Source category → target category with a count / sum weight, as
    {nodes, links}. Self-loops (source == target) carry no flow and are
    dropped; only the SANKEY_MAX_LINKS heaviest links are kept and
    `truncated` says so. Node names are side-prefixed so the two ends never
    collapse into one node (which would make a cycle out of city → city);
    `label` is the value as it appears in the column."""
    src, dst = spec.x, spec.group_by
    data = _clean(df, spec, [spec.y])
    keep = [] if spec.y is None else [pl.col(spec.y)]
    data = data.select(
        pl.col(src).cast(pl.Utf8).alias("_s"), pl.col(dst).cast(pl.Utf8).alias("_t"), *keep
    )
    pairs = data.filter(pl.col("_s") != pl.col("_t"))
    self_loops = data.height - pairs.height
    empty = {"nodes": [], "links": [], "truncated": False, "n_links_total": 0, "self_loops": self_loops, "y_label": _y_label(spec)}
    links = pairs.group_by(["_s", "_t"]).agg(_agg_expr(spec).alias("_v"))
    # a zero / negative / null weight has no ribbon to draw (a sankey encodes magnitude)
    links = links.filter(pl.col("_v") > 0)
    n_links = links.height
    links = links.sort(["_v", "_s", "_t"], descending=[True, False, False]).head(SANKEY_MAX_LINKS)
    if not links.height:
        return empty, False, 0

    rows = [(s, t, v) for s, t, v in links.iter_rows()]
    # a count stays an int, a sum stays a float: the weights are added as they came
    out_total: dict[str, Any] = {}
    in_total: dict[str, Any] = {}
    for s, t, v in rows:
        out_total[s] = out_total.get(s, 0) + v
        in_total[t] = in_total.get(t, 0) + v
    nodes = [
        {"name": f"{side}:{label}", "label": label, "column": column, "side": side_name, "value": jsonify_scalar(total)}
        for side, side_name, column, totals in (("0", "source", src, out_total), ("1", "target", dst, in_total))
        for label, total in ((v, totals[v]) for v in _order_values(list(totals), column, profile, spec))
    ]
    payload = {
        "nodes": nodes,
        "links": [{"source": f"0:{s}", "target": f"1:{t}", "value": jsonify_scalar(v)} for s, t, v in rows],
        "truncated": n_links > len(rows),
        "n_links_total": n_links,
        # rows whose source equals its target: no flow, dropped before aggregation
        "self_loops": self_loops,
        "y_label": _y_label(spec),
    }
    return payload, False, len(rows)


_RENDERERS = {
    "line": _render_line,
    "bar": _render_bar,
    "scatter": _render_scatter,
    "histogram": _render_histogram,
    "box": _render_box,
    "heatmap": _render_heatmap,
    "violin": _render_violin,
    "raincloud": _render_raincloud,
    "ridgeline": _render_ridgeline,
    "dot_interval": _render_dot_interval,
    "lollipop": _render_bar,  # same aggregate; the adapter draws stems
    "dumbbell": _render_two_state,
    "slopegraph": _render_two_state,
    "hexbin": _render_hexbin,
    "polar_line": _render_bar,  # aggregate around the cycle, natural order
    "rose": _render_bar,
    "stacked_bar": _render_stacked_bar,
    "sankey": _render_sankey,
}
