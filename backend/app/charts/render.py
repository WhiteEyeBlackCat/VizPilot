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
from .rules import choose_time_granularity
from .spec import ChartSpec

LINE_MAX_POINTS = 10_000
SCATTER_MAX_POINTS = 10_000
BOX_MAX_OUTLIERS = 100
DEFAULT_TOP_N = 20
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
    return {
        "spec": spec.model_dump(),
        "chart_data": chart_data,
        "sampled": sampled,
        "n_points": n_points,
    }


def _y_label(spec: ChartSpec) -> str:
    # blocking #3: the frontend derives the y-axis title from this alone
    if spec.type == "histogram":
        return "count"
    if spec.type == "heatmap":
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


# --- line -------------------------------------------------------------------


def _render_line(df: pl.DataFrame, spec: ChartSpec, profile: DatasetProfile):
    x, group = spec.x, spec.group_by
    data = _clean(df, spec, [spec.x, spec.y])
    temporal = data.schema[x].is_temporal() if data.height else False

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
    top_n = spec.top_n or DEFAULT_TOP_N
    categories = rank[x].head(top_n).to_list()
    truncated = rank.height > len(categories)

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


def _render_histogram(df: pl.DataFrame, spec: ChartSpec, profile: DatasetProfile):
    x, group = spec.x, spec.group_by
    data = _clean(df, spec, [x])  # nulls and NaN/inf excluded from binning
    if not data.height:
        return {"bins": {"edges": [], "counts": []}, "y_label": _y_label(spec)}, False, 0

    values = data[x]
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

    parts = [("all", data)] if x is None else _sorted_groups(data, x)
    groups, sampled = [], False
    for name, part in parts:
        group, capped = _box_group(name, part[y])
        groups.append(group)
        sampled = sampled or capped
    n_points = sum(5 + len(g["outliers"]) for g in groups)
    return {"groups": groups, "y_label": _y_label(spec)}, sampled, n_points


# --- heatmap ----------------------------------------------------------------


def _render_heatmap(df: pl.DataFrame, spec: ChartSpec, profile: DatasetProfile):
    corr = profile.correlations
    if corr is None:  # validate_spec guarantees >=2 numeric, so this is defensive
        raise RenderError("no correlation matrix is available for this dataset")
    chart_data = {"columns": corr.columns, "matrix": corr.matrix, "y_label": _y_label(spec)}
    return chart_data, False, len(corr.columns) ** 2


_RENDERERS = {
    "line": _render_line,
    "bar": _render_bar,
    "scatter": _render_scatter,
    "histogram": _render_histogram,
    "box": _render_box,
    "heatmap": _render_heatmap,
}
