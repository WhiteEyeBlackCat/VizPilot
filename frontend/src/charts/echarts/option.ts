// Pure RenderResult -> ECharts option conversion. No React, no DOM, no
// echarts runtime import (types only) so it is unit-testable and the only
// place ECharts-specific structure lives. Nothing is re-aggregated or
// re-binned here — the backend RenderResult is the single source of truth.
// The one presentation-side liberty (stage 18) is the reading order of a
// category axis with a natural sequence (hour, month, weekday, date), a pure
// permutation from ./order.ts that keeps every (category, value) pair
// intact. Semantics are documented per chart type below.

import type {
  BarSeriesOption,
  BoxplotSeriesOption,
  CustomSeriesOption,
  HeatmapSeriesOption,
  LineSeriesOption,
  SankeySeriesOption,
  ScatterSeriesOption,
} from "echarts/charts";
// type-only: the runtime registration lives in ./echarts.ts
import type { CustomSeriesRenderItemAPI, CustomSeriesRenderItemParams } from "echarts";

import type {
  BarChartData,
  BoxChartData,
  BoxGroup,
  DensityChartData,
  DotIntervalChartData,
  HeatmapChartData,
  HexbinChartData,
  HistogramChartData,
  LineChartData,
  RenderResult,
  SankeyChartData,
  SankeyLink,
  StackedBarChartData,
  TimeGranularity,
  TwoStateChartData,
} from "../../types";
import type { VizOption } from "./echarts";
import { barOrder, boxOrder } from "./order";
import {
  axisBase,
  COLORS,
  dataZoomInside,
  DIVERGING,
  formatNumber,
  grid,
  gridWithLegend,
  gridWithVisualMap,
  legend,
  PALETTE,
  SEQUENTIAL,
  textStyle,
  title,
  toolbox,
  tooltipBase,
  tooltipValue,
  xAxisName,
  yAxisName,
} from "./theme";

export const isEmpty = (result: RenderResult): boolean => result.n_points === 0;

// scatter switches to ECharts' progressive "large" pipeline above this
export const LARGE_SCATTER_THRESHOLD = 2000;
// beyond this many points a line hides its per-point symbols
export const LINE_SYMBOL_MAX_POINTS = 60;
// bar category labels tilt once the axis gets crowded
export const BAR_LABEL_ROTATE_FROM = 8;
export const HISTOGRAM_GROUP_OPACITY = 0.6;

type Pair = [number | string, number | null];

const pairs = (x: (number | string | null)[], y: (number | null)[]): Pair[] =>
  x.map((xv, i) => [xv ?? "", y[i] ?? null]);

// an ISO date / datetime string on the x axis
const ISO_X = /^\d{4}-\d{2}(-\d{2})?([T ]\d{2}:\d{2})?/;
const isTemporalX = (data: LineChartData): boolean =>
  data.series.some((s) => s.x.some((v) => typeof v === "string" && ISO_X.test(v)));
// stage 19: a line over an ordered category (weekday names, levels) — the
// backend already put the points in the column's natural order
const isCategoricalX = (data: LineChartData): boolean =>
  !isTemporalX(data) && data.series.some((s) => s.x.some((v) => typeof v === "string"));

// Time-axis tick labels per time_granularity (the backend writes the chosen
// granularity back into result.spec when it aggregates; a raw hourly series
// keeps it null, so the intraday case is detected from the timestamps).
// ECharts picks a primary unit per tick and uses that unit's entry, so the
// first tick of a new year (or month at finer granularities) carries the
// coarser context automatically.
export type TimeLabelFormatter = Record<string, string>;

const MONTHLY: TimeLabelFormatter = { year: "{yyyy}-{MM}", month: "{yyyy}-{MM}", day: "{yyyy}-{MM}" };
const DAILY: TimeLabelFormatter = {
  year: "{yyyy}-{MM}-{dd}",
  month: "{MM}-{dd}",
  day: "{MM}-{dd}",
  hour: "{MM}-{dd}",
  minute: "{MM}-{dd}",
  second: "{MM}-{dd}",
  millisecond: "{MM}-{dd}",
};
const INTRADAY: TimeLabelFormatter = {
  year: "{yyyy}-{MM}-{dd}",
  month: "{MM}-{dd}",
  day: "{MM}-{dd}",
  hour: "{MM}-{dd} {HH}:{mm}",
  minute: "{HH}:{mm}",
  second: "{HH}:{mm}:{ss}",
  millisecond: "{HH}:{mm}:{ss}",
};

/** True when any ISO timestamp carries a time of day other than midnight. */
export const hasIntradayX = (data: LineChartData): boolean =>
  data.series.some((s) =>
    s.x.some((v) => typeof v === "string" && /T(?!00:00:00(?:\.0+)?$)\d{2}:\d{2}/.test(v)),
  );

export function timeLabelFormatter(
  granularity: TimeGranularity | null | undefined,
  intraday = false,
): TimeLabelFormatter {
  switch (granularity) {
    case "month":
      return MONTHLY;
    case "day":
    case "week":
      return DAILY;
    default: // "raw" or null: the backend did not bucket the timestamps
      return intraday ? INTRADAY : DAILY;
  }
}

function baseOption(result: RenderResult, zoomable: boolean, legendShown: boolean): VizOption {
  return {
    color: PALETTE,
    textStyle,
    title: title(result.spec.title),
    grid: legendShown ? gridWithLegend : grid,
    legend: legendShown ? legend : { show: false },
    toolbox: toolbox(result.spec.title, zoomable),
    animationDuration: 250,
    animationEasing: "cubicOut",
  };
}

// --- grouped temporal gaps --------------------------------------------------
//
// A grouped line over time (`temp over dteday by season`) has, per group,
// only the buckets that group occupies: the backend (`render.py`
// `_render_line`) emits a point for every (x, group) bucket that occurs and
// none for a month in which a season does not occur, so ECharts joined the
// last 2011 summer point straight to the first 2012 one across nine months
// that other seasons occupy. The fix is presentation-only and keeps every
// backend (x, y) pair in order.
//
// Rule (chart semantics, no calendar): the chart's temporal buckets are the
// x values that actually exist in any series (union across groups, deduped
// on their parsed timestamp). A group's line breaks between two consecutive
// observations iff at least one such bucket lies between them — i.e. the
// group is absent from a bucket where the chart has data. Where that holds
// an invisible null point is inserted at the midpoint and
// `connectNulls: false` draws the break. Nothing is generated: a weekend in
// business-day data, a holiday, or an overnight stretch in hourly data is no
// bucket of the chart when no group has a point there, so the line stays
// connected across it; a season absent from months that other seasons
// occupy breaks. Only grouped temporal lines are touched.
//
// The rule presumes the x values are *aligned buckets* — identities shared
// by every group — and would fragment raw asynchronous timestamps (group A
// at 09:00, 11:00, 13:00 and B at 10:00, 12:00, 14:00 have no bucket in
// common; each group would be cut into isolated points). Two things decide
// whether the x values are aligned buckets, `temporalGrid` records which:
//
// 1. "bucketed": `spec.time_granularity` is day / week / month — the backend
//    truncated x to that unit and aggregated (`_render_line`), so the x
//    values are calendar buckets by construction and the rule applies to
//    every group as it is.
// 2. "grid": the granularity is raw / unset (the rules keep raw up to 500
//    distinct timestamps, e.g. every monthly or hourly dataset of that size),
//    so the x values are the timestamps themselves and the data has to show
//    the grid. The union of x is read on the coarsest exact scale it fits
//    (calendar months when every value is a month start, calendar days when
//    every value is a midnight, else milliseconds; DST-safe for day and
//    month grids), the grid unit is the smallest step between consecutive
//    buckets, and the union is a regular grid iff every step is a whole
//    multiple of the unit and at least half of the steps are exactly one
//    unit (the grid is dense at its own unit, not a coincidence of a few
//    shared instants). On such a grid a group is broken only when it samples
//    at the unit itself — at least half of its own consecutive steps are one
//    unit — because a group whose typical step is coarser (every second
//    hour, quarterly points among monthly ones) is a sparser series, not a
//    series with holes, and is drawn continuous.
// 3. null: anything else (irregular spacing, a single bucket) — raw
//    timestamps, never broken.
//
// Neither chart titles nor column names take part; the decision is a pure
// function of the echoed spec's granularity and the x values.

const DAY_MS = 86_400_000;

// Parsed timestamp (ms) of an x value, the bucket's canonical key. Date-only
// strings are read as local midnight, the way ECharts' time axis reads them
// (`Date.parse` alone would take "2011-01-01" as UTC but
// "2011-01-01T00:00:00" as local, splitting one bucket in two).
const temporalKey = (v: number | string | null): number => {
  if (typeof v !== "string") return NaN;
  if (/^\d{4}-\d{2}$/.test(v)) return Date.parse(`${v}-01T00:00:00`);
  if (/^\d{4}-\d{2}-\d{2}$/.test(v)) return Date.parse(`${v}T00:00:00`);
  return Date.parse(v.replace(/^(\d{4}-\d{2}-\d{2}) /, "$1T"));
};

export type GridScale = "month" | "day" | "ms";

/** Bucket keys on the coarsest exact calendar scale every key fits: month
 *  index when all are local month starts, day number when all are local
 *  midnights, otherwise the millisecond itself. */
export function gridPositions(keys: number[]): { scale: GridScale; position: Map<number, number> } {
  const dates = keys.map((k) => new Date(k));
  const midnight = (d: Date) => d.getHours() === 0 && d.getMinutes() === 0 && d.getSeconds() === 0 && d.getMilliseconds() === 0;
  let scale: GridScale = "ms";
  if (dates.length && dates.every((d) => midnight(d) && d.getDate() === 1)) scale = "month";
  else if (dates.length && dates.every(midnight)) scale = "day";
  const pos = (d: Date, k: number): number =>
    scale === "month" ? d.getFullYear() * 12 + d.getMonth() : scale === "day" ? Math.round(Date.UTC(d.getFullYear(), d.getMonth(), d.getDate()) / DAY_MS) : k;
  return { scale, position: new Map(keys.map((k, i) => [k, pos(dates[i], k)])) };
}

/** Steps between consecutive valid positions of a series. */
const steps = (positions: number[]): number[] => {
  const out: number[] = [];
  for (let i = 1; i < positions.length; i++) if (!Number.isNaN(positions[i - 1]) && !Number.isNaN(positions[i])) out.push(positions[i] - positions[i - 1]);
  return out;
};
const atLeastHalf = (xs: number[], pred: (v: number) => boolean): boolean => xs.length > 0 && 2 * xs.filter(pred).length >= xs.length;

export interface TemporalGrid {
  /** parsed x (ms) → global index in sorted order, over every series;
   *  unparsable / null x values take no part */
  index: Map<number, number>;
  /** why the x values count as aligned buckets: the backend truncated them
   *  ("bucketed"), a regular grid was detected in the data ("grid"), or they
   *  are raw timestamps that are never broken (null) */
  aligned: "bucketed" | "grid" | null;
  /** key → position on the detected grid scale, and the grid unit
   *  (positions); only meaningful when `aligned === "grid"` */
  scale: GridScale;
  position: Map<number, number>;
  unit: number;
}

/** The chart's temporal buckets over every series, with the verdict on
 *  whether they are aligned buckets (see "grouped temporal gaps"). */
export function temporalGrid(data: LineChartData, granularity: TimeGranularity | null | undefined): TemporalGrid {
  const keys = new Set<number>();
  for (const s of data.series) for (const v of s.x) {
    const k = temporalKey(v);
    if (!Number.isNaN(k)) keys.add(k);
  }
  const sorted = Array.from(keys).sort((a, b) => a - b);
  const index = new Map(sorted.map((k, i) => [k, i]));
  const { scale, position } = gridPositions(sorted);
  const unionSteps = steps(sorted.map((k) => position.get(k)!));
  const unit = unionSteps.length ? Math.min(...unionSteps) : 0;
  const bucketed = granularity === "day" || granularity === "week" || granularity === "month";
  const regular = unit > 0 && unionSteps.every((s) => s % unit === 0) && atLeastHalf(unionSteps, (s) => s === unit);
  return { index, aligned: bucketed ? "bucketed" : regular ? "grid" : null, scale, position, unit };
}

/** `pairs()` for one group with a null point inserted between two
 *  consecutive observations whenever another existing bucket of the chart
 *  lies between them (the group is absent there); unchanged when the chart's
 *  x values are not aligned buckets, or — on a detected grid — when this
 *  group does not sample at the grid unit. */
export function breakAbsentBuckets(
  x: (number | string | null)[],
  y: (number | null)[],
  grid: TemporalGrid,
): Pair[] {
  const base = pairs(x, y);
  if (grid.aligned === null) return base;
  const keys = x.map(temporalKey);
  if (grid.aligned === "grid" && !atLeastHalf(steps(keys.map((k) => grid.position.get(k) ?? NaN)), (s) => s === grid.unit)) return base;
  const out: Pair[] = [];
  for (let i = 0; i < base.length; i++) {
    if (i > 0) {
      const gp = grid.index.get(keys[i - 1]);
      const gc = grid.index.get(keys[i]);
      if (gp !== undefined && gc !== undefined && Math.abs(gc - gp) > 1) out.push([(keys[i - 1] + keys[i]) / 2, null]);
    }
    out.push(base[i]);
  }
  return out;
}
// --- line -------------------------------------------------------------------

function lineOption(result: RenderResult): VizOption {
  const d = result.chart_data as LineChartData;
  const temporal = isTemporalX(d);
  const categorical = isCategoricalX(d);
  const categories = categorical ? Array.from(new Set(d.series.flatMap((s) => s.x.map(String)))) : [];
  const many = d.series.some((s) => s.x.length > LINE_SYMBOL_MAX_POINTS);
  // grouped temporal series break where a group is absent from an aligned
  // bucket other groups occupy (see "grouped temporal gaps" above); raw
  // asynchronous timestamps, ungrouped lines and numeric / category axes are
  // passed through as they are
  const grid = temporal && result.spec.group_by != null ? temporalGrid(d, result.spec.time_granularity) : null;
  const series: LineSeriesOption[] = d.series.map((s) => ({
    type: "line",
    name: s.name,
    data: grid ? breakAbsentBuckets(s.x, s.y, grid) : pairs(s.x, s.y),
    showSymbol: !many,
    symbolSize: 5,
    lineStyle: { width: 1.5 },
    emphasis: { focus: d.series.length > 1 ? "series" : "none", lineStyle: { width: 2.5 } },
    connectNulls: false,
  }));
  return {
    ...baseOption(result, true, d.series.length > 1),
    tooltip: { ...tooltipBase, trigger: "axis", axisPointer: { type: "cross" }, valueFormatter: tooltipValue },
    xAxis: temporal
      ? {
          ...axisBase,
          type: "time",
          ...xAxisName(result.spec.x ?? ""),
          axisLabel: {
            ...axisBase.axisLabel,
            formatter: timeLabelFormatter(result.spec.time_granularity, hasIntradayX(d)),
          },
        }
      : categorical
        ? { ...axisBase, type: "category", data: categories, boundaryGap: false, ...xAxisName(result.spec.x ?? ""), splitLine: { show: false } }
        : { ...axisBase, type: "value", scale: true, ...xAxisName(result.spec.x ?? "") },
    yAxis: { ...axisBase, type: "value", scale: true, ...yAxisName(d.y_label) },
    dataZoom: dataZoomInside(["x"]),
    series,
  };
}

// --- scatter ----------------------------------------------------------------

function scatterOption(result: RenderResult): VizOption {
  const d = result.chart_data as LineChartData;
  const series: ScatterSeriesOption[] = d.series.map((s) => ({
    type: "scatter",
    name: s.name,
    data: pairs(s.x, s.y),
    symbolSize: 6,
    itemStyle: { opacity: 0.7 },
    emphasis: { scale: 1.6, itemStyle: { opacity: 1 } },
    large: s.x.length > LARGE_SCATTER_THRESHOLD,
    largeThreshold: LARGE_SCATTER_THRESHOLD,
  }));
  const xName = result.spec.x ?? "";
  return {
    ...baseOption(result, true, d.series.length > 1),
    tooltip: {
      ...tooltipBase,
      trigger: "item",
      formatter: (p) => {
        const item = Array.isArray(p) ? p[0] : p;
        const [x, y] = item.value as Pair;
        const head = d.series.length > 1 ? `${item.seriesName}<br/>` : "";
        return `${head}${xName}: ${fmt(x)}<br/>${d.y_label}: ${fmt(y)}`;
      },
    },
    xAxis: { ...axisBase, type: "value", scale: true, ...xAxisName(xName) },
    yAxis: { ...axisBase, type: "value", scale: true, ...yAxisName(d.y_label) },
    dataZoom: dataZoomInside(["x", "y"]),
    series,
  };
}

// --- bar --------------------------------------------------------------------

function barOption(result: RenderResult): VizOption {
  const d = result.chart_data as BarChartData;
  // category axis with explicit data preserves the backend ranking even for
  // numeric or boolean category values (stage6 blocking #3); null values
  // stay null so a missing (category, group) cell is a gap, not a zero.
  // Categories with a natural sequence are displayed in it (order.ts);
  // every series is permuted with the same index so pairs never split.
  const order = barOrder(d, result.spec.sort);
  const categories = order.map((i) => String(d.categories[i]));
  const series: BarSeriesOption[] = d.series.map((s) => ({
    type: "bar",
    name: s.name,
    data: order.map((i) => s.values[i] ?? null),
    barMaxWidth: 36,
    barCategoryGap: "38%",
    itemStyle: { borderRadius: [3, 3, 0, 0], opacity: 0.92 },
    emphasis: { focus: d.series.length > 1 ? "series" : "none", itemStyle: { opacity: 1 } },
  }));
  return {
    ...baseOption(result, false, d.series.length > 1),
    tooltip: { ...tooltipBase, trigger: "axis", axisPointer: { type: "shadow" }, valueFormatter: tooltipValue },
    xAxis: {
      ...axisBase,
      type: "category",
      data: categories,
      ...xAxisName(result.spec.x ?? ""),
      axisLabel: {
        ...axisBase.axisLabel,
        interval: 0,
        rotate: categories.length > BAR_LABEL_ROTATE_FROM ? 30 : 0,
      },
      splitLine: { show: false },
    },
    yAxis: { ...axisBase, type: "value", ...yAxisName(d.y_label) },
    series,
  };
}

// --- histogram --------------------------------------------------------------

// [lo, hi, count] per bin: a custom series draws each bin as a rectangle
// spanning exactly its edges, so unequal or clipped (display_range) bins are
// drawn faithfully on a true numeric axis.
export type HistogramBin = [number, number, number];

export function histogramBins(edges: number[], counts: number[]): HistogramBin[] {
  return counts.map((count, i) => [edges[i], edges[i + 1], count]);
}

/** renderItem for one histogram series; opacity is baked in per series
 *  (grouped overlay) because a custom series cannot read itemStyle.opacity
 *  through api.visual, and api.style() is deprecated. */
function makeRenderBin(opacity: number) {
  return (_params: CustomSeriesRenderItemParams, api: CustomSeriesRenderItemAPI) => {
    const lo = api.value(0) as number;
    const hi = api.value(1) as number;
    const count = api.value(2) as number;
    const [x0, y1] = api.coord([lo, count]);
    const [x1, y0] = api.coord([hi, 0]);
    return {
      type: "rect" as const,
      shape: { x: x0, y: y1, width: Math.max(x1 - x0 - 1, 1), height: y0 - y1 },
      style: { fill: api.visual("color") as string, opacity },
    };
  };
}

function histogramOption(result: RenderResult): VizOption {
  const d = result.chart_data as HistogramChartData;
  const grouped = (d.series?.length ?? 0) > 0;
  const groups = grouped
    ? d.series!
    : [{ name: result.spec.x ?? "", edges: d.bins.edges, counts: d.bins.counts }];
  const series: CustomSeriesOption[] = groups.map((g) => ({
    type: "custom",
    name: g.name,
    renderItem: makeRenderBin(grouped ? HISTOGRAM_GROUP_OPACITY : 1),
    encode: { x: [0, 1], y: 2, tooltip: [0, 1, 2] },
    data: histogramBins(g.edges, g.counts),
    itemStyle: { opacity: grouped ? HISTOGRAM_GROUP_OPACITY : 1 },
    clip: true,
  }));
  // the axis spans exactly the binned window (the robust display_range when
  // the backend clipped it), so excluded rows are visibly outside the plot
  const edges = groups[0].edges;
  return {
    ...baseOption(result, true, grouped),
    tooltip: {
      ...tooltipBase,
      trigger: "item",
      formatter: (p) => {
        const item = Array.isArray(p) ? p[0] : p;
        const [lo, hi, count] = item.value as HistogramBin;
        const head = grouped ? `${item.seriesName}<br/>` : "";
        return `${head}[${formatNumber(lo)}, ${formatNumber(hi)})<br/>${d.y_label}: ${count}`;
      },
    },
    xAxis: {
      ...axisBase,
      type: "value",
      min: edges.length ? edges[0] : undefined,
      max: edges.length ? edges[edges.length - 1] : undefined,
      ...xAxisName(result.spec.x ?? ""),
    },
    yAxis: { ...axisBase, type: "value", ...yAxisName(d.y_label) },
    dataZoom: dataZoomInside(["x"]),
    series,
  };
}

// --- box --------------------------------------------------------------------

// ECharts boxplot tuples cannot hold null: a missing statistic becomes NaN,
// which ECharts leaves undrawn (a null statistic must never become 0)
export type BoxStats = [number, number, number, number, number];
const stat = (v: number | null): number => (v === null ? NaN : v);

function boxOption(result: RenderResult): VizOption {
  const d = result.chart_data as BoxChartData;
  // groups with a natural sequence are displayed in it (order.ts); the
  // statistics of each group travel with its name
  const groups = boxOrder(d, result.spec.sort).map((i) => d.groups[i]);
  const names = groups.map((g) => g.name);
  // ECharts boxplot order: [min, Q1, median, Q3, max]; the backend's fences
  // play the min/max role (whisker ends), exactly as the backend computed them
  const stats: BoxStats[] = groups.map((g) => [
    stat(g.lower_fence),
    stat(g.q1),
    stat(g.median),
    stat(g.q3),
    stat(g.upper_fence),
  ]);
  const box: BoxplotSeriesOption = {
    type: "boxplot",
    name: d.y_label,
    data: stats,
    itemStyle: { color: COLORS.boxFill, borderColor: PALETTE[0], borderWidth: 1.5 },
    tooltip: {
      formatter: (p) => {
        const item = Array.isArray(p) ? p[0] : p;
        const g = groups[item.dataIndex];
        return (
          `${g.name}<br/>` +
          `upper fence: ${fmt(g.upper_fence)}<br/>Q3: ${fmt(g.q3)}<br/>median: ${fmt(g.median)}<br/>` +
          `Q1: ${fmt(g.q1)}<br/>lower fence: ${fmt(g.lower_fence)}<br/>` +
          `mean: ${fmt(g.mean)} · n=${g.count}`
        );
      },
    },
  };
  const outliers: [string, number][] = [];
  for (const g of groups) for (const v of g.outliers) outliers.push([g.name, v]);
  const series: (BoxplotSeriesOption | ScatterSeriesOption)[] = [box];
  if (outliers.length > 0) {
    series.push({
      type: "scatter",
      name: "outliers",
      data: outliers,
      symbolSize: 5,
      itemStyle: { color: COLORS.outlier, opacity: 0.7 },
      tooltip: {
        formatter: (p) => {
          const item = Array.isArray(p) ? p[0] : p;
          const [name, v] = item.value as [string, number];
          return `${name}<br/>outlier: ${fmt(v)}`;
        },
      },
    });
  }
  const means = groups.filter((g) => g.mean !== null).map((g) => [g.name, g.mean as number] as [string, number]);
  if (means.length > 0) {
    series.push({
      type: "scatter",
      name: "mean",
      data: means,
      symbol: "diamond",
      symbolSize: 8,
      itemStyle: { color: COLORS.mean },
      tooltip: {
        formatter: (p) => {
          const item = Array.isArray(p) ? p[0] : p;
          const [name, v] = item.value as [string, number];
          return `${name}<br/>mean: ${fmt(v)}`;
        },
      },
    });
  }
  return {
    ...baseOption(result, false, false),
    tooltip: { ...tooltipBase, trigger: "item" },
    xAxis: {
      ...axisBase,
      type: "category",
      data: names,
      ...xAxisName(result.spec.x ?? ""),
      splitLine: { show: false },
    },
    yAxis: { ...axisBase, type: "value", scale: true, ...yAxisName(d.y_label) },
    series,
  };
}

// --- heatmap ----------------------------------------------------------------

export type HeatmapCell = [number, number, number];

/** Non-null cells only: a null correlation is left blank, never drawn as 0. */
export function heatmapCells(matrix: (number | null)[][]): HeatmapCell[] {
  const cells: HeatmapCell[] = [];
  matrix.forEach((row, i) =>
    row.forEach((v, j) => {
      if (v !== null) cells.push([j, i, v]);
    }),
  );
  return cells;
}

function heatmapOption(result: RenderResult): VizOption {
  const d = result.chart_data as HeatmapChartData;
  // stage 19 matrix pattern: rows are the group_by categories and the cells
  // hold an aggregate (a sequential scale); the correlation matrix keeps
  // its symmetric diverging scale over [-1, 1]
  const matrix = d.rows !== undefined;
  const rows = d.rows ?? d.columns;
  const values = d.matrix.flat().filter((v): v is number => v !== null);
  const lo = matrix ? Math.min(...values) : -1;
  const hi = matrix ? Math.max(...values) : 1;
  const series: HeatmapSeriesOption = {
    type: "heatmap",
    name: d.y_label,
    data: heatmapCells(d.matrix),
    label: {
      show: !matrix || d.columns.length * rows.length <= 96,
      fontSize: 11,
      color: COLORS.text,
      formatter: (p) => {
        const v = (p.value as HeatmapCell)[2];
        return matrix ? formatNumber(v) : v.toFixed(2);
      },
    },
    itemStyle: { borderColor: COLORS.cellBorder, borderWidth: 1 },
    emphasis: { itemStyle: { borderColor: COLORS.text } },
  };
  return {
    ...baseOption(result, false, false),
    grid: gridWithVisualMap,
    tooltip: {
      ...tooltipBase,
      trigger: "item",
      formatter: (p) => {
        const item = Array.isArray(p) ? p[0] : p;
        const [j, i, v] = item.value as HeatmapCell;
        const head = matrix ? `${result.spec.x}: ${d.columns[j]} · ${result.spec.group_by}: ${rows[i]}` : `${d.columns[i]} × ${d.columns[j]}`;
        return `${head}<br/>${d.y_label}: ${matrix ? formatNumber(v) : v.toFixed(3)}`;
      },
    },
    xAxis: {
      ...axisBase,
      type: "category",
      data: d.columns,
      splitArea: { show: true },
      ...(matrix ? xAxisName(result.spec.x ?? "") : {}),
      axisLabel: { ...axisBase.axisLabel, interval: 0, rotate: d.columns.length > 6 && !matrix ? 45 : 0 },
    },
    // row 0 at the top, like a matrix
    yAxis: { ...axisBase, type: "category", data: rows, inverse: true, splitArea: { show: true }, ...(matrix ? yAxisName(result.spec.group_by ?? "") : {}) },
    visualMap: {
      type: "continuous",
      min: lo,
      max: hi,
      calculable: false,
      orient: "vertical",
      right: 8,
      top: "middle",
      itemHeight: 140,
      itemWidth: 12,
      precision: matrix ? 2 : 1,
      text: matrix ? [formatNumber(hi), formatNumber(lo)] : ["1", "-1"],
      inRange: { color: matrix ? SEQUENTIAL : DIVERGING },
      textStyle: { color: COLORS.muted, fontSize: 11 },
    },
    series: [series],
  };
}

// --- dispatch ---------------------------------------------------------------

const fmt = (v: number | string | null | undefined): string =>
  v === null || v === undefined ? "—" : typeof v === "number" ? formatNumber(v) : String(v);

export function buildOption(result: RenderResult): VizOption {
  switch (result.spec.type) {
    case "line":
      return lineOption(result);
    case "scatter":
      return scatterOption(result);
    case "bar":
      return barOption(result);
    case "histogram":
      return histogramOption(result);
    case "box":
      return boxOption(result);
    case "heatmap":
      return heatmapOption(result);
    case "violin":
    case "raincloud":
    case "ridgeline":
      return densityOption(result, result.spec.type);
    case "dot_interval":
      return dotIntervalOption(result);
    case "lollipop":
      return lollipopOption(result);
    case "dumbbell":
      return dumbbellOption(result);
    case "slopegraph":
      return slopegraphOption(result);
    case "hexbin":
      return hexbinOption(result);
    case "polar_line":
    case "rose":
      return polarOption(result, result.spec.type);
    case "stacked_bar":
      return stackedBarOption(result);
    case "sankey":
      return sankeyOption(result);
  }
}


// ============================================================================
// stage 19 family. Every builder below draws exactly the numbers the backend
// sent (densities, intervals, aggregates, hex counts) with custom / bar /
// line / scatter series; nothing is re-aggregated here. Category order is the
// backend's (ChartSpec.sort).
// ============================================================================

const catAxis = (data: string[], name: string) => ({
  ...axisBase,
  type: "category" as const,
  data,
  ...xAxisName(name),
  axisLabel: { ...axisBase.axisLabel, interval: 0, rotate: data.length > BAR_LABEL_ROTATE_FROM ? 30 : 0 },
  splitLine: { show: false },
});
const valAxis = (name: string) => ({ ...axisBase, type: "value" as const, scale: true, ...yAxisName(name) });

// --- violin / raincloud / ridgeline -------------------------------------------

const DENSITY_HALF_WIDTH = 0.42; // category units: two half-violins never touch
const RIDGE_HEIGHT = 1.35; // ridge peak in row units (rows are 1 apart, so ridges overlap a little)
// deterministic pseudo-jitter in [-1, 1) from the point index
export const jitter = (k: number): number => (((k * 9301 + 49297) % 233280) / 233280) * 2 - 1;

function summaryGlyph(api: CustomSeriesRenderItemAPI, at: number, g: BoxGroup, halfWidth: number) {
  const [x0, yq1] = api.coord([at - halfWidth, stat(g.q1)]);
  const [x1, yq3] = api.coord([at + halfWidth, stat(g.q3)]);
  const [xm, ymed] = api.coord([at, stat(g.median)]);
  const [, ylo] = api.coord([at, stat(g.lower_fence)]);
  const [, yhi] = api.coord([at, stat(g.upper_fence)]);
  return {
    type: "group" as const,
    children: [
      { type: "line" as const, shape: { x1: xm, y1: yhi, x2: xm, y2: ylo }, style: { stroke: COLORS.text, lineWidth: 1, opacity: 0.7 } },
      { type: "rect" as const, shape: { x: x0, y: yq3, width: x1 - x0, height: yq1 - yq3 }, style: { fill: COLORS.elevated, stroke: COLORS.text, lineWidth: 1 } },
      { type: "line" as const, shape: { x1: x0, y1: ymed, x2: x1, y2: ymed }, style: { stroke: COLORS.text, lineWidth: 2 } },
    ],
  };
}

export function densityOption(result: RenderResult, kind: "violin" | "raincloud" | "ridgeline"): VizOption {
  const d = result.chart_data as DensityChartData;
  const groups = d.groups;
  const names = groups.map((g) => g.name);
  const grid = d.grid;
  const ridge = kind === "ridgeline";
  const tooltipFor = (g: BoxGroup) =>
    `${g.name}<br/>median: ${fmt(g.median)} · Q1 ${fmt(g.q1)} · Q3 ${fmt(g.q3)}<br/>mean: ${fmt(g.mean)} · n=${g.count}`;
  // the value axis must span the data (grid + whiskers): a custom series
  // carries no values ECharts could scale from, so the extent is explicit —
  // verifier round 1 found the axis auto-scaled to the group indices
  // the grid already spans the observed values; Tukey fences may lie far
  // outside the data (a count's lower fence is negative), so they only widen
  // the axis up to the observed extremes of the box statistics (verifier r2)
  const stats = groups.flatMap((g) => [g.q1, g.q3, g.median]).filter((v): v is number => v !== null);
  const lo = Math.min(grid[0] ?? 0, ...stats);
  const hi = Math.max(grid[grid.length - 1] ?? 1, ...stats);
  const pad = (hi - lo || 1) * 0.06;
  const valueExtent = { min: lo - pad, max: hi + pad };
  const density: CustomSeriesOption = {
    type: "custom",
    name: d.y_label,
    // [group index, value lo, value hi]: the datum spans the data range so
    // the axis extent is right even before the explicit min / max
    data: groups.map((_, i) => [i, lo, hi]),
    renderItem: (_params, api) => {
      const i = api.value(0) as number;
      const g = groups[i];
      const color = PALETTE[i % PALETTE.length];
      if (ridge) {
        // baseline on the group's row, the curve rising up to RIDGE_HEIGHT rows
        const pts = grid.map((v, k) => api.coord([v, i + g.density[k] * RIDGE_HEIGHT]));
        const base = api.coord([grid[0], i]);
        const end = api.coord([grid[grid.length - 1], i]);
        return {
          type: "polygon" as const,
          shape: { points: [base, ...pts, end] },
          style: { fill: color, opacity: 0.6, stroke: color, lineWidth: 1 },
          z2: groups.length - i, // front rows overlap the ones behind
        };
      }
      const right = grid.map((y, k) => api.coord([i + g.density[k] * DENSITY_HALF_WIDTH, y]));
      const left = grid.map((y, k) => api.coord([kind === "raincloud" ? i : i - g.density[k] * DENSITY_HALF_WIDTH, y]));
      return {
        type: "polygon" as const,
        shape: { points: [...right, ...left.reverse()] },
        style: { fill: color, opacity: kind === "raincloud" ? 0.55 : 0.5, stroke: color, lineWidth: 1 },
      };
    },
    encode: ridge ? { x: [1, 2], y: 0 } : { x: 0, y: [1, 2] },
    tooltip: {
      formatter: (p) => {
        const item = Array.isArray(p) ? p[0] : p;
        return tooltipFor(groups[(item.value as number[])[0]]);
      },
    },
    clip: true,
  };
  const series: (CustomSeriesOption | ScatterSeriesOption)[] = [density];
  if (!ridge) {
    const offset = kind === "raincloud" ? -0.1 : 0;
    series.push({
      type: "custom",
      name: "summary",
      data: groups.map((_, i) => [i, lo, hi]),
      renderItem: (_params, api) => summaryGlyph(api, (api.value(0) as number) + offset, groups[api.value(0) as number], kind === "raincloud" ? 0.06 : 0.05),
      encode: { x: 0, y: [1, 2] },
      tooltip: {
        formatter: (p) => {
          const item = Array.isArray(p) ? p[0] : p;
          return tooltipFor(groups[(item.value as number[])[0]]);
        },
      },
      z: 3,
    });
  }
  if (kind === "raincloud") {
    // the "rain": the backend's point sample, jittered left of the half-violin
    const rain: [number, number][] = [];
    groups.forEach((g, i) => (g.points ?? []).forEach((v, k) => rain.push([i - 0.2 - 0.15 * jitter(k), v])));
    series.push({
      type: "scatter",
      name: "points",
      data: rain,
      symbolSize: 3,
      itemStyle: { color: COLORS.muted, opacity: 0.55 },
      tooltip: { show: false },
      z: 2,
    });
  }
  const xName = result.spec.x ?? "";
  return {
    ...baseOption(result, false, false),
    tooltip: { ...tooltipBase, trigger: "item" },
    xAxis: ridge
      ? { ...axisBase, type: "value", ...valueExtent, ...xAxisName(d.y_label) }
      : { ...axisBase, type: "value", min: -0.6, max: groups.length - 0.4, interval: 1, axisLabel: { ...axisBase.axisLabel, formatter: (v: number) => names[Math.round(v)] ?? "" }, splitLine: { show: false }, ...xAxisName(xName), nameGap: 34 },
    yAxis: ridge
      ? { ...axisBase, type: "value", min: -0.2, max: groups.length - 1 + RIDGE_HEIGHT, interval: 1, axisLabel: { ...axisBase.axisLabel, formatter: (v: number) => names[Math.round(v)] ?? "" }, splitLine: { show: false }, ...yAxisName(xName) }
      : { ...valAxis(d.y_label), ...valueExtent },
    series,
  };
}

// --- dot_interval ------------------------------------------------------------

export function dotIntervalOption(result: RenderResult): VizOption {
  const d = result.chart_data as DotIntervalChartData;
  const names = d.groups.map((g) => g.name);
  const interval: CustomSeriesOption = {
    type: "custom",
    name: "95% interval",
    data: d.groups.map((g, i) => [i, stat(g.lo), stat(g.hi)]),
    renderItem: (_params, api) => {
      const i = api.value(0) as number;
      const lo = api.value(1) as number;
      const hi = api.value(2) as number;
      if (Number.isNaN(lo) || Number.isNaN(hi)) return undefined;
      const [x, y0] = api.coord([i, lo]);
      const [, y1] = api.coord([i, hi]);
      const cap = 5;
      return {
        type: "group" as const,
        children: [
          { type: "line" as const, shape: { x1: x, y1: y0, x2: x, y2: y1 }, style: { stroke: PALETTE[0], lineWidth: 1.5 } },
          { type: "line" as const, shape: { x1: x - cap, y1: y0, x2: x + cap, y2: y0 }, style: { stroke: PALETTE[0], lineWidth: 1.5 } },
          { type: "line" as const, shape: { x1: x - cap, y1: y1, x2: x + cap, y2: y1 }, style: { stroke: PALETTE[0], lineWidth: 1.5 } },
        ],
      };
    },
    encode: { x: 0, y: [1, 2] },
    tooltip: { show: false },
  };
  const dots: ScatterSeriesOption = {
    type: "scatter",
    name: d.y_label,
    data: d.groups.map((g, i) => [i, g.mean] as [number, number | null]),
    symbolSize: 9,
    itemStyle: { color: PALETTE[0] },
    z: 3,
  };
  return {
    ...baseOption(result, false, false),
    tooltip: {
      ...tooltipBase,
      trigger: "item",
      formatter: (p) => {
        const item = Array.isArray(p) ? p[0] : p;
        const g = d.groups[item.dataIndex];
        return `${g.name}<br/>mean: ${fmt(g.mean)}<br/>95% interval: ${fmt(g.lo)} – ${fmt(g.hi)} · n=${g.n}`;
      },
    },
    xAxis: catAxis(names, result.spec.x ?? ""),
    yAxis: valAxis(d.y_label),
    series: [interval, dots],
  };
}

// --- lollipop -----------------------------------------------------------------

export function lollipopOption(result: RenderResult): VizOption {
  const d = result.chart_data as BarChartData;
  const order = barOrder(d, result.spec.sort);
  const categories = order.map((i) => String(d.categories[i]));
  // many categories read better horizontally: categories down the y axis
  const horizontal = categories.length > BAR_LABEL_ROTATE_FROM;
  const series: (BarSeriesOption | ScatterSeriesOption)[] = [];
  d.series.forEach((s, k) => {
    const values = order.map((i) => s.values[i] ?? null);
    series.push({
      type: "bar",
      name: s.name,
      data: values,
      barWidth: 2,
      barGap: "150%",
      itemStyle: { color: PALETTE[k % PALETTE.length], opacity: 0.7 },
      tooltip: { show: false },
    });
    series.push({
      type: "scatter",
      name: s.name,
      data: values.map((v, i) => (horizontal ? [v, i] : [i, v])) as [number | null, number | null][],
      symbolSize: 10,
      itemStyle: { color: PALETTE[k % PALETTE.length] },
      z: 3,
    });
  });
  const cat = { ...catAxis(categories, result.spec.x ?? ""), axisLabel: { ...axisBase.axisLabel, interval: 0, rotate: 0 } };
  const val = { ...axisBase, type: "value" as const, ...yAxisName(d.y_label) };
  return {
    ...baseOption(result, false, d.series.length > 1),
    tooltip: { ...tooltipBase, trigger: "axis", axisPointer: { type: "shadow" }, valueFormatter: tooltipValue },
    xAxis: horizontal ? { ...val, ...xAxisName(d.y_label), nameLocation: "middle", nameGap: 28, nameTextStyle: axisBase.nameTextStyle } : cat,
    yAxis: horizontal ? { ...cat, ...yAxisName(result.spec.x ?? ""), inverse: true } : val,
    series,
  };
}

// --- dumbbell / slopegraph ------------------------------------------------------

export function dumbbellOption(result: RenderResult): VizOption {
  const d = result.chart_data as TwoStateChartData;
  const [a, b] = d.series;
  const connector: CustomSeriesOption = {
    type: "custom",
    name: "change",
    data: d.categories.map((_, i) => [i, stat(a.values[i] ?? null), stat(b.values[i] ?? null)]),
    renderItem: (_params, api) => {
      const i = api.value(0) as number;
      const va = api.value(1) as number;
      const vb = api.value(2) as number;
      if (Number.isNaN(va) || Number.isNaN(vb)) return undefined;
      const [x, ya] = api.coord([i, va]);
      const [, yb] = api.coord([i, vb]);
      return { type: "line" as const, shape: { x1: x, y1: ya, x2: x, y2: yb }, style: { stroke: COLORS.muted, lineWidth: 2, opacity: 0.8 } };
    },
    encode: { x: 0, y: [1, 2] },
    tooltip: { show: false },
  };
  const dots = d.series.map(
    (s, k): ScatterSeriesOption => ({
      type: "scatter",
      name: s.name,
      data: s.values.map((v, i) => [i, v] as [number, number | null]),
      symbolSize: 10,
      itemStyle: { color: PALETTE[k % PALETTE.length] },
      z: 3,
    }),
  );
  return {
    ...baseOption(result, false, true),
    tooltip: {
      ...tooltipBase,
      trigger: "axis",
      axisPointer: { type: "shadow" },
      formatter: (p) => {
        const items = (Array.isArray(p) ? p : [p]).filter((it) => it.seriesType === "scatter");
        if (items.length === 0) return "";
        const i = items[0].dataIndex;
        const lines = d.series.map((s) => `${s.name}: ${fmt(s.values[i])}`);
        const va = a.values[i];
        const vb = b.values[i];
        const delta = va !== null && vb !== null ? `<br/>Δ ${fmt(vb - va)}` : "";
        return `${d.categories[i]}<br/>${lines.join("<br/>")}${delta}`;
      },
    },
    xAxis: catAxis(d.categories, result.spec.x ?? ""),
    yAxis: valAxis(d.y_label),
    series: [connector, ...dots],
  };
}

export function slopegraphOption(result: RenderResult): VizOption {
  const d = result.chart_data as TwoStateChartData;
  // one line per category between the two states; the value labels sit at
  // both ends so the graph reads without a tooltip
  const series: LineSeriesOption[] = d.categories.map((c, i) => ({
    type: "line",
    name: c,
    data: d.series.map((s) => s.values[i] ?? null),
    symbolSize: 7,
    lineStyle: { width: 1.75 },
    emphasis: { focus: "series", lineStyle: { width: 3 } },
    endLabel: { show: true, formatter: (p) => `${c} ${fmt(p.value as number)}`, color: COLORS.muted, fontSize: 11, distance: 8 },
    label: { show: false },
  }));
  return {
    ...baseOption(result, false, false),
    grid: { ...grid, right: 120 },
    tooltip: {
      ...tooltipBase,
      trigger: "axis",
      formatter: (p) => {
        const items = Array.isArray(p) ? p : [p];
        const state = d.states[items[0].dataIndex];
        return `${state}<br/>` + items.map((it) => `${it.seriesName}: ${fmt(it.value as number)}`).join("<br/>");
      },
    },
    xAxis: { ...axisBase, type: "category", data: d.states, boundaryGap: false, ...xAxisName(result.spec.group_by ?? ""), splitLine: { show: true, lineStyle: { color: COLORS.axisLine } } },
    yAxis: valAxis(d.y_label),
    series,
  };
}

// --- hexbin ---------------------------------------------------------------------

export function hexbinOption(result: RenderResult): VizOption {
  const d = result.chart_data as HexbinChartData;
  const [xLo] = d.x_range;
  const [yLo] = d.y_range;
  const counts = d.cells.map((c) => c[2]);
  const max = counts.length ? Math.max(...counts) : 1;
  // cell centre in data space from the offset coordinates the backend used
  const centre = (col: number, row: number): [number, number] => [
    xLo + (col + 0.5) * d.dx + (row % 2 === 1 ? d.dx / 2 : 0),
    yLo + row * d.dy * 0.75 + d.dy / 2,
  ];
  const hexes: CustomSeriesOption = {
    type: "custom",
    name: "count",
    data: d.cells.map(([col, row, count]) => [...centre(col, row), count]),
    renderItem: (_params, api) => {
      const cx = api.value(0) as number;
      const cy = api.value(1) as number;
      const [px, py] = api.coord([cx, cy]);
      // stage 19.1: the backend's cells are regular only when the plot is
      // square (dy is scaled by the data-range aspect). Measure both pitches
      // in pixels and draw the affinely stretched pointy-top hexagon whose
      // half-width is half the column spacing and whose vertical circumradius
      // is half the cell height (row pitch = 0.75 · dy): neighbouring cells
      // then share vertices at any aspect — no hairline gaps, no overlap.
      const [px2, py2] = api.coord([cx + d.dx, cy + d.dy]);
      const w = Math.max((px2 - px) / 2, 0.75);
      const h = Math.max((py - py2) / 2, 0.75);
      const points: number[][] = [
        [px, py - h],
        [px + w, py - h / 2],
        [px + w, py + h / 2],
        [px, py + h],
        [px - w, py + h / 2],
        [px - w, py - h / 2],
      ];
      return { type: "polygon" as const, shape: { points }, style: { fill: api.visual("color") as string, opacity: 0.9 } };
    },
    encode: { x: 0, y: 1, tooltip: 2 },
    clip: true,
  };
  return {
    ...baseOption(result, true, false),
    grid: gridWithVisualMap,
    tooltip: {
      ...tooltipBase,
      trigger: "item",
      formatter: (p) => {
        const item = Array.isArray(p) ? p[0] : p;
        const [cx, cy, count] = item.value as [number, number, number];
        return `${result.spec.x}: ≈ ${formatNumber(cx)}<br/>${result.spec.y}: ≈ ${formatNumber(cy)}<br/>count: ${count}`;
      },
    },
    xAxis: { ...axisBase, type: "value", scale: true, ...xAxisName(result.spec.x ?? "") },
    yAxis: { ...axisBase, type: "value", scale: true, ...yAxisName(d.y_label) },
    visualMap: {
      type: "continuous",
      dimension: 2,
      min: 1,
      max,
      calculable: false,
      orient: "vertical",
      right: 8,
      top: "middle",
      itemHeight: 140,
      itemWidth: 12,
      text: [String(max), "1"],
      inRange: { color: SEQUENTIAL },
      textStyle: { color: COLORS.muted, fontSize: 11 },
    },
    dataZoom: dataZoomInside(["x", "y"]),
    series: [hexes],
  };
}

// --- polar_line / rose ---------------------------------------------------------------

export function polarOption(result: RenderResult, kind: "polar_line" | "rose"): VizOption {
  const d = result.chart_data as BarChartData;
  const order = barOrder(d, result.spec.sort);
  const categories = order.map((i) => String(d.categories[i]));
  const series = d.series.map((s, k): LineSeriesOption | BarSeriesOption =>
    kind === "rose"
      ? {
          type: "bar",
          name: s.name,
          coordinateSystem: "polar",
          data: order.map((i) => s.values[i] ?? null),
          itemStyle: { color: PALETTE[k % PALETTE.length], opacity: 0.85 },
          emphasis: { itemStyle: { opacity: 1 } },
        }
      : {
          type: "line",
          name: s.name,
          coordinateSystem: "polar",
          data: order.map((i) => s.values[i] ?? null),
          symbolSize: 5,
          lineStyle: { width: 1.5 },
          areaStyle: { opacity: 0.12 },
          itemStyle: { color: PALETTE[k % PALETTE.length] },
          connectNulls: false,
        },
  );
  return {
    ...baseOption(result, false, d.series.length > 1),
    grid: undefined,
    tooltip: { ...tooltipBase, trigger: "axis", valueFormatter: (v) => `${tooltipValue(v)} (${d.y_label})` },
    polar: { center: ["50%", "54%"], radius: ["12%", "72%"] },
    // clockwise from the top, like a clock face or a calendar
    angleAxis: { type: "category", data: categories, startAngle: 90, clockwise: true, boundaryGap: kind === "rose", axisLine: { lineStyle: { color: COLORS.axisLine } }, axisTick: { show: false }, axisLabel: { color: COLORS.muted, fontSize: 11 }, splitLine: { show: true, lineStyle: { color: COLORS.splitLine } } },
    // no radial-axis name: it would sit on the 12 o'clock label (verifier N6);
    // the measure is named by the tooltip and the panel
    radiusAxis: { type: "value", axisLine: { show: false }, axisTick: { show: false }, axisLabel: { color: COLORS.muted, fontSize: 10 }, splitLine: { lineStyle: { color: COLORS.splitLine } }, z: 5 },
    series,
  };
}

// ============================================================================
// stage 20b: composition (stacked_bar) and flow (sankey). Both payloads are
// fully aggregated by the backend — the shares, the bar totals and the link
// weights all come from render.py; nothing is summed here.
// ============================================================================

// --- stacked_bar ---------------------------------------------------------------

export function stackedBarOption(result: RenderResult): VizOption {
  const d = result.chart_data as StackedBarChartData;
  const order = barOrder(d, result.spec.sort);
  const categories = order.map((i) => String(d.categories[i]));
  // one stack id: every series piles into the same bar. Null stays null, so a
  // combination that never occurs leaves a gap in the stack instead of a zero.
  const series: BarSeriesOption[] = d.series.map((s, k) => ({
    type: "bar",
    name: s.name,
    stack: "total",
    data: order.map((i) => s.values[i] ?? null),
    barMaxWidth: 48,
    barCategoryGap: "38%",
    itemStyle: { color: PALETTE[k % PALETTE.length], opacity: 0.92 },
    emphasis: { focus: "series", itemStyle: { opacity: 1 } },
  }));
  const totals = order.map((i) => d.totals[i] ?? null);
  return {
    ...baseOption(result, false, true),
    tooltip: {
      ...tooltipBase,
      trigger: "axis",
      axisPointer: { type: "shadow" },
      formatter: (p) => {
        const items = Array.isArray(p) ? p : [p];
        if (items.length === 0) return "";
        const i = items[0].dataIndex;
        const lines = items.map((it) => `${it.marker ?? ""}${it.seriesName}: ${fmt(it.value as number | null)}${d.percent ? " %" : ""}`);
        // a 100 % stack still names the magnitude it is a share of
        const total = totals[i];
        const foot = total === null ? "" : `<br/>${d.percent ? "總計" : "合計"} ${fmt(total)}`;
        return `${categories[i]}<br/>${lines.join("<br/>")}${foot}`;
      },
    },
    xAxis: catAxis(categories, result.spec.x ?? ""),
    yAxis: {
      ...axisBase,
      type: "value",
      ...yAxisName(d.y_label),
      ...(d.percent ? { max: 100, axisLabel: { ...axisBase.axisLabel, formatter: "{value}%" } } : {}),
    },
    series,
  };
}

// --- sankey -------------------------------------------------------------------

export function sankeyOption(result: RenderResult): VizOption {
  const d = result.chart_data as SankeyChartData;
  // node identities are side-prefixed by the backend so the same label on both
  // ends is two nodes; the label the reader sees is the column value itself.
  const label = new Map(d.nodes.map((n) => [n.name, n.label]));
  // one colour per distinct label, so the same category keeps its colour on
  // both ends and a transition is readable as "this went into that"
  const labels = [...new Set(d.nodes.map((n) => n.label))];
  const colorOf = (name: string) => PALETTE[Math.max(0, labels.indexOf(label.get(name) ?? name)) % PALETTE.length];
  const series: SankeySeriesOption = {
    type: "sankey",
    left: 24,
    right: 108,
    top: 48,
    bottom: 24,
    nodeWidth: 14,
    nodeGap: 10,
    draggable: false,
    emphasis: { focus: "adjacency" },
    data: d.nodes.map((n) => ({
      name: n.name,
      value: n.value,
      itemStyle: { color: colorOf(n.name), borderWidth: 0 },
      label: { formatter: () => n.label, color: COLORS.text, fontSize: 11 },
    })),
    links: d.links.map((l) => ({ source: l.source, target: l.target, value: l.value })),
    lineStyle: { color: "gradient", opacity: 0.38, curveness: 0.5 },
  };
  return {
    ...baseOption(result, false, false),
    grid: undefined,
    tooltip: {
      ...tooltipBase,
      trigger: "item",
      formatter: (p) => {
        const item = Array.isArray(p) ? p[0] : p;
        if (item.dataType === "edge") {
          const link = item.data as SankeyLink;
          return `${label.get(link.source) ?? link.source} → ${label.get(link.target) ?? link.target}<br/>${d.y_label}: ${fmt(link.value)}`;
        }
        const node = d.nodes.find((n) => n.name === item.name);
        if (!node) return String(item.name);
        return `${node.label}（${node.column}）<br/>${d.y_label}: ${fmt(node.value)}`;
      },
    },
    series: [series],
  };
}
