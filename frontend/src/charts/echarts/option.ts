// Pure RenderResult -> ECharts option conversion. No React, no DOM, no
// echarts runtime import (types only) so it is unit-testable and the only
// place ECharts-specific structure lives. Nothing is re-aggregated,
// re-sorted or re-binned here — the backend RenderResult is the single
// source of truth. Semantics are documented per chart type below.

import type {
  BarSeriesOption,
  BoxplotSeriesOption,
  CustomSeriesOption,
  HeatmapSeriesOption,
  LineSeriesOption,
  ScatterSeriesOption,
} from "echarts/charts";
// type-only: the runtime registration lives in ./echarts.ts
import type { CustomSeriesRenderItemAPI, CustomSeriesRenderItemParams } from "echarts";

import type {
  BarChartData,
  BoxChartData,
  HeatmapChartData,
  HistogramChartData,
  LineChartData,
  RenderResult,
  TimeGranularity,
} from "../../types";
import type { VizOption } from "./echarts";
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

const isTemporalX = (data: LineChartData): boolean =>
  data.series.some((s) => s.x.some((v) => typeof v === "string"));

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
    animationDuration: 300,
  };
}

// --- line -------------------------------------------------------------------

function lineOption(result: RenderResult): VizOption {
  const d = result.chart_data as LineChartData;
  const temporal = isTemporalX(d);
  const many = d.series.some((s) => s.x.length > LINE_SYMBOL_MAX_POINTS);
  const series: LineSeriesOption[] = d.series.map((s) => ({
    type: "line",
    name: s.name,
    data: pairs(s.x, s.y),
    showSymbol: !many,
    symbolSize: 5,
    lineStyle: { width: 1.5 },
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
    itemStyle: { opacity: 0.75 },
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
  // stay null so a missing (category, group) cell is a gap, not a zero
  const categories = d.categories.map(String);
  const series: BarSeriesOption[] = d.series.map((s) => ({
    type: "bar",
    name: s.name,
    data: s.values.map((v) => v ?? null),
    barMaxWidth: 48,
    itemStyle: { borderRadius: [2, 2, 0, 0] },
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
  const names = d.groups.map((g) => g.name);
  // ECharts boxplot order: [min, Q1, median, Q3, max]; the backend's fences
  // play the min/max role (whisker ends), exactly as the backend computed them
  const stats: BoxStats[] = d.groups.map((g) => [
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
        const g = d.groups[item.dataIndex];
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
  for (const g of d.groups) for (const v of g.outliers) outliers.push([g.name, v]);
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
  const means = d.groups.filter((g) => g.mean !== null).map((g) => [g.name, g.mean as number] as [string, number]);
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
  const series: HeatmapSeriesOption = {
    type: "heatmap",
    name: d.y_label,
    data: heatmapCells(d.matrix),
    label: {
      show: true,
      fontSize: 11,
      color: COLORS.text,
      formatter: (p) => (p.value as HeatmapCell)[2].toFixed(2),
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
        return `${d.columns[i]} × ${d.columns[j]}<br/>${d.y_label}: ${v.toFixed(3)}`;
      },
    },
    xAxis: {
      ...axisBase,
      type: "category",
      data: d.columns,
      splitArea: { show: true },
      axisLabel: { ...axisBase.axisLabel, interval: 0, rotate: d.columns.length > 6 ? 45 : 0 },
    },
    // row 0 at the top, like a matrix
    yAxis: { ...axisBase, type: "category", data: d.columns, inverse: true, splitArea: { show: true } },
    visualMap: {
      type: "continuous",
      min: -1,
      max: 1,
      calculable: false,
      orient: "vertical",
      right: 8,
      top: "middle",
      itemHeight: 140,
      itemWidth: 12,
      precision: 1,
      text: ["1", "-1"],
      inRange: { color: DIVERGING },
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
  }
}
