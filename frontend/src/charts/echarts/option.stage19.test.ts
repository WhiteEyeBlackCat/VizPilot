// Stage 19 builders: every new chart type maps the backend payload to an
// option without recomputing anything, keeps the backend's category order,
// and renders server-side without throwing (custom series included).

import { describe, expect, it } from "vitest";

import type { BarSeriesOption, CustomSeriesOption, LineSeriesOption, ScatterSeriesOption } from "echarts/charts";

import type { BarChartData, DensityChartData, DotIntervalChartData, HeatmapChartData, HexbinChartData, LineChartData, RenderResult, TwoStateChartData } from "../../types";
import { echarts } from "./echarts";
import { buildOption } from "./option";
import { displayOrderNote } from "./order";

const spec = (partial: Partial<RenderResult["spec"]> & { type: RenderResult["spec"]["type"] }): RenderResult["spec"] => ({
  title: "t",
  x: null,
  y: null,
  group_by: null,
  aggregation: null,
  bins: null,
  time_granularity: null,
  top_n: null,
  sort: null,
  reason: "",
  priority: 1,
  ...partial,
});
const result = (s: RenderResult["spec"], chart_data: RenderResult["chart_data"], n = 10): RenderResult => ({
  spec: s,
  chart_data,
  sampled: false,
  n_points: n,
  display_range: null,
});
const series = (r: RenderResult) => {
  const s = buildOption(r).series;
  return (Array.isArray(s) ? s : [s]) as (CustomSeriesOption | ScatterSeriesOption | BarSeriesOption | LineSeriesOption)[];
};
const axis = (a: unknown) => (Array.isArray(a) ? a[0] : a) as { type?: string; data?: unknown[] };

function renderSvg(r: RenderResult): string {
  const chart = echarts.init(null, undefined, { renderer: "svg", ssr: true, width: 600, height: 400 });
  chart.setOption(buildOption(r));
  const svg = chart.renderToSVGString();
  chart.dispose();
  return svg;
}

const grid = Array.from({ length: 8 }, (_, i) => i);
const group = (name: string, peak: number, points?: number[]) => ({
  name,
  q1: 2,
  median: 3,
  q3: 4,
  lower_fence: 0,
  upper_fence: 6,
  outliers: [],
  mean: 3.1,
  count: 40,
  density: grid.map((v) => (v === peak ? 1 : 0.2)),
  bandwidth: 0.5,
  ...(points ? { points } : {}),
});
const density: DensityChartData = { groups: [group("low", 2, [1, 2]), group("medium", 4, [3]), group("high", 6, [5, 6, 7])], grid, y_label: "score" };

describe("density family", () => {
  it("violin: one polygon datum per group in backend order, plus the summary glyphs", () => {
    const r = result(spec({ type: "violin", x: "level", y: "score", sort: "natural" }), density);
    const s = series(r);
    expect(s.map((x) => x.type)).toEqual(["custom", "custom"]);
    expect(((s[0] as CustomSeriesOption).data as number[][]).map((d) => d[0])).toEqual([0, 1, 2]);
    expect(displayOrderNote(r)).toBe("依 level 的類別順序顯示，未依值排序");
    expect(renderSvg(r)).toContain("<polygon");
  });
  it("raincloud adds the backend's point sample as a scatter, never new points", () => {
    const r = result(spec({ type: "raincloud", x: "level", y: "score" }), density);
    const s = series(r);
    const rain = s.find((x) => x.type === "scatter") as ScatterSeriesOption;
    expect((rain.data as number[][]).length).toBe(2 + 1 + 3);
    expect((rain.data as number[][]).map((p) => p[1])).toEqual([1, 2, 3, 5, 6, 7]);
    expect(renderSvg(r)).toContain("<polygon");
  });
  it("the value axis spans the data (grid + whiskers), never the group indices", () => {
    // verifier round 1: a custom series has no values ECharts can scale
    // from, so violin / ridgeline rendered blank on a 14..281 measure
    const big: DensityChartData = {
      ...density,
      grid: grid.map((v) => 14 + v * 38),
      groups: density.groups.map((g) => ({ ...g, q1: 90, median: 120, q3: 160, lower_fence: 10, upper_fence: 290 })),
    };
    const violin = buildOption(result(spec({ type: "violin", x: "level", y: "score" }), big));
    // the grid (14..280) drives the extent; the far-out fences (10 / 290) do not
    const y = axis(violin.yAxis) as { min?: number; max?: number };
    expect(y.min).toBeLessThanOrEqual(14);
    expect(y.min).toBeGreaterThan(-10);
    expect(y.max).toBeGreaterThanOrEqual(280);
    expect(y.max).toBeLessThan(310);
    const ridge = buildOption(result(spec({ type: "ridgeline", x: "level", y: "score" }), big));
    const x = axis(ridge.xAxis) as { min?: number; max?: number };
    expect(x.min).toBeLessThanOrEqual(14);
    expect(x.max).toBeGreaterThanOrEqual(280);
    // every datum carries the data range, so even an auto-scaled axis would be right
    const datum = ((series(result(spec({ type: "violin", x: "level", y: "score" }), big))[0] as CustomSeriesOption).data as number[][])[0];
    expect(datum).toEqual([0, 14, 280]);
  });
  it("ridgeline lays the groups along the vertical axis with the value along x", () => {
    const r = result(spec({ type: "ridgeline", x: "level", y: "score" }), density);
    const o = buildOption(r);
    expect(axis(o.xAxis).type).toBe("value");
    expect(axis(o.yAxis).type).toBe("value"); // rows 0..n-1 labelled by group name
    expect(series(r)).toHaveLength(1);
    expect(renderSvg(r)).toContain("<polygon");
  });
});

describe("dot_interval", () => {
  const data: DotIntervalChartData = {
    groups: [
      { name: "a", mean: 2.5, lo: 1.5, hi: 3.5, n: 4 },
      { name: "b", mean: 10, lo: null, hi: null, n: 1 },
    ],
    y_label: "mean(v)",
  };
  it("draws the mean as a dot and the interval as a custom glyph; a null interval draws nothing", () => {
    const r = result(spec({ type: "dot_interval", x: "g", y: "v" }), data);
    const s = series(r);
    expect(s.map((x) => x.type)).toEqual(["custom", "scatter"]);
    expect((s[1] as ScatterSeriesOption).data).toEqual([[0, 2.5], [1, 10]]);
    expect(axis(buildOption(r).xAxis).data).toEqual(["a", "b"]);
    expect(renderSvg(r)).toMatch(/<(circle|path)/); // symbols render as paths in SSR
  });
});

describe("lollipop", () => {
  const data: BarChartData = { categories: ["C", "B", "A"], series: [{ name: "v", values: [3, 2, null] }], truncated: false, y_label: "mean(v)" };
  it("stems and dots share the bar payload; a ranked spec keeps the backend order", () => {
    const r = result(spec({ type: "lollipop", x: "c", y: "v", aggregation: "mean", sort: "value" }), data);
    const s = series(r);
    expect(s.map((x) => x.type)).toEqual(["bar", "scatter"]);
    expect((s[0] as BarSeriesOption).data).toEqual([3, 2, null]);
    expect(axis(buildOption(r).xAxis).data).toEqual(["C", "B", "A"]);
    expect(displayOrderNote(r)).toBe("依 平均 v 由高到低排序");
    expect(renderSvg(r)).toMatch(/<(circle|path)/); // symbols render as paths in SSR
  });
  it("many categories flip to a horizontal layout", () => {
    const many: BarChartData = {
      categories: Array.from({ length: 12 }, (_, i) => `c${i}`),
      series: [{ name: "v", values: Array.from({ length: 12 }, (_, i) => 12 - i) }],
      truncated: false,
      y_label: "v",
    };
    const o = buildOption(result(spec({ type: "lollipop", x: "c", y: "v", aggregation: "mean" }), many));
    expect(axis(o.yAxis).type).toBe("category");
    expect(axis(o.xAxis).type).toBe("value");
  });
});

describe("two-state charts", () => {
  const data: TwoStateChartData = {
    categories: ["A", "B", "C"],
    states: ["before", "after"],
    series: [
      { name: "before", values: [5, 6, 7] },
      { name: "after", values: [8, 6, null] },
    ],
    y_label: "mean(value)",
  };
  it("dumbbell: a connector per category and one dot series per state", () => {
    const r = result(spec({ type: "dumbbell", x: "cat", y: "value", group_by: "period", aggregation: "mean" }), data);
    const s = series(r);
    expect(s.map((x) => x.type)).toEqual(["custom", "scatter", "scatter"]);
    expect((s[1] as ScatterSeriesOption).name).toBe("before");
    expect((s[2] as ScatterSeriesOption).data).toEqual([[0, 8], [1, 6], [2, null]]);
    expect(axis(buildOption(r).xAxis).data).toEqual(["A", "B", "C"]);
    expect(renderSvg(r)).toMatch(/<(circle|path)/); // symbols render as paths in SSR
  });
  it("slopegraph: one line per category between the two states", () => {
    const r = result(spec({ type: "slopegraph", x: "cat", y: "value", group_by: "period", aggregation: "mean" }), data);
    const s = series(r) as LineSeriesOption[];
    expect(s.map((x) => x.name)).toEqual(["A", "B", "C"]);
    expect(s[0].data).toEqual([5, 8]);
    expect(axis(buildOption(r).xAxis).data).toEqual(["before", "after"]);
    expect(renderSvg(r)).toContain("<path");
  });
});

describe("hexbin", () => {
  const data: HexbinChartData = {
    cells: [
      [0, 0, 5],
      [1, 1, 2],
      [3, 0, 9],
    ],
    x_range: [0, 4],
    y_range: [0, 2],
    dx: 0.1,
    dy: 0.2,
    columns: 40,
    rows: 14,
    n: 16,
    y_label: "y",
  };
  it("one hexagon per cell with the count as the colour dimension", () => {
    const r = result(spec({ type: "hexbin", x: "x", y: "y" }), data);
    const s = series(r);
    expect(s).toHaveLength(1);
    const cells = (s[0] as CustomSeriesOption).data as number[][];
    expect(cells.map((c) => c[2])).toEqual([5, 2, 9]);
    expect(cells[1][0]).toBeCloseTo(0 + 1.5 * 0.1 + 0.05); // odd row shifted by half a cell
    const o = buildOption(r);
    expect((o.visualMap as { max: number }).max).toBe(9);
    expect(renderSvg(r)).toContain("<polygon");
  });
});

describe("polar_line / rose", () => {
  const data: BarChartData = { categories: [1, 2, 3, 4], series: [{ name: "temp", values: [1, 2, 3, 2] }], truncated: false, y_label: "mean(temp)" };
  it("draw on a polar coordinate system in the backend's cycle order", () => {
    const line = result(spec({ type: "polar_line", x: "mnth", y: "temp", aggregation: "mean", sort: "natural" }), data);
    const o = buildOption(line);
    expect(o.polar).toBeDefined();
    expect((o.angleAxis as { data: string[] }).data).toEqual(["1", "2", "3", "4"]);
    expect((series(line)[0] as LineSeriesOption).coordinateSystem).toBe("polar");
    expect(displayOrderNote(line)).toBe("依 mnth 的類別順序顯示，未依值排序");
    const rose = result(spec({ type: "rose", x: "mnth", aggregation: "count", sort: "natural" }), { ...data, series: [{ name: "count", values: [4, 5, 6, 5] }], y_label: "count" });
    expect((series(rose)[0] as BarSeriesOption).coordinateSystem).toBe("polar");
    expect(renderSvg(line)).toContain("<path");
    expect(renderSvg(rose)).toContain("<path");
  });
});

describe("line over an ordered category and the matrix heatmap", () => {
  it("line with weekday names uses a category axis in the backend's order", () => {
    const data: LineChartData = { series: [{ name: "temp", x: ["Mon", "Tue", "Wed"], y: [1, 2, 3] }], y_label: "mean(temp)" };
    const o = buildOption(result(spec({ type: "line", x: "weekday", y: "temp", aggregation: "mean", sort: "natural" }), data));
    expect(axis(o.xAxis)).toMatchObject({ type: "category", data: ["Mon", "Tue", "Wed"] });
  });
  it("ISO timestamps still make a time axis", () => {
    const data: LineChartData = { series: [{ name: "v", x: ["2024-01-01T00:00:00", "2024-01-02T00:00:00"], y: [1, 2] }], y_label: "v" };
    expect(axis(buildOption(result(spec({ type: "line", x: "ts", y: "v" }), data)).xAxis).type).toBe("time");
  });
  it("matrix heatmap: rows from group_by, a sequential scale over the aggregate range", () => {
    const data: HeatmapChartData = { columns: ["1", "2", "3"], rows: ["low", "high"], matrix: [[1, 2, 3], [4, null, 6]], y_label: "mean(temp)" };
    const r = result(spec({ type: "heatmap", x: "mnth", y: "temp", group_by: "level", aggregation: "mean" }), data);
    const o = buildOption(r);
    expect(axis(o.yAxis).data).toEqual(["low", "high"]);
    expect(o.visualMap).toMatchObject({ min: 1, max: 6 });
    expect(renderSvg(r)).toContain("<path");
  });
});
