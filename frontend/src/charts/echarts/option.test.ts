// Contract tests for the ECharts adapter against RenderResult snapshots
// recorded from the real backend (src/charts/__fixtures__, see
// e2e/capture-fixtures.mjs). Every assertion is about semantics the
// backend guarantees and the chart must not alter.

import { describe, expect, it } from "vitest";

import type { BarSeriesOption, BoxplotSeriesOption, CustomSeriesOption, HeatmapSeriesOption, LineSeriesOption, ScatterSeriesOption } from "echarts/charts";
import type { DataZoomComponentOption, LegendComponentOption, ToolboxComponentOption, VisualMapComponentOption } from "echarts/components";

import barCount from "../__fixtures__/bar_count.json";
import barGroupedNull from "../__fixtures__/bar_grouped_null.json";
import barSingle from "../__fixtures__/bar_single.json";
import barTruncated from "../__fixtures__/bar_truncated.json";
import boxOutliers from "../__fixtures__/box_outliers.json";
import boxSingle from "../__fixtures__/box_single.json";
import heatmap from "../__fixtures__/heatmap.json";
import heatmapNullCell from "../__fixtures__/heatmap_null_cell.json";
import histogramDisplayRange from "../__fixtures__/histogram_display_range.json";
import histogramGrouped from "../__fixtures__/histogram_grouped.json";
import histogramSingle from "../__fixtures__/histogram_single.json";
import lineGrouped from "../__fixtures__/line_grouped.json";
import lineNumericX from "../__fixtures__/line_numeric_x.json";
import lineRawTime from "../__fixtures__/line_raw_time.json";
import lineSingle from "../__fixtures__/line_single.json";
import scatterGrouped from "../__fixtures__/scatter_grouped.json";
import scatterSampled from "../__fixtures__/scatter_sampled.json";
import scatterSingle from "../__fixtures__/scatter_single.json";
import type {
  BarChartData,
  BoxChartData,
  HeatmapChartData,
  HistogramChartData,
  LineChartData,
  RenderResult,
} from "../../types";
import { DIVERGING, PALETTE } from "./theme";
import {
  buildOption,
  HISTOGRAM_GROUP_OPACITY,
  heatmapCells,
  histogramBins,
  isEmpty,
  LARGE_SCATTER_THRESHOLD,
  timeLabelFormatter,
} from "./option";

const fx = (json: unknown) => json as RenderResult;

type AnySeries =
  | LineSeriesOption
  | BarSeriesOption
  | ScatterSeriesOption
  | BoxplotSeriesOption
  | HeatmapSeriesOption
  | CustomSeriesOption;

function seriesOf(result: RenderResult): AnySeries[] {
  const s = buildOption(result).series;
  return (Array.isArray(s) ? s : [s]) as AnySeries[];
}

const axis = (a: unknown) => (Array.isArray(a) ? a[0] : a) as Record<string, unknown>;

describe("shared", () => {
  it("every chart carries the spec title, the palette and an image export", () => {
    for (const f of [lineSingle, barSingle, scatterSingle, histogramSingle, boxOutliers, heatmap]) {
      const o = buildOption(fx(f));
      expect((o.title as { text: string }).text).toBe(fx(f).spec.title);
      expect(o.color).toEqual(PALETTE);
      const tb = o.toolbox as ToolboxComponentOption;
      expect(tb.feature?.saveAsImage).toMatchObject({ name: fx(f).spec.title });
    }
  });

  it("zoom tools only where the axes are continuous", () => {
    const zoomable = (f: unknown) =>
      Boolean((buildOption(fx(f)).toolbox as ToolboxComponentOption).feature?.dataZoom);
    expect(zoomable(lineSingle)).toBe(true);
    expect(zoomable(scatterSingle)).toBe(true);
    expect(zoomable(histogramSingle)).toBe(true);
    expect(zoomable(barSingle)).toBe(false);
    expect(zoomable(boxOutliers)).toBe(false);
    expect(zoomable(heatmap)).toBe(false);
  });

  it("isEmpty reflects n_points", () => {
    expect(isEmpty({ ...fx(lineSingle), n_points: 0 })).toBe(true);
    expect(isEmpty(fx(lineSingle))).toBe(false);
  });
});

describe("line", () => {
  it("single series: one line, ISO x -> time axis, y_label as y-axis name", () => {
    const r = fx(lineSingle);
    const d = r.chart_data as LineChartData;
    const o = buildOption(r);
    const s = seriesOf(r);
    expect(s).toHaveLength(1);
    expect(s[0]).toMatchObject({ type: "line", name: d.series[0].name, connectNulls: false });
    expect((s[0].data as unknown[][]).map((p) => p[0])).toEqual(d.series[0].x);
    expect((s[0].data as unknown[][]).map((p) => p[1])).toEqual(d.series[0].y);
    expect(axis(o.xAxis).type).toBe("time");
    expect(axis(o.xAxis).name).toBe("timestamp");
    expect(axis(o.yAxis)).toMatchObject({ type: "value", name: "mean(temperature)" });
    expect((o.legend as LegendComponentOption).show).toBe(false);
    expect((o.dataZoom as DataZoomComponentOption[]).map((z) => z.type)).toEqual(["inside"]);
  });

  it("grouped: one series per group in backend order, legend shown", () => {
    const r = fx(lineGrouped);
    const d = r.chart_data as LineChartData;
    const s = seriesOf(r);
    expect(s.map((x) => x.name)).toEqual(d.series.map((x) => x.name));
    expect(s).toHaveLength(3);
    for (let i = 0; i < 3; i++) {
      expect((s[i].data as unknown[][]).map((p) => p[1])).toEqual(d.series[i].y);
    }
    expect((buildOption(r).legend as LegendComponentOption).show).not.toBe(false);
  });

  it("raw time series with many points hides symbols but keeps every point", () => {
    const r = fx(lineRawTime);
    const s = seriesOf(r);
    expect((s[0] as LineSeriesOption).showSymbol).toBe(false);
    expect((s[0].data as unknown[]).length).toBe(r.n_points);
  });

  it("time axis labels follow the granularity the backend chose (spec.time_granularity)", () => {
    // day granularity: month-day ticks, the first tick of a year carries the year
    const day = axis(buildOption(fx(lineSingle)).xAxis);
    expect(fx(lineSingle).spec.time_granularity).toBe("day");
    expect(day.axisLabel).toMatchObject({ formatter: { day: "{MM}-{dd}", year: "{yyyy}-{MM}-{dd}" } });
    // hourly series left unbucketed by the backend (granularity null): the
    // timestamps are intraday, so hour ticks show the clock time
    expect(fx(lineRawTime).spec.time_granularity ?? null).toBeNull();
    const raw = axis(buildOption(fx(lineRawTime)).xAxis);
    expect(raw.axisLabel).toMatchObject({ formatter: { hour: "{MM}-{dd} {HH}:{mm}", minute: "{HH}:{mm}" } });
    expect(timeLabelFormatter("month")).toMatchObject({ month: "{yyyy}-{MM}", year: "{yyyy}-{MM}" });
    expect(timeLabelFormatter("week")).toEqual(timeLabelFormatter("day"));
    expect(timeLabelFormatter(null, false)).toEqual(timeLabelFormatter("day"));
    expect(timeLabelFormatter("raw", true)).not.toEqual(timeLabelFormatter("day"));
    // a numeric x axis has no time formatter
    expect(axis(buildOption(fx(lineNumericX)).xAxis).axisLabel).not.toHaveProperty("formatter");
  });

  it("numeric x -> value axis (no time parsing of numbers)", () => {
    const r = fx(lineNumericX);
    const o = buildOption(r);
    expect(axis(o.xAxis).type).toBe("value");
    expect(axis(o.xAxis).name).toBe("price");
    const d = r.chart_data as LineChartData;
    expect((seriesOf(r)[0].data as unknown[][])[0]).toEqual([d.series[0].x[0], d.series[0].y[0]]);
  });
});

describe("scatter", () => {
  it("single: value axes named after spec.x / y_label, x-y pairs verbatim", () => {
    const r = fx(scatterSingle);
    const d = r.chart_data as LineChartData;
    const o = buildOption(r);
    const s = seriesOf(r)[0] as ScatterSeriesOption;
    expect(s.type).toBe("scatter");
    expect((s.data as unknown[][]).map((p) => p[0])).toEqual(d.series[0].x);
    expect((s.data as unknown[][]).map((p) => p[1])).toEqual(d.series[0].y);
    expect(axis(o.xAxis)).toMatchObject({ type: "value", name: "temperature" });
    expect(axis(o.yAxis)).toMatchObject({ type: "value", name: "humidity" });
    expect((o.dataZoom as DataZoomComponentOption[]).length).toBe(2);
    expect(s.large).toBe(false); // 720 points: normal pipeline
  });

  it("grouped: series per group, legend on", () => {
    const r = fx(scatterGrouped);
    const d = r.chart_data as LineChartData;
    expect(seriesOf(r).map((s) => s.name)).toEqual(d.series.map((s) => s.name));
    expect((buildOption(r).legend as LegendComponentOption).show).not.toBe(false);
  });

  it("sampled 10k points: large mode on, every sampled point kept", () => {
    const r = fx(scatterSampled);
    expect(r.sampled).toBe(true);
    const s = seriesOf(r)[0] as ScatterSeriesOption;
    expect((s.data as unknown[]).length).toBe(r.n_points);
    expect(r.n_points).toBeGreaterThan(LARGE_SCATTER_THRESHOLD);
    expect(s.large).toBe(true);
  });
});

describe("bar", () => {
  it("category axis keeps the backend ranking; values verbatim", () => {
    const r = fx(barSingle);
    const d = r.chart_data as BarChartData;
    const o = buildOption(r);
    expect(axis(o.xAxis)).toMatchObject({ type: "category", data: d.categories.map(String), name: "station" });
    const s = seriesOf(r)[0] as BarSeriesOption;
    expect(s.data).toEqual(d.series[0].values);
    expect(axis(o.yAxis).name).toBe("mean(pm25)");
  });

  it("count bars: series named count, y_label count, ranking C > B > A untouched", () => {
    const r = fx(barCount);
    const d = r.chart_data as BarChartData;
    expect(d.categories).toEqual(["C", "B", "A"]); // recorded backend ranking
    expect(axis(buildOption(r).xAxis).data).toEqual(["C", "B", "A"]);
    expect(seriesOf(r)[0]).toMatchObject({ name: "count", data: d.series[0].values });
    expect(axis(buildOption(r).yAxis).name).toBe("count");
  });

  it("grouped with a missing cell: null stays null (gap), series aligned to categories", () => {
    const r = fx(barGroupedNull);
    const d = r.chart_data as BarChartData;
    const s = seriesOf(r) as BarSeriesOption[];
    expect(s).toHaveLength(d.series.length);
    const gadget = s.find((x) => x.name === "Gadget")!;
    expect(d.series.find((x) => x.name === "Gadget")!.values[0]).toBeNull(); // East x Gadget
    expect((gadget.data as unknown[])[0]).toBeNull();
    for (const series of s) expect((series.data as unknown[]).length).toBe(d.categories.length);
    expect((buildOption(r).legend as LegendComponentOption).show).not.toBe(false);
  });

  it("truncated top_n: exactly the returned categories, nothing re-sorted", () => {
    const r = fx(barTruncated);
    const d = r.chart_data as BarChartData;
    expect(d.truncated).toBe(true);
    expect(axis(buildOption(r).xAxis).data).toEqual(d.categories);
    expect((seriesOf(r)[0].data as unknown[]).length).toBe(r.spec.top_n);
  });

  it("numeric / boolean categories are stringified, not re-sorted", () => {
    const r: RenderResult = {
      ...fx(barSingle),
      chart_data: { categories: [5, 4, true], series: [{ name: "v", values: [1, 2, 3] }], truncated: false, y_label: "v" },
    };
    expect(axis(buildOption(r).xAxis).data).toEqual(["5", "4", "true"]);
  });
});

describe("histogram", () => {
  it("bins: [lo, hi, count] straight from edges/counts", () => {
    expect(histogramBins([0, 2, 4], [3, 7])).toEqual([
      [0, 2, 3],
      [2, 4, 7],
    ]);
    const r = fx(histogramSingle);
    const d = r.chart_data as HistogramChartData;
    const s = seriesOf(r)[0] as CustomSeriesOption;
    expect(s.type).toBe("custom");
    expect(s.data).toEqual(histogramBins(d.bins.edges, d.bins.counts));
    expect((s.data as unknown[]).length).toBe(d.bins.counts.length);
    expect(s.itemStyle).toMatchObject({ opacity: 1 });
    const o = buildOption(r);
    expect(axis(o.xAxis)).toMatchObject({ type: "value", name: "pm25" });
    expect(axis(o.yAxis).name).toBe("count");
  });

  it("grouped: one translucent series per group with shared edges, overall bins ignored", () => {
    const r = fx(histogramGrouped);
    const d = r.chart_data as HistogramChartData;
    const s = seriesOf(r) as CustomSeriesOption[];
    expect(s.map((x) => x.name)).toEqual(d.series!.map((x) => x.name));
    for (let i = 0; i < s.length; i++) {
      expect(s[i].data).toEqual(histogramBins(d.series![i].edges, d.series![i].counts));
      expect(s[i].itemStyle).toMatchObject({ opacity: HISTOGRAM_GROUP_OPACITY });
    }
  });

  it("display_range: bins cover only the robust window; excluded rows are not drawn", () => {
    const r = fx(histogramDisplayRange);
    const d = r.chart_data as HistogramChartData;
    expect(r.display_range).toMatchObject({ lo: 17.6, hi: 30.5 });
    const bins = (seriesOf(r)[0] as CustomSeriesOption).data as [number, number, number][];
    expect(bins[0][0]).toBe(r.display_range!.lo);
    expect(bins[bins.length - 1][1]).toBe(r.display_range!.hi);
    const drawn = bins.reduce((acc, b) => acc + b[2], 0);
    expect(drawn).toBe(d.bins.counts.reduce((a, b) => a + b, 0));
    // invariant recorded by stage 9: drawn + excluded == valid rows (1000)
    expect(drawn + r.display_range!.excluded_below + r.display_range!.excluded_above).toBe(1000);
  });
});

describe("box", () => {
  it("five-number tuple is [lower_fence, q1, median, q3, upper_fence] per group; outliers and means overlaid", () => {
    const r = fx(boxOutliers);
    const d = r.chart_data as BoxChartData;
    const s = seriesOf(r);
    const box = s[0] as BoxplotSeriesOption;
    expect(box.type).toBe("boxplot");
    expect(box.data).toEqual(d.groups.map((g) => [g.lower_fence, g.q1, g.median, g.q3, g.upper_fence]));
    expect(axis(buildOption(r).xAxis)).toMatchObject({ type: "category", data: d.groups.map((g) => g.name) });
    const outliers = s.find((x) => x.name === "outliers") as ScatterSeriesOption;
    const expected = d.groups.flatMap((g) => g.outliers.map((v) => [g.name, v]));
    expect(outliers.data).toEqual(expected);
    expect(expected.length).toBe(32); // recorded: 10 + 10 + 12
    const mean = s.find((x) => x.name === "mean") as ScatterSeriesOption;
    expect(mean.data).toEqual(d.groups.map((g) => [g.name, g.mean]));
    expect(axis(buildOption(r).yAxis).name).toBe("value");
  });

  it("no x: the single 'all' group", () => {
    const r = fx(boxSingle);
    expect(axis(buildOption(r).xAxis).data).toEqual(["all"]);
    expect((seriesOf(r)[0].data as unknown[]).length).toBe(1);
  });

  it("a null statistic becomes NaN (undrawn), never 0", () => {
    const r: RenderResult = {
      ...fx(boxSingle),
      chart_data: {
        groups: [{ name: "g", q1: null, median: 1, q3: 2, lower_fence: null, upper_fence: 3, outliers: [], mean: null, count: 1 }],
        y_label: "v",
      },
    };
    const box = seriesOf(r)[0] as BoxplotSeriesOption;
    const row = (box.data as number[][])[0];
    expect(Number.isNaN(row[0])).toBe(true);
    expect(Number.isNaN(row[1])).toBe(true);
    expect(row.slice(2)).toEqual([1, 2, 3]);
    expect(seriesOf(r).find((x) => x.name === "mean")).toBeUndefined();
  });
});

describe("heatmap", () => {
  it("cells are [col, row, value]; symmetric -1..1 diverging scale; rows inverted", () => {
    const r = fx(heatmap);
    const d = r.chart_data as HeatmapChartData;
    const o = buildOption(r);
    const s = seriesOf(r)[0] as HeatmapSeriesOption;
    expect(s.type).toBe("heatmap");
    expect(s.data).toEqual(heatmapCells(d.matrix));
    expect((s.data as unknown[]).length).toBe(d.columns.length ** 2);
    expect(axis(o.xAxis)).toMatchObject({ type: "category", data: d.columns });
    expect(axis(o.yAxis)).toMatchObject({ type: "category", data: d.columns, inverse: true });
    const vm = (Array.isArray(o.visualMap) ? o.visualMap[0] : o.visualMap) as VisualMapComponentOption;
    expect(vm).toMatchObject({ type: "continuous", min: -1, max: 1, inRange: { color: DIVERGING } });
  });

  it("a null correlation is left blank (no cell), not drawn as 0", () => {
    const r = fx(heatmapNullCell);
    const d = r.chart_data as HeatmapChartData;
    expect(d.matrix[1][2]).toBeNull();
    const cells = (seriesOf(r)[0] as HeatmapSeriesOption).data as [number, number, number][];
    expect(cells.length).toBe(d.columns.length ** 2 - 2);
    expect(cells.find((c) => c[0] === 2 && c[1] === 1)).toBeUndefined();
    expect(cells.find((c) => c[0] === 1 && c[1] === 2)).toBeUndefined();
    expect(cells.find((c) => c[0] === 0 && c[1] === 0)).toEqual([0, 0, 1]);
  });
});
