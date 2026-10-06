// Stage 20b builders: the composition (stacked_bar) and flow (sankey)
// payloads become options without any re-aggregation — the shares, the bar
// totals and the link weights are the backend's — and both render
// server-side without throwing.

import { describe, expect, it } from "vitest";

import type { BarSeriesOption, SankeySeriesOption } from "echarts/charts";

import sankeyHubFlow from "../__fixtures__/sankey_hub_flow.json";
import sankeyHubRevenue from "../__fixtures__/sankey_hub_revenue.json";
import stackedBarMix from "../__fixtures__/stacked_bar_mix.json";
import stackedBarPercent from "../__fixtures__/stacked_bar_percent.json";
import type { RenderResult, SankeyChartData, StackedBarChartData } from "../../types";
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
  percent: null,
  reason: "",
  priority: 1,
  ...partial,
});
const result = (s: RenderResult["spec"], chart_data: RenderResult["chart_data"], n = 12): RenderResult => ({
  spec: s,
  chart_data,
  sampled: false,
  n_points: n,
  display_range: null,
});
const axis = (a: unknown) => (Array.isArray(a) ? a[0] : a) as { type?: string; data?: unknown[]; max?: number; name?: string };

function renderSvg(r: RenderResult): string {
  const chart = echarts.init(null, undefined, { renderer: "svg", ssr: true, width: 600, height: 400 });
  chart.setOption(buildOption(r));
  const svg = chart.renderToSVGString();
  chart.dispose();
  return svg;
}

// --- composition --------------------------------------------------------------

const stacked: StackedBarChartData = {
  categories: ["low", "high"],
  series: [
    { name: "online", values: [30, 10] },
    { name: "retail", values: [20, null] }, // (high, retail) never occurs
  ],
  totals: [50, 10],
  percent: false,
  truncated: false,
  y_label: "sum(revenue)",
};

const share: StackedBarChartData = {
  ...stacked,
  series: [
    { name: "online", values: [60, 100] },
    { name: "retail", values: [40, null] },
  ],
  percent: true,
  y_label: "share of sum(revenue) (%)",
};

describe("stacked_bar", () => {
  const r = result(spec({ type: "stacked_bar", x: "band", y: "revenue", group_by: "channel", aggregation: "sum", sort: "natural" }), stacked);

  it("stacks every part into one bar and keeps the backend's category order", () => {
    const option = buildOption(r);
    const s = option.series as BarSeriesOption[];
    expect(s.map((x) => x.type)).toEqual(["bar", "bar"]);
    expect(new Set(s.map((x) => x.stack)).size).toBe(1); // one stack id: a single bar
    expect(s.map((x) => x.name)).toEqual(["online", "retail"]);
    expect(axis(option.xAxis).data).toEqual(["low", "high"]);
    expect(axis(option.yAxis).name).toContain("sum(revenue)");
  });

  it("keeps a missing part as a gap, never a zero", () => {
    const s = buildOption(r).series as BarSeriesOption[];
    expect((s[1].data as (number | null)[])[1]).toBeNull();
  });

  it("recomputes nothing: the plotted numbers are exactly the payload's", () => {
    const s = buildOption(r).series as BarSeriesOption[];
    expect(s.map((x) => x.data)).toEqual([
      [30, 10],
      [20, null],
    ]);
  });

  it("a 100 % stack bounds the axis at 100 and labels it as a share", () => {
    const pct = result(spec({ type: "stacked_bar", x: "band", y: "revenue", group_by: "channel", aggregation: "sum", percent: true }), share);
    const option = buildOption(pct);
    expect(axis(option.yAxis).max).toBe(100);
    expect(axis(option.yAxis).name).toContain("share");
    const absolute = buildOption(r);
    expect(axis(absolute.yAxis).max).toBeUndefined();
  });

  it("says the bars are in their category order, not ranked", () => {
    expect(displayOrderNote(r)).toContain("類別順序");
  });

  it("renders server-side", () => {
    expect(renderSvg(r)).toContain("<svg");
  });
});

// --- flow ---------------------------------------------------------------------

const node = (name: string, label: string, column: string, side: "source" | "target", value: number) => ({
  name,
  label,
  column,
  side,
  value,
});
const flow: SankeyChartData = {
  nodes: [
    node("0:alpha", "alpha", "from_hub", "source", 30),
    node("0:bravo", "bravo", "from_hub", "source", 20),
    node("1:alpha", "alpha", "to_hub", "target", 20),
    node("1:bravo", "bravo", "to_hub", "target", 30),
  ],
  links: [
    { source: "0:alpha", target: "1:bravo", value: 30 },
    { source: "0:bravo", target: "1:alpha", value: 20 },
  ],
  truncated: false,
  n_links_total: 2,
  self_loops: 7,
  y_label: "count",
};

describe("sankey", () => {
  const r = result(spec({ type: "sankey", x: "from_hub", group_by: "to_hub", aggregation: "count" }), flow, 2);

  it("maps nodes and links straight onto one sankey series", () => {
    const s = (buildOption(r).series as SankeySeriesOption[])[0];
    expect(s.type).toBe("sankey");
    expect(s.links).toEqual([
      { source: "0:alpha", target: "1:bravo", value: 30 },
      { source: "0:bravo", target: "1:alpha", value: 20 },
    ]);
    expect((s.data ?? []).map((n) => (n as { name: string }).name)).toEqual(["0:alpha", "0:bravo", "1:alpha", "1:bravo"]);
  });

  it("shows the column value as the node label, not the side-prefixed identity", () => {
    const s = (buildOption(r).series as SankeySeriesOption[])[0];
    const labels = (s.data ?? []).map((n) => {
      const formatter = (n as { label?: { formatter?: unknown } }).label?.formatter;
      return typeof formatter === "function" ? (formatter as () => string)() : undefined;
    });
    expect(labels).toEqual(["alpha", "bravo", "alpha", "bravo"]);
  });

  it("gives one label the same colour on both ends", () => {
    const s = (buildOption(r).series as SankeySeriesOption[])[0];
    const color = (i: number) => ((s.data ?? [])[i] as { itemStyle?: { color?: string } }).itemStyle?.color;
    expect(color(0)).toBe(color(2)); // alpha as a source and as a target
    expect(color(0)).not.toBe(color(1));
  });

  it("has no category axis and no grid to draw on", () => {
    const option = buildOption(r);
    expect(option.xAxis).toBeUndefined();
    expect(option.grid).toBeUndefined();
    expect(displayOrderNote(r)).toBeNull();
  });

  it("renders server-side", () => {
    expect(renderSvg(r)).toContain("<svg");
  });

  it("an empty flow is the shared empty state, not a blank chart", () => {
    const empty: SankeyChartData = { nodes: [], links: [], truncated: false, n_links_total: 0, self_loops: 30, y_label: "count" };
    const r0 = result(spec({ type: "sankey", x: "from_hub", group_by: "to_hub", aggregation: "count" }), empty, 0);
    expect(r0.n_points).toBe(0); // ChartView shows the empty state for this
    const s = (buildOption(r0).series as SankeySeriesOption[])[0];
    expect(s.data).toEqual([]);
    expect(s.links).toEqual([]);
  });
});

// --- real-backend fixtures (e2e/capture-fixtures.mjs on e2e/data/flows.csv) ---

describe("recorded stage-20b fixtures", () => {
  const mix = stackedBarMix as unknown as RenderResult;
  const share = stackedBarPercent as unknown as RenderResult;
  const flowCount = sankeyHubFlow as unknown as RenderResult;
  const flowSum = sankeyHubRevenue as unknown as RenderResult;

  it("the absolute stack plots the recorded parts of every region", () => {
    const d = mix.chart_data as StackedBarChartData;
    const s = buildOption(mix).series as BarSeriesOption[];
    expect(s).toHaveLength(d.series.length);
    expect(s.every((x) => x.stack === s[0].stack)).toBe(true);
    expect(s[0].data).toEqual(d.series[0].values);
    expect(axis(buildOption(mix).xAxis).data).toEqual(d.categories.map(String));
  });

  it("the 100 % stack is bounded at 100 and keeps the absolute totals", () => {
    const d = share.chart_data as StackedBarChartData;
    expect(d.percent).toBe(true);
    expect(axis(buildOption(share).yAxis).max).toBe(100);
    expect(d.totals.every((t) => (t ?? 0) > 100)).toBe(true); // real revenue, not a share
  });

  it("both flows build one sankey series over the recorded nodes", () => {
    for (const r of [flowCount, flowSum]) {
      const d = r.chart_data as SankeyChartData;
      const s = (buildOption(r).series as SankeySeriesOption[])[0];
      expect(s.type).toBe("sankey");
      expect(s.data).toHaveLength(d.nodes.length);
      expect(s.links).toHaveLength(d.links.length);
      expect(d.self_loops).toBeGreaterThan(0);
    }
  });

  it("every recorded fixture renders server-side", () => {
    for (const r of [mix, share, flowCount, flowSum]) {
      expect(renderSvg(r)).toContain("<svg");
    }
  });
});
