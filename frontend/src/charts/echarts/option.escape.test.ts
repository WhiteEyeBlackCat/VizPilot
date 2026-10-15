// A3: ECharts injects the string returned by a custom tooltip formatter as
// HTML. Every chart type with such a formatter must escape the user-controlled
// text (column names, categories, group / series names, node labels).

import { describe, expect, it } from "vitest";

import type { ChartData, ChartSpec, RenderResult } from "../../types";
import { escapeHtml } from "./escape";
import { buildOption } from "./option";

const E1 = "<img src=x onerror=alert(1)>";
const E2 = '"><script>alert(2)</script>';

const spec = (type: ChartSpec["type"]): ChartSpec => ({
  title: E1,
  x: E1,
  y: E2,
  group_by: E2,
  aggregation: null,
  bins: null,
  time_granularity: null,
  top_n: null,
  sort: null,
  percent: null,
  reason: "",
  priority: 1,
  type,
});
const res = (type: ChartSpec["type"], chart_data: ChartData): RenderResult => ({
  spec: spec(type),
  chart_data,
  sampled: false,
  n_points: 10,
  display_range: null,
});

const box = (name: string) => ({
  name,
  q1: 1,
  median: 2,
  q3: 3,
  lower_fence: 0,
  upper_fence: 4,
  outliers: [9],
  mean: 2.5,
  count: 5,
});
const dens = (name: string) => ({ ...box(name), density: [0.2, 1, 0.2], bandwidth: 1, points: [1, 2] });

const cases: { type: ChartSpec["type"]; data: ChartData; mustEscape: boolean }[] = [
  { type: "line", data: { series: [{ name: E1, x: [1, 2], y: [1, 2] }], y_label: E2 }, mustEscape: false },
  { type: "scatter", data: { series: [{ name: E1, x: [1, 2], y: [1, 2] }, { name: E2, x: [3], y: [4] }], y_label: E2 } as ChartData, mustEscape: true },
  { type: "bar", data: { categories: [E1, E2], series: [{ name: E2, values: [1, 2] }], truncated: false, y_label: E2 }, mustEscape: false },
  {
    type: "histogram",
    data: {
      bins: { edges: [0, 1, 2], counts: [1, 2] },
      series: [
        { name: E1, edges: [0, 1, 2], counts: [1, 2] },
        { name: E2, edges: [0, 1, 2], counts: [2, 1] },
      ],
      y_label: E2,
    },
    mustEscape: true,
  },
  { type: "box", data: { groups: [box(E1), box(E2)], y_label: E2 }, mustEscape: true },
  { type: "heatmap", data: { columns: [E1, E2], rows: [E2, E1], matrix: [[1, 2], [3, 4]], y_label: E2 }, mustEscape: true },
  { type: "heatmap", data: { columns: [E1, E2], matrix: [[1, 0.5], [0.5, 1]], y_label: E2 }, mustEscape: true },
  { type: "violin", data: { groups: [dens(E1), dens(E2)], grid: [0, 1, 2], y_label: E2 }, mustEscape: true },
  { type: "raincloud", data: { groups: [dens(E1), dens(E2)], grid: [0, 1, 2], y_label: E2 }, mustEscape: true },
  { type: "ridgeline", data: { groups: [dens(E1), dens(E2)], grid: [0, 1, 2], y_label: E2 }, mustEscape: true },
  {
    type: "dot_interval",
    data: { groups: [{ name: E1, mean: 2, lo: 1, hi: 3, n: 4 }, { name: E2, mean: 3, lo: 2, hi: 4, n: 4 }], y_label: E2 },
    mustEscape: true,
  },
  { type: "lollipop", data: { categories: [E1, E2], series: [{ name: E2, values: [1, 2] }], truncated: false, y_label: E2 }, mustEscape: false },
  {
    type: "dumbbell",
    data: { categories: [E1, E2], states: [E1, E2], series: [{ name: E1, values: [1, 2] }, { name: E2, values: [2, 3] }], y_label: E2 },
    mustEscape: true,
  },
  {
    type: "slopegraph",
    data: { categories: [E1, E2], states: [E1, E2], series: [{ name: E1, values: [1, 2] }, { name: E2, values: [2, 3] }], y_label: E2 },
    mustEscape: true,
  },
  {
    type: "hexbin",
    data: { cells: [[0, 0, 3], [1, 1, 5]], x_range: [0, 10], y_range: [0, 10], dx: 1, dy: 1, columns: 10, rows: 10, n: 8, y_label: E2 },
    mustEscape: true,
  },
  { type: "polar_line", data: { categories: [E1, E2], series: [{ name: E2, values: [1, 2] }], truncated: false, y_label: E2 }, mustEscape: false },
  { type: "rose", data: { categories: [E1, E2], series: [{ name: E2, values: [1, 2] }], truncated: false, y_label: E2 }, mustEscape: false },
  {
    type: "stacked_bar",
    data: {
      categories: [E1, E2],
      series: [{ name: E1, values: [1, 2] }, { name: E2, values: [3, null] }],
      totals: [4, 2],
      percent: false,
      truncated: false,
      y_label: E2,
    },
    mustEscape: true,
  },
  {
    type: "sankey",
    data: {
      nodes: [
        { name: "0:a", label: E1, column: E2, side: "source", value: 3 },
        { name: "1:b", label: E2, column: E1, side: "target", value: 3 },
      ],
      links: [{ source: "0:a", target: "1:b", value: 3 }],
      truncated: false,
      n_links_total: 1,
      self_loops: 0,
      y_label: E2,
    },
    mustEscape: true,
  },
];

type Fmt = (p: unknown) => unknown;
type AnyObj = Record<string, unknown>;

/** every custom tooltip formatter of an option, option-level and series-level */
function collect(option: AnyObj): { where: string; fn: Fmt }[] {
  const out: { where: string; fn: Fmt }[] = [];
  const tt = option.tooltip as AnyObj | undefined;
  if (tt && typeof tt.formatter === "function") out.push({ where: "option", fn: tt.formatter as Fmt });
  ((option.series as AnyObj[]) ?? []).forEach((s, i) => {
    const st = s.tooltip as AnyObj | undefined;
    if (st && typeof st.formatter === "function") out.push({ where: `series[${i}]`, fn: st.formatter as Fmt });
  });
  return out;
}

/** representative ECharts params for each series datum, item and axis style */
function paramSets(option: AnyObj): unknown[] {
  const sets: unknown[] = [];
  const series = (option.series as AnyObj[]) ?? [];
  const axisItems: AnyObj[] = [];
  series.forEach((s, seriesIndex) => {
    const data = (s.data as unknown[]) ?? [];
    if (s.type === "sankey") {
      const links = s.links as AnyObj[];
      links.forEach((l) => sets.push({ dataType: "edge", data: l, seriesName: s.name, seriesType: "sankey", marker: "<span>m</span>" }));
      data.forEach((n, dataIndex) =>
        sets.push({ dataType: "node", name: (n as AnyObj).name, data: n, dataIndex, seriesName: s.name, seriesType: "sankey", marker: "<span>m</span>" }),
      );
      return;
    }
    data.slice(0, 3).forEach((datum, dataIndex) => {
      const value = datum && typeof datum === "object" && !Array.isArray(datum) ? (datum as AnyObj).value : datum;
      const p = {
        componentType: "series",
        seriesType: s.type,
        seriesIndex,
        seriesName: s.name,
        name: Array.isArray(value) ? value[0] : undefined,
        dataIndex,
        data: datum,
        value,
        marker: "<span>m</span>",
        dataType: "node",
      };
      sets.push(p, [p]);
      axisItems[dataIndex] = axisItems[dataIndex] ?? { items: [] };
      (axisItems[dataIndex].items as unknown[]).push(p);
    });
  });
  axisItems.forEach((a) => sets.push(a.items));
  return sets;
}

describe("A3 tooltip html is escaped", () => {
  it("escapeHtml escapes the five html characters", () => {
    expect(escapeHtml(`<a href="x" t='1'>&`)).toBe("&lt;a href=&quot;x&quot; t=&#39;1&#39;&gt;&amp;");
    expect(escapeHtml(null)).toBe("");
  });

  for (const c of cases) {
    it(`${c.type}: no formatter emits raw user html`, () => {
      const option = buildOption(res(c.type, c.data)) as unknown as AnyObj;
      const fns = collect(option);
      const outputs: string[] = [];
      for (const { where, fn } of fns) {
        for (const params of paramSets(option)) {
          let out: unknown;
          try {
            out = fn(params);
          } catch {
            continue; // params of another series' shape: not this formatter's input
          }
          if (typeof out !== "string") continue;
          outputs.push(out);
          expect(out, `${c.type} ${where}`).not.toMatch(/<img/i);
          expect(out, `${c.type} ${where}`).not.toMatch(/<script/i);
        }
      }
      if (c.mustEscape) {
        expect(fns.length).toBeGreaterThan(0);
        expect(outputs.some((o) => o.includes("&lt;img"))).toBe(true);
        expect(outputs.some((o) => o.includes("&lt;script") || o.includes("&quot;&gt;&lt;script"))).toBe(true);
      }
    });
  }
});
