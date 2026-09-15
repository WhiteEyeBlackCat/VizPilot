// Renders the adapter's options through ECharts itself (server-side SVG,
// no DOM) so the tests cover what actually gets drawn, not only the option
// object: bar counts, axis labels, heatmap cell counts, histogram rectangles.

import { describe, expect, it } from "vitest";

import barGroupedNull from "../__fixtures__/bar_grouped_null.json";
import barSingle from "../__fixtures__/bar_single.json";
import boxOutliers from "../__fixtures__/box_outliers.json";
import heatmapNullCell from "../__fixtures__/heatmap_null_cell.json";
import histogramDisplayRange from "../__fixtures__/histogram_display_range.json";
import histogramGrouped from "../__fixtures__/histogram_grouped.json";
import lineGrouped from "../__fixtures__/line_grouped.json";
import type { BarChartData, BoxChartData, HistogramChartData, RenderResult } from "../../types";
import { echarts } from "./echarts";
import { buildOption } from "./option";

const fx = (json: unknown) => json as RenderResult;

function renderSvg(result: RenderResult): string {
  const chart = echarts.init(null, null, { renderer: "svg", ssr: true, width: 800, height: 400 });
  chart.setOption({ ...buildOption(result), animation: false });
  const svg = chart.renderToSVGString();
  chart.dispose();
  return svg;
}

const count = (svg: string, re: RegExp) => (svg.match(re) ?? []).length;
const hasText = (svg: string, text: string) => svg.includes(`>${text}<`) || svg.includes(text);
// data marks are tagged ecmeta_ssr_type="chart" (legend icons: "legend"),
// which isolates what was drawn for the data from the chrome around it
const marks = (svg: string, ...needles: string[]) =>
  (svg.match(/<path[^>]*>/g) ?? []).filter(
    (tag) => tag.includes('ecmeta_ssr_type="chart"') && needles.every((n) => tag.includes(n)),
  );
// polylines carry no ecmeta: identify them by stroke colour + width
const polylines = (svg: string, color: string) =>
  (svg.match(/<path[^>]*>/g) ?? []).filter(
    (tag) => tag.includes('fill="none"') && tag.includes(`stroke="${color}"`) && tag.includes('stroke-width="1.5"'),
  );

describe("SSR rendering", () => {
  it("bar: one rect per category, labels in backend order", () => {
    const r = fx(barSingle);
    const d = r.chart_data as BarChartData;
    const svg = renderSvg(r);
    expect(svg.startsWith("<svg")).toBe(true);
    // bars are <path> elements filled with the first palette colour
    expect(marks(svg, 'fill="#2563eb"').length).toBe(d.categories.length);
    for (const c of d.categories) expect(hasText(svg, String(c))).toBe(true);
    expect(hasText(svg, d.y_label)).toBe(true);
    expect(hasText(svg, r.spec.title)).toBe(true);
    // labels appear left-to-right in the recorded order
    const positions = d.categories.map((c) => svg.indexOf(`>${c}<`));
    expect([...positions].sort((a, b) => a - b)).toEqual(positions);
  });

  it("bar grouped with a null cell draws one rect fewer than the grid", () => {
    const r = fx(barGroupedNull);
    const d = r.chart_data as BarChartData;
    const svg = renderSvg(r);
    const drawn = marks(svg, 'fill="#2563eb"').length + marks(svg, 'fill="#f59e0b"').length;
    const nonNull = d.series.reduce((acc, s) => acc + s.values.filter((v) => v !== null).length, 0);
    expect(nonNull).toBe(d.categories.length * d.series.length - 1);
    expect(drawn).toBe(nonNull);
  });

  it("histogram: one rectangle per bin; clipped window drawn, nothing outside", () => {
    const r = fx(histogramDisplayRange);
    const d = r.chart_data as HistogramChartData;
    const svg = renderSvg(r);
    expect(marks(svg, 'fill="#2563eb"').length).toBe(d.bins.counts.length);
    expect(hasText(svg, "count")).toBe(true);
  });

  it("histogram grouped: every group's bins drawn translucent", () => {
    const r = fx(histogramGrouped);
    const d = r.chart_data as HistogramChartData;
    const svg = renderSvg(r);
    expect(marks(svg, 'fill-opacity="0.6"').length).toBe(d.series!.reduce((a, s) => a + s.counts.length, 0));
    for (const s of d.series!) expect(hasText(svg, s.name)).toBe(true); // legend
  });

  it("box: three boxes, red outlier dots, category labels", () => {
    const r = fx(boxOutliers);
    const d = r.chart_data as BoxChartData;
    const svg = renderSvg(r);
    expect(marks(svg, 'fill="#dbeafe"').length).toBe(d.groups.length);
    const outliers = d.groups.reduce((a, g) => a + g.outliers.length, 0);
    expect(marks(svg, 'fill="#ef4444"').length).toBe(outliers);
    for (const g of d.groups) expect(hasText(svg, g.name)).toBe(true);
  });

  it("heatmap: a cell per non-null correlation, value labels, column names on both axes", () => {
    const r = fx(heatmapNullCell);
    const svg = renderSvg(r);
    // 3x3 minus the two null cells; the diagonal is labelled 1.00
    expect(count(svg, />1\.00</g)).toBe(3);
    expect(marks(svg, 'stroke="#fff').length).toBe(7);
    for (const c of ["value", "metric_a", "metric_b"]) expect(count(svg, new RegExp(`>${c}<`, "g"))).toBeGreaterThanOrEqual(2);
  });

  it("line grouped: three polylines and a legend with the group names", () => {
    const r = fx(lineGrouped);
    const svg = renderSvg(r);
    for (const c of ["#2563eb", "#f59e0b", "#10b981"]) {
      expect(polylines(svg, c).length).toBe(1);
      expect(marks(svg, `stroke="${c}"`, 'fill="#fff"').length).toBe(30); // one symbol per day
    }
    for (const name of ["A", "B", "C"]) expect(hasText(svg, name)).toBe(true);
  });
});
