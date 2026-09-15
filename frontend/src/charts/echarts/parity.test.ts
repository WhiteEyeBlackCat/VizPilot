// Stage 11 migration guard: for every recorded RenderResult the numbers the
// ECharts adapter hands to the chart are exactly the numbers the Plotly
// adapter (chartTraces.ts) hands to Plotly. Deleted together with Plotly in
// stage 12.

import { describe, expect, it } from "vitest";

import type { BarSeriesOption, BoxplotSeriesOption, CustomSeriesOption, HeatmapSeriesOption, LineSeriesOption, ScatterSeriesOption } from "echarts/charts";

import { buildPlot } from "../../chartTraces";
import type { RenderResult } from "../../types";
import index from "../__fixtures__/index.json";
import { buildOption } from "./option";

const fixtures = import.meta.glob("../__fixtures__/*.json", { eager: true, import: "default" }) as Record<
  string,
  unknown
>;

const load = (name: string) => fixtures[`../__fixtures__/${name}.json`] as RenderResult;
const seriesOf = (r: RenderResult) => {
  const s = buildOption(r).series;
  return (Array.isArray(s) ? s : [s]) as unknown[];
};

describe("Plotly / ECharts parity on recorded fixtures", () => {
  for (const name of Object.keys(index)) {
    it(name, () => {
      const r = load(name);
      const plot = buildPlot(r);
      const ech = seriesOf(r);
      switch (r.spec.type) {
        case "line":
        case "scatter": {
          expect(ech.length).toBe(plot.data.length);
          plot.data.forEach((t, i) => {
            const s = ech[i] as LineSeriesOption | ScatterSeriesOption;
            expect(s.name).toBe(t.name);
            expect((s.data as unknown[][]).map((p) => p[0])).toEqual(t.x);
            expect((s.data as unknown[][]).map((p) => p[1])).toEqual(t.y);
          });
          break;
        }
        case "bar": {
          plot.data.forEach((t, i) => {
            const s = ech[i] as BarSeriesOption;
            expect(s.name).toBe(t.name);
            expect(s.data).toEqual(t.y);
          });
          expect((buildOption(r).xAxis as { data: string[] }).data).toEqual((plot.data[0].x as unknown[]).map(String));
          break;
        }
        case "histogram": {
          plot.data.forEach((t, i) => {
            const s = ech[i] as CustomSeriesOption;
            const bins = s.data as [number, number, number][];
            expect(bins.map((b) => b[2])).toEqual(t.y); // counts
            expect(bins.map((b) => (b[0] + b[1]) / 2)).toEqual(t.x); // centres
            expect(bins.map((b) => b[1] - b[0])).toEqual(t.width); // widths
          });
          break;
        }
        case "box": {
          const t = plot.data[0];
          const s = ech[0] as BoxplotSeriesOption;
          const rows = s.data as number[][];
          expect(rows.map((x) => x[0])).toEqual(t.lowerfence);
          expect(rows.map((x) => x[1])).toEqual(t.q1);
          expect(rows.map((x) => x[2])).toEqual(t.median);
          expect(rows.map((x) => x[3])).toEqual(t.q3);
          expect(rows.map((x) => x[4])).toEqual(t.upperfence);
          const out = ech.find((x) => (x as { name?: string }).name === "outliers") as ScatterSeriesOption | undefined;
          if (plot.data[1]) {
            expect((out!.data as unknown[][]).map((p) => p[0])).toEqual(plot.data[1].x);
            expect((out!.data as unknown[][]).map((p) => p[1])).toEqual(plot.data[1].y);
          } else {
            expect(out).toBeUndefined();
          }
          break;
        }
        case "heatmap": {
          const z = plot.data[0].z as (number | null)[][];
          const cells = (ech[0] as HeatmapSeriesOption).data as [number, number, number][];
          const nonNull = z.flat().filter((v) => v !== null).length;
          expect(cells.length).toBe(nonNull);
          for (const [j, i, v] of cells) expect(z[i][j]).toBe(v);
          break;
        }
      }
    });
  }
});
