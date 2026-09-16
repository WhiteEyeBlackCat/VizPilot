// Stage 19.1 regression fixtures recorded from the real backend on the
// committed e2e/data inputs (e2e/capture-fixtures.mjs --only …): same mean /
// different dispersion (raincloud, box), an ordered category (ridgeline,
// natural-order bar and line), hour / month cycles (polar line, rose — the
// Explore path — line over a cycle) and a dense XY hexbin whose cells must
// tessellate at any axis aspect.

import { describe, expect, it } from "vitest";

import type { BarSeriesOption, CustomSeriesOption, LineSeriesOption, ScatterSeriesOption } from "echarts/charts";

import barOrderedNatural from "../__fixtures__/bar_ordered_natural.json";
import boxSameMean from "../__fixtures__/box_same_mean.json";
import hexbinDense from "../__fixtures__/hexbin_dense.json";
import lineCyclicMonth from "../__fixtures__/line_cyclic_month.json";
import lineOrderedCategory from "../__fixtures__/line_ordered_category.json";
import polarHour from "../__fixtures__/polar_hour.json";
import raincloudSpread from "../__fixtures__/raincloud_spread.json";
import ridgelineOrdered from "../__fixtures__/ridgeline_ordered.json";
import roseHourSum from "../__fixtures__/rose_hour_sum.json";
import roseMonthCount from "../__fixtures__/rose_month_count.json";
import type { BarChartData, BoxChartData, DensityChartData, HexbinChartData, RenderResult } from "../../types";
import { echarts } from "./echarts";
import { buildOption } from "./option";
import { displayOrderNote } from "./order";

const LEVELS = ["low", "medium", "high", "very high"];
const HOURS = Array.from({ length: 24 }, (_, i) => i);
const MONTHS = Array.from({ length: 12 }, (_, i) => i + 1);

const series = (r: RenderResult) => {
  const s = buildOption(r).series;
  return (Array.isArray(s) ? s : [s]) as (CustomSeriesOption | ScatterSeriesOption | BarSeriesOption | LineSeriesOption)[];
};
const axis = (a: unknown) => (Array.isArray(a) ? a[0] : a) as { type?: string; data?: unknown[] };

/** A recorded statistic the fixture must carry (box quartiles are null only for empty groups). */
function present(value: number | null, what: string): number {
  if (value === null) throw new Error(`fixture is missing ${what}`);
  return value;
}

function renderSvg(r: RenderResult, width = 640, height = 360): string {
  const chart = echarts.init(null, undefined, { renderer: "svg", ssr: true, width, height });
  chart.setOption(buildOption(r));
  const svg = chart.renderToSVGString();
  chart.dispose();
  return svg;
}

describe("same mean, different dispersion", () => {
  const rain = raincloudSpread as RenderResult;
  const data = rain.chart_data as DensityChartData;
  it("raincloud: three groups in backend order, a 64-point density each, the point sample drawn as is", () => {
    expect(data.groups.map((g) => g.name)).toEqual(["a", "b", "c"]);
    expect(data.grid).toHaveLength(64);
    const s = series(rain);
    const rainPoints = s.find((x) => x.type === "scatter") as ScatterSeriesOption;
    const expected = data.groups.reduce((n, g) => n + (g.points?.length ?? 0), 0);
    expect((rainPoints.data as number[][]).length).toBe(expected);
    expect(renderSvg(rain)).toContain("<polygon");
  });
  it("the medians agree while the boxes widen: the finding is the spread, not the level", () => {
    const box = (boxSameMean as RenderResult).chart_data as BoxChartData;
    const medians = box.groups.map((g) => present(g.median, `median of ${g.name}`));
    const iqr = box.groups.map((g) => present(g.q3, `q3 of ${g.name}`) - present(g.q1, `q1 of ${g.name}`));
    expect(Math.max(...medians) - Math.min(...medians)).toBeLessThan(3);
    expect(iqr[2] / iqr[0]).toBeGreaterThan(3);
    expect(axis(buildOption(boxSameMean as RenderResult).xAxis).data).toEqual(["a", "b", "c"]);
  });
});

describe("ordered category", () => {
  it("ridgeline rows follow the scale low → very high", () => {
    const r = ridgelineOrdered as RenderResult;
    expect((r.chart_data as DensityChartData).groups.map((g) => g.name)).toEqual(LEVELS);
    expect(series(r)).toHaveLength(1);
    expect(renderSvg(r)).toContain("<polygon");
  });
  it("bar and line keep the natural order and say so", () => {
    const bar = barOrderedNatural as RenderResult;
    expect((bar.chart_data as BarChartData).categories).toEqual(LEVELS);
    expect(axis(buildOption(bar).xAxis).data).toEqual(LEVELS);
    expect(displayOrderNote(bar)).toBe("依 level 的類別順序顯示，未依值排序");
    const line = lineOrderedCategory as RenderResult;
    const o = buildOption(line);
    const s = series(line)[0] as LineSeriesOption;
    const xs = axis(o.xAxis).data ?? (s.data as unknown[][]).map((d) => d[0]);
    expect(xs.map(String)).toEqual(LEVELS);
  });
});

describe("cycles (polar line, rose, line over a cycle)", () => {
  it("polar line: 24 hours around the circle, a polar line series", () => {
    const o = buildOption(polarHour as RenderResult);
    expect(((o.angleAxis as { data?: unknown[] }).data ?? []).map(String)).toEqual(HOURS.map(String));
    const s = series(polarHour as RenderResult)[0] as LineSeriesOption;
    expect(s.type).toBe("line");
    expect(s.coordinateSystem).toBe("polar");
    expect(renderSvg(polarHour as RenderResult)).toContain("<path");
  });
  it("rose: sector size is the backend's sum, in hour order, commute peak above the small hours", () => {
    const r = roseHourSum as RenderResult;
    const o = buildOption(r);
    expect(((o.angleAxis as { data?: unknown[] }).data ?? []).map(String)).toEqual(HOURS.map(String));
    const s = series(r)[0] as BarSeriesOption;
    expect(s.type).toBe("bar");
    expect(s.coordinateSystem).toBe("polar");
    const values = s.data as number[];
    expect(values[8]).toBeGreaterThan(values[3]);
    expect(values).toEqual((r.chart_data as BarChartData).series[0].values);
  });
  it("rose of a count over months keeps 1..12", () => {
    const o = buildOption(roseMonthCount as RenderResult);
    expect(((o.angleAxis as { data?: unknown[] }).data ?? []).map(String)).toEqual(MONTHS.map(String));
  });
  it("line over a month cycle reads 1..12 in order", () => {
    const r = lineCyclicMonth as RenderResult;
    const o = buildOption(r);
    const s = series(r)[0] as LineSeriesOption;
    const xs = axis(o.xAxis).data ?? (s.data as unknown[][]).map((d) => d[0]);
    expect(xs.map(String)).toEqual(MONTHS.map(String));
  });
});

describe("hexbin", () => {
  it("one polygon per recorded cell, counts intact", () => {
    const r = hexbinDense as RenderResult;
    const d = r.chart_data as HexbinChartData;
    expect(d.cells.reduce((n, c) => n + c[2], 0)).toBe(2000);
    const s = series(r)[0] as CustomSeriesOption;
    expect((s.data as number[][]).length).toBe(d.cells.length);
    // every cell is a polygon (a couple more come from the toolbox / visual-map glyphs)
    expect((renderSvg(r).match(/<polygon/g) ?? []).length).toBeGreaterThanOrEqual(d.cells.length);
  });
  it("cells tessellate at an unequal axis aspect: neighbours share vertices (no hairline gaps)", () => {
    const d: HexbinChartData = {
      cells: [[0, 0, 1], [1, 0, 2], [0, 1, 3]],
      x_range: [0, 4],
      y_range: [0, 4],
      dx: 1,
      dy: 1,
      columns: 4,
      rows: 6,
      n: 6,
      y_label: "y",
    };
    const r: RenderResult = { ...(hexbinDense as RenderResult), chart_data: d };
    const s = series(r)[0] as CustomSeriesOption;
    const data = s.data as number[][];
    // x maps at 10 px per unit, y at 30 px per unit (and flips): the pixel
    // aspect is far from the backend's square assumption
    const api = (datum: number[]) => ({
      value: (i: number) => datum[i],
      coord: ([x, y]: number[]) => [x * 10, 300 - y * 30],
      visual: () => "#000",
    });
    const polygon = (datum: number[]) => {
      const item = (s.renderItem as (p: unknown, a: unknown) => { shape: { points: number[][] } })({}, api(datum));
      return item.shape.points.map(([x, y]) => [Math.round(x * 1000) / 1000, Math.round(y * 1000) / 1000]);
    };
    const [a, b, c] = data.map(polygon); // a=(0,0) b=(1,0) c=(0,1) (odd row: shifted half a cell)
    // pointy-top: [top, upper-right, lower-right, bottom, lower-left, upper-left]
    expect(a[1]).toEqual(b[5]); // a's upper-right is b's upper-left
    expect(a[2]).toEqual(b[4]); // a's lower-right is b's lower-left
    expect(a[0]).toEqual(c[4]); // a's top is c's lower-left
    expect(a[1]).toEqual(c[3]); // a's upper-right is c's bottom
    expect(b[0]).toEqual(c[2]); // b's top is c's lower-right
  });

  // Stage 19.2 (A9) regression: the recorded dense hexbin at three plot
  // aspects — wide, tall, very wide. The backend cells are untouched (the
  // fixture is re-rendered against the backend in test_fixture_regression);
  // here every cell polygon is computed at the pixel scale of each size and
  // the lattice must be seamless (each neighbour pair shares exactly two
  // vertices) and overlap-free (no vertex of one cell inside another).
  describe.each([
    [800, 400],
    [400, 800],
    [1200, 300],
  ])("stretched hexagons tessellate at %i × %i", (width, height) => {
    const r = hexbinDense as RenderResult;
    const d = r.chart_data as HexbinChartData;
    const s = series(r)[0] as CustomSeriesOption;
    const sx = width / (d.x_range[1] - d.x_range[0]);
    const sy = height / (d.y_range[1] - d.y_range[0]);
    const api = (datum: number[]) => ({
      value: (i: number) => datum[i],
      coord: ([x, y]: number[]) => [(x - d.x_range[0]) * sx, height - (y - d.y_range[0]) * sy],
      visual: () => "#000",
    });
    const render = s.renderItem as (p: unknown, a: unknown) => { shape: { points: number[][] } };
    const polygons = (s.data as number[][]).map((datum) =>
      render({}, api(datum)).shape.points.map(([x, y]) => [Math.round(x * 1e6) / 1e6, Math.round(y * 1e6) / 1e6]),
    );
    const same = (a: number[], b: number[]) => Math.abs(a[0] - b[0]) < 1e-6 && Math.abs(a[1] - b[1]) < 1e-6;
    const shared = (a: number[][], b: number[][]) => a.filter((v) => b.some((w) => same(v, w))).length;
    const inside = (pt: number[], poly: number[][]) => {
      // strict point-in-polygon (ray casting); points on an edge do not count
      let hit = false;
      for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
        const [xi, yi] = poly[i];
        const [xj, yj] = poly[j];
        const onEdge =
          Math.abs((xj - xi) * (pt[1] - yi) - (yj - yi) * (pt[0] - xi)) < 1e-6 &&
          pt[0] >= Math.min(xi, xj) - 1e-6 &&
          pt[0] <= Math.max(xi, xj) + 1e-6 &&
          pt[1] >= Math.min(yi, yj) - 1e-6 &&
          pt[1] <= Math.max(yi, yj) + 1e-6;
        if (onEdge) return false;
        if (yi > pt[1] !== yj > pt[1] && pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi) + xi) hit = !hit;
      }
      return hit;
    };
    it("every cell is a hexagon stretched to the pixel pitches, not a regular one", () => {
      expect(polygons).toHaveLength(d.cells.length);
      for (const poly of polygons) expect(poly).toHaveLength(6);
      const xs = polygons[0].map((p) => p[0]);
      const ys = polygons[0].map((p) => p[1]);
      // pointy-top hexagon of the pixel pitches: width = dx·sx (column pitch),
      // height = dy·sy (vertex to vertex; rows are spaced 3/4 of it) — the
      // two extents follow the two axes independently, hence "stretched"
      expect(Math.max(...xs) - Math.min(...xs)).toBeCloseTo(d.dx * sx, 3);
      expect(Math.max(...ys) - Math.min(...ys)).toBeCloseTo(d.dy * sy, 3);
      expect((Math.max(...xs) - Math.min(...xs)) / (Math.max(...ys) - Math.min(...ys))).not.toBeCloseTo(Math.sqrt(3) / 2, 2);
    });
    it("neighbours share exactly two vertices and no cell overlaps another", () => {
      const centre = (poly: number[][]) => [poly.reduce((a, p) => a + p[0], 0) / 6, poly.reduce((a, p) => a + p[1], 0) / 6];
      const centres = polygons.map(centre);
      const reach = 2 * Math.max(d.dx * sx, d.dy * sy); // farther apart than this, two cells cannot touch
      let neighbours = 0;
      const gaps: number[][] = []; // pairs touching at a single vertex (a hairline gap)
      const overlaps: number[][] = []; // pairs sharing 3+ vertices or a vertex inside the other
      for (let i = 0; i < polygons.length; i++) {
        for (let j = i + 1; j < polygons.length; j++) {
          if (Math.abs(centres[i][0] - centres[j][0]) > reach || Math.abs(centres[i][1] - centres[j][1]) > reach) continue;
          const n = shared(polygons[i], polygons[j]);
          if (n === 2) neighbours++;
          else if (n === 1) gaps.push([i, j]);
          else if (n > 2) overlaps.push([i, j]);
          if (polygons[i].some((v) => inside(v, polygons[j])) || polygons[j].some((v) => inside(v, polygons[i]))) overlaps.push([i, j]);
        }
      }
      expect(gaps).toEqual([]);
      expect(overlaps).toEqual([]);
      expect(neighbours).toBeGreaterThan(polygons.length); // a dense lattice, not isolated cells
    });
    it("renders one polygon per cell at this size", () => {
      expect((renderSvg(r, width, height).match(/<polygon/g) ?? []).length).toBeGreaterThanOrEqual(d.cells.length);
    });
  });
});
