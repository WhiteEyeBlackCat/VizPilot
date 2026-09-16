// Grouped temporal lines must not join a group's points across a stretch of
// time in which the group has no observations (hotfix: `temp over dteday by
// season` drew the last 2011 summer point straight to the first 2012 one).
// The break is an invisible null point that `connectNulls: false` renders as
// a gap. line_grouped_gap.json is a real-backend snapshot of
// e2e/data/seasonal_groups.csv (month buckets, four seasons).

import { describe, expect, it } from "vitest";

import type { LineSeriesOption } from "echarts/charts";

import lineGrouped from "../__fixtures__/line_grouped.json";
import lineGroupedGap from "../__fixtures__/line_grouped_gap.json";
import type { LineChartData, RenderResult, XYSeries } from "../../types";
import { buildOption, GAP_BREAK_FACTOR } from "./option";

const fx = (json: unknown) => json as RenderResult;

const line = (series: XYSeries[], spec: Partial<RenderResult["spec"]> = {}): RenderResult => ({
  spec: { title: "t", type: "line", x: "when", y: "v", group_by: "g", ...spec },
  chart_data: { series, y_label: "mean(v)" } satisfies LineChartData,
  sampled: false,
  n_points: series.reduce((n, s) => n + s.x.length, 0),
  display_range: null,
});

const seriesOf = (r: RenderResult): LineSeriesOption[] => {
  const s = buildOption(r).series;
  return (Array.isArray(s) ? s : [s]) as LineSeriesOption[];
};
type Pt = [string | number, number | null];
const points = (s: LineSeriesOption) => s.data as Pt[];
/** contiguous runs of non-null points, as x values */
const segments = (s: LineSeriesOption): (string | number)[][] => {
  const out: (string | number)[][] = [];
  let cur: (string | number)[] = [];
  for (const [x, y] of points(s)) {
    if (y === null) {
      if (cur.length) out.push(cur);
      cur = [];
    } else cur.push(x);
  }
  if (cur.length) out.push(cur);
  return out;
};
const months = (ym: string[]) => ym.map((m) => `${m}-01`);
const monthly = (name: string, ym: string[]): XYSeries => ({ name, x: months(ym), y: ym.map((_, i) => i + 1) });

describe("grouped temporal line: breaks at time discontinuities", () => {
  it("case 1 (real backend fixture): a season absent for most of the year is drawn as separate segments", () => {
    const r = fx(lineGroupedGap);
    const d = r.chart_data as LineChartData;
    const s = seriesOf(r);
    const byName = Object.fromEntries(s.map((x) => [x.name as string, x]));
    // s3: 2011 Jun–Aug, then 2012 Jun–Jul — no line from 2011-08 to 2012-06
    expect(segments(byName.s3)).toEqual([months(["2011-06", "2011-07", "2011-08"]), months(["2012-06", "2012-07"])]);
    // s4: one May 2011 point, then Aug–Dec 2012
    expect(segments(byName.s4)).toEqual([["2011-05-01"], months(["2012-08", "2012-09", "2012-10", "2012-11", "2012-12"])]);
    // the break is an in-gap null point, never connected
    for (const x of s) expect(x.connectNulls).toBe(false);
    const hole = points(byName.s3).find(([, y]) => y === null)!;
    expect(hole).toBeDefined();
    expect(Number(hole[0])).toBeGreaterThan(Date.parse("2011-08-01"));
    expect(Number(hole[0])).toBeLessThan(Date.parse("2012-06-01"));
    // every backend point survives, in order, with its own value
    for (const orig of d.series) {
      const kept = points(byName[orig.name]).filter(([, y]) => y !== null);
      expect(kept).toEqual(orig.x.map((x, i) => [x, orig.y[i]]));
    }
  });

  it("case 2: a continuous grouped series (Jan–Apr) gets no spurious break", () => {
    const r = fx(lineGroupedGap);
    const byName = Object.fromEntries(seriesOf(r).map((x) => [x.name as string, x]));
    expect(segments(byName.s1)).toEqual([months(["2011-01", "2011-02", "2011-03", "2011-04"])]);
    expect(segments(byName.s2)).toHaveLength(1);
    // the pre-existing grouped fixture (three stations, 30 consecutive days) is untouched
    const g = fx(lineGrouped);
    for (const [i, s] of seriesOf(g).entries()) {
      expect(points(s).map((p) => p[1])).toEqual((g.chart_data as LineChartData).series[i].y);
    }
  });

  it("case 3: groups are judged independently — a gap in A does not break B", () => {
    const a = monthly("A", ["2011-06", "2011-07", "2011-08", "2012-06", "2012-07"]);
    const b = monthly("B", ["2011-01", "2011-02", "2011-03", "2011-04", "2011-05", "2011-06", "2011-07"]);
    const [sa, sb] = seriesOf(line([a, b], { time_granularity: "month" }));
    expect(segments(sa)).toHaveLength(2);
    expect(segments(sb)).toHaveLength(1);
    // raw (unbucketed) timestamps: the normal interval is the median gap
    const [ra, rb] = seriesOf(line([a, b], { time_granularity: null }));
    expect(segments(ra)).toHaveLength(2);
    expect(segments(rb)).toHaveLength(1);
  });

  it("does not break on a single mildly irregular step (one missing bucket)", () => {
    const a = monthly("A", ["2011-01", "2011-02", "2011-04", "2011-05", "2011-06"]);
    expect(segments(seriesOf(line([a], { time_granularity: "month" }))[0])).toHaveLength(1);
    expect(segments(seriesOf(line([a], { time_granularity: null }))[0])).toHaveLength(1);
    // daily buckets: a weekend (3-day step) is not a discontinuity
    const days = ["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10", "2025-01-13", "2025-01-14"];
    const d: XYSeries = { name: "A", x: days, y: days.map((_, i) => i) };
    expect(segments(seriesOf(line([d], { time_granularity: "day" }))[0])).toHaveLength(1);
  });

  it("leaves a legitimately irregular raw series alone", () => {
    // event-like inter-arrival times (steps 1–20 h, no dominant interval):
    // the regularity guard rejects the median as a "normal" interval
    const hours = [0, 1, 8, 10, 22, 26, 27, 36, 39, 59, 64, 66, 74, 75];
    const x = hours.map((h) => new Date(Date.UTC(2025, 0, 1, h)).toISOString().slice(0, 19));
    const irregular: XYSeries = { name: "A", x, y: hours.map((h) => h % 3) };
    const [s] = seriesOf(line([irregular], { time_granularity: null }));
    expect(segments(s)).toHaveLength(1);
    expect(points(s)).toHaveLength(hours.length);
  });

  it("ungrouped temporal lines are unchanged even across a long gap", () => {
    const a = monthly("v", ["2011-06", "2011-07", "2011-08", "2012-06", "2012-07"]);
    const [s] = seriesOf(line([a], { group_by: null, time_granularity: "month" }));
    expect(points(s)).toEqual(a.x.map((x, i) => [x, a.y[i]]));
  });

  it("threshold is a small multiple of the normal interval", () => {
    expect(GAP_BREAK_FACTOR).toBeGreaterThanOrEqual(2);
    expect(GAP_BREAK_FACTOR).toBeLessThanOrEqual(5);
  });
});
