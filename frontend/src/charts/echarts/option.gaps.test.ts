// Grouped temporal lines must not join a group's points across buckets of
// the chart in which the group has no observations (hotfix: `temp over
// dteday by season` drew the last 2011 summer point straight to the first
// 2012 one). Rule: the chart's buckets are the x values that exist in any
// series; a group breaks between two consecutive observations iff another
// existing bucket lies between them. Nothing is generated — a weekend,
// holiday or night that no group occupies is not a bucket, so lines stay
// connected across it. The break is an invisible null point that
// `connectNulls: false` renders as a gap. line_grouped_gap.json is a
// real-backend snapshot of e2e/data/seasonal_groups.csv (month buckets,
// four seasons).

import { describe, expect, it } from "vitest";

import type { LineSeriesOption } from "echarts/charts";

import lineGrouped from "../__fixtures__/line_grouped.json";
import lineGroupedGap from "../__fixtures__/line_grouped_gap.json";
import type { LineChartData, RenderResult, XYSeries } from "../../types";
import { breakAbsentBuckets, buildOption, temporalBuckets } from "./option";

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
const xy = (name: string, x: string[]): XYSeries => ({ name, x, y: x.map((_, i) => i + 1) });
const months = (ym: string[]) => ym.map((m) => `${m}-01`);
const monthly = (name: string, ym: string[]): XYSeries => xy(name, months(ym));
/** every backend (x, y) pair survives, in order, and each inserted null lies
 *  strictly between its two neighbours */
const expectPreserved = (r: RenderResult) => {
  const d = r.chart_data as LineChartData;
  const out = seriesOf(r);
  for (const [i, orig] of d.series.entries()) {
    const pts = points(out[i]);
    expect(pts.filter(([, y]) => y !== null)).toEqual(orig.x.map((x, k) => [x, orig.y[k]]));
    for (const [k, [x, y]] of pts.entries()) {
      if (y !== null) continue;
      expect(typeof x).toBe("number");
      expect(Number(x)).toBeGreaterThan(Date.parse(String(pts[k - 1][0])));
      expect(Number(x)).toBeLessThan(Date.parse(String(pts[k + 1][0])));
      expect(pts[k - 1][1]).not.toBeNull();
      expect(pts[k + 1][1]).not.toBeNull();
    }
  }
};

describe("grouped temporal line: breaks where a group is absent from an existing bucket", () => {
  it("case 1: seasonal absence — a group present Jun–Aug then next Jun breaks between Aug and Jun", () => {
    const all = ["2011-01", "2011-02", "2011-03", "2011-04", "2011-05", "2011-06", "2011-07", "2011-08", "2011-09", "2011-10", "2011-11", "2011-12", "2012-01", "2012-02", "2012-03", "2012-04", "2012-05", "2012-06", "2012-07"];
    const a = monthly("A", ["2011-06", "2011-07", "2011-08", "2012-06", "2012-07"]);
    const rest = monthly("rest", all.filter((m) => !["2011-06", "2011-07", "2011-08", "2012-06", "2012-07"].includes(m)));
    const r = line([a, rest], { time_granularity: "month" });
    const [sa, sr] = seriesOf(r);
    expect(segments(sa)).toEqual([months(["2011-06", "2011-07", "2011-08"]), months(["2012-06", "2012-07"])]);
    // the other group is absent from Jun–Aug 2011, which A occupies; it ends
    // in May 2012 so A's 2012 buckets lie beyond its last point
    expect(segments(sr)).toEqual([
      months(["2011-01", "2011-02", "2011-03", "2011-04", "2011-05"]),
      months(["2011-09", "2011-10", "2011-11", "2011-12", "2012-01", "2012-02", "2012-03", "2012-04", "2012-05"]),
    ]);
    expectPreserved(r);
  });

  it("case 2: a weekend absent from the whole chart is not a bucket — Fri→Mon stays connected", () => {
    const days = ["2025-01-09", "2025-01-10", "2025-01-13", "2025-01-14"]; // Thu Fri Mon Tue
    const r = line([xy("A", days), xy("B", days)], { time_granularity: "day" });
    for (const s of seriesOf(r)) {
      expect(segments(s)).toEqual([days]);
      expect(points(s)).toHaveLength(days.length);
    }
  });

  it("case 3: a holiday absent from the whole chart is not a bucket — Dec 24→27 stays connected", () => {
    const days = ["2024-12-23", "2024-12-24", "2024-12-27", "2024-12-28"];
    const r = line([xy("A", days), xy("B", days)], { time_granularity: "day" });
    for (const s of seriesOf(r)) expect(segments(s)).toEqual([days]);
  });

  it("case 4: overnight silence in hourly data across all groups keeps 17:00→09:00 connected", () => {
    const hours = ["2025-03-03T16:00:00", "2025-03-03T17:00:00", "2025-03-04T09:00:00", "2025-03-04T10:00:00"];
    const r = line([xy("A", hours), xy("B", hours)], { time_granularity: null });
    for (const s of seriesOf(r)) {
      expect(segments(s)).toEqual([hours]);
      expect(points(s)).toHaveLength(hours.length);
    }
  });

  it("case 5: a bucket another group occupies breaks the absent group only", () => {
    const hours = ["2025-03-03T09:00:00", "2025-03-03T10:00:00", "2025-03-03T11:00:00", "2025-03-03T12:00:00"];
    const a = xy("A", [hours[0], hours[1], hours[3]]);
    const b = xy("B", hours);
    const r = line([a, b], { time_granularity: null });
    const [sa, sb] = seriesOf(r);
    expect(segments(sa)).toEqual([[hours[0], hours[1]], [hours[3]]]);
    expect(segments(sb)).toEqual([hours]);
    const hole = points(sa).find(([, y]) => y === null)!;
    expect(Number(hole[0])).toBeGreaterThan(Date.parse(hours[1]));
    expect(Number(hole[0])).toBeLessThan(Date.parse(hours[3]));
    expectPreserved(r);
  });

  it("case 6: a fully continuous grouped series gets no nulls (line_grouped fixture unchanged)", () => {
    const g = fx(lineGrouped);
    expect(g.spec.group_by).toBeTruthy();
    for (const [i, s] of seriesOf(g).entries()) {
      const orig = (g.chart_data as LineChartData).series[i];
      expect(points(s)).toEqual(orig.x.map((x, k) => [x, orig.y[k]]));
    }
    // every group on every bucket, sparse global grid (irregular but shared)
    const x = ["2025-01-01", "2025-01-02", "2025-01-06", "2025-02-01", "2025-05-01"];
    for (const s of seriesOf(line([xy("A", x), xy("B", x)], { time_granularity: null }))) expect(points(s)).toHaveLength(x.length);
  });

  it("case 7: several groups with their own gaps are evaluated independently", () => {
    const x = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06"];
    const a = xy("A", [x[0], x[1], x[4], x[5]]); // absent 03, 04
    const b = xy("B", [x[0], x[2], x[3], x[5]]); // absent 02, 05
    const c = xy("C", [x[1], x[2], x[3], x[4]]); // present on its whole span
    const r = line([a, b, c], { time_granularity: "day" });
    const [sa, sb, sc] = seriesOf(r);
    expect(segments(sa)).toEqual([[x[0], x[1]], [x[4], x[5]]]);
    expect(segments(sb)).toEqual([[x[0]], [x[2], x[3]], [x[5]]]);
    expect(segments(sc)).toEqual([[x[1], x[2], x[3], x[4]]]);
    expectPreserved(r);
  });

  it("real backend fixture: cross-year season absence is not joined; within-year runs stay continuous", () => {
    const r = fx(lineGroupedGap);
    const s = seriesOf(r);
    const byName = Object.fromEntries(s.map((x) => [x.name as string, x]));
    // s3: 2011 Jun–Aug, then 2012 Jun–Jul — no line from 2011-08 to 2012-06
    expect(segments(byName.s3)).toEqual([months(["2011-06", "2011-07", "2011-08"]), months(["2012-06", "2012-07"])]);
    // s4: one May 2011 point, then Aug–Dec 2012
    expect(segments(byName.s4)).toEqual([["2011-05-01"], months(["2012-08", "2012-09", "2012-10", "2012-11", "2012-12"])]);
    // s1 / s2 continuous
    expect(segments(byName.s1)).toEqual([months(["2011-01", "2011-02", "2011-03", "2011-04"])]);
    expect(segments(byName.s2)).toHaveLength(1);
    for (const x of s) expect(x.connectNulls).toBe(false);
    expectPreserved(r);
  });

  it("dedupes the same bucket written differently by two groups", () => {
    const a = xy("A", ["2025-01-01", "2025-01-02", "2025-01-04"]);
    const b = xy("B", ["2025-01-01T00:00:00", "2025-01-02T00:00:00", "2025-01-03T00:00:00", "2025-01-04T00:00:00"]);
    const buckets = temporalBuckets({ series: [a, b], y_label: "v" });
    expect(buckets.size).toBe(4);
    expect(segments(seriesOf(line([a, b]))[0])).toEqual([["2025-01-01", "2025-01-02"], ["2025-01-04"]]);
    // a null / unparsable x takes no part and never breaks the line
    expect(breakAbsentBuckets(["2025-01-01", null, "2025-01-04"], [1, 2, 3], buckets)).toEqual([["2025-01-01", 1], ["", 2], ["2025-01-04", 3]]);
  });

  it("ungrouped temporal lines are unchanged even across a long gap", () => {
    const a = monthly("v", ["2011-06", "2011-07", "2011-08", "2012-06", "2012-07"]);
    const [s] = seriesOf(line([a], { group_by: null, time_granularity: "month" }));
    expect(points(s)).toEqual(a.x.map((x, i) => [x, a.y[i]]));
  });

  it("numeric-x and category-x grouped lines are untouched", () => {
    const num: XYSeries[] = [
      { name: "A", x: [1, 2, 6], y: [1, 2, 3] },
      { name: "B", x: [1, 2, 3, 4, 5, 6], y: [1, 2, 3, 4, 5, 6] },
    ];
    for (const [i, s] of seriesOf(line(num)).entries()) expect(points(s)).toEqual(num[i].x.map((x, k) => [x, num[i].y[k]]));
    const cat: XYSeries[] = [
      { name: "A", x: ["Mon", "Tue", "Fri"], y: [1, 2, 3] },
      { name: "B", x: ["Mon", "Tue", "Wed", "Thu", "Fri"], y: [1, 2, 3, 4, 5] },
    ];
    for (const [i, s] of seriesOf(line(cat)).entries()) expect(points(s)).toEqual(cat[i].x.map((x, k) => [x, cat[i].y[k]]));
  });
});
