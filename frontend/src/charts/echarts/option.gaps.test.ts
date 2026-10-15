// Grouped temporal lines must not join a group's points across buckets of
// the chart in which the group has no observations (hotfix: `temp over
// dteday by season` drew the last 2011 summer point straight to the first
// 2012 one). Rule: the chart's buckets are the x values that exist in any
// series; a group breaks between two consecutive observations iff another
// existing bucket lies between them. Nothing is generated — a weekend,
// holiday or night that no group occupies is not a bucket, so lines stay
// connected across it. The break is an invisible null point that
// `connectNulls: false` renders as a gap. The rule applies only to aligned
// buckets: day / week / month granularity from the spec, or a regular grid
// detected in raw / unset-granularity data on which the group itself samples
// at the grid unit (`temporalGrid` in option.ts); raw asynchronous timestamps
// are never broken. line_grouped_gap.json is a real-backend snapshot of
// e2e/data/seasonal_groups.csv (month buckets, four seasons).
//
// The 14 grouped temporal cases are titled "case N: …".

import { describe, expect, it } from "vitest";

import type { LineSeriesOption } from "echarts/charts";

import lineGrouped from "../__fixtures__/line_grouped.json";
import lineGroupedGap from "../__fixtures__/line_grouped_gap.json";
import type { LineChartData, RenderResult, XYSeries } from "../../types";
import { breakAbsentBuckets, buildOption, gridPositions, temporalGrid } from "./option";

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
    // a monthly dataset small enough to keep granularity "raw" (the rules
    // keep raw up to 500 distinct timestamps): the month grid is detected
    // in the data and the same breaks follow
    const raw = line([a, rest], { time_granularity: "raw", aggregation: "mean" });
    expect(temporalGrid(raw.chart_data as LineChartData, "raw")).toMatchObject({ aligned: "grid", scale: "month", unit: 1 });
    expect(segments(seriesOf(raw)[0])).toEqual([months(["2011-06", "2011-07", "2011-08"]), months(["2012-06", "2012-07"])]);
    expectPreserved(raw);
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

  it("case 5: one group skips an aligned bucket another group occupies — the absent group breaks, the other does not", () => {
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

  it("case 6: continuous grouped data is unchanged (line_grouped fixture, shared sparse grid)", () => {
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

  it("case 7: independent group gaps — several groups with their own gaps are evaluated independently", () => {
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

  it("case 1 (real backend fixture): cross-year season absence is not joined; within-year runs stay continuous", () => {
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

  it("case 11: bucket identity — the same bucket written as date and datetime is one bucket; sub-daily buckets stay distinct", () => {
    const a = xy("A", ["2025-01-01", "2025-01-02", "2025-01-04"]);
    const b = xy("B", ["2025-01-01T00:00:00", "2025-01-02T00:00:00", "2025-01-03T00:00:00", "2025-01-04T00:00:00"]);
    const grid = temporalGrid({ series: [a, b], y_label: "v" }, "day");
    expect(grid.index.size).toBe(4);
    expect(grid.aligned).toBe("bucketed");
    expect(segments(seriesOf(line([a, b], { time_granularity: "day" }))[0])).toEqual([["2025-01-01", "2025-01-02"], ["2025-01-04"]]);
    // a null / unparsable x takes no part and never breaks the line
    expect(breakAbsentBuckets(["2025-01-01", null, "2025-01-04"], [1, 2, 3], grid)).toEqual([["2025-01-01", 1], ["", 2], ["2025-01-04", 3]]);
    // sub-daily: 09:00 and 10:00 of one day are two buckets, not one day
    const h = xy("H", ["2025-01-01T09:00:00", "2025-01-01T10:00:00", "2025-01-01T11:00:00", "2025-01-01T12:00:00"]);
    const k = xy("K", ["2025-01-01T09:00:00", "2025-01-01 10:00:00", "2025-01-01T12:00:00"]); // space separator = same instant as H's 10:00
    const sub = temporalGrid({ series: [h, k], y_label: "v" }, "raw");
    expect(sub.index.size).toBe(4);
    expect(sub.scale).toBe("ms");
    expect(sub.unit).toBe(3_600_000);
    expect(segments(seriesOf(line([h, k], { time_granularity: "raw" }))[1])).toEqual([[k.x[0], k.x[1]], [k.x[2]]]);
    // scale detection: month starts → month index, midnights → day number
    expect(gridPositions([Date.parse("2025-01-01T00:00:00"), Date.parse("2025-03-01T00:00:00")]).scale).toBe("month");
    expect(gridPositions([Date.parse("2025-01-01T00:00:00"), Date.parse("2025-01-15T00:00:00")]).scale).toBe("day");
    expect(gridPositions([Date.parse("2025-01-01T00:00:00"), Date.parse("2025-01-01T06:00:00")]).scale).toBe("ms");
  });

  it("case 8: ungrouped temporal lines get no absent-bucket rule — but a long gap beyond the series' own cadence breaks (A5)", () => {
    // before A5 this test pinned the bug: monthly buckets Jun–Aug 2011 then
    // Jun–Jul 2012 were joined across ten empty months. The month cadence
    // (step 1, recurring) now breaks the 10-month step; see option.gaps2.test.ts
    const a = monthly("v", ["2011-06", "2011-07", "2011-08", "2012-06", "2012-07"]);
    const r = line([a], { group_by: null, time_granularity: "month" });
    const [s] = seriesOf(r);
    expect(segments(s)).toEqual([months(["2011-06", "2011-07", "2011-08"]), months(["2012-06", "2012-07"])]);
    expectPreserved(r);
    // no hole: unchanged
    const c = monthly("v", ["2011-06", "2011-07", "2011-08", "2011-09", "2011-10"]);
    expect(points(seriesOf(line([c], { group_by: null, time_granularity: "month" }))[0])).toEqual(c.x.map((x, i) => [x, c.y[i]]));
  });

  it("case 9: numeric-x grouped lines are untouched", () => {
    const num: XYSeries[] = [
      { name: "A", x: [1, 2, 6], y: [1, 2, 3] },
      { name: "B", x: [1, 2, 3, 4, 5, 6], y: [1, 2, 3, 4, 5, 6] },
    ];
    for (const [i, s] of seriesOf(line(num)).entries()) expect(points(s)).toEqual(num[i].x.map((x, k) => [x, num[i].y[k]]));
  });

  it("case 10: category-x grouped lines are untouched", () => {
    const cat: XYSeries[] = [
      { name: "A", x: ["Mon", "Tue", "Fri"], y: [1, 2, 3] },
      { name: "B", x: ["Mon", "Tue", "Wed", "Thu", "Fri"], y: [1, 2, 3, 4, 5] },
    ];
    for (const [i, s] of seriesOf(line(cat)).entries()) expect(points(s)).toEqual(cat[i].x.map((x, k) => [x, cat[i].y[k]]));
  });

  it("case 12: grouped raw asynchronous timestamps (A odd hours, B even hours) are never fragmented", () => {
    const a = xy("A", ["2025-03-03T09:00:00", "2025-03-03T11:00:00", "2025-03-03T13:00:00"]);
    const b = xy("B", ["2025-03-03T10:00:00", "2025-03-03T12:00:00", "2025-03-03T14:00:00"]);
    for (const gran of ["raw", null, undefined] as const) {
      const r = line([a, b], { time_granularity: gran, aggregation: gran === undefined ? undefined : "mean" });
      // the union is an hourly grid, but neither group samples at that unit
      expect(temporalGrid(r.chart_data as LineChartData, gran)).toMatchObject({ aligned: "grid", unit: 3_600_000 });
      const [sa, sb] = seriesOf(r);
      expect(points(sa)).toEqual(a.x.map((x, i) => [x, a.y[i]]));
      expect(points(sb)).toEqual(b.x.map((x, i) => [x, b.y[i]]));
      expect(segments(sa)).toEqual([a.x]);
      expect(segments(sb)).toEqual([b.x]);
    }
    // irregular event-like timestamps with no bucket in common: not a grid at all
    const ea = xy("A", ["2025-03-03T09:00:00", "2025-03-03T09:07:30", "2025-03-03T09:31:05", "2025-03-03T10:02:00"]);
    const eb = xy("B", ["2025-03-03T09:03:00", "2025-03-03T09:20:00", "2025-03-03T09:45:00", "2025-03-03T09:59:59"]);
    const e = line([ea, eb], { time_granularity: "raw" });
    expect(temporalGrid(e.chart_data as LineChartData, "raw").aligned).toBeNull();
    for (const [i, s] of seriesOf(e).entries()) expect(points(s)).toHaveLength([ea, eb][i].x.length);
  });

  it("case 13: partially overlapping raw timestamp groups (some shared instants) stay whole", () => {
    // A and B share 09:00 and 11:00 but otherwise sample at their own times
    const a = xy("A", ["2025-03-03T09:00:00", "2025-03-03T09:30:00", "2025-03-03T10:00:00", "2025-03-03T11:00:00"]);
    const b = xy("B", ["2025-03-03T09:00:00", "2025-03-03T10:15:00", "2025-03-03T11:00:00", "2025-03-03T11:45:00"]);
    const r = line([a, b], { time_granularity: "raw", aggregation: "mean" });
    // union steps 30 / 30 / 15 / 45 / 45 min: multiples of 15 min, but the
    // grid is not dense at that unit → raw timestamps
    expect(temporalGrid(r.chart_data as LineChartData, "raw").aligned).toBeNull();
    const [sa, sb] = seriesOf(r);
    expect(segments(sa)).toEqual([a.x]);
    expect(segments(sb)).toEqual([b.x]);
    // shared instants on an otherwise irregular schedule (seconds-level offsets)
    const ia = xy("A", ["2025-03-03T09:00:00", "2025-03-03T09:00:17", "2025-03-03T09:02:03", "2025-03-03T09:10:00"]);
    const ib = xy("B", ["2025-03-03T09:00:00", "2025-03-03T09:01:00", "2025-03-03T09:05:41", "2025-03-03T09:10:00"]);
    const i = line([ia, ib], { time_granularity: "raw", aggregation: "mean" });
    expect(temporalGrid(i.chart_data as LineChartData, "raw").aligned).toBeNull();
    for (const [k, s] of seriesOf(i).entries()) expect(points(s)).toEqual([ia, ib][k].x.map((x, j) => [x, [ia, ib][k].y[j]]));
  });

  it("case 14: aligned hourly aggregate buckets (raw granularity) with one missing bucket still break there — and only there", () => {
    const day1 = ["09", "10", "11", "12", "13", "14"].map((h) => `2025-03-03T${h}:00:00`);
    const day2 = ["09", "10", "11", "12", "13", "14"].map((h) => `2025-03-04T${h}:00:00`);
    const all = [...day1, ...day2];
    const a = xy("A", all.filter((h) => h !== day2[2])); // absent 2025-03-04 11:00
    const b = xy("B", all);
    const r = line([a, b], { time_granularity: "raw", aggregation: "mean" });
    expect(temporalGrid(r.chart_data as LineChartData, "raw")).toMatchObject({ aligned: "grid", scale: "ms", unit: 3_600_000 });
    const [sa, sb] = seriesOf(r);
    // the overnight 14:00 → 09:00 stretch is no bucket: connected; the
    // skipped 11:00 that B occupies: broken
    expect(segments(sa)).toEqual([[...day1, day2[0], day2[1]], [day2[3], day2[4], day2[5]]]);
    expect(segments(sb)).toEqual([all]);
    const hole = points(sa).find(([, y]) => y === null)!;
    expect(Number(hole[0])).toBeGreaterThan(Date.parse(day2[1]));
    expect(Number(hole[0])).toBeLessThan(Date.parse(day2[3]));
    expectPreserved(r);
    // the same data with the granularity the backend would echo after a
    // downgrade takes the metadata path and breaks identically
    const [da] = seriesOf(line([a, b], { time_granularity: "day" }));
    expect(segments(da)).toEqual(segments(sa));
  });
});
