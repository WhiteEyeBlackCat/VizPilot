// A5: temporal lines break where a series leaves its own cadence. The
// backend emits only buckets that have rows (null-y rows are dropped in
// `_clean`), so an empty week / month used to arrive as two consecutive
// points and be drawn as a straight line. Rule ("temporal own-cadence gaps"
// in option.ts): per series, break where the step between consecutive
// observations is longer than every recurring step (≥ 2 occurrences and
// ≥ 10 % of the steps), counted over the weekdays / times of day the series
// samples; raw / unset granularity only on a regular grid; < 5 points never.
// A gap is inserted as an invisible null point — never a fabricated value.

import { describe, expect, it } from "vitest";

import type { LineSeriesOption } from "echarts/charts";

import type { LineChartData, RenderResult, XYSeries } from "../../types";
import { breakTemporalGaps, buildOption, temporalGrid } from "./option";
import { tooltipValue } from "./theme";

const line = (series: XYSeries[], spec: Partial<RenderResult["spec"]> = {}): RenderResult => ({
  spec: { title: "t", type: "line", x: "when", y: "v", group_by: null, ...spec },
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
const plain = (s: XYSeries): Pt[] => s.x.map((x, i) => [x as string, s.y[i]]);

// calendar helpers (UTC arithmetic on date strings; the option reads them as
// local midnights, which keeps the calendar day)
const iso = (ms: number) => new Date(ms).toISOString().slice(0, 10);
const dayRange = (from: string, to: string): string[] => {
  const out: string[] = [];
  for (let t = Date.parse(`${from}T00:00:00Z`); t <= Date.parse(`${to}T00:00:00Z`); t += 86_400_000) out.push(iso(t));
  return out;
};
const weekday = (d: string) => new Date(`${d}T00:00:00Z`).getUTCDay(); // 0 Sun … 6 Sat
const businessDays = (from: string, to: string) => dayRange(from, to).filter((d) => weekday(d) !== 0 && weekday(d) !== 6);
const monthRange = (fromY: number, fromM: number, n: number): string[] =>
  Array.from({ length: n }, (_, i) => {
    const m = fromM - 1 + i;
    return `${fromY + Math.floor(m / 12)}-${String((m % 12) + 1).padStart(2, "0")}-01`;
  });
const hourRange = (day: string, from: number, to: number) =>
  Array.from({ length: to - from + 1 }, (_, i) => `${day}T${String(from + i).padStart(2, "0")}:00:00`);

/** every backend pair survives in order; each inserted null is a number x
 *  strictly between two real neighbours and carries no value */
const expectPreserved = (r: RenderResult) => {
  const d = r.chart_data as LineChartData;
  for (const [i, s] of seriesOf(r).entries()) {
    const pts = points(s);
    expect(pts.filter(([, y]) => y !== null)).toEqual(plain(d.series[i]));
    for (const [k, [x, y]] of pts.entries()) {
      if (y !== null) continue;
      expect(typeof x).toBe("number");
      expect(Number(x)).toBeGreaterThan(Date.parse(String(pts[k - 1][0]).length === 10 ? `${pts[k - 1][0]}T00:00:00` : String(pts[k - 1][0])));
      expect(Number(x)).toBeLessThan(Date.parse(String(pts[k + 1][0]).length === 10 ? `${pts[k + 1][0]}T00:00:00` : String(pts[k + 1][0])));
      expect(pts[k - 1][1]).not.toBeNull();
      expect(pts[k + 1][1]).not.toBeNull();
    }
  }
};

describe("A5: temporal lines break where a series leaves its own cadence", () => {
  it("ungrouped daily series with no rows for a week (01-20 → 01-28) breaks there, bucketed and raw", () => {
    const x = [...dayRange("2025-01-01", "2025-01-20"), ...dayRange("2025-01-28", "2025-02-10")];
    for (const gran of ["day", "raw", null] as const) {
      const r = line([xy("v", x)], { time_granularity: gran });
      const [s] = seriesOf(r);
      expect(segments(s)).toEqual([dayRange("2025-01-01", "2025-01-20"), dayRange("2025-01-28", "2025-02-10")]);
      expect(s.connectNulls).toBe(false);
      expectPreserved(r);
    }
  });

  it("ungrouped monthly series missing March breaks Feb → Apr, bucketed and raw", () => {
    const all = monthRange(2025, 1, 8);
    const x = all.filter((m) => m !== "2025-03-01");
    for (const gran of ["month", "raw", undefined] as const) {
      const r = line([xy("v", x)], { time_granularity: gran });
      expect(segments(seriesOf(r)[0])).toEqual([["2025-01-01", "2025-02-01"], all.slice(3)]);
      expectPreserved(r);
    }
  });

  it("weekly buckets (Monday-aligned) missing two weeks break once; a full run does not", () => {
    const mondays = dayRange("2025-01-06", "2025-03-31").filter((d) => weekday(d) === 1).map((d) => `${d}T00:00:00`);
    expect(mondays).toHaveLength(13);
    const x = mondays.filter((_, i) => i !== 4 && i !== 5);
    const r = line([xy("v", x)], { time_granularity: "week" });
    expect(segments(seriesOf(r)[0])).toEqual([mondays.slice(0, 4), mondays.slice(6)]);
    expectPreserved(r);
    expect(points(seriesOf(line([xy("v", mondays)], { time_granularity: "week" }))[0])).toEqual(plain(xy("v", mondays)));
  });

  it("business-day daily data (Mon–Fri over several weeks) is not broken at weekends, but a missing full week breaks", () => {
    const bd = businessDays("2025-03-03", "2025-03-28"); // four Mon–Fri weeks
    for (const gran of ["day", "raw"] as const) {
      const full = xy("v", bd);
      expect(points(seriesOf(line([full], { time_granularity: gran }))[0])).toEqual(plain(full));
      const holed = bd.filter((d) => d < "2025-03-17" || d > "2025-03-21"); // week 3 missing
      const r = line([xy("v", holed)], { time_granularity: gran });
      expect(segments(seriesOf(r)[0])).toEqual([businessDays("2025-03-03", "2025-03-14"), businessDays("2025-03-24", "2025-03-28")]);
      expectPreserved(r);
    }
  });

  it("seven-day data that skips one weekend breaks there (it does sample weekends elsewhere)", () => {
    const x = dayRange("2025-03-01", "2025-03-31").filter((d) => d !== "2025-03-15" && d !== "2025-03-16");
    const [s] = seriesOf(line([xy("v", x)], { time_granularity: "day" }));
    expect(segments(s)).toEqual([dayRange("2025-03-01", "2025-03-14"), dayRange("2025-03-17", "2025-03-31")]);
  });

  it("irregular raw timestamps are never broken, however long a stretch", () => {
    const ev = ["2025-03-03T09:00:00", "2025-03-03T09:07:30", "2025-03-03T09:31:05", "2025-03-03T10:02:00", "2025-03-03T10:02:41", "2025-03-05T18:13:00", "2025-03-05T18:20:12", "2025-03-05T19:00:00"];
    for (const gran of ["raw", null, undefined] as const) {
      const s = xy("v", ev);
      expect(points(seriesOf(line([s], { time_granularity: gran }))[0])).toEqual(plain(s));
    }
    // irregular days (event dates): no regular grid, no break
    const days = ["2025-01-01", "2025-01-02", "2025-01-05", "2025-01-12", "2025-01-13", "2025-02-20", "2025-02-27"];
    expect(points(seriesOf(line([xy("v", days)], { time_granularity: "raw" }))[0])).toEqual(plain(xy("v", days)));
  });

  it("raw regular hourly data with a hole breaks; office hours keep overnight and weekend connected", () => {
    const h = [...hourRange("2025-03-03", 0, 23), ...hourRange("2025-03-04", 0, 23)];
    const holed = h.filter((_, i) => i < 30 || i > 32); // 06:00–08:00 on day 2 missing
    const r = line([xy("v", holed)], { time_granularity: "raw" });
    expect(segments(seriesOf(r)[0])).toEqual([h.slice(0, 30), h.slice(33)]);
    expectPreserved(r);
    // 09:00–17:00, Mon–Fri over two weeks: overnight and weekend are the
    // series' own cadence
    const office = businessDays("2025-03-03", "2025-03-14").flatMap((d) => hourRange(d, 9, 17));
    expect(points(seriesOf(line([xy("v", office)], { time_granularity: null }))[0])).toEqual(plain(xy("v", office)));
    // … but a missing whole day (Wed 03-12) breaks
    const noWed = office.filter((t) => !t.startsWith("2025-03-12"));
    const [sw] = seriesOf(line([xy("v", noWed)], { time_granularity: null }));
    expect(segments(sw)).toHaveLength(2);
    expect(segments(sw)[0][segments(sw)[0].length - 1]).toBe("2025-03-11T17:00:00");
    expect(segments(sw)[1][0]).toBe("2025-03-13T09:00:00");
    // raw regular daily with a hole
    const dx = dayRange("2025-01-01", "2025-01-31").filter((d) => d < "2025-01-10" || d > "2025-01-13");
    expect(segments(seriesOf(line([xy("v", dx)], { time_granularity: "raw" }))[0])).toHaveLength(2);
  });

  it("regular series without a hole are deep-equal to the plain pairs", () => {
    const cases: [string[], RenderResult["spec"]["time_granularity"]][] = [
      [dayRange("2025-01-01", "2025-02-28"), "day"],
      [monthRange(2023, 1, 30), "month"],
      [[...hourRange("2025-03-03", 0, 23), ...hourRange("2025-03-04", 0, 23)], "raw"],
      [dayRange("2025-01-01", "2025-03-31").filter((_, i) => i % 2 === 0), "day"], // every other day
      [monthRange(2020, 1, 24).filter((_, i) => i % 3 === 0), null], // quarterly
    ];
    for (const [x, gran] of cases) {
      const s = xy("v", x);
      expect(points(seriesOf(line([s], { time_granularity: gran }))[0])).toEqual(plain(s));
    }
  });

  it("grouped: an internal hole no group occupies breaks every group under bucketed granularity", () => {
    const all = monthRange(2011, 1, 12);
    const x = all.filter((m) => !["2011-03-01", "2011-04-01", "2011-05-01"].includes(m));
    const a = xy("A", x);
    const b = { ...xy("B", x), y: x.map((_, i) => 10 - i) };
    const r = line([a, b], { group_by: "g", time_granularity: "month" });
    // the absent-bucket rule alone sees no bucket between Feb and Jun
    const grid = temporalGrid(r.chart_data as LineChartData, "month");
    expect(grid.index.get(Date.parse("2011-06-01T00:00:00"))! - grid.index.get(Date.parse("2011-02-01T00:00:00"))!).toBe(1);
    for (const s of seriesOf(r)) expect(segments(s)).toEqual([all.slice(0, 2), all.slice(5)]);
    expectPreserved(r);
    // both rules agreeing put a single null, not two
    const c = xy("C", all.filter((m) => m !== "2011-08-01"));
    const sc = breakTemporalGaps(c.x, c.y, "month", temporalGrid({ series: [a, c], y_label: "v" }, "month"));
    expect(sc.filter(([, y]) => y === null)).toHaveLength(1);
  });

  it("fewer than five points: no cadence inferred, unchanged", () => {
    const s = xy("v", ["2025-01-01", "2025-02-01", "2025-06-01", "2025-07-01"]);
    expect(points(seriesOf(line([s], { time_granularity: "month" }))[0])).toEqual(plain(s));
  });

  it("a null-y period (rows dropped by the backend) behaves like a missing period; an explicit null y gets no second null", () => {
    // backend: rows with null y are removed in _clean, so the dates are absent
    const all = dayRange("2025-01-01", "2025-01-31");
    const dropped = all.filter((d) => d < "2025-01-10" || d > "2025-01-16");
    const missing = line([xy("v", dropped)], { time_granularity: "day" });
    expect(segments(seriesOf(missing)[0])).toEqual([dayRange("2025-01-01", "2025-01-09"), dayRange("2025-01-17", "2025-01-31")]);
    // an explicit null y in a payload is already a gap; the cadence is
    // unbroken so nothing more is inserted
    const withNull: XYSeries = { name: "v", x: all, y: all.map((d, i) => (d === "2025-01-12" ? null : i)) };
    expect(points(seriesOf(line([withNull], { time_granularity: "day" }))[0])).toEqual(plain(withNull));
  });

  it("the inserted null is a valueless point: the tooltip formatter shows a dash, never a number", () => {
    const x = [...dayRange("2025-01-01", "2025-01-20"), ...dayRange("2025-01-28", "2025-02-10")];
    const opt = buildOption(line([xy("v", x)], { time_granularity: "day" }));
    const hole = points(seriesOf(line([xy("v", x)], { time_granularity: "day" }))[0]).find(([, y]) => y === null)!;
    expect(hole[1]).toBeNull();
    const tt = opt.tooltip as { valueFormatter: (v: unknown) => string };
    expect(tt.valueFormatter).toBe(tooltipValue);
    expect(tt.valueFormatter(hole[1])).toBe("—");
  });

  it("numeric-x and category-x lines are untouched", () => {
    const num: XYSeries = { name: "v", x: [1, 2, 3, 4, 5, 50, 51], y: [1, 2, 3, 4, 5, 6, 7] };
    expect(points(seriesOf(line([num]))[0])).toEqual(plain(num));
    const cat: XYSeries = { name: "v", x: ["a", "b", "c", "d", "e", "f"], y: [1, 2, 3, 4, 5, 6] };
    expect(points(seriesOf(line([cat]))[0])).toEqual(plain(cat));
  });
});
