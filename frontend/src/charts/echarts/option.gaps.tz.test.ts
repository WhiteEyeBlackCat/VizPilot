// A5 follow-up: the own-cadence rule reads each series on its own clock, so
// the viewer's time zone (and its daylight-saving transitions) never moves a
// break. Run with the process zone forced to America/New_York (Node re-reads
// process.env.TZ on assignment; the first test proves the switch took).

import { afterAll, beforeAll, describe, expect, it } from "vitest";

import type { LineSeriesOption } from "echarts/charts";

import type { LineChartData, RenderResult, XYSeries } from "../../types";
import { breakTemporalGaps, buildOption, cadencePositions, clockKey } from "./option";

const line = (series: XYSeries[], spec: Partial<RenderResult["spec"]> = {}): RenderResult => ({
  spec: { title: "t", type: "line", x: "when", y: "v", group_by: null, ...spec },
  chart_data: { series, y_label: "mean(v)" } satisfies LineChartData,
  sampled: false,
  n_points: series.reduce((n, s) => n + s.x.length, 0),
  display_range: null,
});
type Pt = [string | number, number | null];
const pts = (r: RenderResult) => ((buildOption(r).series as LineSeriesOption[])[0].data as Pt[]);
const nulls = (r: RenderResult) => pts(r).filter(([, y]) => y === null).length;
const xy = (x: string[]): XYSeries => ({ name: "v", x, y: x.map((_, i) => i) });
const HOUR = 3_600_000;
/** n hourly ISO strings from a UTC start, as "…Z" or as naive wall-clock */
const hours = (startUtc: string, n: number, naive = false) =>
  Array.from({ length: n }, (_, i) => {
    const s = new Date(Date.parse(startUtc) + i * HOUR).toISOString().slice(0, 19);
    return naive ? s : `${s}Z`;
  });

let saved: string | undefined;
beforeAll(() => {
  saved = process.env.TZ;
  process.env.TZ = "America/New_York";
});
afterAll(() => {
  if (saved === undefined) delete process.env.TZ;
  else process.env.TZ = saved;
});

describe("A5 own-cadence gaps are independent of the viewer's time zone", () => {
  it("the forced zone is active (EST in January, EDT in July)", () => {
    expect(new Date("2024-01-15T12:00:00Z").getTimezoneOffset()).toBe(300);
    expect(new Date("2024-07-15T12:00:00Z").getTimezoneOffset()).toBe(240);
  });

  it("spring forward: 120 hourly 'Z' timestamps from 2024-03-09 get no spurious break; a real hole still breaks", () => {
    const x = hours("2024-03-09T00:00:00Z", 120);
    for (const gran of ["raw", null] as const) {
      expect(nulls(line([xy(x)], { time_granularity: gran }))).toBe(0);
      const holed = x.filter((_, i) => i < 50 || i > 52);
      const r = line([xy(holed)], { time_granularity: gran });
      expect(nulls(r)).toBe(1);
      const k = pts(r).findIndex(([, y]) => y === null);
      expect(pts(r)[k - 1][0]).toBe(x[49]);
      expect(pts(r)[k + 1][0]).toBe(x[53]);
    }
  });

  it("fall back: hourly 'Z' timestamps across 2024-11-03 get no break, and a hole after it is still found", () => {
    const x = hours("2024-11-02T00:00:00Z", 96);
    expect(nulls(line([xy(x)], { time_granularity: null }))).toBe(0);
    expect(nulls(line([xy(x.filter((_, i) => i !== 70))], { time_granularity: null }))).toBe(1);
  });

  it("naive wall-clock hourly data through the viewer's 02:00 gap is read as spelled — no break", () => {
    const x = hours("2024-03-09T00:00:00Z", 120, true); // includes "2024-03-10T02:00:00"
    expect(x).toContain("2024-03-10T02:00:00");
    expect(nulls(line([xy(x)], { time_granularity: "raw" }))).toBe(0);
  });

  it("a repeated instant (zero step) is skipped, not a reason to drop the rule for the whole series", () => {
    const x = hours("2024-11-02T00:00:00Z", 48, true);
    const dup = [...x.slice(0, 26), x[25], ...x.slice(26)].filter((_, i) => i !== 40); // 01:00 twice, hole later
    expect(breakTemporalGaps(dup, dup.map((_, i) => i), null, null).filter(([, y]) => y === null)).toHaveLength(1);
  });

  it("positions of offset-qualified strings equal their UTC reading, whatever the offset spelling", () => {
    const z = ["2024-03-10T05:00:00Z", "2024-03-10T06:00:00Z", "2024-03-10T07:00:00Z", "2024-03-10T08:00:00Z"];
    const ny = ["2024-03-10T00:00:00-05:00", "2024-03-10T01:00:00-05:00", "2024-03-10T03:00:00-04:00", "2024-03-10T04:00:00-04:00"];
    expect(ny.map(clockKey)).toEqual(z.map(clockKey));
    expect(cadencePositions(ny)).toEqual(cadencePositions(z));
    expect(cadencePositions(z)!.regular).toBe(true);
  });

  it("very long series (300 000 points) do not overflow the call stack", () => {
    const x = hours("2000-01-01T00:00:00Z", 300_000);
    const holed = x.filter((_, i) => i !== 150_000);
    const out = breakTemporalGaps(holed, holed.map((_, i) => i), "raw", null);
    expect(out.filter(([, y]) => y === null)).toHaveLength(1);
  });
});
