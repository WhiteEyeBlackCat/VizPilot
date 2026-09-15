// The display order is a permutation of what the backend returned: values
// stay paired with their category, plain labels keep the backend ranking,
// and the note says which order the reader is looking at.

import { describe, expect, it } from "vitest";

import type { BarSeriesOption, BoxplotSeriesOption } from "echarts/charts";

import barSingle from "../__fixtures__/bar_single.json";
import boxOutliers from "../__fixtures__/box_outliers.json";
import type { BarChartData, BoxChartData, RenderResult } from "../../types";
import { buildOption } from "./option";
import { displayOrderNote, naturalOrder } from "./order";

const fx = (json: unknown) => json as RenderResult;
const axisData = (o: ReturnType<typeof buildOption>) =>
  ((Array.isArray(o.xAxis) ? o.xAxis[0] : o.xAxis) as { data?: unknown }).data as string[];

const bar = (categories: (string | number | boolean)[], values: (number | null)[], spec: Partial<RenderResult["spec"]> = {}): RenderResult => ({
  ...fx(barSingle),
  spec: { ...fx(barSingle).spec, ...spec },
  chart_data: { categories, series: [{ name: "v", values }], truncated: false, y_label: "v" } satisfies BarChartData,
});

describe("naturalOrder", () => {
  it("hours / months as numbers or numeric strings sort numerically, not lexicographically", () => {
    expect(naturalOrder([17, 8, 0, 23])!.index).toEqual([2, 1, 0, 3]);
    expect(naturalOrder(["10", "2", "1"])).toMatchObject({ kind: "numeric", index: [2, 1, 0] });
  });
  it("weekday and month names in any case / abbreviation", () => {
    expect(naturalOrder(["Wed", "mon", "SUNDAY"])).toMatchObject({ kind: "weekday", index: [1, 0, 2] });
    expect(naturalOrder(["Mar", "January", "dec"])).toMatchObject({ kind: "month", index: [1, 0, 2] });
  });
  it("ISO dates sort chronologically; date-shaped codes do not count", () => {
    expect(naturalOrder(["2024-03", "2023-12", "2024-01"])).toMatchObject({ kind: "date", index: [1, 2, 0] });
    expect(naturalOrder(["2024-13-99", "2024-01-01"])).toBeNull();
    expect(naturalOrder(["2024-25", "2023-24"])).toBeNull();
  });
  it("plain labels, booleans, mixed types and single categories have no natural order", () => {
    expect(naturalOrder(["C", "B", "A"])).toBeNull();
    expect(naturalOrder([true, false])).toBeNull();
    expect(naturalOrder([5, 4, true])).toBeNull();
    expect(naturalOrder(["Mon", "B"])).toBeNull();
    expect(naturalOrder([7])).toBeNull();
  });
});

describe("bar display order", () => {
  it("a ranked month axis is shown 1..12 with every value still on its month", () => {
    const r = bar([7, 1, 12, 3], [30, 5, 2, 12], { x: "mnth", y: "temp", aggregation: "mean" });
    const o = buildOption(r);
    expect(axisData(o)).toEqual(["1", "3", "7", "12"]);
    expect((o.series as BarSeriesOption[])[0].data).toEqual([5, 12, 30, 2]);
    expect(displayOrderNote(r)).toBe("mnth 依數值順序顯示"); // numeric months are numbers first
  });
  it("grouped series are permuted together and nulls stay nulls", () => {
    const r: RenderResult = {
      ...bar([2, 1], [1, 2]),
      chart_data: {
        categories: ["Tue", "Mon"],
        series: [
          { name: "a", values: [10, null] },
          { name: "b", values: [null, 20] },
        ],
        truncated: false,
        y_label: "v",
      },
    };
    const s = buildOption(r).series as BarSeriesOption[];
    expect(axisData(buildOption(r))).toEqual(["Mon", "Tue"]);
    expect(s[0].data).toEqual([null, 10]);
    expect(s[1].data).toEqual([20, null]);
  });
  it("plain categories keep the backend ranking and the note says it is ranked", () => {
    const r = fx(barSingle);
    expect(axisData(buildOption(r))).toEqual((r.chart_data as BarChartData).categories.map(String));
    expect(displayOrderNote(r)).toBe("依 平均 pm25 由高到低排序");
    expect(displayOrderNote(bar(["C", "B", "A"], [3, 2, 1], { y: null, aggregation: "count" }))).toBe("依 筆數 由高到低排序");
  });
  it("a truncated natural axis discloses both the cut and the order", () => {
    const r = bar([3, 1, 2], [9, 8, 7], { x: "hr", y: "cnt", aggregation: "mean" });
    (r.chart_data as BarChartData).truncated = true;
    expect(displayOrderNote(r)).toBe("hr 依數值順序顯示；前 3 名（依平均 cnt）");
  });
  it("one category: nothing to order, no note", () => {
    expect(displayOrderNote(bar(["only"], [1]))).toBeNull();
  });
});

describe("box display order", () => {
  it("lexicographic numeric groups are shown numerically with their statistics", () => {
    const base = fx(boxOutliers);
    const d = base.chart_data as BoxChartData;
    const groups = [
      { ...d.groups[0], name: "10", median: 10 },
      { ...d.groups[1], name: "2", median: 2 },
      { ...d.groups[2], name: "1", median: 1 },
    ];
    const r: RenderResult = { ...base, spec: { ...base.spec, x: "hr" }, chart_data: { ...d, groups } };
    const o = buildOption(r);
    expect(axisData(o)).toEqual(["1", "2", "10"]);
    const box = (o.series as BoxplotSeriesOption[])[0];
    expect((box.data as number[][]).map((s) => s[2])).toEqual([1, 2, 10]);
    expect(displayOrderNote(r)).toBe("hr 依數值順序顯示");
  });
  it("named groups keep the backend order and have no note", () => {
    const r = fx(boxOutliers);
    expect(axisData(buildOption(r))).toEqual((r.chart_data as BoxChartData).groups.map((g) => g.name));
    expect(displayOrderNote(r)).toBeNull();
  });
});
