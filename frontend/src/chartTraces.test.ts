import { describe, expect, it } from "vitest";

import { binCenters, binWidths, buildPlot, HEATMAP_COLORSCALE, isEmpty } from "./chartTraces";
import type { RenderResult } from "./types";

function result(type: RenderResult["spec"]["type"], chartData: unknown, nPoints = 5): RenderResult {
  return {
    spec: { title: "t", type, x: "xcol", y: "ycol" },
    chart_data: chartData as RenderResult["chart_data"],
    sampled: false,
    n_points: nPoints,
  };
}

describe("buildPlot", () => {
  it("line: one lines trace per series with y_label axis title", () => {
    const plot = buildPlot(
      result("line", {
        series: [{ name: "a", x: ["2024-01-01T00:00:00"], y: [1] }],
        y_label: "mean(v)",
      }),
    );
    expect(plot.data).toHaveLength(1);
    expect(plot.data[0]).toMatchObject({ type: "scatter", mode: "lines", name: "a" });
    expect(plot.layout.yaxis).toMatchObject({ title: { text: "mean(v)" } });
  });

  it("scatter: scattergl markers", () => {
    const plot = buildPlot(
      result("scatter", { series: [{ name: "y", x: [1, 2], y: [3, 4] }], y_label: "y" }),
    );
    expect(plot.data[0]).toMatchObject({ type: "scattergl", mode: "markers" });
  });

  it("bar: category x-axis preserves backend ordering", () => {
    const plot = buildPlot(
      result("bar", {
        categories: [5, 4, 3], // numeric-backed categories must not be re-sorted
        series: [{ name: "v", values: [10, 20, null] }],
        truncated: false,
        y_label: "mean(v)",
      }),
    );
    expect(plot.layout.barmode).toBe("group");
    expect(plot.layout.xaxis).toMatchObject({ type: "category" });
    expect(plot.data[0]).toMatchObject({ type: "bar", x: [5, 4, 3], y: [10, 20, null] });
  });

  it("histogram: bin centers and widths from edges", () => {
    expect(binCenters([0, 2, 4])).toEqual([1, 3]);
    expect(binWidths([0, 2, 5])).toEqual([2, 3]);
    const plot = buildPlot(
      result("histogram", { bins: { edges: [0, 2, 4], counts: [3, 7] }, y_label: "count" }),
    );
    expect(plot.data[0]).toMatchObject({ type: "bar", x: [1, 3], y: [3, 7], width: [2, 2] });
  });

  it("histogram grouped: uses series with overlay and ignores overall bins", () => {
    const plot = buildPlot(
      result("histogram", {
        bins: { edges: [0, 2], counts: [99] },
        series: [
          { name: "a", edges: [0, 2, 4], counts: [1, 2] },
          { name: "b", edges: [0, 2, 4], counts: [3, 4] },
        ],
        y_label: "count",
      }),
    );
    expect(plot.data).toHaveLength(2);
    expect(plot.layout.barmode).toBe("overlay");
    expect(plot.data[0]).toMatchObject({ name: "a", opacity: 0.6, y: [1, 2] });
  });

  it("box: maps lower_fence/upper_fence to plotly names and adds outlier trace", () => {
    const plot = buildPlot(
      result("box", {
        groups: [
          {
            name: "g1",
            q1: 2,
            median: 3,
            q3: 4,
            lower_fence: -1,
            upper_fence: 7,
            outliers: [100, 200],
            mean: 22,
            count: 5,
          },
        ],
        y_label: "v",
      }),
    );
    expect(plot.data[0]).toMatchObject({
      type: "box",
      q1: [2],
      median: [3],
      q3: [4],
      lowerfence: [-1],
      upperfence: [7],
    });
    expect(plot.data[1]).toMatchObject({ mode: "markers", x: ["g1", "g1"], y: [100, 200] });
    expect(plot.layout.xaxis).toMatchObject({ type: "category" });
  });

  it("heatmap: symmetric z range, custom colorscale, reversed y", () => {
    const plot = buildPlot(
      result("heatmap", {
        columns: ["a", "b"],
        matrix: [
          [1, -0.5],
          [-0.5, 1],
        ],
        y_label: "correlation",
      }),
    );
    expect(plot.data[0]).toMatchObject({
      type: "heatmap",
      zmin: -1,
      zmax: 1,
      colorscale: HEATMAP_COLORSCALE,
    });
    expect(plot.layout.yaxis).toMatchObject({ autorange: "reversed" });
  });

  it("isEmpty reflects n_points", () => {
    expect(isEmpty(result("line", { series: [], y_label: "v" }, 0))).toBe(true);
    expect(isEmpty(result("line", { series: [], y_label: "v" }, 3))).toBe(false);
  });
});
