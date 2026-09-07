// Pure chart_data -> Plotly traces/layout conversion (unit-tested; keep free
// of React and plotly imports).

import type {
  BarChartData,
  BoxChartData,
  HeatmapChartData,
  HistogramChartData,
  LineChartData,
  RenderResult,
} from "./types";

export type Trace = Record<string, unknown>;

export interface PlotDef {
  data: Trace[];
  layout: Record<string, unknown>;
}

// negative blue / zero white / positive red, symmetric around 0
export const HEATMAP_COLORSCALE: [number, string][] = [
  [0, "#2166ac"],
  [0.5, "#f7f7f7"],
  [1, "#b2182b"],
];

export function binCenters(edges: number[]): number[] {
  return edges.slice(0, -1).map((edge, i) => (edge + edges[i + 1]) / 2);
}

export function binWidths(edges: number[]): number[] {
  return edges.slice(0, -1).map((edge, i) => edges[i + 1] - edge);
}

export const isEmpty = (result: RenderResult): boolean => result.n_points === 0;

export function buildPlot(result: RenderResult): PlotDef {
  const spec = result.spec;
  const base: Record<string, unknown> = {
    title: { text: spec.title, font: { size: 14 } },
    margin: { t: 48, r: 24, b: 48, l: 56 },
  };

  switch (spec.type) {
    case "line": {
      const d = result.chart_data as LineChartData;
      return {
        data: d.series.map((s) => ({ type: "scatter", mode: "lines", name: s.name, x: s.x, y: s.y })),
        layout: { ...base, yaxis: { title: { text: d.y_label } } },
      };
    }
    case "scatter": {
      const d = result.chart_data as LineChartData;
      return {
        data: d.series.map((s) => ({
          type: "scattergl",
          mode: "markers",
          name: s.name,
          x: s.x,
          y: s.y,
          marker: { size: 5, opacity: 0.75 },
        })),
        layout: {
          ...base,
          xaxis: { title: { text: spec.x ?? "" } },
          yaxis: { title: { text: d.y_label } },
        },
      };
    }
    case "bar": {
      const d = result.chart_data as BarChartData;
      return {
        data: d.series.map((s) => ({ type: "bar", name: s.name, x: d.categories, y: s.values })),
        layout: {
          ...base,
          barmode: "group",
          // category axis preserves the backend's ranking even for numeric
          // or boolean category values (stage6 blocking #3)
          xaxis: { type: "category", title: { text: spec.x ?? "" } },
          yaxis: { title: { text: d.y_label } },
        },
      };
    }
    case "histogram": {
      const d = result.chart_data as HistogramChartData;
      const grouped = (d.series?.length ?? 0) > 0;
      const groups = grouped
        ? d.series!
        : [{ name: spec.x ?? "", edges: d.bins.edges, counts: d.bins.counts }];
      return {
        data: groups.map((g) => ({
          type: "bar",
          name: g.name,
          x: binCenters(g.edges),
          y: g.counts,
          width: binWidths(g.edges),
          opacity: grouped ? 0.6 : 1,
        })),
        layout: {
          ...base,
          ...(grouped ? { barmode: "overlay" } : {}),
          bargap: 0,
          xaxis: { title: { text: spec.x ?? "" } },
          yaxis: { title: { text: d.y_label } },
        },
      };
    }
    case "box": {
      const d = result.chart_data as BoxChartData;
      const names = d.groups.map((g) => g.name);
      const data: Trace[] = [
        {
          type: "box",
          name: d.y_label,
          x: names,
          q1: d.groups.map((g) => g.q1),
          median: d.groups.map((g) => g.median),
          q3: d.groups.map((g) => g.q3),
          lowerfence: d.groups.map((g) => g.lower_fence),
          upperfence: d.groups.map((g) => g.upper_fence),
          mean: d.groups.map((g) => g.mean),
        },
      ];
      const outlierX: string[] = [];
      const outlierY: number[] = [];
      for (const g of d.groups) {
        for (const v of g.outliers) {
          outlierX.push(g.name);
          outlierY.push(v);
        }
      }
      if (outlierX.length > 0) {
        data.push({
          type: "scatter",
          mode: "markers",
          name: "outliers",
          x: outlierX,
          y: outlierY,
          marker: { size: 5, opacity: 0.7 },
        });
      }
      return {
        data,
        layout: {
          ...base,
          showlegend: false,
          xaxis: { type: "category" },
          yaxis: { title: { text: d.y_label } },
        },
      };
    }
    case "heatmap": {
      const d = result.chart_data as HeatmapChartData;
      return {
        data: [
          {
            type: "heatmap",
            z: d.matrix,
            x: d.columns,
            y: d.columns,
            zmin: -1,
            zmax: 1,
            colorscale: HEATMAP_COLORSCALE,
          },
        ],
        layout: { ...base, yaxis: { autorange: "reversed" } },
      };
    }
  }
}
