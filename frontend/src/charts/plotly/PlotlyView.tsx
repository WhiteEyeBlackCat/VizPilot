import type { Data, Layout } from "plotly.js";
import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";

import { buildPlot } from "../../chartTraces";
import type { RenderResult } from "../../types";

// factory pattern: the default react-plotly.js import would bundle the full
// plotly.js build and break vite build (stage6 blocking #1)
const Plot = createPlotlyComponent(Plotly);

/** Stage-11 fallback renderer; removed with Plotly in stage 12. */
export function PlotlyView({ result, height = 360 }: { result: RenderResult; height?: number }) {
  const plot = buildPlot(result);
  return (
    <Plot
      data={plot.data as Data[]}
      layout={{ ...plot.layout, autosize: true } as Partial<Layout>}
      useResizeHandler
      style={{ width: "100%", height: `${height}px` }}
      config={{ displayModeBar: false }}
    />
  );
}
