import type { Data, Layout } from "plotly.js";
import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";

import { buildPlot, isEmpty } from "../chartTraces";
import type { BarChartData, RenderResult } from "../types";

// factory pattern: the default react-plotly.js import would bundle the full
// plotly.js build and break vite build (stage6 blocking #1)
const Plot = createPlotlyComponent(Plotly);

export function ChartView({ result }: { result: RenderResult }) {
  if (isEmpty(result)) {
    return (
      <div className="flex h-64 items-center justify-center rounded bg-slate-50 text-sm text-slate-400">
        此組合無資料
      </div>
    );
  }

  const plot = buildPlot(result);
  const truncated = result.spec.type === "bar" && (result.chart_data as BarChartData).truncated;
  const range = result.display_range ?? null;
  const excluded = range ? range.excluded_below + range.excluded_above : 0;

  return (
    <div>
      <Plot
        data={plot.data as Data[]}
        layout={{ ...plot.layout, autosize: true } as Partial<Layout>}
        useResizeHandler
        style={{ width: "100%", height: "360px" }}
        config={{ displayModeBar: false }}
      />
      <div className="flex gap-3 px-1 text-xs text-slate-400">
        {result.sampled && <span>已抽樣（顯示 {result.n_points} 點）</span>}
        {truncated && <span>類別過多，僅顯示前 {result.spec.top_n ?? 20} 名</span>}
        {range && (
          <span className="text-amber-700" title={range.reason}>
            顯示範圍 {range.lo}–{range.hi}；另有 {excluded} 筆極端值未在圖中（{range.reason}）
          </span>
        )}
      </div>
    </div>
  );
}
