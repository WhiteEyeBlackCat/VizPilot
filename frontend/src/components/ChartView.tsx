import { useMemo } from "react";

import { EChartsView } from "../charts/echarts/EChartsView";
import { buildOption, isEmpty } from "../charts/echarts/option";
import { PlotlyView } from "../charts/plotly/PlotlyView";
import { getRenderer } from "../charts/renderer";
import type { BarChartData, RenderResult } from "../types";

const CHART_HEIGHT = 360;

export function ChartView({ result }: { result: RenderResult }) {
  // decided once per page load; the option is derived from the immutable
  // RenderResult so it is memoised per chart
  const renderer = useMemo(getRenderer, []);
  const option = useMemo(
    () => (renderer === "echarts" && !isEmpty(result) ? buildOption(result) : null),
    [renderer, result],
  );

  if (isEmpty(result)) {
    return (
      <div className="flex h-64 items-center justify-center rounded bg-slate-50 text-sm text-slate-400">
        此組合無資料
      </div>
    );
  }

  const truncated = result.spec.type === "bar" && (result.chart_data as BarChartData).truncated;
  const range = result.display_range ?? null;
  const excluded = range ? range.excluded_below + range.excluded_above : 0;

  return (
    <div>
      {option ? (
        <EChartsView option={option} height={CHART_HEIGHT} />
      ) : (
        <PlotlyView result={result} height={CHART_HEIGHT} />
      )}
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
