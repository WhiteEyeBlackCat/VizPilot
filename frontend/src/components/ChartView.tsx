import { useMemo } from "react";

import { EChartsView } from "../charts/echarts/EChartsView";
import { buildOption, isEmpty } from "../charts/echarts/option";
import type { VizOption } from "../charts/echarts/echarts";
import type { BarChartData, RenderResult } from "../types";

const CHART_HEIGHT = 360;
// the title row the theme reserves (grid.top 56) shrinks to the y-axis name
// row when the surrounding UI already names the chart
const HIDDEN_TITLE_GRID_TOP = 28;

interface Props {
  result: RenderResult;
  /** px number or any CSS height (the enlarged dialog passes a vh value) */
  height?: number | string;
  /** false: hide the in-canvas title (the preview panel header shows it) */
  showTitle?: boolean;
}

/** Presentation-only override on top of the adapter's option: the data
 *  mapping in option.ts is untouched. */
function withoutTitle(option: VizOption): VizOption {
  const title = Array.isArray(option.title)
    ? option.title.map((t) => ({ ...t, show: false }))
    : { ...(option.title ?? {}), show: false };
  const grid =
    option.grid && !Array.isArray(option.grid) ? { ...option.grid, top: HIDDEN_TITLE_GRID_TOP } : option.grid;
  return { ...option, title, grid };
}

export function ChartView({ result, height = CHART_HEIGHT, showTitle = true }: Props) {
  // the option is derived from the immutable RenderResult: memoised per chart
  const option = useMemo(() => {
    if (isEmpty(result)) return null;
    const built = buildOption(result);
    return showTitle ? built : withoutTitle(built);
  }, [result, showTitle]);

  if (!option) {
    return (
      <div className="flex h-64 items-center justify-center rounded-md bg-muted/40 text-sm text-muted-foreground">
        此組合無資料
      </div>
    );
  }

  const truncated = result.spec.type === "bar" && (result.chart_data as BarChartData).truncated;
  const range = result.display_range ?? null;
  const excluded = range ? range.excluded_below + range.excluded_above : 0;

  return (
    <div data-chart-title={showTitle ? "shown" : "hidden"}>
      <EChartsView option={option} height={height} />
      <div className="flex gap-3 px-1 text-xs text-muted-foreground">
        {result.sampled && <span>已抽樣（顯示 {result.n_points} 點）</span>}
        {truncated && <span>類別過多，僅顯示前 {result.spec.top_n ?? 20} 名</span>}
        {range && (
          <span className="text-warning" title={range.reason}>
            顯示範圍 {range.lo}–{range.hi}；另有 {excluded} 筆極端值未在圖中（{range.reason}）
          </span>
        )}
      </div>
    </div>
  );
}
