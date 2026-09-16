// The one place ECharts modules are registered (tree-shaken build): every
// series type the ChartSpec types need plus the shared components (the
// stage-19 family draws with custom / bar / line / scatter series and polar).
// Import `echarts` from here, never from "echarts" directly.

import {
  BarChart,
  BoxplotChart,
  CustomChart,
  HeatmapChart,
  LineChart,
  ScatterChart,
} from "echarts/charts";
import type {
  BarSeriesOption,
  BoxplotSeriesOption,
  CustomSeriesOption,
  HeatmapSeriesOption,
  LineSeriesOption,
  ScatterSeriesOption,
} from "echarts/charts";
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  PolarComponent,
  TitleComponent,
  ToolboxComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import type {
  DataZoomComponentOption,
  GridComponentOption,
  LegendComponentOption,
  PolarComponentOption,
  TitleComponentOption,
  ToolboxComponentOption,
  TooltipComponentOption,
  VisualMapComponentOption,
} from "echarts/components";
// the polar axis option types are not exported on their own: take them
// from the full option type (type-only import, no runtime cost)
import type { EChartsOption } from "echarts";

type AngleAxisOption = NonNullable<EChartsOption["angleAxis"]>;
type RadiusAxisOption = NonNullable<EChartsOption["radiusAxis"]>;
import * as echarts from "echarts/core";
import type { ComposeOption } from "echarts/core";
import { CanvasRenderer, SVGRenderer } from "echarts/renderers";

echarts.use([
  LineChart,
  BarChart,
  ScatterChart,
  BoxplotChart,
  HeatmapChart,
  CustomChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  DataZoomComponent,
  ToolboxComponent,
  VisualMapComponent,
  TitleComponent,
  PolarComponent, // stage 19: polar_line / rose
  CanvasRenderer, // browser
  SVGRenderer, // server-side rendering in tests (renderToSVGString)
]);

/** The option type this app can produce: only the registered pieces. */
export type VizOption = ComposeOption<
  | LineSeriesOption
  | BarSeriesOption
  | ScatterSeriesOption
  | BoxplotSeriesOption
  | HeatmapSeriesOption
  | CustomSeriesOption
  | GridComponentOption
  | TooltipComponentOption
  | LegendComponentOption
  | DataZoomComponentOption
  | ToolboxComponentOption
  | VisualMapComponentOption
  | TitleComponentOption
  | PolarComponentOption
> & { angleAxis?: AngleAxisOption; radiusAxis?: RadiusAxisOption };

export type { ECharts } from "echarts/core";
export { echarts };
