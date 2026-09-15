// Single source of visual style for every ECharts chart: palette, fonts,
// grid, axes, tooltip, toolbox. Chart builders in option.ts compose these
// and never restate colours or sizes.
//
// Dark theme (stage 16.1). The hex values mirror the design tokens in
// src/index.css (--chart-1..8, --surface, --elevated, --border-subtle,
// --danger …); theme.test.ts fails when they drift apart. The
// chart background stays transparent so a chart takes the surface it sits
// on (card, preview panel, dialog).

import type {
  DataZoomComponentOption,
  GridComponentOption,
  LegendComponentOption,
  TitleComponentOption,
  ToolboxComponentOption,
  TooltipComponentOption,
} from "echarts/components";

// 400-weight hues that read on dark surfaces (each ≥ 3:1 against --surface),
// ordered for maximum separation between neighbours; readable for the
// common red-green deficiencies because neighbours also differ in luminance.
export const PALETTE = [
  "#60a5fa", // blue-400
  "#fbbf24", // amber-400
  "#34d399", // emerald-400
  "#a78bfa", // violet-400
  "#f87171", // red-400
  "#22d3ee", // cyan-400
  "#e879f9", // fuchsia-400
  "#94a3b8", // slate-400
];

export const COLORS = {
  surface: "#171b21", // --surface: card / sidebar
  elevated: "#20242c", // --elevated: tooltip, popover, preview panel
  text: "#e6e9ef", // --foreground: near-white
  muted: "#9ca3b0", // --muted-foreground: cool grey
  axisLine: "#3a4150", // one step above the border: the axis must be visible
  splitLine: "#23272f", // --border-subtle: low-contrast grid
  tooltipBorder: "#2c3140", // --border
  outlier: "#f87171", // --danger
  mean: "#e6e9ef", // diamond marker on the box plot
  boxFill: "#1e3a5f", // deep blue tint under the PALETTE[0] border
  cellBorder: "#171b21", // heatmap cell separator = surface, so cells float
};

// diverging correlation scale for a dark surface: negative blue / zero a
// neutral one step above the surface / positive red. Both ends sit in the
// luminance band that is ≥ 3:1 against the surface AND keeps the near-white
// cell labels ≥ 3:1 on top of them. (visualMap min/max live in option.ts.)
export const DIVERGING = ["#3b82f6", "#2a3040", "#ef4444"];

export const FONT_FAMILY =
  'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Noto Sans TC", sans-serif';

export const textStyle = { fontFamily: FONT_FAMILY, color: COLORS.text };

export function title(text: string): TitleComponentOption {
  return {
    text,
    left: 8,
    top: 6,
    textStyle: { fontSize: 14, fontWeight: 600, color: COLORS.text, fontFamily: FONT_FAMILY },
  };
}

// room for the title row above, the y-axis name on the left and the x-axis
// name below. ECharts 6 default outerBounds layout (outerBoundsMode "auto",
// outerBoundsContain "all") keeps axis labels AND axis names inside the
// canvas; the deprecated containLabel is deliberately not used.
export const grid: GridComponentOption = {
  left: 16,
  right: 24,
  top: 56, // title row (top 6, 14px) + a clear gap above the y-axis name
  bottom: 40,
};

/** A bottom legend needs one more row under the x-axis name. */
export const gridWithLegend: GridComponentOption = { ...grid, bottom: 64 };

/** Extra right margin when a visualMap bar sits beside the plot. */
export const gridWithVisualMap: GridComponentOption = { ...grid, right: 72 };

const axisLabel = { color: COLORS.muted, fontSize: 12, fontFamily: FONT_FAMILY };
const axisName = { color: COLORS.muted, fontSize: 12, fontFamily: FONT_FAMILY };

export const axisBase = {
  axisLine: { lineStyle: { color: COLORS.axisLine } },
  axisTick: { lineStyle: { color: COLORS.axisLine } },
  axisLabel,
  nameTextStyle: axisName,
  splitLine: { lineStyle: { color: COLORS.splitLine } },
};

/** y-axis name rendered at the top-left, x-axis name centred below. */
export const yAxisName = (name: string) => ({
  name,
  nameLocation: "end" as const,
  nameGap: 12,
  nameTextStyle: { ...axisName, align: "left" as const, padding: [0, 0, 0, -4] },
});
export const xAxisName = (name: string) => ({
  name,
  nameLocation: "middle" as const,
  nameGap: 28,
});

export const tooltipBase: TooltipComponentOption = {
  backgroundColor: COLORS.elevated,
  borderColor: COLORS.tooltipBorder,
  borderWidth: 1,
  padding: [6, 10],
  textStyle: { color: COLORS.text, fontSize: 12, fontFamily: FONT_FAMILY },
  extraCssText: "box-shadow: 0 6px 20px rgba(0, 0, 0, 0.45); border-radius: 6px;",
  confine: true,
};

// below the plot, centred: never collides with the title or the toolbox
export const legend: LegendComponentOption = {
  type: "scroll",
  bottom: 4,
  left: "center",
  icon: "roundRect",
  itemWidth: 12,
  itemHeight: 8,
  textStyle: { color: COLORS.muted, fontSize: 12, fontFamily: FONT_FAMILY },
  pageTextStyle: { color: COLORS.muted },
  pageIconColor: COLORS.muted,
  pageIconInactiveColor: COLORS.splitLine,
};

/** Toolbox: image export always; zoom/restore only for continuous axes. */
export function toolbox(imageName: string, zoomable: boolean): ToolboxComponentOption {
  return {
    right: 136, // clear of the workspace card's 放大 / 移除 buttons
    top: 4,
    itemSize: 14,
    iconStyle: { borderColor: COLORS.muted },
    emphasis: { iconStyle: { borderColor: PALETTE[0] } },
    feature: {
      // export on the surface colour: a transparent PNG would be unreadable
      // on a light viewer
      saveAsImage: { name: imageName, title: "存成圖片", pixelRatio: 2, backgroundColor: COLORS.surface },
      ...(zoomable
        ? {
            dataZoom: {
              title: { zoom: "框選縮放", back: "還原縮放" },
              yAxisIndex: "none",
              brushStyle: { color: "rgba(96, 165, 250, 0.15)", borderColor: PALETTE[0] },
            },
            restore: { title: "重設" },
          }
        : {}),
    },
  };
}

/** Wheel / drag zoom on the given axes (no slider: keeps the card compact). */
export function dataZoomInside(axes: ("x" | "y")[]): DataZoomComponentOption[] {
  return axes.map((axis) => ({
    type: "inside",
    ...(axis === "x" ? { xAxisIndex: 0 } : { yAxisIndex: 0 }),
    filterMode: "none",
  }));
}

/** Tooltip number formatting shared by axis-trigger tooltips. */
export const tooltipValue = (v: unknown): string =>
  typeof v === "number" ? formatNumber(v) : v === null || v === undefined ? "—" : String(v);

export const formatNumber = (v: number): string =>
  Number.isInteger(v) ? String(v) : Math.abs(v) >= 1000 ? v.toFixed(1) : v.toFixed(2);
