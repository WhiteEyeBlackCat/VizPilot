// Single source of visual style for every ECharts chart: palette, fonts,
// grid, axes, tooltip, toolbox. Chart builders in option.ts compose these
// and never restate colours or sizes.

import type {
  DataZoomComponentOption,
  GridComponentOption,
  LegendComponentOption,
  TitleComponentOption,
  ToolboxComponentOption,
  TooltipComponentOption,
} from "echarts/components";

// Tailwind 500/600 hues that the shadcn Badge variants already use (info
// blue, warning amber, success emerald, accent violet, danger red), ordered
// for maximum separation between neighbours; readable for the common
// red-green deficiencies because neighbours also differ in luminance.
export const PALETTE = [
  "#2563eb", // blue-600
  "#f59e0b", // amber-500
  "#10b981", // emerald-500
  "#8b5cf6", // violet-500
  "#ef4444", // red-500
  "#0891b2", // cyan-600
  "#d946ef", // fuchsia-500
  "#64748b", // slate-500
];

export const COLORS = {
  text: "#1e293b", // slate-800
  muted: "#64748b", // slate-500
  axisLine: "#cbd5e1", // slate-300
  splitLine: "#f1f5f9", // slate-100
  tooltipBorder: "#e2e8f0", // slate-200
  outlier: "#ef4444", // red-500
  mean: "#1e293b",
};

// diverging correlation scale: negative blue / zero light / positive red
export const DIVERGING = ["#2166ac", "#f7f7f7", "#b2182b"];

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
  backgroundColor: "#ffffff",
  borderColor: COLORS.tooltipBorder,
  borderWidth: 1,
  padding: [6, 10],
  textStyle: { color: COLORS.text, fontSize: 12, fontFamily: FONT_FAMILY },
  extraCssText: "box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08); border-radius: 6px;",
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
      saveAsImage: { name: imageName, title: "存成圖片", pixelRatio: 2 },
      ...(zoomable
        ? {
            dataZoom: { title: { zoom: "框選縮放", back: "還原縮放" }, yAxisIndex: "none" },
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
