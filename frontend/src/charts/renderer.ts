// Which chart library draws a RenderResult. ECharts is the default; Plotly
// stays available as a fallback during the migration (stage 11) via
//   ?renderer=plotly            (this page load)
//   localStorage["vizpilot.renderer"] = "plotly"   (persistent)

export type Renderer = "echarts" | "plotly";

export const RENDERER_STORAGE_KEY = "vizpilot.renderer";

const isRenderer = (v: unknown): v is Renderer => v === "echarts" || v === "plotly";

export function getRenderer(): Renderer {
  try {
    const fromUrl = new URLSearchParams(window.location.search).get("renderer");
    if (isRenderer(fromUrl)) return fromUrl;
    const stored = window.localStorage.getItem(RENDERER_STORAGE_KEY);
    if (isRenderer(stored)) return stored;
  } catch {
    // no window / storage blocked: default below
  }
  return "echarts";
}
