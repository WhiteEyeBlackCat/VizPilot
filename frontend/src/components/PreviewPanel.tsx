import { Bookmark, ChevronDown, ChevronUp, Download, Maximize2, PanelRightClose, Trash2, X } from "lucide-react";
import { useRef, useState } from "react";

import { echarts } from "../charts/echarts/echarts";
import { hideAllTips } from "../charts/echarts/EChartsView";
import { COLORS } from "../charts/echarts/theme";
import type { Preview } from "../store";
import type { ChartSpec, Recommendation, Warning, WarningSeverity } from "../types";
import { ChartView } from "./ChartView";
import { Badge, type BadgeVariant } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { useElementSize } from "@/lib/hooks";
import { cn } from "@/lib/utils";

interface Props {
  preview: Preview;
  /** the previewed spec is already in the workspace */
  saved: boolean;
  /** "side": right column; "drawer": bottom sheet on narrow viewports */
  variant: "side" | "drawer";
  /** how the side variant is placed: resizable split (≥1280px) or an overlay
   *  over the content (1024–1279px); informational (data attribute) */
  mode?: "split" | "overlay" | "drawer";
  /** drawer only: expanded or collapsed to its header */
  open: boolean;
  onToggle: () => void;
  onClose: () => void;
  onSave: () => void;
  onRemove: () => void;
}

const WARNING_VARIANT: Record<WarningSeverity, BadgeVariant> = {
  severe: "danger",
  warning: "warning",
  info: "muted",
};

// ChartView draws a note row under the chart (up to two lines for the
// display_range message); reserve it when sizing the chart to the area.
const NOTE_ROW_PX = 40;

function variables(spec: ChartSpec): string {
  const parts: string[] = [];
  if (spec.x) parts.push(`x = ${spec.x}`);
  if (spec.y) parts.push(`y = ${spec.y}`);
  if (spec.group_by) parts.push(`group = ${spec.group_by}`);
  if (spec.aggregation) parts.push(`agg = ${spec.aggregation}`);
  return parts.join(" · ") || "全部數值欄";
}

function confidenceLine(rec: Recommendation): string | null {
  const c = rec.confidence;
  if (!c) return null;
  return (
    `信心 ${Math.round(c.overall * 100)}% · 樣本 ${Math.round(c.sample_size * 100)}%` +
    ` · 缺失 ${Math.round(c.missingness * 100)}% · 使用 ${c.n_effective}/${c.n_total} 列` +
    `（${c.n_source === "exact" ? "精確" : "估計"}）`
  );
}

function safeFileName(title: string): string {
  return (title.replace(/[\\/:*?"<>|]+/g, "_").trim() || "chart").slice(0, 80);
}

/** The one place a chart is drawn: preview of a recommendation, of an
 *  Explore result, or of a saved workspace chart. Save is the only way a
 *  chart enters the workspace. */
export function PreviewPanel({ preview, saved, variant, mode, open, onToggle, onClose, onSave, onRemove }: Props) {
  const rootRef = useRef<HTMLElement>(null);
  const [areaRef, area] = useElementSize<HTMLDivElement>();
  const [enlarged, setEnlarged] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);

  const { result, source, rec, insightText } = preview;
  const spec = result.spec;
  const chartHeight = Math.max(220, area.height - NOTE_ROW_PX);
  const warnings: Warning[] = rec?.warnings ?? [];
  const confidence = rec ? confidenceLine(rec) : null;
  // the variables line below already states the spec; only a recommendation
  // adds a reason worth repeating
  const description = rec?.spec.reason ?? (source === "explore" ? "手動建圖" : null);

  const chartInstance = () => {
    const host = rootRef.current?.querySelector<HTMLDivElement>("[data-chart-view]");
    return host ? echarts.getInstanceByDom(host) : undefined;
  };

  const exportPng = () => {
    const chart = chartInstance();
    if (!chart) {
      setExportError("圖表尚未就緒");
      return;
    }
    setExportError(null);
    const url = chart.getDataURL({ type: "png", pixelRatio: 2, backgroundColor: COLORS.surface });
    const a = document.createElement("a");
    a.href = url;
    a.download = `${safeFileName(spec.title)}.png`;
    a.rel = "noopener";
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  const collapsedDrawer = variant === "drawer" && !open;

  return (
    <aside
      ref={rootRef}
      aria-label="預覽面板"
      data-preview-panel
      data-variant={variant}
      data-mode={mode ?? (variant === "drawer" ? "drawer" : "split")}
      data-source={source}
      data-seq={preview.seq}
      data-title={spec.title}
      data-open={open ? "true" : "false"}
      className={cn(
        "flex min-h-0 flex-col bg-elevated",
        variant === "side" ? "h-full border-l border-subtle" : "border-t border-subtle",
        variant === "drawer" && (open ? "h-[55vh]" : "h-10"),
      )}
    >
      <header className="flex h-10 shrink-0 items-center gap-2 px-3">
        {variant === "drawer" && (
          <Button
            variant="ghost"
            size="sm"
            className="h-7 w-7 p-0"
            aria-label={open ? "收合預覽面板" : "展開預覽面板"}
            onClick={onToggle}
          >
            {open ? <ChevronDown className="h-4 w-4" /> : <ChevronUp className="h-4 w-4" />}
          </Button>
        )}
        <h2 className="min-w-0 flex-1 truncate text-sm font-medium" title={spec.title}>
          {spec.title}
        </h2>
        <Badge variant="muted" className="font-normal">
          {spec.type}
        </Badge>
        {variant === "side" && (
          <Button
            variant="ghost"
            size="sm"
            className="h-7 w-7 p-0"
            aria-label="收合預覽面板"
            onClick={onToggle}
          >
            <PanelRightClose className="h-4 w-4" />
          </Button>
        )}
        <Button variant="ghost" size="sm" className="h-7 w-7 p-0" aria-label="關閉預覽" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
      </header>

      {!collapsedDrawer && (
        <>
          <div ref={areaRef} className="min-h-0 flex-1 overflow-hidden px-2" data-preview-chart>
            {/* the header names the chart: no second title inside the canvas */}
            <ChartView key={preview.seq} result={result} height={chartHeight} showTitle={false} />
          </div>

          <div className="max-h-[42%] shrink-0 space-y-2 overflow-y-auto border-t border-subtle px-4 py-3 text-xs">
            {description && (
              <p className="text-muted-foreground" data-preview-description>
                {description}
              </p>
            )}
            {insightText && (
              // plain text only — LLM free text must never be rendered as HTML
              <p className="border-l-2 border-primary pl-2 text-foreground/90" data-preview-insight>
                {insightText}
              </p>
            )}
            <p className="text-muted-foreground/80" data-preview-variables>
              {variables(spec)}
              {rec?.source === "llm" && <span className="ml-2 text-ring">AI</span>}
            </p>
            {confidence && (
              <p className="tabular-nums text-muted-foreground" data-preview-confidence>
                {confidence}
              </p>
            )}
            {warnings.length > 0 && (
              <ul className="flex flex-wrap gap-1" data-preview-warnings>
                {warnings.map((w, i) => (
                  <li key={i}>
                    <Badge
                      variant={WARNING_VARIANT[w.severity] ?? "muted"}
                      className="whitespace-normal text-left font-normal leading-snug"
                    >
                      {w.message}
                    </Badge>
                  </li>
                ))}
              </ul>
            )}
            <div className="flex flex-wrap items-center gap-1.5 pt-1" data-preview-actions>
              {source === "workspace" ? (
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 px-2.5 text-xs hover:text-danger"
                  onClick={onRemove}
                >
                  <Trash2 className="mr-1 h-3.5 w-3.5" />
                  Remove
                </Button>
              ) : (
                <Button
                  size="sm"
                  className="h-7 px-2.5 text-xs"
                  disabled={saved}
                  onClick={onSave}
                  data-preview-save
                >
                  <Bookmark className="mr-1 h-3.5 w-3.5" />
                  {saved ? "已保存" : "Save"}
                </Button>
              )}
              <Button variant="outline" size="sm" className="h-7 px-2.5 text-xs" onClick={exportPng}>
                <Download className="mr-1 h-3.5 w-3.5" />
                Export
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="h-7 px-2.5 text-xs"
                onClick={() => {
                  hideAllTips(); // a hovered tooltip must not float above the overlay
                  setEnlarged(true);
                }}
              >
                <Maximize2 className="mr-1 h-3.5 w-3.5" />
                Enlarge
              </Button>
              {exportError && <span className="text-danger">{exportError}</span>}
            </div>
          </div>
        </>
      )}

      <Dialog open={enlarged} onOpenChange={(next) => !next && setEnlarged(false)}>
        <DialogContent className="max-w-[90vw] p-4 pt-10" data-chart-dialog>
          {/* the chart draws its own title here; keep the accessible name for the dialog */}
          <DialogTitle className="sr-only">{spec.title}</DialogTitle>
          <DialogDescription className="sr-only">放大檢視圖表</DialogDescription>
          {enlarged && <ChartView result={result} height="70vh" />}
        </DialogContent>
      </Dialog>
    </aside>
  );
}
