import { Bookmark, ChevronDown, ChevronUp, Download, Maximize2, PanelRightClose, Trash2, X } from "lucide-react";
import { useRef, useState } from "react";

import { echarts } from "../charts/echarts/echarts";
import { hideAllTips } from "../charts/echarts/EChartsView";
import { COLORS } from "../charts/echarts/theme";
import type { Preview } from "../store";
import type { ChartSpec, Recommendation, Warning, WarningSeverity } from "../types";
import { INTENT_LABEL } from "./intentLabels";
import { ChartView } from "./ChartView";
import { evidenceRows } from "./InsightList";
import { specRows } from "./Recommendations";
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
  /** stage 19: render another eligible chart of the same finding into the panel */
  onPreviewSpec?: (spec: ChartSpec) => Promise<void>;
}

const WARNING_VARIANT: Record<WarningSeverity, BadgeVariant> = {
  severe: "danger",
  warning: "warning",
  info: "muted",
};

// ChartView draws a note row under the chart (order / sampling / display
// range); reserve it when sizing the chart to its area.
const NOTE_ROW_PX = 40;
// the chart takes about two thirds of the panel; the rest is the analysis
// (why it matters, evidence, actions) — never a chart wall with a caption
const CHART_SHARE = 0.68; // ≈ 62% of the whole panel once header, note row and actions are counted
const HEADER_PX = 40;
const ACTIONS_PX = 48;

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
 *  Explore result, or of a saved workspace chart — with the analysis that
 *  goes with it (stage 18: analysis detail panel). Save is the only way a
 *  chart enters the workspace. */
export function PreviewPanel({ preview, saved, variant, mode, open, onToggle, onClose, onSave, onRemove, onPreviewSpec }: Props) {
  const [rootRef, root] = useElementSize<HTMLElement>();
  const [enlarged, setEnlarged] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const chartHostRef = useRef<HTMLDivElement>(null);

  const { result, source, rec, insightText } = preview;
  const insight = preview.insight ?? null;
  const spec = result.spec;
  const usable = Math.max(0, root.height - HEADER_PX - ACTIONS_PX);
  const chartHeight = Math.max(200, Math.round(usable * CHART_SHARE) - NOTE_ROW_PX);
  const warnings: Warning[] = rec?.warnings ?? [];
  const confidence = rec ? confidenceLine(rec) : null;
  // context line under the title: the finding's headline when it has one,
  // else the recommendation's analytical purpose, else the manual origin
  // (a headline-less insight is shown once, in the analysis section)
  const subtitle = insight?.headline?.trim() || rec?.spec.reason || (source === "explore" ? "手動建圖" : null);
  const evidence = insight ? evidenceRows(insight) : [];
  // the full finding: an insight's text, or the legacy insightText of a
  // saved chart from before stage 18
  const detailText = insight ? insight.text : (insightText ?? null);
  // stage 19: the other eligible charts of this finding (the recommendation
  // lists them; the one on screen may itself be an alternative)
  const family = rec ? [rec.spec, ...(rec.alternatives ?? [])] : [];
  const alternatives = family.filter((alt) => alt.type !== spec.type || alt.x !== spec.x || alt.y !== spec.y || alt.group_by !== spec.group_by);
  const intentLabel = rec?.intent ? INTENT_LABEL[rec.intent] ?? rec.intent : null;

  const chartInstance = () => {
    const host = chartHostRef.current?.querySelector<HTMLDivElement>("[data-chart-view]");
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
      aria-label="分析面板"
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
        variant === "drawer" && (open ? "h-[60vh]" : "h-10"),
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
        <h2 className="min-w-0 flex-1 truncate text-sm font-semibold tracking-tight" title={spec.title}>
          {spec.title}
        </h2>
        <Badge variant="muted" className="font-normal">
          {spec.type}
        </Badge>
        {intentLabel && (
          <Badge variant="info" className="hidden font-normal sm:inline-flex" data-preview-intent={rec?.intent}>
            {intentLabel}
          </Badge>
        )}
        {rec?.source === "llm" && (
          <Badge variant="accent" className="font-normal">
            AI
          </Badge>
        )}
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
          {subtitle && (
            <p className="line-clamp-2 shrink-0 px-4 pb-1 text-xs leading-snug text-muted-foreground" title={subtitle} data-preview-description>
              {subtitle}
            </p>
          )}

          <div ref={chartHostRef} className="shrink-0 px-2" data-preview-chart>
            {/* the header names the chart: no second title inside the canvas */}
            <ChartView key={preview.seq} result={result} height={chartHeight} showTitle={false} />
          </div>

          <div className="min-h-0 flex-1 space-y-3 overflow-y-auto border-t border-subtle px-4 py-3 text-xs" data-preview-analysis>
            {insight && (
              <section data-preview-insight>
                <h3 className="mb-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                  {insight.why_it_matters ? "Why this matters" : "Finding"}
                </h3>
                {/* plain text only — LLM free text must never be rendered as HTML */}
                <p className="leading-relaxed text-foreground/90">{insight.text}</p>
                {insight.why_it_matters && (
                  <p className="mt-1 leading-relaxed text-muted-foreground" data-preview-why>
                    {insight.why_it_matters}
                  </p>
                )}
              </section>
            )}
            {!insight && detailText && (
              <p className="border-l-2 border-primary pl-2 leading-relaxed text-foreground/90" data-preview-insight>
                {detailText}
              </p>
            )}

            <section data-preview-evidence>
              <h3 className="mb-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Supporting evidence</h3>
              {evidence.length > 0 && (
                <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-0.5">
                  {evidence.map((row) => (
                    <div key={row.label} className="contents">
                      <dt className="text-muted-foreground">{row.label}</dt>
                      <dd className="tabular-nums text-foreground/90">{row.value}</dd>
                    </div>
                  ))}
                </dl>
              )}
              {confidence && (
                <p className={cn("tabular-nums text-muted-foreground", evidence.length > 0 && "mt-1")} data-preview-confidence>
                  {confidence}
                </p>
              )}
              {!confidence && evidence.length === 0 && (
                <p className="text-muted-foreground/80">手動建圖沒有推薦信心分數；圖中的統計由後端聚合。</p>
              )}
              {warnings.length > 0 && (
                <ul className="mt-1.5 flex flex-wrap gap-1" data-preview-warnings>
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
            </section>

            {alternatives.length > 0 && onPreviewSpec && (
              <section data-preview-alternatives>
                <h3 className="mb-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">Alternative views</h3>
                <div className="flex flex-wrap gap-1">
                  {alternatives.map((alt) => (
                    <Button
                      key={`${alt.type}-${alt.x}-${alt.y}-${alt.group_by}`}
                      variant="outline"
                      size="sm"
                      className="h-6 px-2 text-xs"
                      onClick={() => void onPreviewSpec(alt)}
                      data-alternative={alt.type}
                    >
                      {alt.type}
                    </Button>
                  ))}
                </div>
              </section>
            )}

            <details className="group text-muted-foreground/80" data-preview-technical>
              <summary className="cursor-pointer select-none text-[11px] uppercase tracking-wide hover:text-muted-foreground">
                Chart spec
              </summary>
              <p className="mt-1 font-mono text-[11px]" data-preview-variables>
                {variables(spec)}
              </p>
              <dl className="mt-1 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 font-mono text-[11px]">
                {specRows(spec).map((r) => (
                  <div key={r.role} className="contents">
                    <dt>{r.role}</dt>
                    <dd className="text-foreground/80">{r.value}</dd>
                  </div>
                ))}
              </dl>
            </details>
          </div>

          <div className="flex shrink-0 flex-wrap items-center gap-1.5 border-t border-subtle px-3 py-2" data-preview-actions>
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
            <Button variant="ghost" size="sm" className="ml-auto h-7 px-2.5 text-xs text-muted-foreground" onClick={onClose}>
              <X className="mr-1 h-3.5 w-3.5" />
              Close
            </Button>
            {exportError && <span className="basis-full text-danger">{exportError}</span>}
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
