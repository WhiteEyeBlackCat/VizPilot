import { ChevronLeft, ChevronRight, Maximize2, Trash2, X } from "lucide-react";
import { useState } from "react";

import { hideAllTips } from "../charts/echarts/EChartsView";
import type { RenderResult } from "../types";
import { ChartView } from "./ChartView";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { useElementSize } from "@/lib/hooks";
import { cn } from "@/lib/utils";

export interface WorkspaceChart {
  key: string;
  result: RenderResult;
}

interface Props {
  charts: WorkspaceChart[];
  currentKey: string | null;
  onSelect: (key: string) => void;
  onRemove: (key: string) => void;
  onClear: () => void;
}

// ChartView draws a note row under the chart (up to two lines for the
// display_range message); reserve it when sizing the ECharts host to the
// pane (ChartView itself is untouched). The area clips overflow so the
// pane's size never depends on its content.
const NOTE_ROW_PX = 40;

/** RStudio-style plots pane: one chart at a time, ◀ ▶ history navigation,
 *  a thumbnail strip of titles, enlarge / remove / clear. Shared by every
 *  page that can generate charts — the workspace is global. */
export function PlotsPane({ charts, currentKey, onSelect, onRemove, onClear }: Props) {
  const [enlarged, setEnlarged] = useState<WorkspaceChart | null>(null);
  const [areaRef, area] = useElementSize<HTMLDivElement>();

  const index = charts.findIndex((c) => c.key === currentKey);
  const current = index >= 0 ? charts[index] : null;
  const count = charts.length;
  const chartHeight = Math.max(200, area.height - NOTE_ROW_PX);

  const go = (delta: number) => {
    if (count === 0) return;
    const next = ((index < 0 ? 0 : index) + delta + count) % count;
    onSelect(charts[next].key);
  };

  return (
    <div
      className="flex h-full min-h-0 flex-col overflow-hidden rounded-lg border bg-card"
      data-plots-pane
      data-count={count}
      data-current={index >= 0 ? index + 1 : 0}
    >
      <div className="flex items-center justify-between gap-2 border-b px-2 py-1.5">
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="sm"
            className="h-7 w-7 p-0"
            aria-label="上一張"
            disabled={count < 2}
            onClick={() => go(-1)}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="h-7 w-7 p-0"
            aria-label="下一張"
            disabled={count < 2}
            onClick={() => go(1)}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
          <span className="ml-1 text-xs tabular-nums text-muted-foreground" data-plots-counter>
            {count === 0 ? "0 / 0" : `${index + 1} / ${count}`}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="sm"
            className="h-7 px-2 text-xs text-muted-foreground"
            disabled={!current}
            onClick={() => {
              if (!current) return;
              hideAllTips(); // a hovered inline tooltip must not float above the overlay
              setEnlarged(current);
            }}
          >
            <Maximize2 className="mr-1 h-3.5 w-3.5" />
            放大
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="h-7 px-2 text-xs text-muted-foreground hover:text-destructive"
            disabled={!current}
            onClick={() => current && onRemove(current.key)}
          >
            <X className="mr-1 h-3.5 w-3.5" />
            移除
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="h-7 px-2 text-xs text-muted-foreground hover:text-destructive"
            disabled={count === 0}
            onClick={onClear}
          >
            <Trash2 className="mr-1 h-3.5 w-3.5" />
            清空
          </Button>
        </div>
      </div>

      <div ref={areaRef} className="min-h-0 flex-1 overflow-hidden p-2">
        {current ? (
          <ChartView key={current.key} result={current.result} height={chartHeight} />
        ) : (
          <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
            尚未生成圖表——在左側按「Generate」或「生成圖表」
          </div>
        )}
      </div>

      {count > 0 && (
        <div className="flex gap-1.5 overflow-x-auto border-t px-2 py-1.5" data-plots-strip>
          {charts.map((c, i) => (
            <button
              key={c.key}
              type="button"
              className={cn(
                "flex shrink-0 items-center gap-1.5 rounded-md border px-2 py-1 text-left text-xs transition-colors hover:bg-accent",
                c.key === currentKey ? "border-primary bg-accent" : "border-border",
              )}
              onClick={() => onSelect(c.key)}
              aria-current={c.key === currentKey ? "true" : undefined}
            >
              <span className="tabular-nums text-muted-foreground">{i + 1}</span>
              <span className="max-w-[10rem] truncate">{c.result.spec.title}</span>
              <Badge variant="muted" className="font-normal">
                {c.result.spec.type}
              </Badge>
            </button>
          ))}
        </div>
      )}

      <Dialog open={enlarged !== null} onOpenChange={(open) => !open && setEnlarged(null)}>
        <DialogContent className="max-w-[90vw] p-4 pt-10" data-chart-dialog>
          {/* the chart draws its own title; keep the accessible name for the dialog */}
          <DialogTitle className="sr-only">{enlarged?.result.spec.title ?? "圖表"}</DialogTitle>
          <DialogDescription className="sr-only">放大檢視圖表</DialogDescription>
          {enlarged && <ChartView result={enlarged.result} height="70vh" />}
        </DialogContent>
      </Dialog>
    </div>
  );
}
