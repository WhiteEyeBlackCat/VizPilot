import { Trash2 } from "lucide-react";
import { useEffect, useRef } from "react";

import { PageHeader } from "../components/PageHeader";
import type { SavedChart } from "../store";
import type { ChartSpec } from "../types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface Props {
  charts: SavedChart[];
  /** key of the saved chart currently shown in the preview panel */
  currentKey: string | null;
  onPreview: (chart: SavedChart) => void;
  onRemove: (key: string) => void;
}

function variables(spec: ChartSpec): string {
  const parts: string[] = [];
  if (spec.x) parts.push(`x=${spec.x}`);
  if (spec.y) parts.push(`y=${spec.y}`);
  if (spec.group_by) parts.push(`group=${spec.group_by}`);
  if (spec.aggregation) parts.push(`agg=${spec.aggregation}`);
  return parts.join(" · ") || "—";
}

const time = new Intl.DateTimeFormat("zh-Hant", { hour: "2-digit", minute: "2-digit", second: "2-digit" });

/** Only charts the user explicitly saved. Enlarge / Export live in the
 *  preview panel, so a card just selects. */
export function WorkspacePage({ charts, currentKey, onPreview, onRemove }: Props) {
  const listRef = useRef<HTMLUListElement>(null);
  useEffect(() => {
    listRef.current
      ?.querySelector<HTMLElement>("[aria-current=true]")
      ?.scrollIntoView({ block: "nearest" });
  }, [currentKey]);

  return (
    <div data-page="workspace">
      <PageHeader
        title="Workspace"
        subtitle={charts.length === 0 ? "尚未保存任何圖表。" : `${charts.length} 張已保存的圖表`}
      />
      {charts.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          在 Insights 或 Explore 預覽圖表後按 Save，圖表會出現在這裡。
        </p>
      ) : (
        <ul ref={listRef} className="grid gap-2 [grid-template-columns:repeat(auto-fill,minmax(260px,1fr))]" data-workspace-list>
          {charts.map((c) => {
            const active = c.key === currentKey;
            return (
              <li key={c.key}>
                <div
                  role="button"
                  tabIndex={0}
                  aria-current={active ? "true" : undefined}
                  data-workspace-card
                  className={cn(
                    "group flex h-full cursor-pointer flex-col gap-1 rounded-md border border-subtle bg-surface p-3 text-left transition-colors hover:bg-accent/40",
                    active && "border-primary/60 bg-primary/10",
                  )}
                  onClick={() => onPreview(c)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      onPreview(c);
                    }
                  }}
                >
                  <div className="flex items-start justify-between gap-2">
                    <span className="text-sm font-medium">{c.result.spec.title}</span>
                    <Badge variant="muted" className="shrink-0 font-normal">
                      {c.result.spec.type}
                    </Badge>
                  </div>
                  <p className="text-xs text-muted-foreground">{variables(c.result.spec)}</p>
                  <div className="mt-auto flex items-center justify-between pt-1 text-xs text-muted-foreground">
                    <span className="tabular-nums">保存於 {time.format(new Date(c.savedAt))}</span>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 px-1.5 text-xs text-muted-foreground hover:text-danger"
                      aria-label={`移除 ${c.result.spec.title}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        onRemove(c.key);
                      }}
                    >
                      <Trash2 className="mr-1 h-3.5 w-3.5" />
                      Remove
                    </Button>
                  </div>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
