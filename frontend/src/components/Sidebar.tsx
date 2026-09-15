import { Database, Plus } from "lucide-react";
import { useEffect, useRef } from "react";

import { PAGES, type Page } from "../store";
import type { DatasetMeta } from "../types";
import { buildHash } from "@/lib/router";
import { cn } from "@/lib/utils";

interface Props {
  datasets: DatasetMeta[];
  currentId: string | null;
  page: Page;
  workspaceCount: number;
  onNewDataset: () => void;
  /** "side": fixed left column (≥ 1024px); "top": compact bar for narrow viewports */
  variant: "side" | "top";
}

/** Primary navigation: the dataset list and, once one is selected, the four
 *  pages. Links are plain anchors on the hash router so the browser back
 *  button and a reload both work. */
export function Sidebar({ datasets, currentId, page, workspaceCount, onNewDataset, variant }: Props) {
  const listRef = useRef<HTMLUListElement>(null);

  // keep the selected dataset in view when the list is long
  useEffect(() => {
    listRef.current
      ?.querySelector<HTMLElement>("[aria-current=true]")
      ?.scrollIntoView({ block: "nearest" });
  }, [currentId]);

  const pageLinks = currentId && (
    <ul className={cn("flex gap-1", variant === "side" ? "flex-col" : "flex-row")} data-page-nav>
      {PAGES.map((p) => (
        <li key={p.page}>
          <a
            href={buildHash({ datasetId: currentId, page: p.page })}
            aria-current={page === p.page ? "page" : undefined}
            className={cn(
              "flex items-center justify-between rounded-md px-2.5 py-1.5 text-sm transition-colors focus:outline-none focus-visible:ring-1 focus-visible:ring-ring",
              page === p.page
                ? "bg-accent text-foreground"
                : "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
            )}
          >
            <span>{p.label}</span>
            {p.page === "workspace" && (
              <span className="ml-2 tabular-nums text-xs text-muted-foreground" data-workspace-count>
                {workspaceCount}
              </span>
            )}
          </a>
        </li>
      ))}
    </ul>
  );

  if (variant === "top") {
    const current = datasets.find((d) => d.dataset_id === currentId) ?? null;
    return (
      <nav
        aria-label="主導航"
        data-sidebar="top"
        className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-subtle bg-surface px-3 py-2"
      >
        <span className="text-sm font-semibold tracking-tight">VizPilot</span>
        <div className="flex min-w-0 items-center gap-2">
          <select
            aria-label="選擇資料集"
            className="h-8 max-w-[14rem] rounded-md border border-border bg-background px-2 text-sm"
            value={currentId ?? ""}
            onChange={(e) => {
              if (e.target.value) window.location.hash = buildHash({ datasetId: e.target.value, page: "overview" });
            }}
          >
            <option value="" disabled>
              {datasets.length ? "選擇資料集…" : "尚無資料集"}
            </option>
            {datasets.map((d) => (
              <option key={d.dataset_id} value={d.dataset_id}>
                {d.filename} ({d.n_rows}×{d.n_cols})
              </option>
            ))}
          </select>
          <button
            type="button"
            className="flex h-8 items-center gap-1 rounded-md border border-border px-2 text-sm text-muted-foreground hover:bg-accent hover:text-foreground focus:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            onClick={onNewDataset}
          >
            <Plus className="h-3.5 w-3.5" />
            New Dataset
          </button>
        </div>
        {current && <div className="basis-full">{pageLinks}</div>}
      </nav>
    );
  }

  return (
    <nav
      aria-label="主導航"
      data-sidebar="side"
      className="flex h-full w-60 shrink-0 flex-col border-r border-subtle bg-surface"
    >
      <div className="px-4 pb-3 pt-4">
        <span className="text-base font-semibold tracking-tight">VizPilot</span>
        <p className="mt-0.5 text-xs text-muted-foreground">Local data exploration</p>
      </div>

      {/* primary navigation first (only with a dataset), collections below */}
      {currentId && <div className="px-2 pb-3">{pageLinks}</div>}

      <div className={cn("flex items-center justify-between px-4 pb-1 pt-2", currentId && "border-t border-subtle pt-3")}>
        <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
          Datasets
        </span>
      </div>
      <ul ref={listRef} className="max-h-[45vh] min-h-0 overflow-y-auto px-2" data-dataset-list>
        {datasets.length === 0 && (
          <li className="px-2 py-1.5 text-xs text-muted-foreground">尚無資料集</li>
        )}
        {datasets.map((d) => {
          const active = d.dataset_id === currentId;
          return (
            <li key={d.dataset_id}>
              <a
                href={buildHash({ datasetId: d.dataset_id, page: "overview" })}
                aria-current={active ? "true" : undefined}
                title={d.filename}
                className={cn(
                  "flex items-center gap-2 rounded-md px-2 py-1.5 text-sm transition-colors focus:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                  active
                    ? "bg-accent text-foreground"
                    : "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
                )}
              >
                <Database className="h-3.5 w-3.5 shrink-0 opacity-70" />
                <span className="min-w-0 flex-1 truncate">{d.filename}</span>
                <span className="shrink-0 text-[11px] tabular-nums text-muted-foreground">
                  {d.n_rows}×{d.n_cols}
                </span>
              </a>
            </li>
          );
        })}
      </ul>
      <div className="px-2 pb-2 pt-1">
        <button
          type="button"
          className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-accent/60 hover:text-foreground focus:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          onClick={onNewDataset}
        >
          <Plus className="h-3.5 w-3.5" />
          New Dataset
        </button>
      </div>
      <div className="flex-1" />
    </nav>
  );
}
