import { ChevronRight } from "lucide-react";
import { useState } from "react";

import { ApiError } from "../api";
import type {
  ChartSpec,
  Insight,
  Recommendation,
  RecommendationsResponse,
  Tier,
  Warning,
  WarningSeverity,
} from "../types";
import { ErrorList } from "./ErrorList";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge, type BadgeVariant } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Skeleton } from "@/components/ui/skeleton";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

interface Props {
  recs: RecommendationsResponse | null;
  aiPending: boolean;
  /** priority of the recommendation currently shown in the preview panel */
  selectedPriority: number | null;
  /** render the recommendation into the preview panel (never the workspace) */
  onPreview: (rec: Recommendation) => Promise<void>;
  /** the 探索 tier toggle lives in the app store so it survives navigation */
  exploratoryOpen: boolean;
  onExploratoryOpenChange: (open: boolean) => void;
}

const SUPPORTED_BADGE: Record<Insight["supported"], { label: string; variant: BadgeVariant }> = {
  strong: { label: "強證據", variant: "success" },
  weak: { label: "弱證據", variant: "warning" },
  unverified: { label: "未驗證", variant: "muted" },
};

// stage 9 confidence warnings: severe = red, warning = amber, info = grey
const WARNING_VARIANT: Record<WarningSeverity, BadgeVariant> = {
  severe: "danger",
  warning: "warning",
  info: "muted",
};

const TIER_SECTIONS: { tier: Tier; title: string }[] = [
  { tier: "top", title: "推薦重點" },
  { tier: "secondary", title: "次要" },
  { tier: "exploratory", title: "探索" },
];

function warningChips(warnings: Warning[] | undefined) {
  if (!warnings || warnings.length === 0) return null;
  return (
    <ul className="mb-2 flex flex-wrap gap-1">
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
  );
}

function confidenceTitle(rec: Recommendation): string {
  const c = rec.confidence!;
  return (
    `樣本 ${Math.round(c.sample_size * 100)}% · 缺失 ${Math.round(c.missingness * 100)}%` +
    ` · 使用 ${c.n_effective}/${c.n_total} 列（${c.n_source === "exact" ? "精確" : "估計"}）`
  );
}

function variables(spec: ChartSpec): string {
  const parts: string[] = [];
  if (spec.x) parts.push(`x=${spec.x}`);
  if (spec.y) parts.push(`y=${spec.y}`);
  if (spec.group_by) parts.push(`group=${spec.group_by}`);
  if (spec.aggregation) parts.push(`agg=${spec.aggregation}`);
  return parts.join(" · ") || "—";
}

// columns follow the available width (the page shares the row with the
// preview panel), not the viewport; the top tier gets wider cards
const CARD_GRID = "grid gap-2 [grid-template-columns:repeat(auto-fill,minmax(250px,1fr))]";
const TOP_GRID = "grid gap-3 [grid-template-columns:repeat(auto-fill,minmax(320px,1fr))]";

export function Recommendations({
  recs,
  aiPending,
  selectedPriority,
  onPreview,
  exploratoryOpen,
  onExploratoryOpenChange,
}: Props) {
  const [busyPriority, setBusyPriority] = useState<number | null>(null);
  const [error, setError] = useState<string[]>([]);
  const setExploratoryOpen = onExploratoryOpenChange;
  const [highlighted, setHighlighted] = useState<number | null>(null);

  const preview = async (rec: Recommendation) => {
    setBusyPriority(rec.spec.priority ?? -1);
    setError([]);
    try {
      await onPreview(rec);
    } catch (e) {
      setError(e instanceof ApiError ? e.errors : [String(e)]);
    } finally {
      setBusyPriority(null);
    }
  };

  const showChart = (priority: number) => {
    if (!exploratoryOpen && recs?.charts.some((c) => c.spec.priority === priority && c.tier === "exploratory")) {
      setExploratoryOpen(true);
    }
    // defer so a newly expanded section is in the DOM before scrolling
    setTimeout(() => {
      document.getElementById(`rec-card-${priority}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
      setHighlighted(priority);
      setTimeout(() => setHighlighted((current) => (current === priority ? null : current)), 2000);
    }, 0);
    // 16.2: "看圖" also previews the supporting chart
    const rec = recs?.charts.find((c) => c.spec.priority === priority);
    if (rec) void preview(rec);
  };

  const card = (rec: Recommendation, tier: Tier) => {
    const priority = rec.spec.priority ?? 0;
    const selected = selectedPriority === priority;
    const top = tier === "top";
    return (
      // the card body is a pointer convenience; the Preview button inside is
      // the accessible control (no nested interactive elements)
      <div
        key={priority}
        id={`rec-card-${priority}`}
        data-tier={tier}
        data-selected={selected ? "true" : "false"}
        className={cn(
          "group flex cursor-pointer flex-col rounded-md border border-subtle bg-surface text-left transition-colors hover:bg-accent/40",
          top ? "border-l-2 border-l-primary p-4" : "p-3",
          (selected || highlighted === priority) && "border-primary/60 bg-primary/10",
        )}
        onClick={() => void preview(rec)}
      >
        <div className="mb-1 flex items-start justify-between gap-2">
          <span className={cn("font-medium", top ? "text-base" : "text-sm")}>{rec.spec.title}</span>
          <span className="flex shrink-0 gap-1">
            <Badge variant="muted">{rec.spec.type}</Badge>
            {rec.source === "llm" && <Badge variant="accent">AI</Badge>}
          </span>
        </div>
        <p className={cn("mb-1 flex-1 text-muted-foreground", top ? "text-sm" : "text-xs")}>{rec.spec.reason}</p>
        {warningChips(rec.warnings)}
        <p className="mb-2 text-xs text-muted-foreground/80">
          {variables(rec.spec)}
          {rec.confidence && rec.confidence.overall < 1 && (
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="ml-2 cursor-help underline decoration-dotted underline-offset-2">
                  信心 {Math.round(rec.confidence.overall * 100)}%
                </span>
              </TooltipTrigger>
              <TooltipContent>{confidenceTitle(rec)}</TooltipContent>
            </Tooltip>
          )}
        </p>
        <Button
          variant={selected ? "secondary" : "outline"}
          size="sm"
          className="h-7 self-start px-3 text-xs"
          aria-pressed={selected}
          disabled={busyPriority !== null}
          onClick={(e) => {
            e.stopPropagation();
            void preview(rec);
          }}
        >
          {busyPriority === priority ? "生成中…" : selected ? "預覽中" : "Preview"}
        </Button>
      </div>
    );
  };

  const insightCard = (insight: Insight, index: number) => {
    const badge = SUPPORTED_BADGE[insight.supported];
    return (
      <li key={index} className="flex items-start gap-2">
        <Badge variant={badge.variant} className="mt-0.5 shrink-0">
          {badge.label}
        </Badge>
        {/* plain text only — LLM free text must never be rendered as HTML */}
        <span className="flex-1">{insight.text}</span>
        {insight.chart_priority !== null && (
          <Button
            variant="outline"
            size="sm"
            className="h-6 shrink-0 border-primary/40 px-2 text-xs text-ring hover:bg-primary/15 hover:text-ring"
            onClick={() => showChart(insight.chart_priority!)}
          >
            看圖
          </Button>
        )}
      </li>
    );
  };

  const byTier = (tier: Tier) => (recs?.charts ?? []).filter((rec) => rec.tier === tier);

  return (
    <div data-recommendations data-ai-pending={aiPending ? "true" : "false"}>
      {recs?.message && (
        <Alert variant="warning" className="mb-3">
          <AlertDescription>{recs.message}</AlertDescription>
        </Alert>
      )}

      {recs?.warnings && recs.warnings.length > 0 && (
        <Alert variant="muted" className="mb-3">
          <AlertDescription>
            <ul className="space-y-1 text-xs">
              {recs.warnings.map((w, i) => (
                <li key={i}>{w.message}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}

      {recs && recs.insights.length > 0 && (
        <ul className="mb-4 space-y-2 rounded-md border-l-2 border-primary bg-elevated p-3 text-sm">
          {recs.insights.map(insightCard)}
        </ul>
      )}

      <ErrorList errors={error} className="mb-3" />

      {!recs && (
        <div className="space-y-2" aria-label="載入推薦中…">
          <p className="text-sm text-muted-foreground">載入推薦中…</p>
          <div className={CARD_GRID}>
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} className="h-28" />
            ))}
          </div>
        </div>
      )}

      {TIER_SECTIONS.map(({ tier, title }) => {
        const charts = byTier(tier);
        if (charts.length === 0) {
          // an empty top tier is a finding, not a rendering gap: say so
          if (tier === "top" && (recs?.charts.length ?? 0) > 0) {
            return (
              <div key={tier} className="mt-5 first:mt-0">
                <h2 className="mb-2 text-sm font-medium text-muted-foreground">{title}</h2>
                <p className="text-sm text-muted-foreground/80">
                  沒有圖表在資料中展現足夠強的證據（定義性關係不算發現）——以下為次要與探索性建議。
                </p>
              </div>
            );
          }
          return null;
        }
        if (tier === "exploratory") {
          return (
            <Collapsible key={tier} open={exploratoryOpen} onOpenChange={setExploratoryOpen} className="mt-5">
              {/* h2 like the other tiers, with the toggle inside it */}
              <h2 className="mb-2 text-sm font-medium text-muted-foreground">
                <CollapsibleTrigger asChild>
                  <Button variant="ghost" size="sm" className="-ml-2 h-7 px-2 text-sm font-medium text-muted-foreground">
                    <ChevronRight
                      className={cn("h-4 w-4 transition-transform", exploratoryOpen && "rotate-90")}
                    />
                    {title}（{charts.length}）
                  </Button>
                </CollapsibleTrigger>
              </h2>
              <CollapsibleContent>
                <div className={CARD_GRID}>{charts.map((rec) => card(rec, tier))}</div>
              </CollapsibleContent>
            </Collapsible>
          );
        }
        return (
          <div key={tier} className="mt-5 first:mt-0">
            <h2 className={cn("mb-2 text-sm font-medium", tier === "top" ? "text-foreground" : "text-muted-foreground")}>
              {title}
            </h2>
            <div className={tier === "top" ? TOP_GRID : CARD_GRID}>{charts.map((rec) => card(rec, tier))}</div>
          </div>
        );
      })}
    </div>
  );
}
