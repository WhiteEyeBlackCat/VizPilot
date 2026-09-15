import { ChevronRight, Sparkles } from "lucide-react";
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
import { SectionCard } from "./SectionCard";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge, type BadgeVariant } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Skeleton } from "@/components/ui/skeleton";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

interface Props {
  recs: RecommendationsResponse | null;
  aiPending: boolean;
  onGenerate: (spec: ChartSpec) => Promise<void>;
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

// columns follow the available width (the cards live in a half-width pane
// since stage 15), not the viewport
const CARD_GRID = "grid gap-3 [grid-template-columns:repeat(auto-fill,minmax(250px,1fr))]";

export function Recommendations({ recs, aiPending, onGenerate }: Props) {
  const [busyPriority, setBusyPriority] = useState<number | null>(null);
  const [error, setError] = useState<string[]>([]);
  const [exploratoryOpen, setExploratoryOpen] = useState(false);
  const [highlighted, setHighlighted] = useState<number | null>(null);

  const generate = async (spec: ChartSpec) => {
    setBusyPriority(spec.priority ?? -1);
    setError([]);
    try {
      await onGenerate(spec);
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
  };

  const card = (rec: Recommendation) => {
    const priority = rec.spec.priority ?? 0;
    return (
      <Card
        key={priority}
        id={`rec-card-${priority}`}
        className={cn(
          "flex flex-col p-3 shadow-none transition-colors",
          highlighted === priority && "border-primary bg-primary/10",
        )}
      >
        <div className="mb-1 flex items-start justify-between gap-2">
          <span className="text-sm font-medium">{rec.spec.title}</span>
          <span className="flex shrink-0 gap-1">
            <Badge variant="muted">{rec.spec.type}</Badge>
            <Badge variant={rec.source === "llm" ? "accent" : "muted"}>
              {rec.source === "llm" ? "AI" : "rules"}
            </Badge>
          </span>
        </div>
        <p className="mb-1 flex-1 text-xs text-muted-foreground">{rec.spec.reason}</p>
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
          size="sm"
          className="h-7 self-start px-3 text-xs"
          disabled={busyPriority !== null}
          onClick={() => void generate(rec.spec)}
        >
          {busyPriority === priority ? "生成中…" : "Generate"}
        </Button>
      </Card>
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

  const title = (
    <span className="flex items-center gap-2">
      推薦圖表
      {aiPending && (
        <Badge variant="accent" className="animate-pulse font-normal">
          <Sparkles className="mr-1 h-3 w-3" />
          AI 分析中…
        </Badge>
      )}
    </span>
  );

  return (
    <SectionCard title={title}>
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
        <ul className="mb-3 space-y-2 rounded-md border-l-2 border-primary bg-elevated p-3 text-sm">
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
              <div key={tier} className="mt-4 first:mt-0">
                <h3 className="mb-2 text-sm font-medium text-muted-foreground">{title}</h3>
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
            <Collapsible key={tier} open={exploratoryOpen} onOpenChange={setExploratoryOpen} className="mt-4">
              {/* h3 like the other tiers, with the toggle inside it */}
              <h3 className="mb-2 text-sm font-medium text-muted-foreground">
                <CollapsibleTrigger asChild>
                  <Button variant="ghost" size="sm" className="-ml-2 h-7 px-2 text-sm font-medium text-muted-foreground">
                    <ChevronRight
                      className={cn("h-4 w-4 transition-transform", exploratoryOpen && "rotate-90")}
                    />
                    {title}（{charts.length}）
                  </Button>
                </CollapsibleTrigger>
              </h3>
              <CollapsibleContent>
                <div className={CARD_GRID}>{charts.map(card)}</div>
              </CollapsibleContent>
            </Collapsible>
          );
        }
        return (
          <div key={tier} className="mt-4 first:mt-0">
            <h3 className="mb-2 text-sm font-medium text-muted-foreground">{title}</h3>
            <div className={CARD_GRID}>{charts.map(card)}</div>
          </div>
        );
      })}
    </SectionCard>
  );
}
