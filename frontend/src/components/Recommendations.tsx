import { ChevronRight, Info } from "lucide-react";
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
import { InsightList } from "./InsightList";
import { INTENT_LABEL } from "./intentLabels";
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
  /** text of the insight whose chart is in the preview panel */
  selectedInsightText: string | null;
  /** render the recommendation into the preview panel (never the workspace) */
  onPreview: (rec: Recommendation) => Promise<void>;
  /** render an insight's supporting chart into the preview panel */
  onPreviewInsight: (insight: Insight) => Promise<void>;
  /** the 探索 tier toggle lives in the app store so it survives navigation */
  exploratoryOpen: boolean;
  onExploratoryOpenChange: (open: boolean) => void;
}

// stage 9 confidence warnings: severe = red, warning = amber, info = grey
const WARNING_VARIANT: Record<WarningSeverity, BadgeVariant> = {
  severe: "danger",
  warning: "warning",
  info: "muted",
};

const TIER_SECTIONS: { tier: Tier; title: string; hint: string }[] = [
  { tier: "top", title: "推薦重點", hint: "證據分數最高的候選圖" },
  { tier: "secondary", title: "次要", hint: "" },
  { tier: "exploratory", title: "探索", hint: "" },
];

const AGG_LABEL: Record<NonNullable<ChartSpec["aggregation"]>, string> = {
  mean: "平均",
  sum: "總和",
  count: "筆數",
  median: "中位數",
  min: "最小值",
  max: "最大值",
};

function warningChips(warnings: Warning[] | undefined) {
  if (!warnings || warnings.length === 0) return null;
  return (
    <ul className="flex flex-wrap gap-1">
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

/** The reliability factor is a heuristic discount (sample size, missing
 *  values, sentinels / extremes), not a probability that the chart's claim is
 *  true; it is shown as a 0–1 index, never as a percentage (audit B1). */
export const RELIABILITY_NOTE = "啟發式折減係數（樣本數、缺失、極端值），不是結論正確的機率";

function confidenceTitle(rec: Recommendation): string {
  const c = rec.confidence!;
  return (
    `樣本係數 ${c.sample_size.toFixed(2)} · 缺失係數 ${c.missingness.toFixed(2)}` +
    ` · 使用 ${c.n_effective}/${c.n_total} 列（${c.n_source === "exact" ? "精確" : "估計"}）` +
    ` — ${RELIABILITY_NOTE}`
  );
}

/** The full ChartSpec, one role per line (tooltip / technical detail). */
export function specRows(spec: ChartSpec): { role: string; value: string }[] {
  const rows: { role: string; value: string }[] = [{ role: "type", value: spec.type }];
  if (spec.x) rows.push({ role: "x", value: spec.x });
  if (spec.y) rows.push({ role: "y", value: spec.y });
  if (spec.group_by) rows.push({ role: "group", value: spec.group_by });
  if (spec.aggregation) rows.push({ role: "aggregation", value: spec.aggregation });
  if (spec.bins) rows.push({ role: "bins", value: String(spec.bins) });
  if (spec.time_granularity) rows.push({ role: "time", value: spec.time_granularity });
  if (spec.top_n) rows.push({ role: "top_n", value: String(spec.top_n) });
  return rows;
}

/** Short metadata for the card: chart type and, when there is one, the
 *  aggregation in words — the column roles live in the tooltip. */
export function chartMeta(spec: ChartSpec): string {
  const parts: string[] = [spec.type];
  if (spec.aggregation) parts.push(AGG_LABEL[spec.aggregation] ?? spec.aggregation);
  if (spec.time_granularity && spec.time_granularity !== "raw") parts.push(`by ${spec.time_granularity}`);
  return parts.join(" · ");
}

// columns follow the available width (the page shares the row with the
// preview panel), not the viewport; the top tier gets wider cards
const CARD_GRID = "grid gap-2 [grid-template-columns:repeat(auto-fill,minmax(240px,1fr))]";
const TOP_GRID = "grid gap-3 [grid-template-columns:repeat(auto-fill,minmax(300px,1fr))]";

export function Recommendations({
  recs,
  aiPending,
  selectedPriority,
  selectedInsightText,
  onPreview,
  onPreviewInsight,
  exploratoryOpen,
  onExploratoryOpenChange,
}: Props) {
  const [busyPriority, setBusyPriority] = useState<number | null>(null);
  const [error, setError] = useState<string[]>([]);
  const setExploratoryOpen = onExploratoryOpenChange;

  const run = async (priority: number, task: () => Promise<void>) => {
    setBusyPriority(priority);
    setError([]);
    try {
      await task();
    } catch (e) {
      setError(e instanceof ApiError ? e.errors : [String(e)]);
    } finally {
      setBusyPriority(null);
    }
  };

  const preview = (rec: Recommendation) => run(rec.spec.priority ?? -1, () => onPreview(rec));
  const viewInsightChart = (insight: Insight) => run(insight.chart_priority ?? -1, () => onPreviewInsight(insight));

  const card = (rec: Recommendation, tier: Tier) => {
    const priority = rec.spec.priority ?? 0;
    const selected = selectedPriority === priority;
    const top = tier === "top";
    const compact = tier === "exploratory";
    const rows = specRows(rec.spec);
    return (
      // the card body is a pointer convenience; the Preview button inside is
      // the accessible control (no nested interactive elements)
      <div
        key={priority}
        id={`rec-card-${priority}`}
        data-tier={tier}
        data-selected={selected ? "true" : "false"}
        data-x={rec.spec.x ?? ""}
        data-y={rec.spec.y ?? ""}
        className={cn(
          "group flex cursor-pointer flex-col gap-1.5 rounded-md border border-subtle bg-surface text-left transition-colors hover:bg-accent/40",
          top ? "border-l-2 border-l-primary p-4" : compact ? "p-2.5" : "p-3",
          selected && "border-primary/60 bg-primary/10",
        )}
        onClick={() => void preview(rec)}
      >
        <div className="flex items-start justify-between gap-2">
          <span className={cn("font-medium leading-snug", top ? "text-base" : "text-sm")}>{rec.spec.title}</span>
          {rec.source === "llm" && (
            <Badge variant="accent" className="shrink-0 font-normal">
              AI
            </Badge>
          )}
        </div>
        {!compact && rec.spec.reason && (
          <p className={cn("text-muted-foreground", top ? "text-sm" : "text-xs line-clamp-2")}>{rec.spec.reason}</p>
        )}
        {warningChips(rec.warnings)}
        <div className="mt-auto flex flex-wrap items-center gap-x-2 gap-y-1 pt-1 text-xs text-muted-foreground/90">
          <span data-chart-meta>{chartMeta(rec.spec)}</span>
          {rec.intent && (
            <span className="text-info/90" data-card-intent={rec.intent}>
              {INTENT_LABEL[rec.intent] ?? rec.intent}
            </span>
          )}
          {rec.confidence && rec.confidence.overall < 1 && (
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="cursor-help underline decoration-dotted underline-offset-2">
                  資料可靠度 {rec.confidence.overall.toFixed(2)}
                </span>
              </TooltipTrigger>
              <TooltipContent>{confidenceTitle(rec)}</TooltipContent>
            </Tooltip>
          )}
          <span className="ml-auto flex items-center gap-1">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 w-7 p-0 text-muted-foreground"
                  aria-label={`圖表設定：${rows.map((r) => `${r.role} ${r.value}`).join("，")}`}
                  onClick={(e) => e.stopPropagation()}
                >
                  <Info className="h-3.5 w-3.5" />
                </Button>
              </TooltipTrigger>
              <TooltipContent className="p-2">
                <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 font-mono text-[11px]" data-spec-detail>
                  {rows.map((r) => (
                    <div key={r.role} className="contents">
                      <dt className="text-muted-foreground">{r.role}</dt>
                      <dd>{r.value}</dd>
                    </div>
                  ))}
                </dl>
              </TooltipContent>
            </Tooltip>
            <Button
              variant={selected ? "secondary" : "outline"}
              size="sm"
              className="h-7 px-3 text-xs"
              aria-pressed={selected}
              disabled={busyPriority !== null}
              onClick={(e) => {
                e.stopPropagation();
                void preview(rec);
              }}
            >
              {busyPriority === priority ? "生成中…" : selected ? "預覽中" : "Preview"}
            </Button>
          </span>
        </div>
      </div>
    );
  };

  const byTier = (tier: Tier) => (recs?.charts ?? []).filter((rec) => rec.tier === tier);
  const insights = recs?.insights ?? [];

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

      {/* Key Findings: validated insights, headline first. Rendered ahead of
          every rule chart so validated claims never share the weight of
          unverified candidates. */}
      {(insights.length > 0 || (aiPending && recs)) && (
        <section className="mb-7" data-key-findings>
          <div className="mb-2 flex items-baseline justify-between gap-3">
            <h2 className="text-sm font-semibold tracking-tight">
              Key Findings
              {insights.length > 0 && <span className="ml-2 font-normal text-muted-foreground">{insights.length}</span>}
            </h2>
            <p className="text-xs text-muted-foreground">後端統計驗證過的發現；點整列展開細節與證據</p>
          </div>
          {insights.length > 0 ? (
            <InsightList
              insights={insights}
              selectedText={selectedInsightText}
              busy={busyPriority !== null}
              onViewChart={viewInsightChart}
            />
          ) : (
            <div className="space-y-2" aria-label="AI 驗證假設中…" data-insights-pending>
              <Skeleton className="h-9" />
              <Skeleton className="h-9 w-4/5" />
            </div>
          )}
        </section>
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

      {recs && recs.charts.length > 0 && (
        <section data-more-charts>
          <div className="mb-3 flex items-baseline justify-between gap-3 border-t border-subtle pt-4">
            <h2 className="text-sm font-semibold tracking-tight text-muted-foreground">
              {insights.length > 0 ? "More charts" : "Charts"}
              <span className="ml-2 font-normal">{recs.charts.length}</span>
            </h2>
            <p className="text-xs text-muted-foreground/80">規則引擎依證據分數提出的候選圖；未經假設驗證</p>
          </div>

          {TIER_SECTIONS.map(({ tier, title }) => {
            const charts = byTier(tier);
            if (charts.length === 0) {
              // an empty top tier is a finding, not a rendering gap: say so
              if (tier === "top") {
                return (
                  <div key={tier} className="mt-4 first:mt-0">
                    <h3 className="mb-1.5 text-xs font-medium uppercase tracking-wide text-muted-foreground">{title}</h3>
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
                  <h3 className="mb-1.5 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                    <CollapsibleTrigger asChild>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="-ml-2 h-7 px-2 text-xs font-medium uppercase tracking-wide text-muted-foreground"
                      >
                        <ChevronRight className={cn("h-4 w-4 transition-transform", exploratoryOpen && "rotate-90")} />
                        {title}（{charts.length}）
                      </Button>
                    </CollapsibleTrigger>
                  </h3>
                  <CollapsibleContent>
                    <div className={CARD_GRID}>{charts.map((rec) => card(rec, tier))}</div>
                  </CollapsibleContent>
                </Collapsible>
              );
            }
            return (
              <div key={tier} className="mt-4 first:mt-0">
                <h3
                  className={cn(
                    "mb-1.5 text-xs font-medium uppercase tracking-wide",
                    tier === "top" ? "text-foreground/80" : "text-muted-foreground",
                  )}
                >
                  {title}
                </h3>
                <div className={tier === "top" ? TOP_GRID : CARD_GRID}>{charts.map((rec) => card(rec, tier))}</div>
              </div>
            );
          })}
        </section>
      )}
    </div>
  );
}
