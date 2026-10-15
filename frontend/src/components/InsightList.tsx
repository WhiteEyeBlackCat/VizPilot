import { ChevronRight, LineChart } from "lucide-react";
import { useState } from "react";

import type { Insight, Supported } from "../types";
import { Badge, type BadgeVariant } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface Props {
  insights: Insight[];
  /** text of the insight whose chart is in the preview panel */
  selectedText: string | null;
  /** a preview render is in flight (disables the View chart buttons) */
  busy: boolean;
  onViewChart: (insight: Insight) => void;
}

export const SUPPORTED_BADGE: Record<Supported, { label: string; variant: BadgeVariant }> = {
  strong: { label: "強證據", variant: "success" },
  weak: { label: "弱證據", variant: "warning" },
  unverified: { label: "未驗證", variant: "muted" },
};

const VALIDATION_LABEL: Record<string, string> = {
  existing_evidence: "既有證據表（profile 統計）",
  probe: "針對性檢定（probe）",
};

const nf = new Intl.NumberFormat("en-US");

/** The headline is the backend's own short claim; without one the full
 *  text stands in. The UI never cuts a sentence to fake a headline. */
export const headlineOf = (i: Insight): string => i.headline?.trim() || i.text;

/** Evidence rows for one insight: only what the backend measured. */
export function evidenceRows(i: Insight): { label: string; value: string }[] {
  const rows: { label: string; value: string }[] = [];
  const e = i.effect;
  if (e && e.value !== null && e.value !== undefined) rows.push({ label: e.label, value: e.value.toFixed(2) });
  if (e && e.n !== null && e.n !== undefined) rows.push({ label: "樣本數", value: nf.format(e.n) });
  rows.push({ label: "證據強度", value: SUPPORTED_BADGE[i.supported].label });
  if (i.validation) rows.push({ label: "驗證方式", value: VALIDATION_LABEL[i.validation] ?? i.validation });
  return rows;
}

/** Visual weight: one lead — the strong finding that has both a chart and a
 *  headline, else one with a chart, else the first strong one; other strong
 *  findings are secondary; weak / unverified ones stay compact. */
export function insightWeight(insights: Insight[]): ("lead" | "secondary" | "compact")[] {
  const strong = insights.map((i, index) => ({ i, index })).filter(({ i }) => i.supported === "strong");
  const lead =
    strong.find(({ i }) => i.chart_priority !== null && i.headline)?.index ??
    strong.find(({ i }) => i.chart_priority !== null)?.index ??
    strong[0]?.index;
  return insights.map((i, index) => (index === lead ? "lead" : i.supported === "strong" ? "secondary" : "compact"));
}

/** Key Findings: headline-first rows. The whole row is one button that
 *  expands the detail (full text, why it matters, evidence, View chart);
 *  nothing interactive is nested inside it. */
export function InsightList({ insights, selectedText, busy, onViewChart }: Props) {
  const [open, setOpen] = useState<Set<number>>(() => new Set());
  const weights = insightWeight(insights);

  const toggle = (index: number) =>
    setOpen((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });

  return (
    <ol className="divide-y divide-subtle" data-insight-list>
      {insights.map((insight, index) => {
        const weight = weights[index];
        const expanded = open.has(index);
        const selected = selectedText !== null && insight.text === selectedText;
        const badge = SUPPORTED_BADGE[insight.supported];
        const headline = headlineOf(insight);
        const detailId = `insight-detail-${index}`;
        const showText = headline !== insight.text; // the text is the detail only when a headline stands above it
        const rows = evidenceRows(insight);
        return (
          <li
            key={index}
            data-insight
            data-weight={weight}
            data-expanded={expanded ? "true" : "false"}
            data-selected={selected ? "true" : "false"}
            className={cn(
              "transition-colors",
              weight === "lead" && "border-l-2 border-l-primary bg-elevated/60",
              selected && "bg-primary/10",
            )}
          >
            <button
              type="button"
              aria-expanded={expanded}
              aria-controls={detailId}
              onClick={() => toggle(index)}
              className={cn(
                "flex w-full items-start gap-3 text-left transition-colors hover:bg-accent/40 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-ring",
                weight === "lead" ? "px-4 py-3.5" : weight === "secondary" ? "px-4 py-3" : "px-4 py-2",
              )}
              data-insight-toggle
            >
              <Badge variant={badge.variant} className="mt-0.5 shrink-0 font-normal">
                {badge.label}
              </Badge>
              {/* plain text only — LLM free text must never be rendered as HTML */}
              <span
                className={cn(
                  "flex-1 leading-snug",
                  weight === "lead" && "text-base font-semibold tracking-tight",
                  weight === "secondary" && "text-sm font-medium",
                  weight === "compact" && "text-sm text-foreground/85",
                )}
                data-insight-headline
              >
                {headline}
              </span>
              <ChevronRight
                aria-hidden
                className={cn("mt-1 h-4 w-4 shrink-0 text-muted-foreground transition-transform", expanded && "rotate-90")}
              />
            </button>

            {expanded && (
              <div id={detailId} className="space-y-3 px-4 pb-4 pl-[4.75rem] text-sm" data-insight-detail>
                {showText && <p className="leading-relaxed text-foreground/90">{insight.text}</p>}
                {insight.why_it_matters && (
                  <p className="leading-relaxed text-muted-foreground" data-insight-why>
                    <span className="mr-1.5 font-medium text-foreground/80">為何重要</span>
                    {insight.why_it_matters}
                  </p>
                )}
                <div>
                  <p className="mb-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">Evidence</p>
                  <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-0.5 text-xs" data-insight-evidence>
                    {rows.map((row) => (
                      <div key={row.label} className="contents">
                        <dt className="text-muted-foreground">{row.label}</dt>
                        <dd className="tabular-nums text-foreground/90">{row.value}</dd>
                      </div>
                    ))}
                  </dl>
                </div>
                <div className="flex items-center gap-2">
                  {insight.chart_priority !== null ? (
                    <Button
                      variant={selected ? "secondary" : "outline"}
                      size="sm"
                      className="h-7 border-primary/40 px-2.5 text-xs text-ring hover:bg-primary/15 hover:text-ring"
                      aria-pressed={selected}
                      disabled={busy}
                      onClick={() => onViewChart(insight)}
                      data-insight-view-chart
                    >
                      <LineChart className="mr-1 h-3.5 w-3.5" />
                      {selected ? "Showing chart" : "View chart"}
                    </Button>
                  ) : (
                    <span className="text-xs text-muted-foreground">此發現沒有可對應的圖表</span>
                  )}
                </div>
              </div>
            )}
          </li>
        );
      })}
    </ol>
  );
}
