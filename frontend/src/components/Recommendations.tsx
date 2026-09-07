import { useState } from "react";

import { ApiError } from "../api";
import type { ChartSpec, Insight, Recommendation, RecommendationsResponse, Tier } from "../types";

interface Props {
  recs: RecommendationsResponse | null;
  aiPending: boolean;
  onGenerate: (spec: ChartSpec) => Promise<void>;
}

const SUPPORTED_BADGE = {
  strong: { label: "強證據", className: "bg-emerald-100 text-emerald-700" },
  weak: { label: "弱證據", className: "bg-amber-100 text-amber-700" },
  unverified: { label: "未驗證", className: "bg-slate-200 text-slate-500" },
} as const;

const TIER_SECTIONS: { tier: Tier; title: string }[] = [
  { tier: "top", title: "推薦重點" },
  { tier: "secondary", title: "次要" },
  { tier: "exploratory", title: "探索" },
];

function variables(spec: ChartSpec): string {
  const parts: string[] = [];
  if (spec.x) parts.push(`x=${spec.x}`);
  if (spec.y) parts.push(`y=${spec.y}`);
  if (spec.group_by) parts.push(`group=${spec.group_by}`);
  if (spec.aggregation) parts.push(`agg=${spec.aggregation}`);
  return parts.join(" · ") || "—";
}

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
      <div
        key={priority}
        id={`rec-card-${priority}`}
        className={`flex flex-col rounded border p-3 transition-colors ${
          highlighted === priority ? "border-violet-500 bg-violet-50" : "border-slate-200"
        }`}
      >
        <div className="mb-1 flex items-start justify-between gap-2">
          <span className="text-sm font-medium">{rec.spec.title}</span>
          <span className="flex shrink-0 gap-1">
            <span className="rounded bg-slate-100 px-1.5 py-0.5 text-xs text-slate-600">{rec.spec.type}</span>
            <span
              className={`rounded px-1.5 py-0.5 text-xs ${
                rec.source === "llm" ? "bg-violet-100 text-violet-700" : "bg-slate-100 text-slate-500"
              }`}
            >
              {rec.source === "llm" ? "AI" : "rules"}
            </span>
          </span>
        </div>
        <p className="mb-1 flex-1 text-xs text-slate-600">{rec.spec.reason}</p>
        <p className="mb-2 text-xs text-slate-400">{variables(rec.spec)}</p>
        <button
          className="self-start rounded bg-blue-600 px-3 py-1 text-xs text-white hover:bg-blue-700 disabled:opacity-50"
          disabled={busyPriority !== null}
          onClick={() => void generate(rec.spec)}
        >
          {busyPriority === priority ? "生成中…" : "Generate"}
        </button>
      </div>
    );
  };

  const insightCard = (insight: Insight, index: number) => {
    const badge = SUPPORTED_BADGE[insight.supported];
    return (
      <li key={index} className="flex items-start gap-2">
        <span className={`mt-0.5 shrink-0 rounded px-1.5 py-0.5 text-xs ${badge.className}`}>{badge.label}</span>
        {/* plain text only — LLM free text must never be rendered as HTML */}
        <span className="flex-1">{insight.text}</span>
        {insight.chart_priority !== null && (
          <button
            className="shrink-0 rounded border border-violet-300 px-2 py-0.5 text-xs text-violet-700 hover:bg-violet-100"
            onClick={() => showChart(insight.chart_priority!)}
          >
            看圖
          </button>
        )}
      </li>
    );
  };

  const byTier = (tier: Tier) =>
    (recs?.charts ?? []).filter((rec) => rec.tier === tier);

  return (
    <section className="rounded-lg bg-white p-4 shadow">
      <h2 className="mb-2 font-semibold">
        C. 推薦圖表
        {aiPending && (
          <span className="ml-2 animate-pulse rounded bg-violet-100 px-2 py-0.5 text-xs font-normal text-violet-700">
            AI 分析中…
          </span>
        )}
      </h2>

      {recs?.message && (
        <div className="mb-3 rounded border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
          {recs.message}
        </div>
      )}

      {recs && recs.insights.length > 0 && (
        <ul className="mb-3 space-y-2 rounded bg-violet-50 p-3 text-sm text-violet-900">
          {recs.insights.map(insightCard)}
        </ul>
      )}

      {error.length > 0 && (
        <ul className="mb-2 list-inside list-disc text-sm text-red-600">
          {error.map((e, i) => (
            <li key={i}>{e}</li>
          ))}
        </ul>
      )}

      {!recs && <p className="text-sm text-slate-400">載入推薦中…</p>}

      {TIER_SECTIONS.map(({ tier, title }) => {
        const charts = byTier(tier);
        if (charts.length === 0) {
          // an empty top tier is a finding, not a rendering gap: say so
          if (tier === "top" && (recs?.charts.length ?? 0) > 0) {
            return (
              <div key={tier} className="mt-4">
                <h3 className="mb-2 text-sm font-semibold text-slate-700">{title}</h3>
                <p className="text-sm text-slate-400">
                  沒有圖表在資料中展現足夠強的證據——以下為次要與探索性建議。
                </p>
              </div>
            );
          }
          return null;
        }
        if (tier === "exploratory") {
          return (
            <div key={tier} className="mt-4">
              <button
                className="mb-2 flex items-center gap-1 text-sm font-medium text-slate-500 hover:text-slate-700"
                onClick={() => setExploratoryOpen((open) => !open)}
              >
                <span>{exploratoryOpen ? "▾" : "▸"}</span>
                {title}（{charts.length}）
              </button>
              {exploratoryOpen && (
                <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
                  {charts.map(card)}
                </div>
              )}
            </div>
          );
        }
        return (
          <div key={tier} className="mt-4 first:mt-0">
            <h3 className="mb-2 text-sm font-medium text-slate-500">{title}</h3>
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">{charts.map(card)}</div>
          </div>
        );
      })}
    </section>
  );
}
