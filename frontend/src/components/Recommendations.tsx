import { useState } from "react";

import { ApiError } from "../api";
import type { ChartSpec, RecommendationsResponse } from "../types";

interface Props {
  recs: RecommendationsResponse | null;
  aiPending: boolean;
  onGenerate: (spec: ChartSpec) => Promise<void>;
}

function variables(spec: ChartSpec): string {
  const parts: string[] = [];
  if (spec.x) parts.push(`x=${spec.x}`);
  if (spec.y) parts.push(`y=${spec.y}`);
  if (spec.group_by) parts.push(`group=${spec.group_by}`);
  if (spec.aggregation) parts.push(`agg=${spec.aggregation}`);
  return parts.join(" · ") || "—";
}

export function Recommendations({ recs, aiPending, onGenerate }: Props) {
  const [busyIdx, setBusyIdx] = useState<number | null>(null);
  const [error, setError] = useState<string[]>([]);

  const generate = async (spec: ChartSpec, idx: number) => {
    setBusyIdx(idx);
    setError([]);
    try {
      await onGenerate(spec);
    } catch (e) {
      setError(e instanceof ApiError ? e.errors : [String(e)]);
    } finally {
      setBusyIdx(null);
    }
  };

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
        <ul className="mb-3 space-y-1 rounded bg-violet-50 p-3 text-sm text-violet-900">
          {recs.insights.map((insight, i) => (
            // plain text only — LLM free text must never be rendered as HTML
            <li key={i}>💡 {insight}</li>
          ))}
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

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
        {recs?.charts.map((rec, idx) => (
          <div key={idx} className="flex flex-col rounded border border-slate-200 p-3">
            <div className="mb-1 flex items-start justify-between gap-2">
              <span className="text-sm font-medium">{rec.spec.title}</span>
              <span className="flex shrink-0 gap-1">
                <span className="rounded bg-slate-100 px-1.5 py-0.5 text-xs text-slate-600">
                  {rec.spec.type}
                </span>
                <span
                  className={`rounded px-1.5 py-0.5 text-xs ${
                    rec.source === "llm"
                      ? "bg-violet-100 text-violet-700"
                      : "bg-slate-100 text-slate-500"
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
              disabled={busyIdx !== null}
              onClick={() => void generate(rec.spec, idx)}
            >
              {busyIdx === idx ? "生成中…" : "Generate"}
            </button>
          </div>
        ))}
      </div>
    </section>
  );
}
