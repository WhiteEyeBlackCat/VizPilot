import { useMemo } from "react";

import { ManualBuilder } from "../components/ManualBuilder";
import { PageHeader } from "../components/PageHeader";
import type { ExploreField, ExploreForm, FieldUpdate } from "../store";
import type { ChartSpec, DatasetProfile } from "../types";
import { Skeleton } from "@/components/ui/skeleton";

interface Props {
  profile: DatasetProfile | null;
  form: ExploreForm;
  onField: (field: ExploreField, value: FieldUpdate<string>) => void;
  onGenerate: (spec: ChartSpec) => Promise<void>;
  /** the spec last generated here (recap; the chart itself is in the panel) */
  lastSpec: ChartSpec | null;
}

function specSummary(spec: ChartSpec): string {
  const parts: string[] = [spec.type];
  if (spec.x) parts.push(`x = ${spec.x}`);
  if (spec.y) parts.push(`y = ${spec.y}`);
  if (spec.group_by) parts.push(`group = ${spec.group_by}`);
  if (spec.aggregation) parts.push(`agg = ${spec.aggregation}`);
  return parts.join(" · ");
}

export function ExplorePage({ profile, form, onField, onGenerate, lastSpec }: Props) {
  // which columns the builder can offer, by role — the same rules the
  // ManualBuilder applies (numeric axes, categorical / boolean groups,
  // datetime x for line charts); ids, text and unknown columns never chart
  const hints = useMemo(() => {
    if (!profile) return null;
    const cols = profile.columns;
    const count = (pred: (c: (typeof cols)[number]) => boolean) => cols.filter(pred).length;
    return {
      numeric: count((c) => c.semantic_type === "numeric"),
      categorical: count((c) => c.semantic_type === "categorical" || c.semantic_type === "boolean"),
      datetime: count((c) => c.semantic_type === "datetime"),
      excluded: count((c) => c.semantic_type === "id" || c.semantic_type === "text" || c.semantic_type === "unknown"),
    };
  }, [profile]);

  return (
    <div data-page="explore" className="space-y-5">
      <PageHeader
        title="Explore"
        subtitle="自行選擇圖表類型與欄位；生成後在預覽面板檢視，按 Save 才會加入 Workspace。"
      />
      {profile && hints ? (
        <>
          <ManualBuilder profile={profile} form={form} onField={onField} onGenerate={onGenerate} />
          <dl className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-muted-foreground" data-explore-hints>
            <div>
              <dt className="inline">可作數值軸：</dt>
              <dd className="inline tabular-nums text-foreground" data-hint="numeric">
                {hints.numeric}
              </dd>
            </div>
            <div>
              <dt className="inline">可作類別 / 分組：</dt>
              <dd className="inline tabular-nums text-foreground" data-hint="categorical">
                {hints.categorical}
              </dd>
            </div>
            <div>
              <dt className="inline">時間軸：</dt>
              <dd className="inline tabular-nums text-foreground" data-hint="datetime">
                {hints.datetime}
              </dd>
            </div>
            {hints.excluded > 0 && (
              <div>
                <dt className="inline">不作圖（id / 文字 / 未知）：</dt>
                <dd className="inline tabular-nums" data-hint="excluded">
                  {hints.excluded}
                </dd>
              </div>
            )}
          </dl>
          <div className="text-xs text-muted-foreground" data-explore-last>
            {lastSpec ? (
              <>
                最近一次生成：<span className="text-foreground">{specSummary(lastSpec)}</span>
              </>
            ) : (
              "尚未生成圖表。選好欄位後按「生成圖表」，結果會出現在預覽面板。"
            )}
          </div>
        </>
      ) : (
        <Skeleton className="h-16" aria-label="載入 profile 中…" />
      )}
    </div>
  );
}
