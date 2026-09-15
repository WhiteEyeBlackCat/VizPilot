import { useMemo, useState } from "react";

import { DatasetOverview, hasQualityFlags, TYPE_VARIANT } from "../components/DatasetOverview";
import { PageHeader } from "../components/PageHeader";
import type {
  DatasetMeta,
  DatasetProfile,
  DerivedColumn,
  NearDuplicateGroup,
  SemanticType,
  Warning,
  WarningSeverity,
} from "../types";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

interface Props {
  meta: DatasetMeta;
  profile: DatasetProfile | null;
  /** dataset-level warnings from the recommendations response; null until loaded */
  warnings: Warning[] | null;
}

const TYPE_ORDER: SemanticType[] = ["numeric", "categorical", "datetime", "boolean", "text", "id", "unknown"];

const SEVERITY_ORDER: Record<WarningSeverity, number> = { severe: 0, warning: 1, info: 2 };
const SEVERITY_EDGE: Record<WarningSeverity, string> = {
  severe: "border-danger text-foreground",
  warning: "border-warning text-foreground",
  info: "border-border text-muted-foreground",
};
const SEVERITY_LABEL: Record<WarningSeverity, string> = { severe: "嚴重", warning: "注意", info: "說明" };

const KIND_LABEL: Record<DerivedColumn["kind"], string> = {
  product: "乘積",
  product_discount: "乘積（含折扣）",
  sum: "加總",
  difference: "差",
  ratio: "比值",
  near_copy: "近似複製",
};

const dateTime = new Intl.DateTimeFormat("zh-Hant", { dateStyle: "medium", timeStyle: "short" });

function pct(value: number): string {
  return `${(value * 100).toFixed(value < 0.1 && value > 0 ? 2 : 1)}%`;
}

function Tile({ label, value, note, name }: { label: string; value: string; note?: string; name: string }) {
  return (
    <div className="min-w-[8rem] border-l border-subtle pl-3" data-summary-tile={name}>
      <dt className="text-[11px] uppercase tracking-wider text-muted-foreground">{label}</dt>
      <dd className="mt-0.5 text-lg font-semibold tabular-nums" data-summary-value>
        {value}
      </dd>
      {note && <dd className="text-xs text-muted-foreground">{note}</dd>}
    </div>
  );
}

function SectionTitle({ children, aside }: { children: string; aside?: string }) {
  return (
    <div className="mb-2 flex items-baseline justify-between gap-3">
      <h2 className="text-sm font-medium">{children}</h2>
      {aside && <span className="text-xs text-muted-foreground">{aside}</span>}
    </div>
  );
}

/** Understand a dataset in thirty seconds: size, types, quality, what is
 *  definitional (derived / duplicate columns) and what the engine warns
 *  about. Typography and spacing carry the hierarchy; one surface at most. */
export function OverviewPage({ meta, profile, warnings }: Props) {
  const [typeFilter, setTypeFilter] = useState<SemanticType | null>(null);

  const stats = useMemo(() => {
    if (!profile) return null;
    const cols = profile.columns;
    const missing = cols.length ? cols.reduce((s, c) => s + c.missing_ratio, 0) / cols.length : 0;
    const counts = new Map<SemanticType, number>();
    for (const c of cols) counts.set(c.semantic_type, (counts.get(c.semantic_type) ?? 0) + 1);
    const flagged = cols.filter(hasQualityFlags).length;
    const evidence = profile.evidence ?? {};
    const derived: DerivedColumn[] = (evidence.derived_columns ?? []).filter((d) => d.kind !== "near_copy");
    const nearDup: NearDuplicateGroup[] = evidence.near_duplicate_groups ?? [];
    return { missing, counts, flagged, derived, nearDup };
  }, [profile]);

  const sortedWarnings = useMemo(
    () => (warnings ? [...warnings].sort((a, b) => SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity]) : null),
    [warnings],
  );

  const uploaded = (() => {
    const d = new Date(meta.uploaded_at);
    return Number.isNaN(d.getTime()) ? meta.uploaded_at : dateTime.format(d);
  })();

  return (
    <div data-page="overview" className="space-y-7">
      <PageHeader
        title={meta.filename}
        subtitle={
          <>
            上傳於 {uploaded}
            {profile?.sampled && " · 統計基於抽樣"}
          </>
        }
      />

      {/* --- summary ---------------------------------------------------- */}
      {stats && profile ? (
        <dl className="flex flex-wrap gap-x-8 gap-y-4" data-summary>
          <Tile name="rows" label="Rows" value={meta.n_rows.toLocaleString()} />
          <Tile name="columns" label="Columns" value={String(meta.n_cols)} />
          <Tile
            name="missing"
            label="Missing"
            value={pct(stats.missing)}
            note={`${profile.columns.filter((c) => c.missing_count > 0).length} 欄有缺失`}
          />
          <Tile
            name="profiled"
            label="Profiled rows"
            value={(profile.profiled_rows || profile.n_rows).toLocaleString()}
            note={profile.sampled ? "抽樣" : "全量"}
          />
          <Tile
            name="quality"
            label="Quality flags"
            value={String(stats.flagged)}
            note={stats.flagged === 0 ? "無異常" : "欄位有旗標"}
          />
        </dl>
      ) : (
        <div className="flex gap-8" aria-label="載入 profile 中…">
          {[0, 1, 2, 3, 4].map((i) => (
            <Skeleton key={i} className="h-12 w-28" />
          ))}
        </div>
      )}

      {/* --- column types + table ---------------------------------------- */}
      {stats && profile && (
        <section aria-labelledby="overview-columns">
          <div className="mb-2 flex flex-wrap items-baseline justify-between gap-3">
            <h2 id="overview-columns" className="text-sm font-medium">
              欄位
            </h2>
            <span className="text-xs text-muted-foreground">
              {typeFilter ? `顯示 ${stats.counts.get(typeFilter) ?? 0} / ${profile.n_cols} 欄` : `${profile.n_cols} 欄`}
            </span>
          </div>
          <div className="mb-3 flex flex-wrap gap-1.5" role="group" aria-label="依型別篩選欄位" data-type-summary>
            {TYPE_ORDER.filter((t) => stats.counts.has(t)).map((t) => {
              const active = typeFilter === t;
              return (
                <button
                  key={t}
                  type="button"
                  aria-pressed={active}
                  data-type-chip={t}
                  onClick={() => setTypeFilter(active ? null : t)}
                  className={cn(
                    "flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs transition-colors focus:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                    active
                      ? "border-primary/60 bg-primary/10 text-foreground"
                      : "border-subtle text-muted-foreground hover:bg-accent/60 hover:text-foreground",
                  )}
                >
                  <Badge variant={TYPE_VARIANT[t]} className="px-1 py-0">
                    {t}
                  </Badge>
                  <span className="tabular-nums" data-type-count>
                    {stats.counts.get(t)}
                  </span>
                </button>
              );
            })}
            {typeFilter && (
              <button
                type="button"
                className="rounded-md px-2 py-1 text-xs text-muted-foreground hover:text-foreground focus:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                onClick={() => setTypeFilter(null)}
              >
                清除篩選
              </button>
            )}
          </div>
          <DatasetOverview profile={profile} filter={typeFilter} />
        </section>
      )}

      {/* --- derived fields ---------------------------------------------- */}
      {stats && (
        <section aria-labelledby="overview-derived" data-derived-section>
          <SectionTitle aside="定義性關係：圖表引擎不把它們當發現">Derived fields</SectionTitle>
          <span id="overview-derived" className="sr-only">
            Derived fields
          </span>
          {stats.derived.length === 0 && stats.nearDup.length === 0 ? (
            <p className="text-sm text-muted-foreground" data-derived-empty>
              未偵測到定義性關係。
            </p>
          ) : (
            <ul className="space-y-1.5 text-sm">
              {stats.derived.map((d) => (
                <li key={d.target} className="flex flex-wrap items-baseline gap-x-2" data-derived-column={d.target}>
                  <span className="font-medium">{d.target}</span>
                  <span className="text-muted-foreground">=</span>
                  <code className="rounded bg-muted px-1.5 py-0.5 text-xs">{d.formula}</code>
                  <span className="text-xs text-muted-foreground">
                    {KIND_LABEL[d.kind]} · {pct(d.match_ratio)} 列成立 · n = {d.n.toLocaleString()}
                  </span>
                </li>
              ))}
              {stats.nearDup.map((g) => (
                <li key={g.representative} className="flex flex-wrap items-baseline gap-x-2" data-near-duplicate={g.representative}>
                  <span className="font-medium">{g.duplicates.join(", ")}</span>
                  <span className="text-muted-foreground">≈</span>
                  <span className="font-medium">{g.representative}</span>
                  <span className="text-xs text-muted-foreground">
                    近似複製（rank correlation{" "}
                    {g.duplicates.map((d) => (g.rho[d] ?? 0).toFixed(2)).join(", ")}）· 推薦只用 {g.representative}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </section>
      )}

      {/* --- warnings ---------------------------------------------------- */}
      <section aria-labelledby="overview-warnings" data-warnings-section>
        <SectionTitle aside={sortedWarnings ? `${sortedWarnings.length} 則` : "載入中"}>Warnings</SectionTitle>
        <span id="overview-warnings" className="sr-only">
          Warnings
        </span>
        {sortedWarnings === null ? (
          <Skeleton className="h-8 w-2/3" />
        ) : sortedWarnings.length === 0 ? (
          <p className="text-sm text-muted-foreground" data-warnings-empty>
            推薦引擎沒有針對這份資料集的警告。
          </p>
        ) : (
          <ul className="space-y-1.5">
            {sortedWarnings.map((w, i) => (
              <li
                key={`${w.code}-${i}`}
                data-warning-severity={w.severity}
                className={cn("border-l-2 pl-3 text-sm", SEVERITY_EDGE[w.severity] ?? SEVERITY_EDGE.info)}
              >
                <span className="mr-2 text-xs uppercase tracking-wider text-muted-foreground">
                  {SEVERITY_LABEL[w.severity] ?? w.severity}
                </span>
                {w.message}
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
