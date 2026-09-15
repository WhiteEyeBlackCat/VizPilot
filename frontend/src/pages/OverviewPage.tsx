import { DatasetOverview } from "../components/DatasetOverview";
import { PageHeader } from "../components/PageHeader";
import type { DatasetMeta, DatasetProfile } from "../types";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";

interface Props {
  meta: DatasetMeta;
  profile: DatasetProfile | null;
}

/** Stage 16.2 keeps the existing column table; summary cards, data quality,
 *  derived fields and warnings arrive in 16.3. */
export function OverviewPage({ meta, profile }: Props) {
  return (
    <div data-page="overview">
      <PageHeader
        title={meta.filename}
        subtitle={
          <>
            {meta.n_rows} 列 × {meta.n_cols} 欄
            {profile?.sampled && "（統計基於抽樣）"}
          </>
        }
      />
      <div className="mb-5 flex flex-wrap gap-1">
        {meta.columns.map((c) => (
          <Badge key={c.name} variant="muted" className="font-normal">
            {c.name}
            <span className="ml-1 text-muted-foreground">({c.dtype})</span>
          </Badge>
        ))}
      </div>
      {profile ? (
        <DatasetOverview profile={profile} />
      ) : (
        <div className="space-y-2" aria-label="載入 profile 中…">
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-8" />
          ))}
        </div>
      )}
    </div>
  );
}
