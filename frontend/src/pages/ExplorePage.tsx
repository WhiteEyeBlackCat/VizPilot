import { ManualBuilder } from "../components/ManualBuilder";
import { PageHeader } from "../components/PageHeader";
import type { ChartSpec, DatasetProfile } from "../types";
import { Skeleton } from "@/components/ui/skeleton";

interface Props {
  profile: DatasetProfile | null;
  onGenerate: (spec: ChartSpec) => Promise<void>;
}

export function ExplorePage({ profile, onGenerate }: Props) {
  return (
    <div data-page="explore">
      <PageHeader
        title="Explore"
        subtitle="自行選擇圖表類型與欄位；生成後在預覽面板檢視，按 Save 才會加入 Workspace。"
      />
      {profile ? (
        <ManualBuilder profile={profile} onGenerate={onGenerate} />
      ) : (
        <Skeleton className="h-16" aria-label="載入 profile 中…" />
      )}
    </div>
  );
}
