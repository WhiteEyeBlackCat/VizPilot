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
}

export function ExplorePage({ profile, form, onField, onGenerate }: Props) {
  return (
    <div data-page="explore">
      <PageHeader
        title="Explore"
        subtitle="自行選擇圖表類型與欄位；生成後在預覽面板檢視，按 Save 才會加入 Workspace。"
      />
      {profile ? (
        <ManualBuilder profile={profile} form={form} onField={onField} onGenerate={onGenerate} />
      ) : (
        <Skeleton className="h-16" aria-label="載入 profile 中…" />
      )}
    </div>
  );
}
