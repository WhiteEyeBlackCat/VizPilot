import { Sparkles } from "lucide-react";

import { PageHeader } from "../components/PageHeader";
import { Recommendations } from "../components/Recommendations";
import type { Recommendation, RecommendationsResponse } from "../types";
import { Badge } from "@/components/ui/badge";

interface Props {
  recs: RecommendationsResponse | null;
  aiPending: boolean;
  selectedPriority: number | null;
  onPreview: (rec: Recommendation) => Promise<void>;
}

export function InsightsPage({ recs, aiPending, selectedPriority, onPreview }: Props) {
  return (
    <div data-page="insights">
      <PageHeader
        title="Insights"
        subtitle="系統依統計證據推薦的圖表；點選卡片在預覽面板檢視，按 Save 加入 Workspace。"
        aside={
          aiPending && (
            <Badge variant="accent" className="animate-pulse font-normal">
              <Sparkles className="mr-1 h-3 w-3" />
              AI 分析中…
            </Badge>
          )
        }
      />
      <Recommendations
        recs={recs}
        aiPending={aiPending}
        selectedPriority={selectedPriority}
        onPreview={onPreview}
      />
    </div>
  );
}
