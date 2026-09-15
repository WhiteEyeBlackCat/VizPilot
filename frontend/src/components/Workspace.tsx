import { X } from "lucide-react";

import type { RenderResult } from "../types";
import { ChartView } from "./ChartView";
import { SectionCard } from "./SectionCard";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

export interface WorkspaceChart {
  key: string;
  result: RenderResult;
}

interface Props {
  charts: WorkspaceChart[];
  onRemove: (key: string) => void;
}

export function ChartsWorkspace({ charts, onRemove }: Props) {
  return (
    <SectionCard title="已生成圖表">
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        {charts.map((chart) => (
          <Card key={chart.key} className="relative p-2 shadow-none">
            <Button
              variant="ghost"
              size="sm"
              className="absolute right-2 top-2 z-10 h-7 px-2 text-xs text-muted-foreground hover:text-destructive"
              onClick={() => onRemove(chart.key)}
            >
              <X className="h-3.5 w-3.5" />
              移除
            </Button>
            <ChartView result={chart.result} />
          </Card>
        ))}
      </div>
    </SectionCard>
  );
}
