import { Maximize2, X } from "lucide-react";
import { useState } from "react";

import { hideAllTips } from "../charts/echarts/EChartsView";
import type { RenderResult } from "../types";
import { ChartView } from "./ChartView";
import { SectionCard } from "./SectionCard";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";

export interface WorkspaceChart {
  key: string;
  result: RenderResult;
}

interface Props {
  charts: WorkspaceChart[];
  onRemove: (key: string) => void;
}

export function ChartsWorkspace({ charts, onRemove }: Props) {
  // the enlarged view re-renders the same RenderResult in a second ECharts
  // instance (with its own toolbox); closing the dialog unmounts and
  // disposes it
  const [enlarged, setEnlarged] = useState<WorkspaceChart | null>(null);

  return (
    <SectionCard title="已生成圖表">
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        {charts.map((chart) => (
          <Card key={chart.key} className="relative p-2 shadow-none">
            <div className="absolute right-2 top-2 z-10 flex gap-1">
              <Button
                variant="ghost"
                size="sm"
                className="h-7 px-2 text-xs text-muted-foreground"
                onClick={() => {
                  hideAllTips(); // a hovered inline tooltip must not float above the overlay
                  setEnlarged(chart);
                }}
              >
                <Maximize2 className="h-3.5 w-3.5" />
                放大
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="h-7 px-2 text-xs text-muted-foreground hover:text-destructive"
                onClick={() => onRemove(chart.key)}
              >
                <X className="h-3.5 w-3.5" />
                移除
              </Button>
            </div>
            <ChartView result={chart.result} />
          </Card>
        ))}
      </div>

      <Dialog open={enlarged !== null} onOpenChange={(open) => !open && setEnlarged(null)}>
        <DialogContent className="max-w-[90vw] p-4 pt-10" data-chart-dialog>
          {/* the chart draws its own title; keep the accessible name for the dialog */}
          <DialogTitle className="sr-only">{enlarged?.result.spec.title ?? "圖表"}</DialogTitle>
          <DialogDescription className="sr-only">放大檢視圖表</DialogDescription>
          {enlarged && <ChartView result={enlarged.result} height="70vh" />}
        </DialogContent>
      </Dialog>
    </SectionCard>
  );
}
