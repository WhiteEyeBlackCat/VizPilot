import { ChartView } from "./ChartView";
import type { RenderResult } from "../types";

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
    <section className="rounded-lg bg-white p-4 shadow">
      <h2 className="mb-3 font-semibold">已生成圖表</h2>
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        {charts.map((chart) => (
          <div key={chart.key} className="relative rounded border border-slate-200 p-2">
            <button
              className="absolute right-2 top-2 z-10 rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-500 hover:bg-red-100 hover:text-red-600"
              onClick={() => onRemove(chart.key)}
            >
              移除
            </button>
            <ChartView result={chart.result} />
          </div>
        ))}
      </div>
    </section>
  );
}
