import type { ColumnProfile, DatasetProfile } from "../types";

const TYPE_COLORS: Record<string, string> = {
  numeric: "bg-blue-100 text-blue-700",
  categorical: "bg-emerald-100 text-emerald-700",
  datetime: "bg-purple-100 text-purple-700",
  boolean: "bg-amber-100 text-amber-700",
  text: "bg-slate-200 text-slate-600",
  id: "bg-slate-200 text-slate-600",
  unknown: "bg-slate-200 text-slate-500",
};

function fmt(value: number | string | null): string {
  if (value === null) return "—";
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(2);
  }
  return value;
}

function summary(c: ColumnProfile): string {
  switch (c.semantic_type) {
    case "numeric":
      return `mean ${fmt(c.mean)} · std ${fmt(c.std)} · 範圍 [${fmt(c.min)}, ${fmt(c.max)}]`;
    case "categorical":
    case "boolean":
      return `${c.n_categories ?? 0} 類 · top: ${(c.top_values ?? [])
        .slice(0, 3)
        .map((t) => String(t.value))
        .join(", ")}`;
    case "datetime":
      return `${fmt(c.min)} → ${fmt(c.max)} · ${c.inferred_frequency ?? "unknown"}`;
    case "text":
      return `平均長度 ${fmt(c.avg_length)} · 最長 ${c.max_length ?? "—"}`;
    default:
      return "—";
  }
}

export function DatasetOverview({ profile }: { profile: DatasetProfile }) {
  return (
    <section className="rounded-lg bg-white p-4 shadow">
      <h2 className="mb-2 font-semibold">
        B. 資料總覽
        <span className="ml-2 text-sm font-normal text-slate-500">
          {profile.n_rows} 列 × {profile.n_cols} 欄
          {profile.sampled && "（統計基於抽樣）"}
        </span>
      </h2>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-xs uppercase text-slate-500">
              <th className="py-1.5 pr-3">欄位</th>
              <th className="py-1.5 pr-3">型別</th>
              <th className="py-1.5 pr-3">缺失</th>
              <th className="py-1.5 pr-3">唯一值</th>
              <th className="py-1.5">統計摘要</th>
            </tr>
          </thead>
          <tbody>
            {profile.columns.map((c) => (
              <tr key={c.name} className="border-b border-slate-100">
                <td className="py-1.5 pr-3 font-medium">{c.name}</td>
                <td className="py-1.5 pr-3">
                  <span className={`rounded px-1.5 py-0.5 text-xs ${TYPE_COLORS[c.semantic_type]}`}>
                    {c.semantic_type}
                  </span>
                </td>
                <td className="py-1.5 pr-3 text-slate-500">{(c.missing_ratio * 100).toFixed(1)}%</td>
                <td className="py-1.5 pr-3 text-slate-500">{c.unique_count}</td>
                <td className="py-1.5 text-slate-600">{summary(c)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
