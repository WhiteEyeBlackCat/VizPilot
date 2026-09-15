import type { ColumnProfile, DatasetProfile, SemanticType } from "../types";
import { SectionCard } from "./SectionCard";
import { Badge, type BadgeVariant } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

const TYPE_VARIANT: Record<SemanticType, BadgeVariant> = {
  numeric: "info",
  categorical: "success",
  datetime: "accent",
  boolean: "warning",
  text: "muted",
  id: "muted",
  unknown: "muted",
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
  const title = (
    <>
      B. 資料總覽
      <span className="ml-2 text-sm font-normal text-muted-foreground">
        {profile.n_rows} 列 × {profile.n_cols} 欄
        {profile.sampled && "（統計基於抽樣）"}
      </span>
    </>
  );
  return (
    <SectionCard title={title}>
      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>欄位</TableHead>
              <TableHead>型別</TableHead>
              <TableHead>缺失</TableHead>
              <TableHead>唯一值</TableHead>
              <TableHead>統計摘要</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {profile.columns.map((c) => (
              <TableRow key={c.name}>
                <TableCell className="font-medium">{c.name}</TableCell>
                <TableCell>
                  <Badge variant={TYPE_VARIANT[c.semantic_type] ?? "muted"}>
                    {c.semantic_type}
                    {c.nominal && <span className="ml-1 font-normal opacity-70">code</span>}
                  </Badge>
                </TableCell>
                <TableCell className="text-muted-foreground">
                  {(c.missing_ratio * 100).toFixed(1)}%
                </TableCell>
                <TableCell className="text-muted-foreground">{c.unique_count}</TableCell>
                <TableCell className="text-muted-foreground">{summary(c)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </SectionCard>
  );
}
