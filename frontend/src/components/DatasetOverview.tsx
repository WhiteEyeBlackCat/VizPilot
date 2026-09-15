import type { ColumnProfile, DatasetProfile, SemanticType } from "../types";
import { Badge, type BadgeVariant } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

export const TYPE_VARIANT: Record<SemanticType, BadgeVariant> = {
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

// an extreme-value share below this is ordinary tail behaviour, not a flag
const EXTREME_FLAG_RATIO = 0.01;

export interface QualityFlag {
  kind: "missing_token" | "invalid" | "sentinel" | "extreme" | "nominal";
  label: string;
  detail?: string;
  variant: BadgeVariant;
}

/** Data-quality flags for one column (stage 9 quality block + nominal). */
export function qualityFlags(c: ColumnProfile): QualityFlag[] {
  const q = c.quality;
  const flags: QualityFlag[] = [];
  if (c.nominal) flags.push({ kind: "nominal", label: "code", detail: "數值編碼的類別：是標籤，不是數量", variant: "muted" });
  if (!q) return flags;
  if (q.missing_token_count > 0) {
    flags.push({ kind: "missing_token", label: `缺失符號 ${q.missing_token_count}`, detail: "N/A、-、空白等被視為缺失的值", variant: "warning" });
  }
  if (q.invalid_count > 0) {
    flags.push({ kind: "invalid", label: `無法解析 ${q.invalid_count}`, detail: "無法轉成數值 / 日期的值（例如 12kg、A123）", variant: "warning" });
  }
  if (q.suspected_sentinels.length > 0) {
    const values = q.suspected_sentinels.map((s) => `${fmt(s.value)} ×${s.count}（${s.signals.join(", ")}）`).join("；");
    flags.push({
      kind: "sentinel",
      label: `疑似 sentinel ${q.suspected_sentinels.map((s) => fmt(s.value)).join(", ")}`,
      detail: `${values}；共 ${q.sentinel_row_count} 列。這些值離群且重複／符合常見填充值，統計未剔除，圖表以 robust 範圍顯示。`,
      variant: "danger",
    });
  }
  if (q.extreme_value_ratio >= EXTREME_FLAG_RATIO) {
    flags.push({
      kind: "extreme",
      label: `極端值 ${(q.extreme_value_ratio * 100).toFixed(1)}%`,
      detail: `${q.extreme_value_count} 列落在 far-out fences 之外（合法的長尾，只影響可讀性）`,
      variant: "muted",
    });
  }
  return flags;
}

export function hasQualityFlags(c: ColumnProfile): boolean {
  return qualityFlags(c).some((f) => f.kind !== "nominal");
}

interface Props {
  profile: DatasetProfile;
  /** show only columns of this semantic type */
  filter?: SemanticType | null;
}

export function DatasetOverview({ profile, filter = null }: Props) {
  const rows = filter ? profile.columns.filter((c) => c.semantic_type === filter) : profile.columns;
  return (
    <div data-column-table data-filter={filter ?? ""}>
      {/* the wrapper scrolls horizontally on narrow pages: keyboard users
          need to reach it */}
      <Table
        className="min-w-[720px]"
        wrapperProps={{
          tabIndex: 0,
          role: "region",
          "aria-label": "欄位表（可橫向捲動）",
          className: "focus:outline-none focus-visible:ring-1 focus-visible:ring-ring",
        }}
      >
        <TableHeader>
          <TableRow>
            <TableHead>欄位</TableHead>
            <TableHead>型別</TableHead>
            {/* quality flags sit before the wide summary column so they stay
                in view when the preview panel narrows the page */}
            <TableHead>品質</TableHead>
            <TableHead className="text-right">缺失</TableHead>
            <TableHead className="text-right">唯一值</TableHead>
            <TableHead>統計摘要</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((c) => {
            const flags = qualityFlags(c);
            return (
              <TableRow key={c.name} data-column-row={c.name}>
                <TableCell className="font-medium">{c.name}</TableCell>
                <TableCell>
                  <Badge variant={TYPE_VARIANT[c.semantic_type] ?? "muted"}>{c.semantic_type}</Badge>
                </TableCell>
                <TableCell>
                  {flags.length === 0 ? (
                    <span className="text-muted-foreground/60">—</span>
                  ) : (
                    <span className="flex flex-wrap gap-1">
                      {flags.map((f) => (
                        <Tooltip key={f.kind}>
                          <TooltipTrigger asChild>
                            <Badge
                              variant={f.variant}
                              className="cursor-help font-normal"
                              data-quality-flag={f.kind}
                              tabIndex={0}
                            >
                              {f.label}
                            </Badge>
                          </TooltipTrigger>
                          {f.detail && <TooltipContent className="max-w-xs">{f.detail}</TooltipContent>}
                        </Tooltip>
                      ))}
                    </span>
                  )}
                </TableCell>
                <TableCell className="text-right tabular-nums text-muted-foreground">
                  {(c.missing_ratio * 100).toFixed(1)}%
                </TableCell>
                <TableCell className="text-right tabular-nums text-muted-foreground">{c.unique_count}</TableCell>
                <TableCell className="text-muted-foreground">{summary(c)}</TableCell>
              </TableRow>
            );
          })}
          {rows.length === 0 && (
            <TableRow>
              <TableCell colSpan={6} className="text-center text-muted-foreground">
                沒有此型別的欄位
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </div>
  );
}
