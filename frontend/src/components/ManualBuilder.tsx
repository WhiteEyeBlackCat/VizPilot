import { useEffect, useMemo, useState } from "react";

import { ApiError } from "../api";
import type { Aggregation, ChartSpec, ChartType, ColumnProfile, DatasetProfile } from "../types";
import { ErrorList } from "./ErrorList";
import { FieldSelect } from "./FieldSelect";
import { SectionCard } from "./SectionCard";
import { Button } from "@/components/ui/button";

interface Props {
  profile: DatasetProfile;
  onGenerate: (spec: ChartSpec) => Promise<void>;
}

const CHART_TYPES: ChartType[] = ["line", "bar", "scatter", "histogram", "box", "heatmap"];
const AGGREGATIONS: Aggregation[] = ["mean", "sum", "count", "median", "min", "max"];
const NUMERIC_DTYPE = /^(U?Int|Float)\d+$/;

const names = (cols: ColumnProfile[]) => cols.map((c) => c.name);

export function ManualBuilder({ profile, onGenerate }: Props) {
  const [type, setType] = useState<ChartType>("bar");
  const [x, setX] = useState("");
  const [y, setY] = useState("");
  const [group, setGroup] = useState("");
  const [agg, setAgg] = useState("");
  const [busy, setBusy] = useState(false);
  const [errors, setErrors] = useState<string[]>([]);

  const cols = profile.columns;
  const options = useMemo(() => {
    const numeric = cols.filter((c) => c.semantic_type === "numeric");
    // mirrors backend spec._is_numeric_y: numeric-backed categoricals are an
    // acceptable y unless they are nominal codes (stage 9 #4 — never a quantity)
    const numericY = cols.filter(
      (c) =>
        c.semantic_type === "numeric" ||
        (c.semantic_type === "categorical" && !c.nominal && NUMERIC_DTYPE.test(c.original_dtype)),
    );
    const datetime = cols.filter((c) => c.semantic_type === "datetime");
    const categorical = cols.filter(
      (c) => c.semantic_type === "categorical" || c.semantic_type === "boolean",
    );
    const groupable = categorical.filter((c) => (c.n_categories ?? c.unique_count) <= 20);
    const boxX = categorical.filter((c) => {
      const n = c.n_categories ?? c.unique_count;
      return n >= 2 && n <= 50;
    });
    return {
      x: {
        line: names([...datetime, ...numeric]),
        bar: names(categorical),
        scatter: names(numeric),
        histogram: names(numeric),
        box: names(boxX),
        heatmap: [] as string[],
      }[type],
      y: {
        line: names(numericY),
        bar: names(numericY),
        scatter: names(numeric),
        histogram: [] as string[],
        box: names(numericY),
        heatmap: [] as string[],
      }[type],
      group: type === "box" || type === "heatmap" ? [] : names(groupable),
    };
  }, [cols, type]);

  const showAgg = type === "line" || type === "bar";

  // axes must be pairwise distinct (stage3 validation): hide already-picked
  // columns from the other selects instead of round-tripping a 422
  const yOptions = options.y.filter((n) => n !== x);
  const groupOptions = options.group.filter((n) => n !== x && n !== y);

  // drop selections that are no longer valid for the chosen type
  useEffect(() => {
    setX((prev) => (options.x.includes(prev) ? prev : ""));
    setY((prev) => (options.y.includes(prev) ? prev : ""));
    setGroup((prev) => (options.group.includes(prev) ? prev : ""));
    if (!showAgg) setAgg("");
  }, [options, showAgg]);

  useEffect(() => {
    setY((prev) => (prev === x ? "" : prev));
    setGroup((prev) => (prev === x || prev === y ? "" : prev));
  }, [x, y]);

  // aggregation prefill (stage6 blocking #2): group_by, or a line x with
  // duplicate values (long-format data), requires an aggregation upstream
  useEffect(() => {
    if (type === "bar") {
      if (!y) setAgg("count");
      else setAgg((prev) => (prev && prev !== "count" ? prev : "mean"));
      return;
    }
    if (type !== "line") return;
    const xCol = cols.find((c) => c.name === x);
    const duplicateX =
      xCol !== undefined && xCol.unique_count < profile.n_rows - xCol.missing_count;
    if (group !== "" || duplicateX) setAgg((prev) => prev || "mean");
  }, [type, x, y, group, cols, profile.n_rows]);

  const generate = async () => {
    setBusy(true);
    setErrors([]);
    const spec: ChartSpec = {
      title: `${type}: ${[y, x].filter(Boolean).join(" by ") || "correlations"}`,
      type,
      x: x || null,
      y: y || null,
      group_by: group || null,
      aggregation: showAgg && agg ? (agg as Aggregation) : null,
    };
    try {
      await onGenerate(spec);
    } catch (e) {
      setErrors(e instanceof ApiError ? e.errors : [String(e)]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <SectionCard title="手動建圖">
      <div className="flex flex-wrap items-end gap-3">
        <FieldSelect
          label="圖表類型"
          value={type}
          onChange={(v) => setType(v as ChartType)}
          choices={CHART_TYPES}
          allowNone={false}
        />
        <FieldSelect label="X 軸" value={x} onChange={setX} choices={options.x} />
        <FieldSelect label="Y 軸" value={y} onChange={setY} choices={yOptions} />
        <FieldSelect label="分組" value={group} onChange={setGroup} choices={groupOptions} />
        {showAgg && (
          <FieldSelect
            label="聚合"
            value={agg}
            onChange={setAgg}
            choices={AGGREGATIONS}
            disabled={type === "bar" && !y}
          />
        )}
        <Button size="sm" className="h-8" disabled={busy} onClick={() => void generate()}>
          {busy ? "生成中…" : "生成圖表"}
        </Button>
      </div>
      <ErrorList errors={errors} />
    </SectionCard>
  );
}
