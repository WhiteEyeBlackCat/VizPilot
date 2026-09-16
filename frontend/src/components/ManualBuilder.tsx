import { useCallback, useEffect, useMemo, useState } from "react";

import { ApiError } from "../api";
import type { ExploreField, ExploreForm, FieldUpdate } from "../store";
import { CHART_TYPES, type Aggregation, type ChartSpec, type ChartType, type ColumnProfile, type DatasetProfile } from "../types";
import { ErrorList } from "./ErrorList";
import { FieldSelect } from "./FieldSelect";
import { Button } from "@/components/ui/button";

interface Props {
  profile: DatasetProfile;
  /** form values live in the app store (stage 16.2) so they survive
   *  navigation and preview-panel changes */
  form: ExploreForm;
  onField: (field: ExploreField, value: FieldUpdate<string>) => void;
  onGenerate: (spec: ChartSpec) => Promise<void>;
}

// which types take an aggregation / a group column (mirrors backend spec.py)
const AGG_TYPES: ChartType[] = ["line", "bar", "lollipop", "polar_line", "rose", "dumbbell", "slopegraph"];
const BAR_LIKE: ChartType[] = ["bar", "lollipop", "polar_line", "rose"];
const GROUP_TYPES: ChartType[] = ["line", "bar", "scatter", "histogram", "lollipop", "dumbbell", "slopegraph"];
const AGGREGATIONS: Aggregation[] = ["mean", "sum", "count", "median", "min", "max"];
const NUMERIC_DTYPE = /^(U?Int|Float)\d+$/;

const names = (cols: ColumnProfile[]) => cols.map((c) => c.name);

export function ManualBuilder({ profile, form, onField, onGenerate }: Props) {
  // same names and setter semantics as the former useState pairs (value or
  // updater), backed by the store; the logic below is unchanged
  const { type, x, y, group, agg } = form;
  const setType = useCallback((v: FieldUpdate<string>) => onField("type", v), [onField]);
  const setX = useCallback((v: FieldUpdate<string>) => onField("x", v), [onField]);
  const setY = useCallback((v: FieldUpdate<string>) => onField("y", v), [onField]);
  const setGroup = useCallback((v: FieldUpdate<string>) => onField("group", v), [onField]);
  const setAgg = useCallback((v: FieldUpdate<string>) => onField("agg", v), [onField]);
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
    const cats = (lo: number, hi: number) =>
      categorical.filter((c) => {
        const n = c.n_categories ?? c.unique_count;
        return n >= lo && n <= hi;
      });
    const boxX = cats(2, 50);
    const twoState = cats(2, 2);
    // stage 19: the field matrix per type mirrors backend validate_spec; the
    // ordered / cyclic role of a category (line over month, rose over hour) is
    // the backend's call — its 422 message says so when a column has none
    const xByType: Record<ChartType, string[]> = {
      line: names([...datetime, ...numeric, ...groupable]),
      bar: names(categorical),
      scatter: names(numeric),
      histogram: names(numeric),
      box: names(boxX),
      heatmap: [],
      violin: names(boxX),
      raincloud: names(cats(2, 8)),
      ridgeline: names(cats(2, 12)),
      dot_interval: names(boxX),
      lollipop: names(categorical),
      dumbbell: names(boxX),
      slopegraph: names(cats(2, 8)),
      hexbin: names(numeric),
      polar_line: names([...groupable, ...numeric]),
      rose: names([...groupable, ...numeric]),
    };
    const yByType: Record<ChartType, string[]> = {
      line: names(numericY),
      bar: names(numericY),
      scatter: names(numeric),
      histogram: [],
      box: names(numericY),
      heatmap: [],
      violin: names(numericY),
      raincloud: names(numericY),
      ridgeline: names(numericY),
      dot_interval: names(numericY),
      lollipop: names(numericY),
      dumbbell: names(numericY),
      slopegraph: names(numericY),
      hexbin: names(numeric),
      polar_line: names(numericY),
      rose: names(numericY),
    };
    const groupChoices = type === "dumbbell" || type === "slopegraph" ? names(twoState) : GROUP_TYPES.includes(type) ? names(groupable) : [];
    return { x: xByType[type], y: yByType[type], group: groupChoices };
  }, [cols, type]);

  const showAgg = AGG_TYPES.includes(type);

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
    if (BAR_LIKE.includes(type)) {
      if (!y) setAgg("count");
      else if (type === "rose") setAgg((prev) => (prev === "sum" || prev === "count" ? prev : "sum")); // a rose is a magnitude
      else setAgg((prev) => (prev && prev !== "count" ? prev : "mean"));
      return;
    }
    if (type === "dumbbell" || type === "slopegraph") {
      setAgg((prev) => prev || "mean");
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
    <div className="space-y-3" data-manual-builder>
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
            disabled={BAR_LIKE.includes(type) && !y}
          />
        )}
        <Button size="sm" className="h-8" disabled={busy} onClick={() => void generate()}>
          {busy ? "生成中…" : "生成圖表"}
        </Button>
      </div>
      <ErrorList errors={errors} />
    </div>
  );
}
