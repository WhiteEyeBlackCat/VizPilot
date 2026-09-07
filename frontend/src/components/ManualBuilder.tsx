import { useEffect, useMemo, useState } from "react";

import { ApiError } from "../api";
import type { Aggregation, ChartSpec, ChartType, ColumnProfile, DatasetProfile } from "../types";

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
    const numericY = cols.filter(
      (c) =>
        c.semantic_type === "numeric" ||
        (c.semantic_type === "categorical" && NUMERIC_DTYPE.test(c.original_dtype)),
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

  const select = (
    label: string,
    value: string,
    setter: (v: string) => void,
    choices: string[],
    disabled = false,
  ) => (
    <label className="flex flex-col gap-1 text-xs text-slate-500">
      {label}
      <select
        className="rounded border border-slate-300 px-2 py-1 text-sm text-slate-800 disabled:bg-slate-100"
        value={value}
        disabled={disabled || choices.length === 0}
        onChange={(e) => setter(e.target.value)}
      >
        <option value="">—</option>
        {choices.map((c) => (
          <option key={c} value={c}>
            {c}
          </option>
        ))}
      </select>
    </label>
  );

  return (
    <section className="rounded-lg bg-white p-4 shadow">
      <h2 className="mb-3 font-semibold">D. 手動建圖</h2>
      <div className="flex flex-wrap items-end gap-3">
        <label className="flex flex-col gap-1 text-xs text-slate-500">
          圖表類型
          <select
            className="rounded border border-slate-300 px-2 py-1 text-sm text-slate-800"
            value={type}
            onChange={(e) => setType(e.target.value as ChartType)}
          >
            {CHART_TYPES.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </label>
        {select("X 軸", x, setX, options.x)}
        {select("Y 軸", y, setY, yOptions)}
        {select("分組", group, setGroup, groupOptions)}
        {showAgg && select("聚合", agg, setAgg, AGGREGATIONS, type === "bar" && !y)}
        <button
          className="rounded bg-blue-600 px-4 py-1.5 text-sm text-white hover:bg-blue-700 disabled:opacity-50"
          disabled={busy}
          onClick={() => void generate()}
        >
          {busy ? "生成中…" : "生成圖表"}
        </button>
      </div>
      {errors.length > 0 && (
        <ul className="mt-2 list-inside list-disc text-sm text-red-600">
          {errors.map((e, i) => (
            <li key={i}>{e}</li>
          ))}
        </ul>
      )}
    </section>
  );
}
