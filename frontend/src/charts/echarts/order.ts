// Display order of a category axis (stage 18). The backend ranks bar
// categories by aggregated value (render.py) and sorts box groups
// lexicographically; both are the right *data* order but the wrong *reading*
// order for columns with a natural sequence (hour, month, weekday, date).
// This module only permutes what the backend returned — no value, label or
// count changes — and says in words which order the chart is in, so a
// ranked axis is never mistaken for a chronological one.

import type { Aggregation, BarChartData, BoxChartData, RenderResult, SortOrder } from "../../types";

export type OrderKind = "numeric" | "weekday" | "month" | "date";

export interface DisplayOrder {
  kind: OrderKind;
  /** index[i] = position in the backend arrays of the i-th displayed category */
  index: number[];
}

type Category = string | number | boolean;

const WEEKDAYS: Record<string, number> = {
  mon: 0, monday: 0, tue: 1, tues: 1, tuesday: 1, wed: 2, wednesday: 2, thu: 3, thur: 3, thurs: 3, thursday: 3,
  fri: 4, friday: 4, sat: 5, saturday: 5, sun: 6, sunday: 6,
  週一: 0, 週二: 1, 週三: 2, 週四: 3, 週五: 4, 週六: 5, 週日: 6,
  星期一: 0, 星期二: 1, 星期三: 2, 星期四: 3, 星期五: 4, 星期六: 5, 星期日: 6,
};
const MONTHS: Record<string, number> = {
  jan: 0, january: 0, feb: 1, february: 1, mar: 2, march: 2, apr: 3, april: 3, may: 4, jun: 5, june: 5,
  jul: 6, july: 6, aug: 7, august: 7, sep: 8, sept: 8, september: 8, oct: 9, october: 9, nov: 10, november: 10,
  dec: 11, december: 11,
};
const NUMERIC = /^-?\d+(\.\d+)?$/;
// month 01–12 and day 01–31 are bounded so a code shaped like a date
// ("2024-13-99", a "2024-25" season) is never re-sorted as one
const ISO_DATE = /^\d{4}-(0[1-9]|1[0-2])(-(0[1-9]|[12]\d|3[01]))?([T ]\d{2}:\d{2}(:\d{2})?)?$/;

const key = (c: Category) => String(c).trim().toLowerCase();

function rank(categories: Category[], table: Record<string, number>): number[] | null {
  const ranks = categories.map((c) => table[key(c)]);
  return ranks.every((r) => r !== undefined) ? (ranks as number[]) : null;
}

function sortedIndex(keys: number[]): number[] {
  return keys.map((k, i) => [k, i] as const).sort((a, b) => a[0] - b[0] || a[1] - b[1]).map(([, i]) => i);
}

/** The natural order of the categories, or null when they have none (plain
 *  labels, booleans, mixed types): then the backend order stands. */
export function naturalOrder(categories: Category[]): DisplayOrder | null {
  if (categories.length < 2) return null;
  if (categories.some((c) => typeof c === "boolean")) return null;
  if (categories.every((c) => typeof c === "number" || NUMERIC.test(String(c).trim()))) {
    return { kind: "numeric", index: sortedIndex(categories.map((c) => Number(c))) };
  }
  const weekday = rank(categories, WEEKDAYS);
  if (weekday) return { kind: "weekday", index: sortedIndex(weekday) };
  const month = rank(categories, MONTHS);
  if (month) return { kind: "month", index: sortedIndex(month) };
  if (categories.every((c) => typeof c === "string" && ISO_DATE.test(c.trim()))) {
    const sorted = categories.map((c, i) => [String(c), i] as const).sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
    return { kind: "date", index: sorted.map(([, i]) => i) };
  }
  return null;
}

const identity = (n: number) => Array.from({ length: n }, (_, i) => i);

/** Order in which a bar's categories are displayed. A spec that names its
 *  order (stage 19 `sort`) is already ordered by the backend and is kept as
 *  is; only an unordered spec falls back to the natural-sequence heuristic. */
export function barOrder(d: BarChartData, sort?: SortOrder | null): number[] {
  if (sort) return identity(d.categories.length);
  return naturalOrder(d.categories)?.index ?? identity(d.categories.length);
}

/** Order in which a box plot's groups are displayed (same rule as the bar). */
export function boxOrder(d: BoxChartData, sort?: SortOrder | null): number[] {
  if (sort) return identity(d.groups.length);
  return naturalOrder(d.groups.map((g) => g.name))?.index ?? identity(d.groups.length);
}

const AGG_LABEL: Record<Aggregation, string> = {
  mean: "平均",
  sum: "總和",
  count: "筆數",
  median: "中位數",
  min: "最小值",
  max: "最大值",
};
const KIND_LABEL: Record<OrderKind, string> = {
  numeric: "數值順序",
  weekday: "星期順序",
  month: "月份順序",
  date: "時間順序",
};

/** One sentence naming the order of the category axis, or null when the
 *  chart has no category axis worth explaining. */
export function displayOrderNote(result: RenderResult): string | null {
  const { spec } = result;
  if (spec.type === "bar" || spec.type === "lollipop" || spec.type === "polar_line" || spec.type === "rose" || spec.type === "stacked_bar") {
    const d = result.chart_data as BarChartData;
    if (d.categories.length < 2) return null;
    const agg = spec.aggregation ? AGG_LABEL[spec.aggregation] : null;
    const ranked = agg && spec.y ? `${agg} ${spec.y}` : agg === "筆數" ? "筆數" : null;
    const kept = d.truncated && ranked ? `前 ${d.categories.length} 名（依${ranked}）` : null;
    // stage 19: the spec says which order the backend produced
    if (spec.sort === "value") return ranked ? `依 ${ranked} 由高到低排序` : null;
    if (spec.sort === "natural") return `依 ${spec.x ?? "類別"} 的類別順序顯示，未依值排序${kept ? `；${kept}` : ""}`;
    const natural = naturalOrder(d.categories);
    if (natural) return `${spec.x ?? "類別"} 依${KIND_LABEL[natural.kind]}顯示${kept ? `；${kept}` : ""}`;
    return ranked ? `依 ${ranked} 由高到低排序` : null;
  }
  if (spec.type === "box" || spec.type === "violin" || spec.type === "raincloud" || spec.type === "ridgeline" || spec.type === "dot_interval") {
    const groups = (result.chart_data as { groups: { name: string }[] }).groups;
    if (groups.length < 2 || spec.x === null || spec.x === undefined) return null;
    if (spec.sort === "natural") return `依 ${spec.x} 的類別順序顯示，未依值排序`;
    if (spec.sort === "value") return null;
    const natural = naturalOrder(groups.map((g) => g.name));
    return natural ? `${spec.x} 依${KIND_LABEL[natural.kind]}顯示` : null;
  }
  if ((spec.type === "dumbbell" || spec.type === "slopegraph") && spec.sort === "natural") {
    return `依 ${spec.x} 的類別順序顯示，未依值排序`;
  }
  if (spec.type === "line" && spec.sort === "natural") return `依 ${spec.x} 的順序顯示（週期／等級）`;
  return null;
}
