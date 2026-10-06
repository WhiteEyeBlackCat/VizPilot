// Types mirroring the backend JSON contracts (Stages 1-5).

export interface ColumnMeta {
  name: string;
  dtype: string;
}

export interface DatasetMeta {
  dataset_id: string;
  filename: string;
  uploaded_at: string;
  n_rows: number;
  n_cols: number;
  columns: ColumnMeta[];
}

export type SemanticType =
  | "numeric"
  | "categorical"
  | "datetime"
  | "boolean"
  | "text"
  | "id"
  | "unknown";

export interface TopValue {
  value: unknown;
  count: number;
}

export interface ColumnProfile {
  name: string;
  original_dtype: string;
  semantic_type: SemanticType;
  missing_count: number;
  missing_ratio: number;
  unique_count: number;
  cast_params: unknown;
  nominal?: boolean; // numeric-backed categorical that is a code, never a quantity (stage 9)
  quality?: ColumnQuality | null; // per-column data-quality counts (stage 9; additive)
  min: number | string | null;
  max: number | string | null;
  mean: number | null;
  median: number | null;
  std: number | null;
  q25: number | null;
  q75: number | null;
  skewness: number | null;
  top_values: TopValue[] | null;
  n_categories: number | null;
  /** stage 19.2: every distinct label (sorted) when the column has <= 13 categories; role detection input */
  categories?: string[] | null;
  inferred_frequency: string | null;
  avg_length: number | null;
  max_length: number | null;
}

export interface Correlations {
  columns: string[];
  matrix: (number | null)[][];
  truncated: boolean;
}

export interface DatasetProfile {
  profile_version: number;
  dataset_id: string;
  n_rows: number;
  n_cols: number;
  sampled: boolean;
  columns: ColumnProfile[];
  correlations: Correlations | null;
  sample_rows: Record<string, unknown>[];
  /** rows every statistic was computed on (== n_rows unless sampled); additive */
  profiled_rows?: number;
  /** measured evidence (stages 7-17); additive, the UI reads only the parts it shows */
  evidence?: Evidence;
  /** stage 19.1 (additive, `/profile` response only): order semantics per column (backend charts/roles.py) */
  column_roles?: ColumnRole[];
}

// backend charts/roles.py ColumnRole — how a column's values are ordered
export type Role = "numeric" | "datetime" | "boolean" | "nominal" | "ordinal" | "cyclic";

export interface ColumnRole {
  name: string;
  role: Role;
  ordered: boolean; // ordinal / cyclic / datetime / numeric
  cyclic: boolean;
  period: number | null; // cyclic only
  order: string[] | null; // display order of the categories when known
  n_categories: number | null;
  two_state: boolean; // exactly two categories
}

// the six original types plus the stage-19 family (backend charts/spec.py)
export type ChartType =
  | "line"
  | "bar"
  | "scatter"
  | "histogram"
  | "box"
  | "heatmap"
  | "violin"
  | "raincloud"
  | "ridgeline"
  | "dot_interval"
  | "lollipop"
  | "dumbbell"
  | "slopegraph"
  | "hexbin"
  | "polar_line"
  | "rose"
  // stage 20b: composition (part-to-whole) and flow
  | "stacked_bar"
  | "sankey";
export const CHART_TYPES: ChartType[] = [
  "line",
  "bar",
  "scatter",
  "histogram",
  "box",
  "heatmap",
  "violin",
  "raincloud",
  "ridgeline",
  "dot_interval",
  "lollipop",
  "dumbbell",
  "slopegraph",
  "hexbin",
  "polar_line",
  "rose",
  "stacked_bar",
  "sankey",
];
export type Aggregation = "mean" | "sum" | "count" | "median" | "min" | "max";
export type TimeGranularity = "raw" | "day" | "week" | "month";
/** category order on a category axis (stage 19): the column's natural order
 *  or ranked by value; null = backend default (ordered roles natural, else ranked) */
export type SortOrder = "natural" | "value";

export interface ChartSpec {
  title: string;
  type: ChartType;
  x?: string | null;
  y?: string | null;
  group_by?: string | null;
  aggregation?: Aggregation | null;
  bins?: number | null;
  time_granularity?: TimeGranularity | null;
  top_n?: number | null;
  sort?: SortOrder | null; // stage 19, additive
  /** stage 20b, additive: stacked_bar only — every bar normalised to 100 % */
  percent?: boolean | null;
  reason?: string;
  priority?: number;
}

/** Why a chart family was chosen (backend charts/intent.py); additive. */
export type VisualizationIntent =
  | "trend"
  | "distribution"
  | "distribution_by_group"
  | "group_comparison"
  | "ranking"
  | "numeric_relationship"
  | "cyclic_pattern"
  | "matrix_pattern"
  | "two_state_comparison"
  | "composition"
  | "flow";

export type Tier = "top" | "secondary" | "exploratory";
export type Supported = "strong" | "weak" | "unverified";

// Stage 9 confidence layer (additive fields; older backends omit them)
export type WarningSeverity = "info" | "warning" | "severe";

export interface Warning {
  code: string;
  severity: WarningSeverity;
  message: string;
  meta: Record<string, unknown>;
}

export interface Confidence {
  sample_size: number;
  missingness: number;
  robustness: number;
  overall: number;
  n_total: number;
  n_effective: number;
  missing_ratio: number;
  min_group_n: number | null;
  n_source: "exact" | "estimated";
}

export interface Recommendation {
  spec: ChartSpec;
  score: number; // final score = base evidence score x confidence.overall
  source: "rules" | "llm";
  tier: Tier;
  confidence?: Confidence | null;
  warnings?: Warning[];
  /** stage 19 (additive): the analytical intent behind the chart and the
   *  other eligible charts of the same finding (the card shows the preferred) */
  intent?: VisualizationIntent | null;
  alternatives?: ChartSpec[];
}

/** Backend measurement behind an insight (stage 17.3; additive). The label
 *  and numbers are the backend's own — the UI formats, never computes. */
export interface InsightEffect {
  label: string;
  value: number | null;
  n: number | null;
}

export interface Insight {
  text: string;
  supported: Supported;
  chart_priority: number | null;
  /** stage 18: short number-free claim shown collapsed; null → the UI shows
   *  `text` as the headline (never a truncated string) */
  headline?: string | null;
  /** stage 17.3 (additive; older backends omit them) */
  why_it_matters?: string | null;
  /** "existing_evidence" (evidence tables) or "probe" (targeted test) */
  validation?: string | null;
  priority?: number | null;
  effect?: InsightEffect | null;
}

export interface RecommendationsResponse {
  charts: Recommendation[];
  insights: Insight[];
  message: string | null;
  warnings?: Warning[]; // dataset-level (e.g. columns excluded for missingness)
}

export interface XYSeries {
  name: string;
  x: (number | string | null)[];
  y: (number | null)[];
}

export interface LineChartData {
  series: XYSeries[];
  y_label: string;
}

export interface BarChartData {
  categories: (string | number | boolean)[];
  series: { name: string; values: (number | null)[] }[];
  truncated: boolean;
  y_label: string;
}

export interface HistogramBins {
  edges: number[];
  counts: number[];
}

export interface HistogramChartData {
  bins: HistogramBins;
  series?: { name: string; edges: number[]; counts: number[] }[];
  y_label: string;
}

export interface BoxGroup {
  name: string;
  q1: number | null;
  median: number | null;
  q3: number | null;
  lower_fence: number | null;
  upper_fence: number | null;
  outliers: number[];
  mean: number | null;
  count: number;
}

export interface BoxChartData {
  groups: BoxGroup[];
  y_label: string;
}

export interface HeatmapChartData {
  columns: string[];
  matrix: (number | null)[][];
  y_label: string;
  /** stage 19 matrix pattern (x × group_by → aggregate): the row labels;
   *  absent for the correlation matrix (rows == columns) */
  rows?: string[];
}

// --- stage 19 payloads (backend charts/render.py) --------------------------

/** violin / raincloud / ridgeline: one normalised density per group over a
 *  shared grid plus the box statistics; raincloud carries sampled points */
export interface DensityGroup extends BoxGroup {
  density: number[]; // peak-normalised to 1, one value per grid centre
  bandwidth: number;
  points?: number[];
}

export interface DensityChartData {
  groups: DensityGroup[];
  grid: number[];
  y_label: string;
}

export interface DotIntervalGroup {
  name: string;
  mean: number | null;
  lo: number | null;
  hi: number | null;
  n: number;
}

export interface DotIntervalChartData {
  groups: DotIntervalGroup[];
  y_label: string;
}

/** dumbbell / slopegraph: aggregate per category for exactly two states */
export interface TwoStateChartData {
  categories: string[];
  states: string[];
  series: { name: string; values: (number | null)[] }[];
  y_label: string;
}

/** hexbin: [column, row, count] cells in offset coordinates over the ranges */
export interface HexbinChartData {
  cells: [number, number, number][];
  x_range: [number, number];
  y_range: [number, number];
  dx: number;
  dy: number;
  columns: number;
  rows: number;
  n: number;
  y_label: string;
}

// --- stage 20b payloads (backend charts/render.py) --------------------------

/** stacked_bar: the bar payload plus the total of every bar. `percent` says
 *  whether `series[].values` are shares (0..100) — `totals` always carries
 *  the absolute magnitude behind them. */
export interface StackedBarChartData extends BarChartData {
  totals: (number | null)[];
  percent: boolean;
}

/** One end of a sankey link. `name` is the node identity (side-prefixed, so
 *  the same label on both ends is two nodes and the graph never cycles) and
 *  `label` is the value as it appears in the column. */
export interface SankeyNode {
  name: string;
  label: string;
  column: string;
  side: "source" | "target";
  value: number;
}

export interface SankeyLink {
  source: string; // a node `name`
  target: string;
  value: number;
}

/** sankey: aggregated flow from one category column to another. Self-loops
 *  are dropped before aggregation (`self_loops` counts the rows) and only the
 *  heaviest links are kept (`truncated` / `n_links_total`). */
export interface SankeyChartData {
  nodes: SankeyNode[];
  links: SankeyLink[];
  truncated: boolean;
  n_links_total: number;
  self_loops: number;
  y_label: string;
}

export type ChartData =
  | LineChartData
  | BarChartData
  | HistogramChartData
  | BoxChartData
  | HeatmapChartData
  | DensityChartData
  | DotIntervalChartData
  | TwoStateChartData
  | HexbinChartData
  | StackedBarChartData
  | SankeyChartData;

// Stage 9: histogram over a sentinel-laden column is binned over the robust
// range; the rows outside it are counted here, never dropped silently
export interface DisplayRange {
  lo: number;
  hi: number;
  excluded_below: number;
  excluded_above: number;
  reason: string;
}

export interface RenderResult {
  spec: ChartSpec;
  chart_data: ChartData;
  sampled: boolean;
  n_points: number;
  display_range?: DisplayRange | null;
}

// --- profiling extras mirrored from backend profiling/models.py (stage 16.3,
// additive: the Overview page reads them; nothing else depends on them) ----

export interface SentinelCandidate {
  value: number;
  count: number;
  /** subset of {extreme, repeated, pattern, scale} */
  signals: string[];
}

export interface RobustRange {
  lo: number;
  hi: number;
}

/** Per-column data-quality counts (sample-level, like every profile count). */
export interface ColumnQuality {
  profiled_rows: number;
  missing_count: number;
  missing_token_count: number;
  invalid_count: number;
  valid_count: number;
  valid_ratio: number;
  q1: number | null;
  q3: number | null;
  iqr: number | null;
  mad: number | null;
  robust_z_max: number | null;
  extreme_value_count: number;
  extreme_value_ratio: number;
  suspected_sentinels: SentinelCandidate[];
  sentinel_row_count: number;
  robust_range: RobustRange | null;
}

export type DerivedKind = "product" | "product_discount" | "sum" | "difference" | "ratio" | "near_copy" | "bucket";

/** A column that is (almost) a deterministic function of others (stage 13). */
export interface DerivedColumn {
  target: string;
  components: string[];
  formula: string;
  kind: DerivedKind;
  match_ratio: number;
  n: number;
}

/** Columns that rank (almost) identically; only the representative charts (stage 14). */
export interface NearDuplicateGroup {
  representative: string;
  duplicates: string[];
  rho: Record<string, number>;
  n: number;
}

export type DispersionStrength = "strong" | "weak";

/** Spread difference of `num` across the groups of `cat` (stage 19.1 / 19.2), mirroring
 *  backend `DispersionEffect`: Brown–Forsythe eta² / F / p-value over the compared groups,
 *  robust spread ratios, which groups took part, and the definitional flag. */
export interface DispersionEffect {
  cat: string;
  num: string;
  n_total: number; // rows in the compared groups
  n_groups: number; // compared groups
  n_min: number; // rows in the smallest compared group
  bf_eta_squared: number;
  std_ratio: number | null;
  iqr_ratio: number | null;
  mad_ratio: number | null;
  spread_ratio: number;
  widest_group: string;
  narrowest_group: string;
  strength: DispersionStrength;
  // stage 19.2, additive
  bf_f?: number | null;
  bf_p_value?: number | null;
  compared_groups?: string[];
  excluded_groups?: Record<string, number>; // label -> rows, too small to take part
  definitional?: boolean;
  definitional_kind?: DerivedKind | null;
  definition?: string | null; // e.g. "price_band = bucket(price)"
}

/** Only the evidence the UI shows is typed; the statistical layers stay opaque. */
export interface Evidence {
  cat_num?: unknown[];
  time_effects?: unknown[];
  num_num_spearman?: Correlations | null;
  interactions?: unknown[];
  slope_heterogeneity?: unknown[];
  derived_columns?: DerivedColumn[];
  near_duplicate_groups?: NearDuplicateGroup[];
  dispersion?: DispersionEffect[]; // stage 19.1, additive
  layer2?: unknown;
}
