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
}

export type ChartType = "line" | "bar" | "scatter" | "histogram" | "box" | "heatmap";
export type Aggregation = "mean" | "sum" | "count" | "median" | "min" | "max";
export type TimeGranularity = "raw" | "day" | "week" | "month";

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
  reason?: string;
  priority?: number;
}

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
}

export interface Insight {
  text: string;
  supported: Supported;
  chart_priority: number | null;
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
}

export type ChartData =
  | LineChartData
  | BarChartData
  | HistogramChartData
  | BoxChartData
  | HeatmapChartData;

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

export type DerivedKind = "product" | "product_discount" | "sum" | "difference" | "ratio" | "near_copy";

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

/** Only the evidence the UI shows is typed; the statistical layers stay opaque. */
export interface Evidence {
  cat_num?: unknown[];
  time_effects?: unknown[];
  num_num_spearman?: Correlations | null;
  interactions?: unknown[];
  slope_heterogeneity?: unknown[];
  derived_columns?: DerivedColumn[];
  near_duplicate_groups?: NearDuplicateGroup[];
  layer2?: unknown;
}
