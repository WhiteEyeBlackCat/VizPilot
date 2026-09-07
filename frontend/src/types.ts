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

export interface Recommendation {
  spec: ChartSpec;
  score: number;
  source: "rules" | "llm";
}

export interface RecommendationsResponse {
  charts: Recommendation[];
  insights: string[];
  message: string | null;
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

export interface RenderResult {
  spec: ChartSpec;
  chart_data: ChartData;
  sampled: boolean;
  n_points: number;
}
