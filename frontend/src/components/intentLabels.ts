import type { VisualizationIntent } from "../types";

/** Plain-language label of the analytical intent behind a chart (stage 19). */
export const INTENT_LABEL: Record<VisualizationIntent, string> = {
  trend: "時間趨勢",
  distribution: "分布",
  distribution_by_group: "分組分布",
  group_comparison: "組間比較",
  ranking: "排名",
  numeric_relationship: "數值關係",
  cyclic_pattern: "週期模式",
  matrix_pattern: "矩陣模式",
  two_state_comparison: "兩態比較",
  composition: "組成",
  flow: "流向",
};
