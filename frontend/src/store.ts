// App-level state (stage 16.2): one reducer owns the dataset selection, the
// two-phase recommendation load, the shared preview and the per-dataset
// workspace. Pages dispatch intents; nothing pushes charts anywhere except
// an explicit Save.
//
//   Insights card / Explore Generate  --api.render-->  SET_PREVIEW
//   PreviewPanel Save                 ---------------> SAVE_PREVIEW (workspace)
//   Workspace card                    ---------------> SET_PREVIEW (source workspace)
//
// Race guards live here: every dataset-scoped payload carries the dataset id
// it was requested for and is dropped when the selection has moved on.

import type {
  ChartSpec,
  DatasetMeta,
  DatasetProfile,
  Recommendation,
  RecommendationsResponse,
  RenderResult,
} from "./types";

export type Page = "overview" | "insights" | "explore" | "workspace";

export const PAGES: { page: Page; label: string }[] = [
  { page: "overview", label: "Overview" },
  { page: "insights", label: "Insights" },
  { page: "explore", label: "Explore" },
  { page: "workspace", label: "Workspace" },
];

export type PreviewSource = "insight" | "explore" | "workspace";

export interface Preview {
  result: RenderResult;
  source: PreviewSource;
  /** the recommendation behind the chart (reason, confidence, warnings) */
  rec?: Recommendation;
  /** LLM insight text that this chart supports (plain text) */
  insightText?: string;
  /** the workspace entry being viewed when source === "workspace" */
  savedKey?: string;
  /** monotonically increasing; lets the UI / e2e see a replaced preview */
  seq: number;
}

export interface SavedChart {
  key: string;
  result: RenderResult;
  savedAt: string;
  rec?: Recommendation;
  insightText?: string;
}

export interface AppState {
  datasets: DatasetMeta[] | null; // null until the list request answered
  meta: DatasetMeta | null;
  profile: DatasetProfile | null;
  recs: RecommendationsResponse | null;
  recsFinal: boolean; // the full (LLM) response has replaced the rules-only one
  aiPending: boolean;
  preview: Preview | null;
  panelOpen: boolean;
  workspaceByDataset: Record<string, SavedChart[]>;
}

export const initialState: AppState = {
  datasets: null,
  meta: null,
  profile: null,
  recs: null,
  recsFinal: false,
  aiPending: false,
  preview: null,
  panelOpen: false,
  workspaceByDataset: {},
};

export type Action =
  | { type: "DATASETS_LOADED"; datasets: DatasetMeta[] }
  | { type: "DATASET_ADDED"; meta: DatasetMeta }
  | { type: "SELECT_DATASET"; meta: DatasetMeta }
  | { type: "PROFILE_LOADED"; datasetId: string; profile: DatasetProfile }
  | { type: "RECS_LOADED"; datasetId: string; recs: RecommendationsResponse; final: boolean }
  | { type: "AI_DONE"; datasetId: string }
  | { type: "SET_PREVIEW"; datasetId: string; preview: Omit<Preview, "seq"> }
  | { type: "CLOSE_PREVIEW" }
  | { type: "SET_PANEL_OPEN"; open: boolean }
  | { type: "SAVE_PREVIEW"; key: string; savedAt: string }
  | { type: "REMOVE_SAVED"; datasetId: string; key: string };

/** The fields that make two specs the same chart (title / reason / priority
 *  are presentation). Used to keep the workspace free of duplicates. */
export function specKey(spec: ChartSpec): string {
  return JSON.stringify([
    spec.type,
    spec.x ?? null,
    spec.y ?? null,
    spec.group_by ?? null,
    spec.aggregation ?? null,
    spec.bins ?? null,
    spec.time_granularity ?? null,
    spec.top_n ?? null,
  ]);
}

export function workspaceOf(state: AppState, datasetId: string | null | undefined): SavedChart[] {
  return datasetId ? (state.workspaceByDataset[datasetId] ?? []) : [];
}

export function isSaved(state: AppState, spec: ChartSpec): boolean {
  const key = specKey(spec);
  return workspaceOf(state, state.meta?.dataset_id).some((c) => specKey(c.result.spec) === key);
}

let previewSeq = 0;

export function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case "DATASETS_LOADED":
      return { ...state, datasets: action.datasets };

    case "DATASET_ADDED":
      return {
        ...state,
        datasets: [action.meta, ...(state.datasets ?? []).filter((d) => d.dataset_id !== action.meta.dataset_id)],
      };

    case "SELECT_DATASET":
      if (state.meta?.dataset_id === action.meta.dataset_id) return state;
      return {
        ...state,
        meta: action.meta,
        profile: null,
        recs: null,
        recsFinal: false,
        aiPending: true,
        preview: null, // the workspace of the previous dataset is kept, keyed by id
      };

    case "PROFILE_LOADED":
      if (state.meta?.dataset_id !== action.datasetId) return state;
      return { ...state, profile: action.profile };

    case "RECS_LOADED":
      if (state.meta?.dataset_id !== action.datasetId) return state;
      // two-phase load: the rules-only answer must not overwrite the full one
      if (!action.final && state.recsFinal) return state;
      return {
        ...state,
        recs: action.recs,
        recsFinal: state.recsFinal || action.final,
        aiPending: action.final ? false : state.aiPending,
      };

    case "AI_DONE":
      if (state.meta?.dataset_id !== action.datasetId) return state;
      return { ...state, aiPending: false };

    case "SET_PREVIEW":
      if (state.meta?.dataset_id !== action.datasetId) return state; // dataset switched while rendering
      previewSeq += 1;
      return { ...state, preview: { ...action.preview, seq: previewSeq }, panelOpen: true };

    case "CLOSE_PREVIEW":
      return { ...state, preview: null, panelOpen: false };

    case "SET_PANEL_OPEN":
      return { ...state, panelOpen: action.open };

    case "SAVE_PREVIEW": {
      const id = state.meta?.dataset_id;
      const preview = state.preview;
      if (!id || !preview) return state;
      if (isSaved(state, preview.result.spec)) return state; // already in the workspace
      const saved: SavedChart = {
        key: action.key,
        result: preview.result,
        savedAt: action.savedAt,
        rec: preview.rec,
        insightText: preview.insightText,
      };
      return {
        ...state,
        workspaceByDataset: {
          ...state.workspaceByDataset,
          [id]: [...workspaceOf(state, id), saved],
        },
      };
    }

    case "REMOVE_SAVED": {
      const list = workspaceOf(state, action.datasetId).filter((c) => c.key !== action.key);
      const viewingRemoved =
        state.preview?.source === "workspace" && state.preview.savedKey === action.key;
      return {
        ...state,
        workspaceByDataset: { ...state.workspaceByDataset, [action.datasetId]: list },
        preview: viewingRemoved ? null : state.preview,
        panelOpen: viewingRemoved ? false : state.panelOpen,
      };
    }
  }
}
