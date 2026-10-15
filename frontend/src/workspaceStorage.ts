// Workspace persistence (A8). Saved charts survive a page reload through one
// localStorage key.
//
//   key     vizpilot.workspace.v1
//   value   { version: 1, datasets: { [datasetId]: PersistedChart[] } }
//
// Only what is needed to rebuild a card is stored: the ChartSpec (the chart is
// re-rendered through api.render on load, so no data arrays are persisted),
// the key, savedAt, a display title, and the small recommendation / insight
// context. `notes` is an optional passthrough field reserved for a later UI.
//
// Failure policy: every storage access is wrapped; corrupted JSON, a wrong
// version or malformed entries are ignored (valid entries of the same blob are
// kept); a failed write (quota, private mode) only warns. Entries that cannot
// be re-rendered (dataset deleted, schema changed) stay in storage, are not
// shown, and are written back untouched ("orphans"), so a restored dataset
// brings its charts back. Nothing here is ever deleted silently.

import type { SavedChart } from "./store";
import type { ChartSpec, Insight, Recommendation, RenderResult } from "./types";

export const STORAGE_KEY = "vizpilot.workspace.v1";
export const STORAGE_VERSION = 1;

export interface PersistedChart {
  key: string;
  spec: ChartSpec;
  savedAt: string;
  title?: string;
  rec?: Recommendation;
  insightText?: string;
  insight?: Insight;
  notes?: string;
}

interface Blob {
  version: number;
  datasets: Record<string, PersistedChart[]>;
}

/** The subset of the Storage API used here (lets tests pass an in-memory one). */
export interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem?(key: string): void;
}

function defaultStorage(): StorageLike | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null; // access itself can throw (blocked site data)
  }
}

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);

export function isPersistedChart(v: unknown): v is PersistedChart {
  if (!isObject(v)) return false;
  if (typeof v.key !== "string" || v.key === "" || typeof v.savedAt !== "string") return false;
  if (!isObject(v.spec) || typeof v.spec.type !== "string") return false;
  if (v.rec !== undefined && !isObject(v.rec)) return false;
  if (v.insight !== undefined && !isObject(v.insight)) return false;
  if (v.insightText !== undefined && typeof v.insightText !== "string") return false;
  if (v.title !== undefined && typeof v.title !== "string") return false;
  if (v.notes !== undefined && typeof v.notes !== "string") return false;
  return true;
}

/** Parse a stored blob; returns null for unusable input. `foreignVersion`
 *  reports a well-formed blob of another version (never overwritten). */
function parse(raw: string | null): { blob: Blob | null; foreignVersion: boolean } {
  if (raw === null) return { blob: null, foreignVersion: false };
  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch {
    return { blob: null, foreignVersion: false };
  }
  if (!isObject(data) || typeof data.version !== "number") return { blob: null, foreignVersion: false };
  if (data.version !== STORAGE_VERSION) return { blob: null, foreignVersion: data.version > STORAGE_VERSION };
  const datasets: Record<string, PersistedChart[]> = {};
  if (isObject(data.datasets)) {
    for (const [id, list] of Object.entries(data.datasets)) {
      if (!Array.isArray(list)) continue;
      const seen = new Set<string>();
      const valid: PersistedChart[] = [];
      for (const entry of list) {
        if (!isPersistedChart(entry) || seen.has(entry.key)) continue;
        seen.add(entry.key);
        valid.push(entry);
      }
      if (valid.length > 0) datasets[id] = valid;
    }
  }
  return { blob: { version: STORAGE_VERSION, datasets }, foreignVersion: false };
}

function read(storage: StorageLike | null): { blob: Blob | null; foreignVersion: boolean } {
  if (!storage) return { blob: null, foreignVersion: false };
  try {
    return parse(storage.getItem(STORAGE_KEY));
  } catch (e) {
    console.warn("workspace: cannot read localStorage", e);
    return { blob: null, foreignVersion: false };
  }
}

/** The persisted charts of one dataset (empty when none / unreadable). */
export function loadWorkspace(datasetId: string, storage: StorageLike | null = defaultStorage()): PersistedChart[] {
  return read(storage).blob?.datasets[datasetId] ?? [];
}

/** Re-render the persisted charts of a dataset. Entries whose render fails
 *  (dataset gone: 404, schema changed: 422, network) are returned as orphans:
 *  skipped in the UI, kept in storage. Sequential, so a long list never
 *  floods the backend. */
export async function hydrateWorkspace(
  datasetId: string,
  render: (datasetId: string, spec: ChartSpec) => Promise<RenderResult>,
  storage: StorageLike | null = defaultStorage(),
): Promise<{ charts: SavedChart[]; orphans: PersistedChart[] }> {
  const charts: SavedChart[] = [];
  const orphans: PersistedChart[] = [];
  for (const p of loadWorkspace(datasetId, storage)) {
    try {
      const result = await render(datasetId, p.spec);
      charts.push({ key: p.key, result, savedAt: p.savedAt, rec: p.rec, insightText: p.insightText, insight: p.insight });
    } catch (e) {
      console.warn(`workspace: cannot restore saved chart "${p.title ?? p.key}"; kept in storage`, e);
      orphans.push(p);
    }
  }
  return { charts, orphans };
}

/** Replace one dataset's persisted charts (read-modify-write so the other
 *  datasets are kept). Returns false when nothing was written. */
export function saveWorkspace(
  datasetId: string,
  charts: PersistedChart[],
  storage: StorageLike | null = defaultStorage(),
): boolean {
  if (!storage) return false;
  const { blob, foreignVersion } = read(storage);
  if (foreignVersion) {
    console.warn("workspace: storage written by a newer version; not overwriting");
    return false;
  }
  const next: Blob = { version: STORAGE_VERSION, datasets: { ...(blob?.datasets ?? {}) } };
  if (charts.length > 0) next.datasets[datasetId] = charts;
  else delete next.datasets[datasetId];
  try {
    storage.setItem(STORAGE_KEY, JSON.stringify(next));
    return true;
  } catch (e) {
    console.warn("workspace: cannot write localStorage (quota or blocked); continuing without persistence", e);
    return false;
  }
}
