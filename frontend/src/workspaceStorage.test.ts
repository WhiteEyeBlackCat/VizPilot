import { afterEach, describe, expect, it, vi } from "vitest";

import { initialState, persistableDatasets, persistableWorkspace, reducer, toPersisted, type Action, type AppState, type SavedChart } from "./store";
import type { ChartSpec, DatasetMeta, RenderResult } from "./types";
import {
  STORAGE_KEY,
  hydrateWorkspace,
  loadWorkspace,
  saveWorkspace,
  type PersistedChart,
  type StorageLike,
} from "./workspaceStorage";

const memory = (initial?: string): StorageLike & { data: Map<string, string>; writes: number } => {
  const data = new Map<string, string>();
  if (initial !== undefined) data.set(STORAGE_KEY, initial);
  const s = {
    data,
    writes: 0,
    getItem: (k: string) => data.get(k) ?? null,
    setItem: (k: string, v: string) => {
      s.writes += 1;
      data.set(k, v);
    },
  };
  return s;
};

const spec = (x: string): ChartSpec =>
  ({ type: "bar", title: `chart ${x}`, x, y: "v", group_by: null, aggregation: "sum", bins: null, time_granularity: null, top_n: null }) as unknown as ChartSpec;
const persisted = (key: string, x = key): PersistedChart => ({ key, spec: spec(x), savedAt: "2026-10-01T00:00:00Z", title: `chart ${x}` });
const rendered = (s: ChartSpec): RenderResult => ({ spec: s, chart_data: {}, sampled: false, n_points: 1 }) as unknown as RenderResult;
const saved = (key: string, x = key): SavedChart => ({ key, result: rendered(spec(x)), savedAt: "2026-10-01T00:00:00Z" });
const blob = (datasets: unknown, version = 1) => JSON.stringify({ version, datasets });

afterEach(() => vi.restoreAllMocks());

describe("workspaceStorage", () => {
  it("round-trips a dataset and keeps the others", () => {
    const st = memory();
    expect(saveWorkspace("d1", [persisted("a"), persisted("b")], st)).toBe(true);
    expect(saveWorkspace("d2", [persisted("c")], st)).toBe(true);
    expect(loadWorkspace("d1", st).map((c) => c.key)).toEqual(["a", "b"]);
    expect(loadWorkspace("d2", st).map((c) => c.key)).toEqual(["c"]);
    expect(loadWorkspace("nope", st)).toEqual([]);
  });

  it("saving an empty list removes the dataset entry", () => {
    const st = memory();
    saveWorkspace("d1", [persisted("a")], st);
    saveWorkspace("d1", [], st);
    expect(loadWorkspace("d1", st)).toEqual([]);
    expect(JSON.parse(st.data.get(STORAGE_KEY)!).datasets).toEqual({});
  });

  it("stores specs, not render data", () => {
    const p = toPersisted({ ...saved("a"), result: { ...rendered(spec("a")), chart_data: { huge: [1, 2, 3] } as never } });
    expect(JSON.stringify(p)).not.toContain("huge");
    expect(p.spec.type).toBe("bar");
  });

  it("ignores corrupted JSON", () => {
    expect(loadWorkspace("d1", memory("{not json"))).toEqual([]);
    expect(loadWorkspace("d1", memory("null"))).toEqual([]);
    expect(loadWorkspace("d1", memory('"str"'))).toEqual([]);
  });

  it("ignores a wrong version", () => {
    expect(loadWorkspace("d1", memory(blob({ d1: [persisted("a")] }, 2)))).toEqual([]);
    expect(loadWorkspace("d1", memory(blob({ d1: [persisted("a")] }, 0)))).toEqual([]);
  });

  it("filters malformed entries but keeps the valid ones of the same blob", () => {
    const st = memory(
      blob({
        d1: [persisted("ok"), { key: "nospec", savedAt: "x" }, null, 5, { ...persisted("badrec"), rec: "oops" }, persisted("ok"), persisted("ok2")],
        d2: "not a list",
      }),
    );
    expect(loadWorkspace("d1", st).map((c) => c.key)).toEqual(["ok", "ok2"]);
    expect(loadWorkspace("d2", st)).toEqual([]);
  });

  it("swallows a quota error on write", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const st: StorageLike = {
      getItem: () => null,
      setItem: () => {
        throw new DOMException("full", "QuotaExceededError");
      },
    };
    expect(saveWorkspace("d1", [persisted("a")], st)).toBe(false);
    expect(warn).toHaveBeenCalled();
  });

  it("swallows a throwing getItem and a missing storage", () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const st: StorageLike = {
      getItem: () => {
        throw new Error("blocked");
      },
      setItem: () => {},
    };
    expect(loadWorkspace("d1", st)).toEqual([]);
    expect(loadWorkspace("d1", null)).toEqual([]);
    expect(saveWorkspace("d1", [persisted("a")], null)).toBe(false);
  });

  it("recovers from a corrupted blob by overwriting it, but never overwrites a newer version", () => {
    const bad = memory("{oops");
    expect(saveWorkspace("d1", [persisted("a")], bad)).toBe(true);
    expect(loadWorkspace("d1", bad).map((c) => c.key)).toEqual(["a"]);
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const newer = memory(blob({ d1: [persisted("z")] }, 2));
    expect(saveWorkspace("d1", [persisted("a")], newer)).toBe(false);
    expect(JSON.parse(newer.data.get(STORAGE_KEY)!).version).toBe(2);
  });
});

describe("hydrateWorkspace", () => {
  it("re-renders every stored spec; failures become orphans that stay in storage", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const st = memory();
    saveWorkspace("d1", [persisted("a"), persisted("gone"), persisted("b")], st);
    const render = vi.fn(async (_id: string, s: ChartSpec) => {
      if (s.x === "gone") throw new Error("422");
      return rendered(s);
    });
    const { charts, orphans } = await hydrateWorkspace("d1", render, st);
    expect(charts.map((c) => c.key)).toEqual(["a", "b"]);
    expect(orphans.map((o) => o.key)).toEqual(["gone"]);
    expect(st.data.get(STORAGE_KEY)).toContain("gone"); // hydration never deletes
  });
});

// --- reducer side -------------------------------------------------------------

const meta = (id: string) => ({ dataset_id: id, filename: `${id}.csv` }) as unknown as DatasetMeta;
const run = (actions: Action[], start: AppState = initialState): AppState => actions.reduce(reducer, start);
const hydrate = (datasetId: string, charts: SavedChart[], orphans: PersistedChart[] = []): Action => ({
  type: "HYDRATE_WORKSPACE",
  datasetId,
  charts,
  orphans,
});
const selected = (id = "d1") => run([{ type: "SELECT_DATASET", meta: meta(id) }]);

describe("HYDRATE_WORKSPACE", () => {
  it("fills the workspace and marks the dataset hydrated", () => {
    const s = run([hydrate("d1", [saved("a")])], selected());
    expect(s.workspaceByDataset.d1.map((c) => c.key)).toEqual(["a"]);
    expect(s.hydrated.d1).toBe(true);
  });

  it("dedupes by key and by chart; charts saved meanwhile stay, restored ones go first", () => {
    const start = run([hydrate("d1", [])], selected());
    const withLive: AppState = { ...start, workspaceByDataset: { d1: [saved("live", "x1")] } };
    const s = run([hydrate("d1", [saved("old", "x0"), saved("live", "x2"), saved("dup", "x1"), saved("old", "x3")])], withLive);
    expect(s.workspaceByDataset.d1.map((c) => c.key)).toEqual(["old", "live"]);
  });

  it("is dropped when the selection moved on", () => {
    const s = run([{ type: "SELECT_DATASET", meta: meta("d2") }, hydrate("d1", [saved("a")])], selected());
    expect(s.workspaceByDataset.d1).toBeUndefined();
    expect(s.hydrated.d1).toBeUndefined();
  });

  it("write guard: nothing is writable before hydration, the dataset is after", () => {
    const before = selected();
    expect(persistableDatasets(before)).toEqual([]); // empty list must not overwrite storage
    const withPreview: AppState = { ...before, preview: { result: rendered(spec("x")), source: "explore", seq: 1 } };
    const savedEarly = reducer(withPreview, { type: "SAVE_PREVIEW", key: "k", savedAt: "t" });
    expect(persistableDatasets(savedEarly)).toEqual([]);
    const after = reducer(savedEarly, hydrate("d1", [saved("old", "x0")]));
    expect(persistableDatasets(after)).toEqual(["d1"]);
    expect(persistableWorkspace(after, "d1").map((c) => c.key)).toEqual(["old", "k"]); // early save merged, stored one kept
  });

  it("orphans are written back next to the visible charts, never lost", () => {
    const s = run([hydrate("d1", [saved("a")], [persisted("gone")])], selected());
    expect(persistableWorkspace(s, "d1").map((c) => c.key)).toEqual(["gone", "a"]);
    const removed = reducer(s, { type: "REMOVE_SAVED", datasetId: "d1", key: "a" });
    expect(persistableWorkspace(removed, "d1").map((c) => c.key)).toEqual(["gone"]);
  });

  it("SAVE_PREVIEW still refuses a duplicate spec after hydration", () => {
    const base = run([hydrate("d1", [saved("a", "x")])], selected());
    const withPreview: AppState = { ...base, preview: { result: rendered(spec("x")), source: "explore", seq: 1 } };
    expect(reducer(withPreview, { type: "SAVE_PREVIEW", key: "new", savedAt: "t" }).workspaceByDataset.d1).toHaveLength(1);
  });
});
