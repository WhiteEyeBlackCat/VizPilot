import { describe, expect, it } from "vitest";

import { createPreviewGate, initialState, reducer, type Action, type AppState, type Preview } from "./store";
import type { DatasetMeta, RenderResult } from "./types";

const meta = (id: string) => ({ dataset_id: id, filename: `${id}.csv` }) as unknown as DatasetMeta;
const result = (title: string): RenderResult =>
  ({ spec: { type: "bar", title, x: "a", y: "b" }, chart_data: {}, sampled: false, n_points: 1 }) as unknown as RenderResult;
const body = (title: string): Omit<Preview, "seq"> => ({ result: result(title), source: "explore" });

const run = (actions: Action[], start: AppState = initialState): AppState => actions.reduce(reducer, start);
const selected = (id = "d1") => run([{ type: "SELECT_DATASET", meta: meta(id) }]);
const request = (requestId: number, datasetId = "d1"): Action => ({ type: "PREVIEW_REQUESTED", datasetId, requestId });
const done = (title: string, requestId: number, datasetId = "d1"): Action => ({
  type: "SET_PREVIEW",
  datasetId,
  preview: body(title),
  requestId,
});
const title = (s: AppState) => s.preview?.result.spec.title;

describe("createPreviewGate (error side of latest-wins)", () => {
  it("A starts, B starts, B succeeds, A fails -> A may not report", () => {
    const g = createPreviewGate();
    const a = g.begin();
    const b = g.begin();
    expect(g.isCurrent(b)).toBe(true);
    expect(g.isCurrent(a)).toBe(false); // A's failure is swallowed, B's chart stays
  });

  it("the newest request reports its own failure", () => {
    const g = createPreviewGate();
    g.begin();
    const b = g.begin();
    expect(g.isCurrent(b)).toBe(true);
  });

  it("A succeeds after B failed -> still not A (reducer drops A)", () => {
    const g = createPreviewGate();
    const a = g.begin();
    const b = g.begin();
    expect(g.isCurrent(b)).toBe(true); // B fails and reports
    const s = run([request(a), request(b), done("A", a)], selected());
    expect(s.preview).toBeNull();
  });

  it("invalidate (sync preview, close, dataset switch) supersedes pending requests", () => {
    const g = createPreviewGate();
    const a = g.begin();
    g.invalidate();
    expect(g.isCurrent(a)).toBe(false);
  });
});

describe("preview: latest request wins", () => {
  it("A starts, B starts, B finishes, A finishes -> B", () => {
    const s = run([request(1), request(2), done("B", 2), done("A", 1)], selected());
    expect(title(s)).toBe("B");
  });

  it("A starts, B starts, A finishes first -> A is dropped, B shows when it arrives", () => {
    const mid = run([request(1), request(2), done("A", 1)], selected());
    expect(mid.preview).toBeNull();
    expect(mid.panelOpen).toBe(false);
    expect(title(run([done("B", 2)], mid))).toBe("B");
  });

  it("a stale result does not reopen a panel the user closed", () => {
    const s = run([request(1), { type: "CLOSE_PREVIEW" }, done("A", 1)], selected());
    expect(s.preview).toBeNull();
    expect(s.panelOpen).toBe(false);
  });

  it("a single request still lands", () => {
    const s = run([request(1), done("A", 1)], selected());
    expect(title(s)).toBe("A");
    expect(s.panelOpen).toBe(true);
  });

  it("dataset switch mid-flight drops the result (also by request id)", () => {
    const s = run([request(1), { type: "SELECT_DATASET", meta: meta("d2") }, done("A", 1)], selected());
    expect(s.preview).toBeNull();
    expect(s.previewRequestId).toBeNull();
    // the dataset-id guard alone still holds
    expect(run([done("A", 1, "d1")], run([request(1, "d2")], selected("d2"))).preview).toBeNull();
  });

  it("clearing the selection drops a pending result", () => {
    const s = run([request(1), { type: "CLEAR_SELECTION" }, done("A", 1)], selected());
    expect(s.preview).toBeNull();
  });

  it("a synchronous preview (saved chart) invalidates a pending render", () => {
    const saved: Action = { type: "SET_PREVIEW", datasetId: "d1", preview: { ...body("S"), source: "workspace" } };
    const s = run([request(1), saved, done("A", 1)], selected());
    expect(title(s)).toBe("S");
  });

  it("close invalidates a pending render", () => {
    const s = run([request(1), { type: "CLOSE_PREVIEW" }], selected());
    expect(s.previewRequestId).toBeNull();
  });

  it("a request for another dataset is ignored", () => {
    expect(run([request(1, "other")], selected()).previewRequestId).toBeNull();
  });
});
