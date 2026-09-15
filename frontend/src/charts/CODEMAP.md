---
mode: learning
generated_at: 2026-09-15
---

> Chart rendering layer: the ECharts adapter and the recorded `RenderResult` fixtures it is tested against.

## Task Guide

| Task | Domain | Target | Also Check |
|---|---|---|---|
| Anything about how charts are drawn | Chart Layer | `echarts/CODEMAP.md` | — |
| Need a real backend payload for a chart type | Fixtures | `__fixtures__/index.json` then the named json | `../../e2e/capture-fixtures.mjs` (how they were recorded) |

## Subdirectories

| Dir | Domain | Depends On | Purpose |
|---|---|---|---|
| `echarts/` | Chart Layer | echarts, ../types.ts | Option builders, theme, binding, tests. |
| `__fixtures__/` | Fixtures | — | 18 `RenderResult` JSON files captured from the real backend (some >1000 lines; data, not code — do not read whole). |
