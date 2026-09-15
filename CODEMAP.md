---
mode: learning
commit: ea1140a
ignore: built-ins + .gitignore (node_modules, dist, .venv, backend/data, dataset/*.csv|parquet, .claude/* (except the four context docs), e2e/screenshots, e2e/.xlibs)
generated_at: 2026-09-15
stats:
  total_files: 148
  total_lines: 49265   # ~15.7k source; the rest is frontend/src/charts/__fixtures__/*.json
  total_size: 1.35 MB
---

> VizPilot — local AI data-exploration assistant: upload tabular data → Polars profiling + evidence layers → rule engine (+ optional local LLM) → validated `ChartSpec` → backend-aggregated `RenderResult` → ECharts in a React SPA. Raw data never leaves the machine; the LLM sees metadata only and is optional.

## Task Guide

| Task | Domain | Target | Also Check |
|---|---|---|---|
| Understand the end-to-end request flow | Architecture | `.claude/ARCHITECTURE.md` | `backend/app/CODEMAP.md` |
| Understand a cross-module data shape (profile, evidence, spec, recommendation, render) | Contracts | `.claude/CONTRACTS.md` | `backend/app/profiling/models.py`, `backend/app/charts/spec.py`, `frontend/src/types.ts` |
| Understand current stage / what is unfinished | State | `.claude/CURRENT_STATE.md` | `.claude/docs/` (local only, not in repo) |
| Understand semantic typing / dtype inference | Profiling | `backend/app/profiling/CODEMAP.md` → `types.py` | `backend/tests/test_types.py` |
| Understand evidence statistics (L1 effects, derived/near-dup, L2 structure) | Profiling | `backend/app/profiling/CODEMAP.md` → `evidence.py`, `layer2.py.analysis.md` | — |
| Understand why a chart is (not) recommended, scores, tiers, caps | Recommendation | `backend/app/charts/CODEMAP.md` → `rules.py`, `confidence.py` | `backend/tests/test_recommendation_quality.py` |
| Understand ChartSpec validation (422) | Recommendation | `backend/app/charts/spec.py` | `frontend/src/components/ManualBuilder.tsx` |
| Understand chart data aggregation | Render | `backend/app/charts/render.py` | `frontend/src/charts/echarts/option.ts` |
| Understand LLM prompts, provider, two-stage hypothesis → coverage → probe → wording workflow, merge | LLM | `backend/app/llm/CODEMAP.md` | `backend/app/probes/CODEMAP.md`, `backend/tests/test_insight_benchmark.py` |
| Understand targeted validation probes | Probes | `backend/app/probes/CODEMAP.md` | `backend/app/profiling/layer2.py.analysis.md` |
| Understand HTTP endpoints, upload, storage | API | `backend/app/datasets/CODEMAP.md` | `backend/tests/test_upload.py` |
| Understand app shell, state, routing, preview panel | Frontend Shell | `frontend/src/CODEMAP.md` → `App.tsx`, `store.ts` | `frontend/src/components/CODEMAP.md` |
| Understand how charts are drawn / themed | Chart Layer | `frontend/src/charts/echarts/CODEMAP.md` | `frontend/src/charts/__fixtures__/index.json` |
| Understand pages / feature components / shadcn kit | Frontend UI | `frontend/src/pages/CODEMAP.md`, `frontend/src/components/CODEMAP.md` | `frontend/src/components/ui/CODEMAP.md` |
| Run tests / e2e / find a test for a module | Tests | `backend/tests/CODEMAP.md`, `frontend/e2e/CODEMAP.md` | `frontend/package.json` |
| Get a dataset with a planted pattern | Synthetic Data | `dataset/syn/CODEMAP.md` | — |

## Subdirectories

| Dir | Domain | Depends On | Purpose |
|---|---|---|---|
| `backend/` | API, Profiling, Recommendation, Render, LLM, Probes | polars, pandas, duckdb, fastapi, pydantic | FastAPI app + pytest suite. |
| `frontend/` | Frontend Shell, Chart Layer, Frontend UI, E2E | react, echarts, radix, tailwind, playwright | Vite SPA + e2e harness. |
| `dataset/` | Synthetic Data | polars | Benchmark data generators (outputs git-ignored). |

## Key Exports

| Symbol | Source | Line |
|---|---|---|
| `create_app` | `backend/app/main.py` | L:18 |
| `DatasetProfile`, `Evidence`, `PROFILE_VERSION` | `backend/app/profiling/models.py` | L:413, L:401, L:7 |
| `ChartSpec`, `validate_spec` | `backend/app/charts/spec.py` | L:28, L:61 |
| `Recommendation`, `recommend_charts`, `assign_tiers` | `backend/app/charts/rules.py` | L:54, L:284, L:321 |
| `render_chart` | `backend/app/charts/render.py` | L:50 |
| `RecommendationService` | `backend/app/llm/service.py` | L:106 |
| `run_probes` | `backend/app/probes/engine.py` | L:137 |
| `buildOption` | `frontend/src/charts/echarts/option.ts` | L:466 |
| `reducer` / `AppState` | `frontend/src/store.ts` | L:143 / L:70 |

## Files

| File | Domain | Function |
|---|---|---|
| `README.md` | Docs | Purpose, stack, run/build commands (ports 8100 / 5173). |
| `.claude/CLAUDE.md` | Agent Rules | Behaviour, navigation protocol, subagent + test rules. |
| `.claude/ARCHITECTURE.md` | Architecture | High-level data flow and module boundaries. |
| `.claude/CONTRACTS.md` | Contracts | Cross-module data contracts and invariants. |
| `.claude/CURRENT_STATE.md` | State | Completed stage, blockers, test status, next task. |
| `.gitignore` | — | `.claude/*` is ignored except the four context docs above; `.claude/docs/` (stage notes) and worktrees stay local. |
