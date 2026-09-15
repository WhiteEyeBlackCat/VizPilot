---
mode: learning
generated_at: 2026-09-15
---

> React app source: single reducer store, hash routes, app shell with sidebar + resizable preview panel; API types mirror the backend contracts.

## Task Guide

| Task | Domain | Target | Also Check |
|---|---|---|---|
| Understand app shell, data fetching sequence after selecting a dataset, panel layout | Shell | `App.tsx` | `store.ts`, `lib/router.ts` |
| Understand global state, actions, save semantics (`specKey`, `isSaved`) | State | `store.ts` | `components/PreviewPanel.tsx` |
| Understand API endpoints called and error type | API Client | `api.ts` | backend `datasets/router.py` |
| Understand TS shapes of profile/recommendation/render payloads | Types | `types.ts` | `.claude/CONTRACTS.md` |
| Change how a chart looks | Chart Layer | `charts/CODEMAP.md` | — |
| Change a page or a feature component | Pages / Components | `pages/CODEMAP.md`, `components/CODEMAP.md` | — |
| Change colours / dark tokens | Theme | `index.css` (CSS variables) | `charts/echarts/theme.ts`, `tailwind.config.*` |

## Subdirectories

| Dir | Domain | Depends On | Purpose |
|---|---|---|---|
| `charts/` | Chart Layer | echarts, types.ts | ECharts adapter + fixtures. |
| `components/` | Components | store, types, api, ui | Feature components. |
| `components/ui/` | UI Kit | radix | shadcn primitives. |
| `pages/` | Pages | components, store | Four pages. |
| `lib/` | Utils | store | Router, hooks, `cn`. |

## Key Exports

| Symbol | Source | Line |
|---|---|---|
| `App` | `App.tsx` | L:33 |
| `reducer`, `initialState`, `AppState`, `Action`, `PAGES`, `Preview`, `SavedChart`, `specKey`, `isSaved` | `store.ts` | L:143, L:86, L:70, L:101, L:25, L:34, L:47, L:119, L:136 |
| `api`, `ApiError` | `api.ts` | L:49, L:9 |
| `ChartSpec`, `Recommendation`, `Insight`, `RecommendationsResponse`, `RenderResult`, `DatasetProfile` | `types.ts` | L:81, L:120, L:129, L:135, L:211, L:62 |

## Files

| File | Domain | Deps | Function |
|---|---|---|---|
| `App.tsx` | Shell | `← components/*, pages/*, lib/* \| → main.tsx` | Loads datasets; on select fetches profile + recs(llm=false) + recs(llm=true) in parallel; routes pages; keeps panel group mounted. |
| `store.ts` | State | `→ 8 files (foundational); rg "from \"../store\"\|@/store" frontend/src -l` | `AppState`/`Action`/`reducer`; per-dataset workspace; explore form. |
| `api.ts` | API Client | `→ App.tsx, components/UploadPanel.tsx, ManualBuilder.tsx, Recommendations.tsx` | fetch wrappers for `/api/*`; `ApiError` carries 422 errors. |
| `types.ts` | Types | `→ many` | Contract mirrors (keep in sync with backend models). |
| `main.tsx` | Shell | — | React root. |
| `index.css` | Theme | — | Tailwind layers + light/dark CSS variable tokens (16.1). |
