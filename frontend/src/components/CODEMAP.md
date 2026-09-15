---
mode: learning
generated_at: 2026-09-15
---

> Feature components composed by the pages and the app shell: sidebar, preview panel, recommendations list, manual chart builder, dataset overview, upload.

## Task Guide

| Task | Domain | Target | Also Check |
|---|---|---|---|
| Understand the analysis detail panel (title/context, chart ≈64% height, why it matters, evidence, chart spec details, Save/Export/Enlarge/Close) | Preview | `PreviewPanel.tsx` | `../store.ts` (`Preview`, `SavedChart`), `ChartView.tsx`, `InsightList.tsx` (`evidenceRows`) |
| Understand Key Findings: headline-first insight rows, expand/collapse, evidence list, View chart | Insights | `InsightList.tsx` | `../types.ts` (`Insight`), `Recommendations.tsx` |
| Understand the More charts section: tier sub-sections, simplified cards (title, purpose, type · aggregation, spec tooltip, Preview) | Insights | `Recommendations.tsx` | `../pages/InsightsPage.tsx`, `../types.ts` |
| Understand manual chart building rules (field filtering, nominal never y) | Explore | `ManualBuilder.tsx`, `FieldSelect.tsx` | backend `charts/spec.py` (`validate_spec` mirror) |
| Understand column table, type badges, quality flags | Overview | `DatasetOverview.tsx` | `../pages/OverviewPage.tsx` |
| Understand upload flow / drop zone / new dataset dialog | Upload | `UploadPanel.tsx`, `NewDatasetDialog.tsx` | `../api.ts` |
| Understand navigation list and dataset switcher | Shell | `Sidebar.tsx` | `../lib/router.ts`, `../store.ts` (`PAGES`) |
| Render one chart from a RenderResult | Chart | `ChartView.tsx` | `../charts/echarts/CODEMAP.md` |

## Subdirectories

| Dir | Domain | Depends On | Purpose |
|---|---|---|---|
| `ui/` | UI Kit | radix-ui, tailwind | shadcn primitives. |

## Key Exports

| Symbol | Source | Line |
|---|---|---|
| `PreviewPanel` | `PreviewPanel.tsx` | L:77 |
| `Recommendations`, `specRows`, `chartMeta` | `Recommendations.tsx` | L:115, L:89, L:103 |
| `InsightList`, `headlineOf`, `evidenceRows`, `insightWeight`, `SUPPORTED_BADGE` | `InsightList.tsx` | L:63, L:33, L:36, L:48, L:18 |
| `ManualBuilder` | `ManualBuilder.tsx` | L:25 |
| `DatasetOverview`, `qualityFlags`, `hasQualityFlags`, `TYPE_VARIANT` | `DatasetOverview.tsx` | L:102, L:61, L:92, L:13 |
| `Sidebar` | `Sidebar.tsx` | L:37 |
| `ChartView` | `ChartView.tsx` | L:32 |
| `useUploader`, `DropZone`, `NewDatasetDialog` | `UploadPanel.tsx`, `NewDatasetDialog.tsx` | L:8, L:38, L:20 |
| `PageHeader`, `ErrorList`, `FieldSelect` | respective files | L:10, L:8, L:26 |

## Files

| File | Domain | Deps | Function |
|---|---|---|---|
| `PreviewPanel.tsx` | Preview | `← ../store.ts, ../types.ts, ../charts/echarts/*, InsightList.tsx, Recommendations.tsx \| → ../App.tsx` | Analysis detail panel (stage 18): header + context line, chart sized to ~64% of the panel, "Why this matters" (insight text / why_it_matters), "Supporting evidence" (effect, n, confidence, warnings), collapsible chart spec, actions row (Save/Remove, Export, Enlarge, Close); enlarge Dialog re-renders the same RenderResult at 70vh. |
| `InsightList.tsx` | Insights | `← ../types.ts \| → Recommendations.tsx, PreviewPanel.tsx` | Key Findings rows: badge + headline (`headline ?? text`, never truncated) as one `aria-expanded` button; expanded detail = full text, why it matters, evidence `dl`, View chart. Weight lead/secondary/compact by `supported`. |
| `Recommendations.tsx` | Insights | `← ../api.ts, InsightList.tsx \| → ../pages/InsightsPage.tsx, ../pages/WorkspacePage.tsx` | Key Findings section first, then "More charts" (h2) with tier h3s; cards show title, purpose, `type · aggregation`, confidence tooltip, spec tooltip (Info button), Preview; `data-x`/`data-y` for e2e; empty-top honesty text. |
| `ManualBuilder.tsx` | Explore | `← ../store.ts, ../types.ts, ../api.ts \| → ../pages/ExplorePage.tsx` | Spec form with type-dependent field filters; 422 errors listed. |
| `DatasetOverview.tsx` | Overview | `← ../types.ts \| → ../pages/OverviewPage.tsx` | Column table with semantic-type badges and quality flags. |
| `Sidebar.tsx` | Shell | `← ../store.ts, ../lib/router.ts \| → ../App.tsx` | Page nav + dataset list; hash links. |
| `UploadPanel.tsx`, `NewDatasetDialog.tsx` | Upload | `← ../api.ts \| → ../App.tsx` | Upload hook/dropzone; modal wrapper. |
| `ChartView.tsx` | Chart | `← ../charts/echarts/* \| → PreviewPanel.tsx` | `buildOption` + `EChartsView` with height prop and empty state. |
| `PageHeader.tsx`, `ErrorList.tsx`, `FieldSelect.tsx` | Shared | — | Small shared pieces. |
