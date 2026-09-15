---
mode: learning
generated_at: 2026-09-15
---

> Feature components composed by the pages and the app shell: sidebar, preview panel, recommendations list, manual chart builder, dataset overview, upload.

## Task Guide

| Task | Domain | Target | Also Check |
|---|---|---|---|
| Understand the shared right-hand preview panel (chart, warnings, save, enlarge dialog) | Preview | `PreviewPanel.tsx` | `../store.ts` (`Preview`, `SavedChart`), `ChartView.tsx` |
| Understand tier sections, supported badges, insight list | Insights | `Recommendations.tsx` | `../pages/InsightsPage.tsx`, `../types.ts` |
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
| `PreviewPanel` | `PreviewPanel.tsx` | L:69 |
| `Recommendations` | `Recommendations.tsx` | L:94 |
| `ManualBuilder` | `ManualBuilder.tsx` | L:25 |
| `DatasetOverview`, `qualityFlags`, `hasQualityFlags`, `TYPE_VARIANT` | `DatasetOverview.tsx` | L:102, L:61, L:92, L:13 |
| `Sidebar` | `Sidebar.tsx` | L:37 |
| `ChartView` | `ChartView.tsx` | L:32 |
| `useUploader`, `DropZone`, `NewDatasetDialog` | `UploadPanel.tsx`, `NewDatasetDialog.tsx` | L:8, L:38, L:20 |
| `PageHeader`, `ErrorList`, `FieldSelect` | respective files | L:10, L:8, L:26 |

## Files

| File | Domain | Deps | Function |
|---|---|---|---|
| `PreviewPanel.tsx` | Preview | `← ../store.ts, ../types.ts, ../charts/echarts/* \| → ../App.tsx` | Shows current `Preview`; save/unsave; enlarge Dialog re-rendering same RenderResult at 70vh; warning chips. |
| `Recommendations.tsx` | Insights | `← ../api.ts \| → ../pages/InsightsPage.tsx` | Top/secondary/exploratory sections, LLM pending skeleton, confidence labels, empty-top honesty text. |
| `ManualBuilder.tsx` | Explore | `← ../store.ts, ../types.ts, ../api.ts \| → ../pages/ExplorePage.tsx` | Spec form with type-dependent field filters; 422 errors listed. |
| `DatasetOverview.tsx` | Overview | `← ../types.ts \| → ../pages/OverviewPage.tsx` | Column table with semantic-type badges and quality flags. |
| `Sidebar.tsx` | Shell | `← ../store.ts, ../lib/router.ts \| → ../App.tsx` | Page nav + dataset list; hash links. |
| `UploadPanel.tsx`, `NewDatasetDialog.tsx` | Upload | `← ../api.ts \| → ../App.tsx` | Upload hook/dropzone; modal wrapper. |
| `ChartView.tsx` | Chart | `← ../charts/echarts/* \| → PreviewPanel.tsx` | `buildOption` + `EChartsView` with height prop and empty state. |
| `PageHeader.tsx`, `ErrorList.tsx`, `FieldSelect.tsx` | Shared | — | Small shared pieces. |
