---
mode: learning
generated_at: 2026-09-15
---

> One component per sidebar page (Overview / Insights / Explore / Workspace); pages receive state and dispatch from `App.tsx`, they own no fetching.

## Task Guide

| Task | Domain | Target | Also Check |
|---|---|---|---|
| Understand dataset summary page (stats tiles, quality, column table) | Overview | `OverviewPage.tsx` | `../components/DatasetOverview.tsx` |
| Understand recommendations page wrapper (LLM pending, message) | Insights | `InsightsPage.tsx` | `../components/Recommendations.tsx` |
| Understand manual builder page (recap of last spec) | Explore | `ExplorePage.tsx` | `../components/ManualBuilder.tsx`, `../store.ts` (`ExploreForm`) |
| Understand saved charts list per dataset | Workspace | `WorkspacePage.tsx` | `../store.ts` (`SavedChart`, `workspaceOf`) |

## Files

| File | Domain | Deps | Function |
|---|---|---|---|
| `OverviewPage.tsx` | Overview | `← ../components/DatasetOverview.tsx \| → ../App.tsx` | Tiles + column table + warnings. |
| `InsightsPage.tsx` | Insights | `← ../components/Recommendations.tsx \| → ../App.tsx` | Tier sections + insights; `RecommendationsResponse` consumer. |
| `ExplorePage.tsx` | Explore | `← ../components/ManualBuilder.tsx, ../store.ts \| → ../App.tsx` | Form state lifted to store (16.2 fix). |
| `WorkspacePage.tsx` | Workspace | `← ../store.ts \| → ../App.tsx` | Saved chart cards; click → preview. |
