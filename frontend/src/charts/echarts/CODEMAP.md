---
mode: learning
generated_at: 2026-09-15
---

> ECharts adapter: pure `buildOption(RenderResult) → VizOption` for the six chart types, a single theme source, tree-shaken registration, and a 50-line React binding. Verified against 18 real-backend fixtures.

## Task Guide

| Task | Domain | Target | Also Check |
|---|---|---|---|
| Understand how a RenderResult becomes an ECharts option (per type) | Option | `option.ts` (`buildOption` L:466 dispatches to per-type builders) | `../__fixtures__/index.json` |
| Understand time-axis labels / histogram bins / box stats / heatmap cells | Option | `option.ts` (`timeLabelFormatter`, `histogramBins`, `heatmapCells`) | `option.test.ts` |
| Understand colours, fonts, grid, tooltip, toolbox, dark theme | Theme | `theme.ts` | `theme.test.ts` (contrast) |
| Understand which ECharts modules are registered | Registration | `echarts.ts` | — |
| Understand mount/resize/dispose and tooltip hiding | Binding | `EChartsView.tsx` | `../../components/ChartView.tsx` |

## Key Exports

| Symbol | Source | Line |
|---|---|---|
| `buildOption`, `isEmpty` | `option.ts` | L:466, L:48 |
| `EChartsView`, `hideAllTips`, `EChartsHost` | `EChartsView.tsx` | L:26, L:18, L:12 |
| `echarts`, `VizOption`, `ECharts` | `echarts.ts` | L:79, L:62 |
| `PALETTE`, `COLORS`, `DIVERGING` | `theme.ts` | L:23, L:34, L:52 |

## Files

| File | Domain | Deps | Function |
|---|---|---|---|
| `option.ts` | Option | `← ../../types.ts \| → ../../components/ChartView.tsx` | Builders for line/bar/scatter/histogram/box/heatmap; semantics mirror the old Plotly layer (category order, null gaps, display_range). |
| `theme.ts` | Theme | `→ ../../components/PreviewPanel.tsx` | Palette, text styles, grid presets, tooltip/legend/toolbox/dataZoom factories. |
| `echarts.ts` | Registration | — | `echarts/core` + used charts/components/renderers; `VizOption` composed type. |
| `EChartsView.tsx` | Binding | `→ ../../components/*` | Init/setOption/resize/dispose; exposes host element. |
| `option.test.ts`, `option.ssr.test.ts`, `theme.test.ts` | Tests | `← ../__fixtures__/*.json` | vitest: option values, SSR render, contrast. |

## File Dependencies

| File | Imports (in-dir) | Exposed To (in-dir) |
|---|---|---|
| `echarts.ts` | — | `option`, `EChartsView`, tests |
| `theme.ts` | — | `option`, tests |
| `option.ts` | `echarts`, `theme` | tests |
| `EChartsView.tsx` | `echarts` | — |
