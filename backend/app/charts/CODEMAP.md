---
mode: learning
generated_at: 2026-09-15
---

> Chart layer: `ChartSpec` contract + validation, rule-based recommendation/scoring/tiering, confidence layer, and backend aggregation into `RenderResult`.

## Task Guide

| Task | Domain | Target | Also Check |
|---|---|---|---|
| Understand the ChartSpec fields and the validation matrix (422 errors) | Spec | `spec.py` | `../../tests/test_spec.py` |
| Understand how rule candidates are generated, scored, capped and tiered | Recommendation | `rules.py` (`recommend_charts`, `assign_tiers`, `apply_diversity_caps`) | `confidence.py`, `../profiling/evidence.py` |
| Understand how an LLM-proposed spec is verified against evidence | Recommendation | `rules.py` (`evaluate_llm_spec`, `canonicalize_spec`) | `../llm/service.py` |
| Understand derived / near-duplicate demotion and warnings | Recommendation | `rules.py` (`apply_derived_caps`, `definitional_reason`, `dedup_equivalent`) | `../profiling/evidence.py` |
| Understand `score = base × sample_size × missingness × robustness` and tier caps | Confidence | `confidence.py` (`assess`, `column_robustness`) | DECISIONS D11 |
| Understand `Warning` / `Confidence` payload shapes | Confidence | `confidence.py` L:122–150 | — |
| Understand how a chart's data is aggregated (line/bar/scatter/histogram/box/heatmap) | Render | `render.py` | `../profiling/types.py` (`apply_semantic_casts`) |
| Understand time bucketing on line charts | Render | `rules.py` (`choose_time_granularity`) → `render.py` | — |

## Key Exports

| Symbol | Source | Line |
|---|---|---|
| `ChartSpec`, `validate_spec` | `spec.py` | L:28, L:61 |
| `Recommendation` | `rules.py` | L:54 |
| `recommend_charts`, `assign_tiers`, `apply_diversity_caps` | `rules.py` | L:284, L:321, L:762 |
| `evaluate_llm_spec`, `canonicalize_spec`, `dedup_equivalent` | `rules.py` | L:583, L:666, L:717 |
| `apply_confidence`, `apply_derived_caps`, `derived_column_warnings` | `rules.py` | L:77, L:92, L:157 |
| `slope_spread_threshold`, `choose_time_granularity` | `rules.py` | L:211, L:193 |
| `Confidence`, `Warning`, `assess`, `TOP_CONFIDENCE_FLOOR`, `MAX_MISSING_RATIO` | `confidence.py` | L:129, L:122, L:377, L:45, L:52 |
| `with_caution`, `excluded_column_warnings`, `has_suspected_sentinels`, `sentinel_summary` | `confidence.py` | L:682, L:692, L:552, L:563 |
| `render_chart`, `RenderError`, `CastedFrameCache` | `render.py` | L:50, L:27, L:31 |

## Files

| File | Domain | Deps | Function |
|---|---|---|---|
| `spec.py` | Spec | `← profiling/models.py \| → 7 files (foundational); rg "charts.spec" backend -l` | `ChartSpec` model, `ChartType/Aggregation/TimeGranularity` literals, `validate_spec` (type/role/semantic matrix; no pie). |
| `confidence.py` | Confidence | `← profiling/models.py \| → rules.py, render.py, llm/service.py, probes/*` | Confidence factors (1/√n curves, missingness, sentinel/extreme robustness), `Warning`, tier caps, reason `Caution:` suffix. |
| `rules.py` | Recommendation | `← profiling/models.py \| → llm/*, probes/engine.py, datasets/router.py` | Rule engine: candidates → evidence score → derived/near-dup caps → diversity caps (per-x ≤2, per-y ≤3) → score-based tiers (top/secondary/exploratory, 0.68 floor); LLM spec verification. |
| `render.py` | Render | `← profiling/types.py, serialization.py \| → datasets/router.py, main.py` | Per-type aggregation into `{spec, chart_data, sampled, n_points, display_range}`; `HARD_MAX_POINTS`; sentinel-aware histogram range. |

## File Dependencies

| File | Imports (in-dir) | Exposed To (in-dir) |
|---|---|---|
| `spec.py` | — | `confidence`, `rules`, `render` |
| `confidence.py` | `spec` | `rules`, `render` |
| `rules.py` | `spec`, `confidence` | `render` (`choose_time_granularity`) |
| `render.py` | `spec`, `confidence`, `rules` | — |
