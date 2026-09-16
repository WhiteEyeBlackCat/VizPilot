---
mode: learning
generated_at: 2026-09-15
---

> Chart layer: `ChartSpec` contract + validation, semantic roles → visualization intent → deterministic chart selection (stage 19), rule-based recommendation/scoring/tiering, confidence layer, and backend aggregation into `RenderResult` (16 chart types).

## Task Guide

| Task | Domain | Target | Also Check |
|---|---|---|---|
| Understand the ChartSpec fields and the validation matrix (422 errors) | Spec | `spec.py` | `../../tests/test_spec.py` |
| Understand a column's order semantics (nominal / ordinal / cyclic; month, weekday, hour) | Roles | `roles.py` (`column_role`, `category_key`) | `../../tests/test_roles.py` |
| Understand why a finding is drawn as this chart family (intent, guards, preferred + alternatives) | Intent | `intent.py` (`classify_intent`, `eligible_charts`, `select_chart`) | `../../tests/test_intent.py`, `../../tests/test_intent_pipeline.py` |
| Understand how a suggested (LLM / probe) chart is re-selected by intent | Intent | `rules.py` (`select_for_spec`, `_variant`) | `../llm/service.py` (`_Merger.integrate_spec`) |
| Understand how rule candidates are generated, scored, capped and tiered | Recommendation | `rules.py` (`recommend_charts`, `assign_tiers`, `apply_diversity_caps`) | `confidence.py`, `../profiling/evidence.py` |
| Understand how an LLM-proposed spec is verified against evidence | Recommendation | `rules.py` (`evaluate_llm_spec`, `canonicalize_spec`) | `../llm/service.py` |
| Understand derived / near-duplicate demotion and warnings | Recommendation | `rules.py` (`apply_derived_caps`, `definitional_reason`, `dedup_equivalent`) | `../profiling/evidence.py` |
| Understand `score = base × sample_size × missingness × robustness` and tier caps | Confidence | `confidence.py` (`assess`, `column_robustness`) | DECISIONS D11 |
| Understand `Warning` / `Confidence` payload shapes | Confidence | `confidence.py` L:122–150 | — |
| Understand how a chart's data is aggregated (16 types; density / interval / two-state / hexbin / matrix payloads) | Render | `render.py` | `../profiling/types.py` (`apply_semantic_casts`), `../../tests/test_render_stage19.py` |
| Understand category order at render time (natural vs ranked, `ChartSpec.sort`) | Render | `render.py` (`_natural`, `_order_values`, `_ordered_groups`) | `roles.py` (`category_key`) |
| Understand time bucketing on line charts | Render | `rules.py` (`choose_time_granularity`) → `render.py` | — |

## Key Exports

| Symbol | Source | Line |
|---|---|---|
| `ChartSpec`, `validate_spec`, `ChartType`, `SortOrder`, `CATEGORY_AXIS_TYPES` | `spec.py` | L:65, L:99, L:18, L:41, L:53 |
| `ColumnRole`, `column_role`, `roles_for`, `category_key` | `roles.py` | L:69, L:141, L:196, L:216 |
| `Intent`, `IntentContext`, `ChartChoice`, `classify_intent`, `eligible_charts`, `select_chart` | `intent.py` | L:19, L:48, L:64, L:77, L:120, L:178 |
| `select_for_spec` | `rules.py` | `rg "def select_for_spec" backend/app/charts/rules.py` |
| `Recommendation` | `rules.py` | L:90 |
| `recommend_charts`, `assign_tiers`, `apply_diversity_caps` | `rules.py` | L:347, L:393, L:1077 |
| `QUESTION_DECAY`, `TYPE_DECAY`, `TYPE_FREE_SLOTS` (stage 19.1 slot value) | `rules.py` | L:46 |
| `evaluate_llm_spec`, `canonicalize_spec`, `dedup_equivalent` | `rules.py` | L:877, L:981, L:1032 |
| `apply_confidence`, `apply_derived_caps`, `derived_column_warnings` | `rules.py` | L:117, L:132, L:197 |
| `slope_spread_threshold`, `choose_time_granularity` | `rules.py` | L:251, L:233 |
| `Confidence`, `Warning`, `assess`, `TOP_CONFIDENCE_FLOOR`, `MAX_MISSING_RATIO` | `confidence.py` | L:129, L:122, L:377, L:45, L:52 |
| `with_caution`, `excluded_column_warnings`, `has_suspected_sentinels`, `sentinel_summary` | `confidence.py` | L:682, L:692, L:552, L:563 |
| `render_chart`, `RenderError`, `CastedFrameCache` | `render.py` | L:50, L:27, L:31 |

## Files

| File | Domain | Deps | Function |
|---|---|---|---|
| `spec.py` | Spec | `← profiling/models.py, roles.py \| → 7 files (foundational); rg "charts.spec" backend -l` | `ChartSpec` model (16 types, `sort`), `validate_spec` (type/role/semantic matrix; polar needs a cyclic x, two-state needs exactly 2 states, density x ranges; no pie). |
| `roles.py` | Roles | `← profiling/models.py \| → spec.py, intent.py, rules.py, render.py` | Deterministic `ColumnRole` from a ColumnProfile (name hints + value range for cycles, label scales / integer codes for ordinals); `category_key` for render-time order. |
| `intent.py` | Intent | `← roles.py, spec.py \| → rules.py` | Intent literal, guards (`DENSITY_MIN_N`, `HEXBIN_MIN_N`, …), `classify_intent`, `eligible_charts`, `select_chart` (hint promotes only eligible charts). |
| `confidence.py` | Confidence | `← profiling/models.py \| → rules.py, render.py, llm/service.py, probes/*` | Confidence factors (1/√n curves, missingness, sentinel/extreme robustness), `Warning`, tier caps, reason `Caution:` suffix. |
| `rules.py` | Recommendation | `← profiling/models.py, roles.py, intent.py \| → llm/*, probes/engine.py, datasets/router.py` | Rule engine: one candidate per finding (`_category_charts`, `_cyclic_numeric_charts`, `_two_state_charts`, `_matrix_charts`, lines, scatter/hexbin, histogram, heatmap) → evidence score → derived/near-dup caps → diversity caps (per family ≤3, per-x ≤2, per-y ≤3; stage 19.1: slots filled by slot value = score × `QUESTION_DECAY`^same-question × `TYPE_DECAY`^same-raw-type-beyond-2, list stays score-ordered) → tiers; `Recommendation.intent/alternatives`; `distribution_by_group` reads `evidence.dispersion` (strong) via `_EvidenceIndex.spread_differs`; `select_for_spec` + `evaluate_llm_spec` (family scoring) for LLM charts. |
| `render.py` | Render | `← profiling/types.py, serialization.py, roles.py \| → datasets/router.py, main.py` | Per-type aggregation into `{spec, chart_data, sampled, n_points, display_range}`; natural category order for ordered roles; density (binned Gaussian KDE), dot interval, two-state, hexbin, matrix heatmap; `HARD_MAX_POINTS`; sentinel-aware histogram range. |

## File Dependencies

| File | Imports (in-dir) | Exposed To (in-dir) |
|---|---|---|
| `roles.py` | — | `spec`, `intent`, `rules`, `render` |
| `spec.py` | `roles` | `confidence`, `intent`, `rules`, `render` |
| `intent.py` | `roles`, `spec` | `rules` |
| `confidence.py` | `spec` | `rules`, `render` |
| `rules.py` | `spec`, `confidence`, `intent`, `roles` | `render` (`choose_time_granularity`) |
| `render.py` | `spec`, `confidence`, `rules`, `roles` | — |
