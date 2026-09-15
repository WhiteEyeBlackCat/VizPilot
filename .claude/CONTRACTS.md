# CONTRACTS.md — cross-module data contracts

Source of truth: backend Pydantic models. `frontend/src/types.ts` mirrors them and must change in lockstep. All changes are additive; removals need a user decision.

## DatasetProfile — `profiling/models.py` L:413
- Producer: `profile_dataset` (`profiling/profiler.py`), cached by `ProfileService`, keyed by dataset id, invalidated when `PROFILE_VERSION` (now 10) changes.
- Consumers: `charts/rules.py`, `charts/render.py`, `charts/spec.py` (validation), `llm/*`, `probes/*`, router `/profile`, frontend Overview/Explore.
- Fields: `profile_version, dataset_id, n_rows, n_cols, sampled, profiled_rows, columns[ColumnProfile], correlations, evidence, sample_rows(≤5)`.
- Invariants: every statistic was computed on `profiled_rows` rows (== `n_rows` unless sampled >100k); `sample_rows` is the only raw data anywhere downstream; `semantic_type ∈ numeric|categorical|datetime|boolean|text|id|unknown`; `nominal=True` categoricals are never a numeric y; `quality` carries sentinels/robust range/invalid counts and never mutates the stored data.

## Evidence — `profiling/models.py` L:401 (+ `EvidenceLayer2` L:391)
- Producer: `compute_evidence` (L1) and `compute_layer2` (L2), both deterministic Polars.
- Consumers: L1 → `rules.py` scoring, `evaluate_llm_spec`, prompts, probes. L2 → prompts and probes only (rules never read L2).
- L1: `cat_num[CatNumEffect(adjusted eta², n_total, n_min, group_counts)]`, `time_effects`, `num_num_spearman`, `interactions`, `slope_heterogeneity`, `derived_columns` (row-wise identities like `a×b×(1−c)`; ≥99% match), `near_duplicate_groups` (|ρ|≥0.995, or ≥0.98 with related names; representative column chosen).
- Invariants: derived pairs are "definitional, not a finding" (score ×0.6, tier cap exploratory, warning); near-duplicate non-representatives are excluded from candidates (heatmap keeps all); every effect carries the n it was computed on.

## ChartSpec — `charts/spec.py` L:28
- Producers: rules engine, LLM (via canonicalize + `validate_spec`), manual builder (frontend), fixtures.
- Consumers: `render_chart`, `Recommendation.spec`, frontend preview/workspace (`specKey` identity).
- Fields: `title, type(line|bar|scatter|histogram|box|heatmap), x, y, group_by, aggregation, bins(histogram), time_granularity(line+datetime x), top_n(bar; default 20), reason, priority`.
- Invariants: `validate_spec(spec, profile)` is the only gate; errors return 422 `{detail:{errors:[…]}}`; no pie; bar never on high-cardinality categoricals; column roles must match semantic types.

## Recommendation — `charts/rules.py` L:54
- Producer: `recommend_charts` → caps → `apply_confidence` → `assign_tiers` (rules); `RecommendationService` for merged LLM output.
- Consumers: router `/recommendations`, frontend Insights/Preview.
- Fields: `spec, score, source(rules|llm), tier(top|secondary|exploratory), confidence{sample_size, missingness, robustness, overall, n_total, n_effective, missing_ratio, min_group_n, n_source}, warnings[{code, severity, message, meta}]`; `tier_cap` internal only (excluded from JSON).
- Invariants: `score = base × confidence.overall` (base recoverable); tiers are score-based with a 0.68 top floor, decoupled from display order; diversity caps per-x ≤2, per-y ≤3; a rec with `tier_cap` never exceeds it; sentinel-sensitive aggregations (mean/sum/min/max) are discounted, median/count are not.

## Insight / RecommendationsResponse — `llm/service.py` `get()`
- Producer: `RecommendationService.get(profile, use_llm)`.
- Consumers: router, frontend Insights page.
- Shape: `{charts: Recommendation[], insights: [{text, supported(strong|weak|unverified; backend-internal "neutral" is shown as weak), chart_priority|null}], message|null, warnings: Warning[]}`.
- Invariants: with `use_llm=False` the output is a pure function of the profile (byte-identical across LLM changes); weak insights never make top; an insight's chart must pass `validate_spec` or `chart_priority` is null; `message` explains an empty result honestly; dataset-level warnings disclose excluded columns (`MAX_MISSING_RATIO`), derived and near-duplicate columns.

## RenderResult — `charts/render.py` L:50 → frontend `types.ts` L:211
- Producer: `render_chart(df, spec, profile)` via `POST /api/charts/render {dataset_id, ...spec}`.
- Consumers: `buildOption` (frontend), fixtures in `frontend/src/charts/__fixtures__/`.
- Shape: `{spec, chart_data, sampled, n_points, display_range|null}`; `chart_data` per type: line/scatter `{series[{name, x, y}], y_label}`, bar `{categories, series[{name, values}], truncated, y_label}`, histogram `{bins{edges, counts}, series?, y_label}`, box `{groups[{name, q1, median, q3, min, max, outliers, mean}], y_label}`, heatmap `{columns, matrix(−1..1, null allowed), y_label}`.
- Invariants: category order is backend order; nulls are gaps, never zeros; histogram binned over robust range with `display_range` disclosing excluded rows (`sum(counts)+excluded == valid rows`); `y_label` alone drives the axis title; `HARD_MAX_POINTS` → `RenderError` (422), scatter sampled with `sampled=true`.

## ProbeRequest / ProbeResult — `probes/schemas.py` L:49 / L:70
- Producer: LLM workflow `llm/service.py` (`_Workflow`, stage 17.3); engine `run_probes`.
- Shape: request `{type, columns{role: name}}` with `extra="forbid"`; result `{type, columns, effect_size, effect_label, n, n_min_group, confidence, verdict(pass|weak|fail), thresholds{pass, weak}, evidence(structured only), chart|null, notes, cached}`; rejected `{type, columns, reason, rejected: true}`.
- Invariants: never returns raw rows; thresholds come from the engine, LLM-stated numbers are ignored; ≤5 probes per dataset; cache key = (type, sorted columns).

## HTTP API — `datasets/router.py`
`GET /api/health` · `POST /api/datasets` (multipart) → `DatasetMeta` · `GET /api/datasets` · `GET /api/datasets/{id}` · `/preview` · `/profile` → DatasetProfile · `/recommendations?llm=true|false&debug=0|1` → RecommendationsResponse (`debug=1` adds the optional `debug` workflow trace) · `POST /api/charts/render` → RenderResult. Errors: 404 unknown id, 422 `{detail:{errors}}` for spec/loader problems.
