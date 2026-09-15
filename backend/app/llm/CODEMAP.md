---
mode: learning
generated_at: 2026-09-16
---

> Optional local-LLM layer, two-stage since 17.3: LLM #1 proposes hypotheses from metadata → hallucination gate → coverage against evidence L1/L2 → targeted probes (≤5) for uncovered claims → LLM #2 words only the validated findings (quoting only backend numbers) → merge with rule recommendations. Disabled provider or any LLM failure = rules only; `?llm=false` output is byte-identical to pre-17.3.

## Task Guide

| Task | Domain | Target | Also Check |
|---|---|---|---|
| Understand the recommendations response assembly (rules + workflow, tiers, insights, `message`) | Service | `service.py` (`RecommendationService.get`, `_shape`) | `../charts/rules.py` (`assign_tiers`), `../datasets/router.py` (`?debug=1`) |
| Follow one hypothesis: gate → coverage → probe → wording | Workflow | `service.py` (`_Workflow`: `validate` → `_gate` → `_validated`, `_run_probes`, `needs_final_call`, `finish` → `_final_wording`, `_merge`) | `coverage.py` (`check_coverage`, `definitional_conflict`), `../probes/engine.py` |
| Understand what counts as "already answered by the tables" and its thresholds | Coverage | `coverage.py` (`check_coverage`, `apply_confidence_cap`, `corroborated`) | `../probes/engine.py` (`THRESHOLDS`), `../charts/confidence.py` (`assess`) |
| Understand which numbers the LLM may quote and the neutral fallback | Wording gate | `service.py` (`_wording_problem`, `quotable_numbers`, `_neutral`, `_template`) | `tests/test_llm_workflow.py` (number/date cases) |
| Understand what LLM #1 / LLM #2 are shown | Prompt | `prompts.py` (`build_hypothesis_messages`, `build_final_messages`, `PROBE_TYPE_GUIDE`) | `../profiling/models.py` (layer 2 signals) |
| Understand provider abstraction / disabled fallback / usage accounting | Provider | `provider.py` | `../config.py`, DECISIONS D4, D7 |
| Understand tolerant parsing of LLM JSON | Schema | `schemas.py`, `service.py` (`_parse_hypotheses`, `_parse_final`, `_parse_suggestion`) | — |
| Chart-only suggestions (no probe type) and bare chart lists | Service | `service.py` (`_Merger`, `coverage.infer_probe`) | `../charts/rules.py` (`evaluate_llm_spec`) |

## Key Exports

| Symbol | Source | Line |
|---|---|---|
| `RecommendationService` | `service.py` | L:106 |
| `NO_PATTERNS_MESSAGE`, `FALLBACK_MESSAGE`, `MAX_INSIGHTS`, `MAX_HYPOTHESES` | `service.py` | L:71, L:70, L:66, L:67 |
| `quotable_numbers` | `service.py` | L:913 |
| `check_coverage`, `CoverageHit`, `ValidatedHypothesis` | `coverage.py` | L:87, L:38, L:49 |
| `apply_confidence_cap`, `corroborated`, `infer_probe`, `suggest_chart`, `required_roles`, `definitional_conflict` | `coverage.py` | L:298, L:334, L:393, L:419, L:451, L:456 |
| `build_hypothesis_messages`, `build_final_messages`, `PROBE_TYPE_GUIDE` | `prompts.py` | L:145, L:400, L:43 |
| `LLMProvider`, `DisabledProvider`, `OpenAICompatProvider`, `LLMError` | `provider.py` | L:35, L:45, L:57, L:29 |
| `Hypothesis`, `HypothesisResponse`, `FinalInsight`, `FinalResponse`, `LLMChartSuggestion`, `LLMUsage` | `schemas.py` | L:32, L:50, L:59, L:72, L:17, L:77 |

## Files

| File | Domain | Deps | Function |
|---|---|---|---|
| `service.py` | Service / Workflow | `← charts/rules.py, charts/confidence.py, charts/spec.py, probes/*, profiling/models.py, coverage, provider, schemas \| → main.py, datasets/router.py` | `get(profile, use_llm, df, include_debug)` → `{charts, insights, message, warnings, debug?}`. `_Workflow`: cap 5 hypotheses → gate (columns exist, probe type known, roles typed, no definitional/near-dup claim, unknown snake_case tokens) → dedupe (test, columns) → coverage hit or probe (≤5, cached, n≥30) → LLM #2 once iff a probe validated something or ≥2 validated → wording gate (only backend numbers/dates quotable, else template/neutral) → `_Merger` integrates backend or LLM charts through the same caps/tiers as rules. `_Trace` = `debug` payload. |
| `coverage.py` | Coverage | `← charts/confidence.py, charts/rules.py, charts/spec.py, probes/engine.py, probes/schemas.py, profiling/models.py` | `check_coverage(test, columns, profile)` looks a claim up in evidence L1 (η², correlation, slope spread, interaction, time buckets) and L2 (nonlinear, directed `(x, y)`; conditional; change points; distributions) and returns a `CoverageHit` with the probe thresholds' verdict; confidence cap for tiny n; `corroborated` for slope_difference; `infer_probe` maps a chart-only suggestion to a test; `suggest_chart` builds the backend chart for a validated claim. |
| `prompts.py` | Prompt | `← charts/rules.py, probes/engine.py, profiling/models.py, coverage` | Hypothesis prompt: user rules, probe-type guide with roles, column briefs, evidence L1 top-k, selected L2 signals (bounded, no series dumps), ≤5 sample rows. Final prompt: only validated facts with backend numbers and the columns involved. |
| `provider.py` | Provider | `← charts/rules.py, profiling/models.py, coverage, prompts, schemas` | `LLMProvider` protocol (`generate_hypotheses`, `finalize_insights`); OpenAI-compatible HTTP client (base_url/model from Settings, per-call `LLMUsage`); disabled stub. |
| `schemas.py` | Schema | — | Loose Pydantic models for both calls; `validate_spec` remains the chart gate; probe requests are re-validated by `probes/schemas.py` (`extra="forbid"`). |

## File Dependencies

| File | Imports (in-dir) | Exposed To (in-dir) |
|---|---|---|
| `schemas.py` | — | `provider`, `service` |
| `coverage.py` | — | `prompts`, `provider`, `service` |
| `prompts.py` | `coverage` | `provider` |
| `provider.py` | `coverage`, `prompts`, `schemas` | `service` |
| `service.py` | `coverage`, `provider`, `schemas` | — |

Invariants: raw rows never leave `prompts.py` beyond `MAX_SAMPLE_ROWS`; every LLM-stated number is stripped or must equal a backend number for that hypothesis; the LLM never chooses thresholds; `use_llm=False` takes the pre-17.3 code path (no trace, no workflow fields).
