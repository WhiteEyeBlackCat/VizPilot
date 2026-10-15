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
| Understand which numbers the LLM may quote and the neutral fallback | Wording gate | `service.py` (`_wording_problem`, `_mask_column_tokens`, `quotable_numbers`, `_neutral`, `_template`) | `tests/test_llm_workflow.py` (number/date cases) |
| Understand what a finding may say (direction, group order, number-to-group binding, scope, no causation) and the claim template | Claims | `claims.py` (`build_claim`, `render_claim`, `semantic_problem`) | `service.py` (`claim_of`, `_template`, `_caption`), `tests/test_semantic_claims.py` |
| Understand recommendation caching (failure TTL, key, `?refresh=1`) | Service | `service.py` (`RecommendationService.get`, `_cache_key`, `FAILURE_TTL_SECONDS`) | `provider.py` (`cache_identity`), `prompts.py` (`PROMPT_VERSION`), `tests/test_llm_cache.py` |
| Understand what LLM #1 / LLM #2 are shown | Prompt | `prompts.py` (`build_hypothesis_messages`, `build_final_messages`, `PROBE_TYPE_GUIDE`) | `../profiling/models.py` (layer 2 signals) |
| Understand provider abstraction / disabled fallback / usage accounting | Provider | `provider.py` | `../config.py`, DECISIONS D4, D7 |
| Understand tolerant parsing of LLM JSON | Schema | `schemas.py`, `service.py` (`_parse_hypotheses`, `_parse_final`, `_parse_suggestion`) | — |
| Chart-only suggestions (no probe type) and bare chart lists | Service | `service.py` (`_Merger`, `coverage.infer_probe`) | `../charts/rules.py` (`evaluate_llm_spec`) |

## Key Exports

| Symbol | Source | Line |
|---|---|---|
| `RecommendationService` | `service.py` | L:118 |
| `NO_PATTERNS_MESSAGE`, `FALLBACK_MESSAGE`, `MAX_INSIGHTS`, `MAX_HYPOTHESES`, `FAILURE_TTL_SECONDS` | `service.py` | L:76, L:75, L:70, L:71, L:79 |
| `quotable_numbers`, `claim_of` | `service.py` | L:1112, L:1089 |
| `Claim`, `build_claim`, `render_claim`, `semantic_problem` | `claims.py` | L:88, L:202, L:326, L:545 |
| `check_coverage`, `CoverageHit`, `ValidatedHypothesis` | `coverage.py` | L:90, L:38, L:49 |
| `apply_confidence_cap`, `corroborated`, `infer_probe`, `suggest_chart`, `required_roles`, `definitional_conflict` | `coverage.py` | L:326, L:362, L:421, L:447, L:479, L:484 |
| `build_hypothesis_messages`, `build_final_messages`, `PROBE_TYPE_GUIDE`, `PROMPT_VERSION` | `prompts.py` | L:179, L:485, L:59, L:38 |
| `_roles_section`, `_dispersion_lines` (stage 19.1 metadata for LLM #1) | `prompts.py` | L:221, L:244 |
| `LLMProvider`, `DisabledProvider`, `OpenAICompatProvider`, `LLMError` | `provider.py` | L:35, L:45, L:57, L:29 |
| `Hypothesis`, `HypothesisResponse`, `FinalInsight`, `FinalResponse`, `LLMChartSuggestion`, `LLMUsage` | `schemas.py` | L:32, L:50, L:59, L:72, L:17, L:77 |

## Files

| File | Domain | Deps | Function |
|---|---|---|---|
| `service.py` | Service / Workflow | `← charts/rules.py, charts/confidence.py, charts/spec.py, probes/*, profiling/models.py, coverage, provider, schemas \| → main.py, datasets/router.py` | `get(profile, use_llm, df, include_debug)` → `{charts, insights, message, warnings, debug?}`. `_Workflow`: cap 5 hypotheses → gate (columns exist, probe type known, roles typed, no definitional/near-dup claim, unknown snake_case tokens) → dedupe (test, columns) → coverage hit or probe (≤5, cached, n≥30) → LLM #2 once iff a probe validated something or ≥2 validated → wording gate (only backend numbers/dates quotable, else template/neutral) → `_Merger` integrates backend or LLM charts through the same caps/tiers as rules. `_Trace` = `debug` payload. |
| `claims.py` | Claims | `← probes/engine.py (NEAR_ZERO_CORR) \| → service, prompts` | Audit A1/A2: `build_claim(v, profile)` turns validated evidence (coverage- or probe-shaped, normalised) into a `Claim` per probe type — group order by mean/median, per-group numbers, per-group correlation signs (near-zero |r| < 0.10 has no direction), same_sign / direction / unmeasured groups, Simpson flag, curve shape, level-shift rise/fall; a grouped_relationship with opposite signs becomes kind `sign_flip`. `render_claim` = deterministic wording (the template). `semantic_problem(text, claim)` rejects causal words, comparisons / superlatives against the evidence order, numbers next to the wrong group, direction words against the measured sign, universal quantifiers unless every group was estimated with one sign, generalisations over non-claim groupings; claim types it cannot bind (interaction, grouped time pattern, group difference without per-group means, non-monotone curves) accept only text without such words. Fail-closed (review round 2): non-English letters, comparison symbols, negated assertions, unknown entities next to an assertion, and any evaluative word (broad class incl. unknown -er/-est adjectives) the verifier does not recognise and bind reject the sentence; regression set in `tests/test_wording_attacks.py`. |
| `coverage.py` | Coverage | `← charts/confidence.py, charts/rules.py, charts/spec.py, probes/engine.py, probes/schemas.py, profiling/models.py` | `check_coverage(test, columns, profile)` looks a claim up in evidence L1 (η², correlation, slope spread, interaction, time buckets) and L2 (nonlinear, directed `(x, y)`; conditional; change points; distributions) and returns a `CoverageHit` with the probe thresholds' verdict; confidence cap for tiny n; `corroborated` for slope_difference; `infer_probe` maps a chart-only suggestion to a test; `suggest_chart` builds the backend chart for a validated claim. |
| `prompts.py` | Prompt | `← charts/roles.py, charts/rules.py, probes/engine.py, profiling/models.py, coverage` | Hypothesis prompt: user rules, probe-type guide with roles, column briefs, "Column roles" (cyclic period / order, ordered scales, two-state; stage 19.1) and "Spread differences" (`evidence.dispersion`) sections, evidence L1 top-k, selected L2 signals (bounded, no series dumps), ≤5 sample rows; chart hints may name polar_line / rose / raincloud / ridgeline. Final prompt: only validated facts with backend numbers and the columns involved. |
| `provider.py` | Provider | `← charts/rules.py, profiling/models.py, coverage, prompts, schemas` | `LLMProvider` protocol (`generate_hypotheses`, `finalize_insights`); OpenAI-compatible HTTP client (base_url/model from Settings, per-call `LLMUsage`); disabled stub. |
| `schemas.py` | Schema | — | Loose Pydantic models for both calls; `validate_spec` remains the chart gate; probe requests are re-validated by `probes/schemas.py` (`extra="forbid"`). |

## File Dependencies

| File | Imports (in-dir) | Exposed To (in-dir) |
|---|---|---|
| `schemas.py` | — | `provider`, `service` |
| `coverage.py` | — | `prompts`, `provider`, `service` |
| `prompts.py` | `claims`, `coverage` | `provider`, `service` |
| `provider.py` | `coverage`, `prompts`, `schemas` | `service` |
| `claims.py` | — (`service._number_supported`, lazily) | `prompts`, `service` |
| `service.py` | `claims`, `coverage`, `prompts`, `provider`, `schemas` | — |

Invariants: raw rows never leave `prompts.py` beyond `MAX_SAMPLE_ROWS`; every LLM-stated number is stripped or must equal a backend number for that hypothesis; every LLM-written sentence that reaches the user (LLM #2 text / headline / why_it_matters, LLM #1 reason, chart captions and titles) passes `semantic_problem` against the claim; the template never prints LLM #1's statement; the LLM never chooses thresholds; `use_llm=False` takes the pre-17.3 code path (no trace, no workflow fields). Cache: deterministic results never expire; results with an LLM transport failure (`llm1:` / `llm2:` errors) live `FAILURE_TTL_SECONDS`; the key includes PROFILE_VERSION, PROMPT_VERSION and the provider's `cache_identity`; `?refresh=1` rebuilds.
