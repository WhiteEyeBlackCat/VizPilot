---
mode: learning
generated_at: 2026-09-15
---

> Optional local-LLM layer: builds metadata-only prompts, calls an OpenAI-compatible endpoint, parses tolerant JSON, and merges verified LLM suggestions/insights with rule recommendations. Disabled provider = rules only.

## Task Guide

| Task | Domain | Target | Also Check |
|---|---|---|---|
| Understand the recommendations response assembly (rules + LLM merge, tiers, insights) | Service | `service.py` (`RecommendationService.get`) | `../charts/rules.py` (`evaluate_llm_spec`, `assign_tiers`) |
| Understand what the LLM is shown (evidence summary, derived/near-dup rules, few-shots) | Prompt | `prompts.py` (`build_messages`) | `../profiling/models.py` |
| Understand provider abstraction / disabled fallback / errors | Provider | `provider.py` | `../config.py`, DECISIONS D4, D7 |
| Understand tolerant parsing of LLM JSON | Schema | `schemas.py` | `service.py` (per-item validation) |
| Understand insight verification (strong/weak/unverified/neutral → tier cap) | Service | `service.py` (`integrate`) | `../charts/rules.py` (`evaluate_llm_spec`) |

## Key Exports

| Symbol | Source | Line |
|---|---|---|
| `RecommendationService` | `service.py` | L:67 |
| `LLMProvider`, `DisabledProvider`, `OpenAICompatProvider`, `LLMError` | `provider.py` | L:28, L:34, L:41, L:22 |
| `LLMResponse`, `LLMChartSuggestion`, `LLMInsight` | `schemas.py` | L:30, L:6, L:21 |
| `build_messages` | `prompts.py` | L:66 |

## Files

| File | Domain | Deps | Function |
|---|---|---|---|
| `service.py` | Service | `← charts/rules.py, charts/confidence.py, charts/spec.py, profiling/models.py \| → main.py, datasets/router.py` | `get(profile, use_llm)` → `{charts, insights, message, warnings}`; per-dataset lock/cache; canonicalize → verify → merge → diversity caps → tiers. |
| `prompts.py` | Prompt | `← charts/rules.py, profiling/models.py` | System/user messages from profile metadata (no raw rows beyond tiny sample), evidence top-k, forbidden insight rules. |
| `provider.py` | Provider | `← charts/rules.py, profiling/models.py` | `LLMProvider` ABC; OpenAI-compatible HTTP client (base_url/model from Settings); disabled stub. |
| `schemas.py` | Schema | — | Loose Pydantic models; `validate_spec` is the real gate. |

## File Dependencies

| File | Imports (in-dir) | Exposed To (in-dir) |
|---|---|---|
| `schemas.py` | — | `provider`, `service` |
| `prompts.py` | — | `provider` |
| `provider.py` | `prompts`, `schemas` | `service` |
| `service.py` | `provider`, `schemas` | — |

Note: stage 17.3 (branch `worktree-agent-a7a3af35e9c056762`, not on main) adds `coverage.py` and a two-stage hypothesis→probe→final workflow in this directory. Regenerate this map after merge.
