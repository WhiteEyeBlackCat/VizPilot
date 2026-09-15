---
mode: learning
generated_at: 2026-09-15
---

> Profiling pipeline: DataFrame → semantic typing → column stats/quality → evidence layer 1 (effects, correlations, derived/near-dup columns) → evidence layer 2 (structural summaries for the LLM). Output is `DatasetProfile` (Pydantic, `PROFILE_VERSION`).

## Task Guide

| Task | Domain | Target | Also Check |
|---|---|---|---|
| Understand how a column gets its semantic type (numeric/categorical/datetime/id/…) | Typing | `types.py` | `quality.py` (dirty numerics), `../../tests/test_types.py` |
| Understand what `DatasetProfile` / `Evidence` contain | Models | `models.py` | `.claude/CONTRACTS.md` |
| Understand how profiles are computed, cached and versioned | Orchestration | `profiler.py` | `models.py` (`PROFILE_VERSION`) |
| Understand evidence layer 1 (eta², Spearman, interactions, slope het, derived, near-dup) | Evidence L1 | `evidence.py` | `pairwise.py`, `layer2.py` (consumes L1) |
| Understand evidence layer 2 (distributions, group summaries, nonlinear, change points, anomalies) | Evidence L2 | `layer2.py.analysis.md` → `layer2.py` ranges | `evidence.py` (definitional pairs) |
| Understand sentinel / robust-range / invalid-value detection | Quality | `quality.py` | `types.py` (`missing_token_*`) |
| Understand batched correlation matrix | Evidence L1 | `pairwise.py` | — |

## Key Exports

| Symbol | Source | Line |
|---|---|---|
| `DatasetProfile`, `ColumnProfile`, `Evidence`, `EvidenceLayer2` | `models.py` | L:413, L:79, L:401, L:391 |
| `PROFILE_VERSION` | `models.py` | L:7 |
| `ProfileService` (cache + per-dataset lock + atomic json) | `profiler.py` | L:182 |
| `profile_dataset` | `profiler.py` | L:33 |
| `apply_semantic_casts`, `apply_casts`, `SAMPLE_SEED` | `types.py` | L:312, L:301 |
| `infer_column`, `infer_semantic_type` | `types.py` | L:167, L:162 |
| `compute_evidence`, `adjusted_eta_squared`, `span_bucket` | `evidence.py` | L:78, L:108, L:215 |
| `MAX_CAT_CATEGORIES` | `evidence.py` | L:31 |
| `compute_layer2` | `layer2.py` | L:128 |
| `robust_quality` | `quality.py` | L:52 |
| `pairwise_pearson` | `pairwise.py` | L:19 |

## Files

| File | Domain | Deps | Function |
|---|---|---|---|
| `models.py` | Models | `→ 12 files (foundational); rg "profiling.models" backend -l` | All Pydantic profile/evidence models; `PROFILE_VERSION` bump invalidates cached profiles. |
| `types.py` | Typing | `→ charts/render.py, probes/engine.py` | Semantic type inference (multi-signal id detection, dirty-numeric normalisation, datetime probing) and cast application. |
| `quality.py` | Quality | — | Robust range (MAD), suspected sentinels, extreme-value share per numeric column. |
| `pairwise.py` | Evidence L1 | — | Polars-native batched Pearson matrix with pair counts (no numpy). |
| `evidence.py` | Evidence L1 | `→ charts/rules.py, probes/*, llm/*` | Layer-1 evidence: cat×num eta², time effects, Spearman, interactions, slope heterogeneity, derived-column identities (stage 13), near-duplicate groups (stage 14). |
| `layer2.py` | Evidence L2 | `→ probes/engine.py` | Layer-2 structural summaries, LLM-only (stage 17.1). → see `layer2.py.analysis.md` |
| `profiler.py` | Orchestration | `← serialization.py \| → datasets/router.py, main.py` | `profile_dataset` (sampling >100k rows, calls types/quality/pairwise/evidence/layer2) and `ProfileService` cache. |

## File Dependencies

| File | Imports (in-dir) | Exposed To (in-dir) |
|---|---|---|
| `models.py` | — | all |
| `types.py` | `models` | `profiler`, (external) |
| `quality.py` | `models` | `profiler` |
| `pairwise.py` | — | `evidence`, `profiler` |
| `evidence.py` | `models`, `pairwise` | `layer2`, `profiler` |
| `layer2.py` | `models`, `evidence` | `profiler` |
| `profiler.py` | all above | — (entry point) |
