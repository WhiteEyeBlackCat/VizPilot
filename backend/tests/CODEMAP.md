---
mode: learning
generated_at: 2026-09-15
---

> pytest suite (690 tests after stage 19.1): one module per backend concern plus HTTP tests via TestClient; `data/` holds small loader fixtures (run `data/make_fixtures.py` to regenerate).

## Task Guide

| Task | Domain | Target | Also Check |
|---|---|---|---|
| Find tests for a module | Tests | `test_<module>.py` (same stem as `app/*/<module>.py`) | `conftest.py` (fixtures, fake provider) |
| End-to-end recommendation quality assertions on synthetic frames | Tests | `test_recommendation_quality.py` | `test_rules.py` |
| HTTP contract (upload, 422 shapes, render) | Tests | `test_upload.py`, `test_render.py`, `test_spec.py` | `../app/datasets/router.py` |
| LLM merge / insight state machine | Tests | `test_llm_service.py`, `test_llm_provider.py` | `conftest.py` |
| Two-stage workflow: gate, coverage, probes, LLM #2 trigger, wording gate | Tests | `test_llm_workflow.py` (`TwoStageFake`, `_planted`, `_h`) | `test_probes.py` |
| Planted end-to-end scenarios with known right answers (sales formula, hour_like, noise, tiny, hallucination) | Tests | `test_insight_benchmark.py` | `test_llm_workflow.py`, `test_recommendation_quality.py` |
| Semantic roles, intent classifier, chart eligibility guards (stage 19) | Tests | `test_roles.py`, `test_intent.py` | `test_intent_pipeline.py` (planted month / region / hexbin / dumbbell / ridgeline / tiny / noise cases) |
| New renderers and category order (density, dot interval, two-state, hexbin, matrix, polar) | Tests | `test_render_stage19.py` | `test_render.py` |
| Composition / flow payloads (stacked bar totals + 100 % shares, sankey nodes / links / self-loops / truncation) | Tests | `test_render_stage20.py` | `test_spec.py`, `test_intent.py`, `test_intent_pipeline.py` |
| Dispersion evidence (cases A–D: means vs spread, small samples) and the distribution family it triggers (stage 19.1) | Tests | `test_dispersion.py` | `../app/profiling/evidence.py` (`DISPERSION_*`) |
| Dispersion calibration: seeded null false-positive / power / mean-shift / outlier grids, F critical values (stage 19.2) | Tests | `test_dispersion_calibration.py` | `test_dispersion.py` (cases E small groups, F definitional / bucket) |
| Month / weekday names as cycles from the full label set, `/profile` roles for the Explore builder (stage 19.2) | Tests | `test_roles.py`, `test_stage191_explore.py` | `../app/charts/roles.py`, `../app/profiling/profiler.py` (`categories`) |
| Recommendation diversity: same-question / same-raw-type diminishing returns, hour_like line count (stage 19.1) | Tests | `test_diversity.py` | `test_rules.py` |
| Explore / LLM metadata: prompt roles + spread lines, `/profile` `column_roles`, LLM rose kept or dropped (stage 19.1) | Tests | `test_stage191_explore.py` | `../app/llm/prompts.py`, `../app/datasets/router.py` |
| Real-backend render fixtures re-rendered and compared with float tolerance (stage 19.1) | Tests | `test_fixture_regression.py` | `../../frontend/e2e/capture-fixtures.mjs`, `../../frontend/e2e/data/` |

## Files

| File | Domain | Function |
|---|---|---|
| `conftest.py` | Tests | Shared fixtures (app client, frames, fake LLM provider). |
| `test_types.py` (48), `test_profiler.py` (45), `test_quality.py` (14) | Tests | Typing, profile, quality. |
| `test_evidence.py` (43), `test_layer2.py` (31), `test_probes.py` (25) | Tests | Evidence L1/L2, probe engine. |
| `test_rules.py` (29), `test_confidence.py` (25), `test_spec.py` (69), `test_recommendation_quality.py` (17), `test_roles.py` (13), `test_intent.py` (29), `test_intent_pipeline.py` (18) | Tests | Recommendation engine, roles, intents. |
| `test_render.py` (35), `test_render_stage19.py` (13), `test_render_stage20.py` (10), `test_upload.py` (22), `test_loader.py` (12) | Tests | Render + API + loader. |
| `test_llm_service.py` (26), `test_llm_provider.py` (20), `test_llm_workflow.py` (36), `test_insight_benchmark.py` (8) | Tests | LLM layer: merge, prompts/provider, two-stage workflow, insight benchmark. |
| `test_dispersion.py` (29), `test_dispersion_calibration.py` (30), `test_diversity.py` (6), `test_stage191_explore.py` (10), `test_fixture_regression.py` (21) | Tests | Stage 19.1 / 19.2: dispersion evidence + chart family (+ small groups, definitional / bucket), seeded calibration grids, diminishing-returns diversity (incl. `hour_like.csv`), Explore / LLM metadata + LLM rose + month names, real-backend fixture regression (inputs in `frontend/e2e/data/`). |
| `data/` | Tests | csv/xlsx/parquet fixtures incl. corrupt/empty/latin1/nan_inf. |

Targeted run: `cd backend && .venv/bin/pytest tests/test_rules.py -q`. Full: `.venv/bin/pytest tests -q` (~1–2 min).
