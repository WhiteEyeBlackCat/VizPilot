---
mode: learning
generated_at: 2026-09-15
---

> pytest suite (535 tests on main): one module per backend concern plus HTTP tests via TestClient; `data/` holds small loader fixtures (run `data/make_fixtures.py` to regenerate).

## Task Guide

| Task | Domain | Target | Also Check |
|---|---|---|---|
| Find tests for a module | Tests | `test_<module>.py` (same stem as `app/*/<module>.py`) | `conftest.py` (fixtures, fake provider) |
| End-to-end recommendation quality assertions on synthetic frames | Tests | `test_recommendation_quality.py` | `test_rules.py` |
| HTTP contract (upload, 422 shapes, render) | Tests | `test_upload.py`, `test_render.py`, `test_spec.py` | `../app/datasets/router.py` |
| LLM merge / insight state machine | Tests | `test_llm_service.py`, `test_llm_provider.py` | `conftest.py` |

## Files

| File | Domain | Function |
|---|---|---|
| `conftest.py` | Tests | Shared fixtures (app client, frames, fake LLM provider). |
| `test_types.py` (48), `test_profiler.py` (45), `test_quality.py` (14) | Tests | Typing, profile, quality. |
| `test_evidence.py` (43), `test_layer2.py` (31), `test_probes.py` (25) | Tests | Evidence L1/L2, probe engine. |
| `test_rules.py` (29), `test_confidence.py` (25), `test_spec.py` (34), `test_recommendation_quality.py` (17) | Tests | Recommendation engine. |
| `test_render.py` (35), `test_upload.py` (22), `test_loader.py` (12) | Tests | Render + API + loader. |
| `test_llm_service.py` (23), `test_llm_provider.py` (16) | Tests | LLM layer. |
| `data/` | Tests | csv/xlsx/parquet fixtures incl. corrupt/empty/latin1/nan_inf. |

Targeted run: `cd backend && .venv/bin/pytest tests/test_rules.py -q`. Full: `.venv/bin/pytest tests -q` (~1–2 min).
