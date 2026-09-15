---
mode: learning
generated_at: 2026-09-15
---

> FastAPI application package: app factory wiring store/profile/recommendation services, settings, and the five domain subpackages.

## Task Guide

| Task | Domain | Target | Also Check |
|---|---|---|---|
| Understand app startup, state wiring, SPA serving | App | `main.py` | `config.py` |
| Understand env settings (data dir, LLM base_url/model/enabled) | App | `config.py` | `llm/provider.py` |
| Understand JSON-safe conversion of Polars values | App | `serialization.py` | — |
| Follow a request end to end (upload → profile → recs → render) | API | `datasets/CODEMAP.md` | `profiling/`, `llm/`, `charts/` |

## Subdirectories

| Dir | Domain | Depends On | Purpose |
|---|---|---|---|
| `datasets/` | API | charts, config, serialization | Routes, loader, store. |
| `profiling/` | Profiling | serialization | Typing, stats, quality, evidence L1/L2, `DatasetProfile`. |
| `charts/` | Recommendation / Render | profiling | ChartSpec, rules, confidence, aggregation. |
| `llm/` | LLM | charts, probes, profiling | Two-stage hypothesis → coverage → probe → wording workflow; merge with rules. |
| `probes/` | Probes | charts, profiling | Targeted validation tests (17.2), called by `llm/service.py`. |

## Key Exports

| Symbol | Source | Line |
|---|---|---|
| `create_app` (factory; `app.state.{store,profiles,recommendations,frames}`) | `main.py` | L:18 |
| `Settings` | `config.py` | L:8 |
| `df_to_records`, `jsonify_scalar` | `serialization.py` | L:15, L:19 |

## Files

| File | Domain | Function |
|---|---|---|
| `main.py` | App | Builds Settings, DatasetStore, ProfileService, RecommendationService (Disabled/OpenAICompat provider), CastedFrameCache; mounts router and built SPA. |
| `config.py` | App | `Settings` from env (`VIZPILOT_*`). |
| `serialization.py` | App | Row/scalar JSON coercion (NaN/inf/datetime). |
