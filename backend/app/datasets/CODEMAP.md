---
mode: learning
generated_at: 2026-09-15
---

> HTTP API: upload/list/preview/profile/recommendations/render endpoints, file loading, and the uuid+Parquet dataset store.

## Task Guide

| Task | Domain | Target | Also Check |
|---|---|---|---|
| Understand the REST surface and response/error shapes | API | `router.py` | `.claude/CONTRACTS.md` (API section) |
| Understand how CSV/XLSX/Parquet are read | Loader | `loader.py` | `../../tests/test_loader.py`, `tests/data/` |
| Understand dataset persistence (no DB, Parquet temp + uuid) | Store | `store.py` | DECISIONS D5 |

## Key Exports

| Symbol | Source | Line |
|---|---|---|
| `router` (prefix `/api`) | `router.py` | L:14 |
| `DatasetStore`, `DatasetNotFoundError` | `store.py` | L:17, L:13 |
| `load_dataframe`, `LoaderError`, `SUPPORTED_EXTENSIONS` | `loader.py` | L:13, L:9 |

## Files

| File | Domain | Deps | Function |
|---|---|---|---|
| `router.py` | API | `← charts/render.py, charts/spec.py, config.py, serialization.py \| → main.py` | Endpoints: `GET /health`, `POST/GET /datasets`, `GET /datasets/{id}`, `/preview`, `/profile`, `/recommendations?llm=`, `POST /charts/render`; unified 422 `{detail:{errors:[…]}}`. |
| `loader.py` | Loader | — | Extension-dispatched Polars/Pandas readers; `LoaderError` for corrupt/empty files. |
| `store.py` | Store | — | In-memory index + Parquet file per dataset id under the data dir. |

## File Dependencies

| File | Imports (in-dir) | Exposed To (in-dir) |
|---|---|---|
| `loader.py` | — | `router` |
| `store.py` | — | `router` |
| `router.py` | `loader`, `store` | — |
