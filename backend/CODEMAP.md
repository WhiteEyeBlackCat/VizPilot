---
mode: learning
generated_at: 2026-09-15
---

> Python backend (FastAPI + Polars + Pydantic): `app/` package and `tests/`; venv at `backend/.venv`, serves on port 8100.

## Task Guide

| Task | Domain | Target | Also Check |
|---|---|---|---|
| Any backend code question | Backend | `app/CODEMAP.md` | — |
| Run / find tests | Tests | `tests/CODEMAP.md` | — |
| Dependencies / run command | Backend | `requirements.txt`, `../README.md` | — |

## Subdirectories

| Dir | Domain | Depends On | Purpose |
|---|---|---|---|
| `app/` | Backend | polars, pandas, duckdb, fastapi, pydantic, httpx | Application package. |
| `tests/` | Tests | app | pytest suite + fixtures. |

## Files

| File | Domain | Function |
|---|---|---|
| `requirements.txt` | Backend | Pinned runtime + test dependencies. |
| `pyproject.toml` | Backend | pytest/ruff config. |
