---
mode: learning
generated_at: 2026-09-15
---

> Playwright smoke harness against a real backend and real `dataset/` files (no mocks); also records the chart fixtures used by vitest.

## Task Guide

| Task | Domain | Target | Also Check |
|---|---|---|---|
| Run or extend the e2e smoke steps | E2E | `run.mjs`, `README.md` | `../../dataset/syn/` (input files) |
| Re-record `src/charts/__fixtures__` after a backend render change | E2E | `capture-fixtures.mjs` | `../src/charts/__fixtures__/index.json` |
| Headless Chromium setup without root on this host | E2E | `README.md` (Requirements) | — |

## Files

| File | Domain | Function |
|---|---|---|
| `run.mjs` | E2E | Upload → overview → insights (Key Findings expand / keyboard / View chart via `checkKeyFindings`) → explore → workspace → month-order bar steps; screenshots per tag; stdout JSON summary `ok: true|false`. |
| `capture-fixtures.mjs` | E2E | Calls `/api/charts/render` for each fixture spec and writes JSON; `--only a,b` records a subset and merges it into `index.json` (stage 19.1). |
| `README.md` | E2E | Requirements and coverage list. |
| `data/sparse_groups.csv` | E2E | Committed input for null-gap fixtures. |
| `data/make_stage191.py` → `data/spread_groups.csv`, `data/cycles.csv`, `data/dense_xy.csv` | E2E | Stage 19.1 deterministic inputs (same mean / different dispersion + ordered level; hour / month cycles; dense XY) behind the raincloud / ridgeline / box / natural-order bar + line / polar_line / rose / hexbin fixtures and `backend/tests/test_fixture_regression.py`. |
| `data/seasonal_groups.csv` | E2E | Committed input for `line_grouped_gap` (24 monthly rows; each season occupies only part of each year). |
| `data/make_stage20b.py` → `data/flows.csv` | E2E | Stage 20b deterministic input (600 shipments: region x channel revenue with a mix that shifts by region, and a declared from_hub/to_hub pair with self-loops) behind the stacked_bar / sankey fixtures and `backend/tests/test_fixture_regression.py`. |
