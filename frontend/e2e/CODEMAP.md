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
| `run.mjs` | E2E | Upload → overview → insights → explore → workspace steps; screenshots per tag; stdout JSON summary `ok: true|false`. |
| `capture-fixtures.mjs` | E2E | Calls `/api/charts/render` for each fixture spec and writes JSON. |
| `README.md` | E2E | Requirements and coverage list. |
| `data/sparse_groups.csv` | E2E | Committed input for null-gap fixtures. |
