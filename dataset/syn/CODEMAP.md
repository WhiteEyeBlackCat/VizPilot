---
mode: learning
generated_at: 2026-09-15
---

> Generators for the synthetic benchmark datasets (one script → one csv/parquet in `dataset/`, git-ignored). Each targets one profiling/recommendation behaviour.

## Task Guide

| Task | Domain | Target | Also Check |
|---|---|---|---|
| Regenerate all datasets | Synthetic Data | run each `*.py` with `backend/.venv/bin/python` | `../../frontend/e2e/README.md` |
| Add a planted-pattern dataset for a benchmark | Synthetic Data | copy `hour_like.py` (polars) or `sales_basic.py` | backend `tests/test_recommendation_quality.py` |

## Files

| File | Domain | Function |
|---|---|---|
| `sales_basic.py` | Synthetic Data | `sales = unit_price × quantity × (1−discount)`; region has zero effect (derived/definitional benchmark). |
| `hour_like.py` | Synthetic Data | Bike-sharing-like; temp/atemp near-duplicates, hourly multimodality. |
| `air_qulity.py`, `wide_sensors.py`, `large_dataset.py` | Synthetic Data | Time series, 60+ column wide table (perf), >100k rows (sampling). |
| `customers_high_cardinality.py`, `product.py`, `dirty_types.py`, `missing_values.py`, `outliers.py` | Synthetic Data | Id/high-cardinality, nominal codes, dirty numerics, missingness, sentinels/extremes. |
| `single_numeric.py`, `text_only.py`, `tiny_dataset.py` | Synthetic Data | Degenerate shapes (one column, text only, n<30). |
