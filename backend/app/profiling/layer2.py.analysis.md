---
source: layer2.py
lines: 1008
generated_at: 2026-09-15
---

> Evidence layer 2 (stage 17.1): deterministic structural summaries (distribution shape, group summaries, conditional relationships, group time patterns, nonlinear curves, change points, subgroup anomalies) written into `DatasetProfile.evidence.layer2`; consumed by the LLM prompt and the probe engine, never by the rules engine.

## Feature Index

| Intent | Lines | Notes |
|---|---|---|
| Tune any cap / threshold (all numbers live here) | 58–124 | constants only; each has an inline comment |
| Entry point and ordering of the seven summaries | 128–156 | `compute_layer2(df, profile, layer1)` |
| Shared helpers (finite filter, quantiles, definitional pairs, median) | 157–189 | `_definitional_pairs` excludes stage-13 derived pairs |
| Distribution shape (skew label, bimodality coefficient, peak count) | 190–320 | uses `DIST_BINS`, `BIMODALITY_THRESHOLD` |
| Group summaries (cat×num, top by eta²) and pooled SD | 321–432 | `GroupStat` per group; `MAX_GROUPS_LISTED` |
| Subgroup anomalies (robust z of group means) | 433–472 | depends on `_group_stats`, `_pooled_std` |
| Conditional relationships (correlation within groups) | 473–582 | `CONDITIONAL_TOP_GROUPS/NUMS` |
| Group time patterns (per-group bucketed series, thinning) | 583–693 | `_one_group_time_pattern`, `_thin_indices` |
| Nonlinear signals (equal-frequency bins, binned eta, curve shape, quadratic fit) | 694–895 | `_curve_shape` labels: flat/linear/monotone/peak/valley… |
| Change points (best single split on bucket means) | 896–1008 | `_best_split`, `_choose_change_bucket` |

## Symbols

| Symbol | Type | Line |
|---|---|---|
| `MIN_ROWS` … `MAD_SCALE` | constants | 60–122 |
| `compute_layer2` | function | 128 |

## Logical Sections

| Lines | Content |
|---|---|
| 1–57 | module docstring, imports |
| 58–124 | caps and thresholds |
| 128–156 | `compute_layer2` orchestrator |
| 157–189 | helpers |
| 190–320 | distributions |
| 321–472 | group summaries + anomalies |
| 473–582 | conditional relationships |
| 583–693 | group time patterns |
| 694–895 | nonlinear signals |
| 896–1008 | change points |
