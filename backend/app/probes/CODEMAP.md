---
mode: learning
generated_at: 2026-09-15
---

> Targeted validation probe engine (stage 17.2): runs a bounded set of typed statistical tests requested by the LLM workflow, returns structured verdicts. Not yet wired to an HTTP route or the service on main.

## Task Guide

| Task | Domain | Target | Also Check |
|---|---|---|---|
| Understand the probe request/result contract and validation rules | Probe Schema | `schemas.py` | `../charts/confidence.py` (`Confidence`) |
| Understand how each probe type is computed and thresholded | Probe Engine | `engine.py` (`run_probes`, `THRESHOLDS`) | `../profiling/layer2.py`, `../profiling/evidence.py` |
| Understand caching / de-duplication of probes | Probe Engine | `engine.py` (`ProbeCache`, `ProbeRequest.key`) | — |

## Key Exports

| Symbol | Source | Line |
|---|---|---|
| `run_probes`, `prepare_frame`, `ProbeCache` | `engine.py` | L:137, L:128, L:93 |
| `ProbeRequest`, `ProbeResult`, `ProbeRejected`, `ProbeOutcome`, `validate_request` | `schemas.py` | L:49, L:70, L:86, L:93, L:99 |

## Files

| File | Domain | Deps | Function |
|---|---|---|---|
| `engine.py` | Probe Engine | `← charts/confidence.py, charts/rules.py, charts/spec.py, profiling/evidence.py, profiling/layer2.py, profiling/models.py, profiling/types.py` | Probe implementations (group difference, correlation, nonlinear, time pattern, interaction, distribution, …), effect sizes with pass/weak thresholds, ≤N cap, error containment. |
| `schemas.py` | Probe Schema | `← charts/confidence.py, charts/spec.py, profiling/*` | `extra="forbid"` request (no code/sql smuggling), result with structured evidence only, role/column validation against the profile. |

## File Dependencies

| File | Imports (in-dir) | Exposed To (in-dir) |
|---|---|---|
| `schemas.py` | — | `engine` |
| `engine.py` | `schemas` | — |
