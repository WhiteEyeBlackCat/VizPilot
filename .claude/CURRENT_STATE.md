# CURRENT_STATE.md — VizPilot (update after each major stage)

Updated: 2026-09-16 · main = stage 19 merge (see git log) · detailed stage notes in `.claude/docs/stages/` (local, git-ignored)

## Completed
- Stages 1–8: upload → profiler → rule engine → ECharts render → LLM → manual builder; evidence L1 + tier system + insight/chart contract.
- 9/9b: unified confidence layer (D11), dirty-numeric/id/sentinel handling, per-dataset profile lock.
- 10–12: Playwright e2e harness, shadcn/ui, Plotly → ECharts (bundle 1.03 MB).
- 13/14: derived-column detection (definitional relationships demoted), near-duplicate suppression, per-y caps.
- 16.1–16.3: dark tokens, sidebar + hash routes + shared preview panel + save semantics, overview page, accessibility.
- 17.1/17.1b: Evidence layer 2 (`PROFILE_VERSION` 10) — PASS.
- 17.2: probe engine (`backend/app/probes/`) — merged `f42f362`.
- **17.3 two-stage LLM workflow** — merged `7516e0d` after verifier round 4 PASS (rounds 1–3 FAIL, each fixed: chart-only leniency, LLM#2 number/column gate, statement number/date gate, directed nonlinear coverage, LLM#1 `reason` gate, literal ISO-date check, gated caption on new LLM charts). `PROFILE_VERSION` 11; `llm_timeout_seconds` 30 → 60; `?debug=1` adds the workflow trace. Reports: `stage17.3.md`, `stage17.3-verify3.md`, `stage17.3-verify4.md`.
- **17.4 insight benchmark** — merged `aa1696c`: `backend/tests/test_insight_benchmark.py`, eight planted scenarios (sales formula, hour_like definitions, Pearson-weak U shape, planted interaction, noise, tiny, probe failure, hallucination) through the full pipeline with FakeProvider. Notes: `stage17.4.md`.
- Docs refreshed for the merge: `backend/app/llm/CODEMAP.md` (two-stage workflow), `ARCHITECTURE.md` pipeline, `CONTRACTS.md` probes producer + `debug` field, tests CODEMAP.
- **18 frontend UI** — verifier PASS (0 blocking, 4 non-blocking fixed): Key Findings headline-first rows (`InsightList.tsx`, whole row = `aria-expanded` button, evidence `dl`, View chart), Insight → analysis panel (`Preview.insight`), "More charts" section with simplified cards (no `x=/y=` on the face; spec tooltip), PreviewPanel = analysis detail panel (chart ≈ 60 % + Why this matters / Supporting evidence / Chart spec / actions), ECharts theme refinement, natural category order for bar/box (`charts/echarts/order.ts` + `[data-order-note]`). Only backend change: LLM #2 `insights[].headline` (optional, gated, null-safe). Notes: `stage18.md`, `stage18-verify.md`.

- **19 Visualization Intent → Chart Selection → Renderer** — verifier round 2 MERGE (round 1 DO NOT MERGE: violin/ridgeline axis extent, duplicate hr finding; both fixed): `charts/roles.py` (semantic roles: nominal / ordinal / cyclic / datetime / numeric / boolean, deterministic from the profile), `charts/intent.py` (11 intents, guards, `classify_intent` → `select_chart` → preferred + alternatives), `ChartSpec` 16 types + `sort`, 10 new renderers (violin, raincloud, ridgeline, dot_interval, lollipop, dumbbell, slopegraph, hexbin, polar_line, rose) backend + ECharts, matrix heatmap, line over ordered categories, natural category order in the backend, one recommendation per finding (`Recommendation.intent/alternatives`), LLM chart optional (`select_for_spec` re-selects suggested types). Notes: `stage19.md`.

- **hotfix: grouped temporal gaps** — a grouped line over time no longer joins a group's points across a stretch with no observations (`temp over dteday by season`): frontend-only, `option.ts` "grouped temporal gaps" inserts an in-gap null point (`connectNulls: false`) when a step exceeds 3 × the group's normal interval (bucket unit, else median step with a regularity guard); ungrouped lines unchanged; real fixture `line_grouped_gap` (`e2e/data/seasonal_groups.csv`), `option.gaps.test.ts` (7).

## Architecture state
Pipeline as in `.claude/ARCHITECTURE.md`. Charts (stage 19): evidence + roles → intent → eligible family → preferred `ChartSpec` (+ alternatives) → `RenderResult` → ECharts adapter; the LLM never sees the chart inventory. LLM side: LLM #1 hypotheses → hallucination gate → coverage against evidence L1/L2 → typed probes (≤5, cached, n ≥ 30) for uncovered claims → LLM #2 wording only when a probe validated or ≥2 validated → wording gate (only backend numbers / change-point dates quotable; else template or neutral sentence) → merge with rules. `use_llm=False` takes the pre-17.3 path (byte-identical on 16 datasets). Frontend (stage 18) renders `insights[].headline/text/why_it_matters/effect/validation` in Key Findings and the analysis panel; the `debug` trace is not shown.

## In progress
- Nothing mid-flight; stage 19 merged.

## Recent decisions
- Definitional / near-duplicate / trivial relationships are not findings; 0 insights is legitimate.
- LLM does hypothesis + wording only; every claim gated by backend statistics; LLM-stated numbers ignored; LLM #1 `reason` and LLM #2 text pass the same wording gate.
- `?llm=false` output must stay byte-identical across LLM-side changes (16 dataset snapshots incl. `hour.parquet`, `noise.csv`, `planted_probe.csv`). Stage 19 changed the rules on purpose: new snapshot md5 `6cc5e2a27cbfd8842d2163b8e6614933` (stage 18: `d7e09d2b1820a5238fe94101286ad771`).
- Stage 19: chart types are never core/advanced in product logic; every type has a named guard in `intent.py`; only the `ranking` intent sorts by value; a nominal column is never a rose / polar; composition / flow intents exist as contract only (no generator, no renderer).
- Multi-agent work uses separate git worktrees; no Claude co-author trailer in commits; independent verifier subagent for statistics/LLM-gate changes.

## Known blockers / risks (carry-forward from verifier round 4, non-blocking)
- main is unpushed since 16.3 (17.x + 18 commits). Push needs the user's credentials (VS Code askpass).
- Stage 19 limitations (verifier): family-based diversity cap can give 5–6 charts of one raw type (hour.parquet: 3 cyclic + 3 trend lines); `distribution_by_group` spread evidence is gated by the L2 group-summary η² threshold (a spread-only difference with equal means stays a bar); `rose` is Explore-only in practice (rules offer it only for sum/count cycles); hexbin rows show hairline gaps (pixel radius vs data row pitch); `composition` / `flow` intents have no generator or renderer yet.
- Stage 18 non-blocking (verifier): recommendation cards are `div onClick` (keyboard reaches only the inner buttons; pre-existing); bar categories are still value-ranked by the backend — the frontend re-orders natural sequences for display (stage 19 moves ordering into the spec via intents).
- Column names containing digits (`pm25`) make the number gate reject the whole sentence → neutral wording and `why_it_matters = null` more often (air_quality).
- Group labels can pass the number gate by coincidence (`n_groups`, per-group q25); `value*100` percentage rule is wide; natural-language mappings (month names, weekday names) are unchecked.
- LLM #2 prompt shows column mean/min/max and bin ranges the LLM may not quote → its wording gets rejected and the template is used; either strip them from the prompt or whitelist them.
- Real qwen2.5:14b has produced probe fail/rejected but never a probe pass end to end; the probe → LLM #2 chain is proven by FakeProvider.
- `_neutral()` capitalises column names (`Pm25 varies by station`); `?debug=1` has no environment switch; digit-leading snake_case tokens escape the unknown-column gate.
- Running backend on 8100 may be stale; restart after backend changes. Leading zeros lost on CSV load; `pct` mixed scales; no histogram range opt-out.
- pytest rewrites `backend/tests/data/basic.xlsx`; `git checkout -- backend/tests/data/basic.xlsx` before committing.

## Test status (stage 19, main)
- backend: 649 pytest passed (new: `test_roles` 6, `test_intent` 22, `test_intent_pipeline` 13, `test_render_stage19` 13); rules-only snapshot md5 `6cc5e2a27cbfd8842d2163b8e6614933`.
- frontend: tsc 0 errors, vitest 64 (`option.stage19.test.ts` 14), build green; e2e `ok: true` (60 steps incl. cyclic intent → polar alternative, rose guard) against a throw-away LLM backend on 8118; verifier reports `stage19-verify1.md` / `stage19-verify2.md`.

## Next planned task
1. Stage 19 follow-ups: composition / flow renderers (Sankey) once a dataset shape calls for it; Explore UI hints for cyclic / ordered roles (today the backend 422 explains); record real-backend fixtures for the new chart types (`capture-fixtures.mjs`).
2. Optional LLM quality follow-ups: mask column-name tokens before the number scan; trim un-quotable numbers from the LLM #2 prompt; lowercase-preserving `_neutral()`.
3. Push main to origin (user).
