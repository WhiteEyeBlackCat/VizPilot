# CURRENT_STATE.md — VizPilot (update after each major stage)

Updated: 2026-09-16 · main = stage 18 merge (see git log) · detailed stage notes in `.claude/docs/stages/` (local, git-ignored)

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

## Architecture state
Pipeline as in `.claude/ARCHITECTURE.md`: LLM #1 hypotheses → hallucination gate → coverage against evidence L1/L2 → typed probes (≤5, cached, n ≥ 30) for uncovered claims → LLM #2 wording only when a probe validated or ≥2 validated → wording gate (only backend numbers / change-point dates quotable; else template or neutral sentence) → merge with rules. `use_llm=False` takes the pre-17.3 path (byte-identical on 16 datasets). Frontend (stage 18) renders `insights[].headline/text/why_it_matters/effect/validation` in Key Findings and the analysis panel; the `debug` trace is not shown.

## In progress
- Nothing mid-flight. Worktrees for 17.3/17.4 were removed after merge.

## Recent decisions
- Definitional / near-duplicate / trivial relationships are not findings; 0 insights is legitimate.
- LLM does hypothesis + wording only; every claim gated by backend statistics; LLM-stated numbers ignored; LLM #1 `reason` and LLM #2 text pass the same wording gate.
- `?llm=false` output must stay byte-identical across LLM-side changes (16 dataset snapshots incl. `hour.parquet`, `noise.csv`, `planted_probe.csv`).
- Multi-agent work uses separate git worktrees; no Claude co-author trailer in commits; independent verifier subagent for statistics/LLM-gate changes.

## Known blockers / risks (carry-forward from verifier round 4, non-blocking)
- main is unpushed since 16.3 (17.x + 18 commits). Push needs the user's credentials (VS Code askpass).
- Stage 18 non-blocking (verifier): recommendation cards are `div onClick` (keyboard reaches only the inner buttons; pre-existing); bar categories are still value-ranked by the backend — the frontend re-orders natural sequences for display (stage 19 moves ordering into the spec via intents).
- Column names containing digits (`pm25`) make the number gate reject the whole sentence → neutral wording and `why_it_matters = null` more often (air_quality).
- Group labels can pass the number gate by coincidence (`n_groups`, per-group q25); `value*100` percentage rule is wide; natural-language mappings (month names, weekday names) are unchecked.
- LLM #2 prompt shows column mean/min/max and bin ranges the LLM may not quote → its wording gets rejected and the template is used; either strip them from the prompt or whitelist them.
- Real qwen2.5:14b has produced probe fail/rejected but never a probe pass end to end; the probe → LLM #2 chain is proven by FakeProvider.
- `_neutral()` capitalises column names (`Pm25 varies by station`); `?debug=1` has no environment switch; digit-leading snake_case tokens escape the unknown-column gate.
- Running backend on 8100 may be stale; restart after backend changes. Leading zeros lost on CSV load; `pct` mixed scales; no histogram range opt-out.
- pytest rewrites `backend/tests/data/basic.xlsx`; `git checkout -- backend/tests/data/basic.xlsx` before committing.

## Test status (stage 18)
- backend: 595 pytest passed; rules-only snapshot md5 `d7e09d2b1820a5238fe94101286ad771` unchanged (16 datasets).
- frontend: tsc 0 errors, vitest 50 (order 11), build green; e2e `ok: true` (56 steps, incl. Key Findings expand / keyboard / View chart, month-order bar) against a throw-away LLM backend on 8118.

## Next planned task
1. **Stage 19** (in progress): Visualization Intent → chart selection → renderer (`charts/roles.py`, `charts/intent.py`, new chart types, `ChartSpec.sort`, preferred + alternatives, LLM chart optional). Design: `.claude/docs/stages/stage19.md`.
2. Optional LLM quality follow-ups: mask column-name tokens before the number scan; trim un-quotable numbers from the LLM #2 prompt; lowercase-preserving `_neutral()`.
3. Push main to origin (user).
