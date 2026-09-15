# CURRENT_STATE.md — VizPilot (update after each major stage)

Updated: 2026-09-16 · main `aa1696c` · detailed stage notes in `.claude/docs/stages/` (local, git-ignored)

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

## Architecture state
Pipeline as in `.claude/ARCHITECTURE.md`: LLM #1 hypotheses → hallucination gate → coverage against evidence L1/L2 → typed probes (≤5, cached, n ≥ 30) for uncovered claims → LLM #2 wording only when a probe validated or ≥2 validated → wording gate (only backend numbers / change-point dates quotable; else template or neutral sentence) → merge with rules. `use_llm=False` takes the pre-17.3 path (byte-identical on 16 datasets). Frontend is the 16.x shell; it does not yet render `insights[].validation`, `effect`, `why_it_matters`, or the `debug` trace.

## In progress
- Nothing mid-flight. Worktrees for 17.3/17.4 were removed after merge.

## Recent decisions
- Definitional / near-duplicate / trivial relationships are not findings; 0 insights is legitimate.
- LLM does hypothesis + wording only; every claim gated by backend statistics; LLM-stated numbers ignored; LLM #1 `reason` and LLM #2 text pass the same wording gate.
- `?llm=false` output must stay byte-identical across LLM-side changes (16 dataset snapshots incl. `hour.parquet`, `noise.csv`, `planted_probe.csv`).
- Multi-agent work uses separate git worktrees; no Claude co-author trailer in commits; independent verifier subagent for statistics/LLM-gate changes.

## Known blockers / risks (carry-forward from verifier round 4, non-blocking)
- main is 9 commits ahead of `origin/main` (unpushed since 16.3). Push needs the user's credentials (VS Code askpass).
- Column names containing digits (`pm25`) make the number gate reject the whole sentence → neutral wording and `why_it_matters = null` more often (air_quality).
- Group labels can pass the number gate by coincidence (`n_groups`, per-group q25); `value*100` percentage rule is wide; natural-language mappings (month names, weekday names) are unchecked.
- LLM #2 prompt shows column mean/min/max and bin ranges the LLM may not quote → its wording gets rejected and the template is used; either strip them from the prompt or whitelist them.
- Real qwen2.5:14b has produced probe fail/rejected but never a probe pass end to end; the probe → LLM #2 chain is proven by FakeProvider.
- `_neutral()` capitalises column names (`Pm25 varies by station`); `?debug=1` has no environment switch; digit-leading snake_case tokens escape the unknown-column gate.
- Running backend on 8100 may be stale; restart after backend changes. Leading zeros lost on CSV load; `pct` mixed scales; no histogram range opt-out.
- pytest rewrites `backend/tests/data/basic.xlsx`; `git checkout -- backend/tests/data/basic.xlsx` before committing.

## Test status (main `aa1696c`)
- backend: 594 pytest passed (0 skipped when `dataset/*.csv` are generated and `frontend/dist` exists).
- frontend: tsc 0 errors, vitest 39, build green (frontend untouched by 17.3/17.4).

## Next planned task
1. Wire probe/insight `validation`, `supported`, `effect`, `why_it_matters` into the frontend Insights page (optional fields only; `types.ts` mirrors backend).
2. Optional quality follow-ups from the verifier list: mask known column-name tokens before the number scan; trim un-quotable numbers from the LLM #2 prompt; lowercase-preserving `_neutral()`.
3. Push main to origin (user).
