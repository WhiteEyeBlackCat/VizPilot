# CLAUDE.md — VizPilot agent rules

Local AI data-exploration assistant (FastAPI + Polars backend, React + ECharts frontend, optional local LLM). Read `.claude/CURRENT_STATE.md` first, then `CODEMAP.md`. `.claude/ARCHITECTURE.md` / `.claude/CONTRACTS.md` only when the task touches data flow or a shared shape.

## Product principles (do not violate)
- Raw dataset rows never go to the LLM; only profile metadata (+ ≤5 sample rows). No cloud providers.
- The app must work fully with the LLM disabled; rules engine is the floor; LLM failures fall back silently to rules.
- LLM output is structured JSON validated by Pydantic (`validate_spec` is the gate); the LLM never produces code/SQL/expressions.
- Statistics and verification live in the backend; the LLM only proposes hypotheses and wording.
- Definitional / near-duplicate / trivial relationships are not findings. Zero insights is a legitimate answer; say so honestly.
- No pie charts; no bar charts on high-cardinality categoricals.

## Architecture boundaries
- `ChartSpec`, `RenderResult`, `render.py`, and the confidence formula (DECISIONS D11) change only with an explicit user decision.
- Response fields are additive: never remove or rename an existing API field; new fields are optional.
- Bump `PROFILE_VERSION` whenever `DatasetProfile`/`Evidence` shape or statistics change (it invalidates cached profiles).
- `frontend/src/types.ts` mirrors backend models; change both or neither.
- Rules-only path (`?llm=false`) output must stay byte-identical across LLM-side changes.

## Conventions
- Backend: Python 3, Polars-native (avoid per-row Python and numpy in hot paths), Pydantic models, pytest in `backend/tests/` (`test_<module>.py`). Run with `backend/.venv/bin/pytest`.
- Frontend: TypeScript strict, Tailwind 3.4 + shadcn/ui, ECharts via `charts/echarts/` only. `npm run build` = tsc + vite. vitest in `src/charts/echarts/*.test.ts`. Node 20 at `/data-10/users/re6141011/opt/node20/bin`.
- Backend port 8100 (8000 is taken). e2e needs a built frontend and a running backend (`frontend/e2e/README.md`).
- Commits: no Claude co-author trailer in this repo. Stage docs live in `.claude/docs/` (git-ignored).

## Repo navigation protocol
Before reading source code:
1. Read `.claude/CURRENT_STATE.md`.
2. Read root `CODEMAP.md`.
3. Use the CODEMAP Task Guide / Domain information to identify the minimum relevant file set.
4. Read the nearest subdirectory `CODEMAP.md` before reading multiple files in that directory.
5. Do not recursively scan unrelated directories.
6. Do not read the entire repository merely to gain context.
7. Prefer symbol search (`rg`) and exact relevant files/ranges.
8. Only follow dependencies when the current change affects their public contract or semantics.

If more than 5 previously-unrelated source files appear necessary, explain why before expanding the working set.
Large files: read their `.analysis.md` feature index first (`backend/app/profiling/layer2.py.analysis.md`) and load only matching line ranges. `frontend/src/charts/__fixtures__/*.json` are data; never read whole.

## Per-stage workflow
1. Read `.claude/CLAUDE.md`, `.claude/CURRENT_STATE.md`, root `CODEMAP.md`; `.claude/ARCHITECTURE.md`/`.claude/CONTRACTS.md` only if relevant.
2. Identify task files via CODEMAP.
3. Run targeted tests first (`pytest tests/test_x.py -q`, `vitest run <file>`).
4. Implement only the current stage.
5. Targeted tests again. Do not rerun the full suite during implementation.
6. Review subagent only if justified (see below).
7. Full regression once at the end: `pytest tests -q` (backend), `npm run build && npm test` (frontend), e2e if UI/render changed.
8. Commit the stage.
9. Update `.claude/CURRENT_STATE.md`.
10. Update CODEMAP incrementally only if files/dirs were added/moved/removed or a public symbol signature changed (see CODEMAP update rules).
11. Recommend `/compact` after a substantial stage; for an unrelated next task recommend a new session or `/clear`.

## Subagent rules
Main agent is the default executor. Do not spawn Agent A + Agent B by default.
Use a subagent only for: analysis/statistics core changes; public data contract changes; architecture changes; high-risk renderer/API semantic changes; a genuinely independent final review; clearly parallel tasks with little shared context.
Normal UI work, styling, docs, small refactors: main agent + tests only.
When using one: cheapest suitable model for review/search; minimum context = task goal + relevant CODEMAP excerpt + `git diff` + affected contracts + relevant tests. Do not fork a long session; do not spawn from a session already carrying large context. Parallel implementers use separate git worktrees.

## Testing / output rules
Prefer targeted, quiet commands (`-q`, `--tb=short`, `| tail`). On success keep a one-line summary; on failure keep only failing tests and the relevant traceback. Never paste full logs into the conversation.
