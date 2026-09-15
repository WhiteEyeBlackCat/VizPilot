# CURRENT_STATE.md — VizPilot (update after each major stage)

Updated: 2026-09-15 · main `ea1140a` · detailed stage notes in `.claude/docs/stages/` (local, git-ignored)

## Completed
- Stages 1–8: upload → profiler → rule engine → ECharts render → LLM → manual builder; evidence L1 + tier system + insight/chart contract.
- 9/9b: unified confidence layer (D11), dirty-numeric/id/sentinel handling, per-dataset profile lock.
- 10–12: Playwright e2e harness, shadcn/ui, Plotly → ECharts (bundle 1.03 MB).
- 13/14: derived-column detection (definitional relationships demoted), near-duplicate suppression, per-y caps.
- 16.1–16.3: dark tokens, sidebar + hash routes + shared preview panel + save semantics, overview page, accessibility (16.2 remount bug fixed in `ca19ab5`).
- 17.1/17.1b: Evidence layer 2 (`PROFILE_VERSION` 10) — PASS.
- 17.2: probe engine (`backend/app/probes/`) — merged `f42f362` with fixes (boolean groups, error containment). Not yet routed/used on main.

## Architecture state
Pipeline as in `.claude/ARCHITECTURE.md`. On main the LLM path is still single-call (`build_messages` → merge). The probe engine exists but has no caller. Frontend is the 16.x shell (4 pages, preview panel).

## In progress
- **17.3 two-stage LLM workflow** on branch `worktree-agent-a7a3af35e9c056762` (worktree `.claude/worktrees/agent-a7a3af35e9c056762`, 3 commits ahead of main, base `f42f362`): hypotheses → coverage check (`llm/coverage.py`) → targeted probes (≤5) → final analyst call; `PROFILE_VERSION` 11; 561 tests there. Verifier rounds 1 and 2 FAILED (chart-only path leniency; ungated LLM numbers/column names; then bare numbers/dates in LLM#1 statements; undirected nonlinear coverage). Round 3 fixes are committed (`42afa37`); **round 3 verification was interrupted by a rate limit and has not run**.
- Branch is based on `f42f362`, so it lacks the 16.2-fix/16.3 frontend commits; merge main into it (or rebase) before verification.

## Recent decisions
- Definitional / near-duplicate / trivial relationships are not findings; 0 insights is legitimate.
- LLM does hypothesis + wording only; every claim gated by backend statistics; LLM-stated numbers ignored.
- `?llm=false` output must stay byte-identical across LLM-side changes (15 dataset snapshots).
- Multi-agent work uses separate git worktrees; no Claude co-author trailer in commits.

## Known blockers / risks
- main is 4 commits ahead of `origin/main` (unpushed: 16.3 + 17.2 fix merges). Push needs the user's credentials (VS Code askpass).
- Running backend on 8100 may be stale; restart after backend changes.
- Real model (qwen2.5:14b) rarely proposes distribution/interaction probes; probe path mainly proven by FakeProvider.
- Leading zeros lost on CSV load (`"001"` → 1); `pct` mixed scales; no histogram range opt-out.

## Test status (main `ea1140a`)
- backend: 535 pytest collected (last full run green at 16.3/17.2 merge).
- frontend: tsc 0 errors, vitest 39, build green; e2e stage16.3 ok.

## Next planned task
1. Merge/rebase 17.3 branch onto main, run full pytest + 15-dataset `llm=false` snapshot check, then verifier round 3 (single reviewer, diff + `tests/test_llm_workflow.py` + `stage17.3.md` as context).
2. 17.4 benchmark + regression (8 planted datasets incl. `dataset/syn/noise.py`).
3. Then wire probe/insight `validation` fields into the frontend (optional fields only).
