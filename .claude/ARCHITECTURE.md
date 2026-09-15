# ARCHITECTURE.md — VizPilot data flow and module boundaries

Read this only when a task crosses module boundaries. For file-level navigation use `CODEMAP.md`; for exact shapes use `.claude/CONTRACTS.md`.

## Pipeline

```
upload (CSV/XLSX/Parquet)
  → DatasetStore            uuid + Parquet temp file, in-memory index, no DB
  → profile_dataset         Polars; sample >100k rows; semantic typing; quality; stats
      → Evidence L1         eta², Spearman, time effects, interactions, slope het,
                            derived-column identities, near-duplicate groups
      → Evidence L2         distributions, group summaries, conditional relationships,
                            group time patterns, nonlinear curves, change points, anomalies
  → DatasetProfile (PROFILE_VERSION) cached as json per dataset
  → RecommendationService.get(profile, use_llm)
      rules: recommend_charts → evidence score → derived/near-dup caps
             → confidence (score × sample × missingness × robustness)
             → diversity caps (per-x ≤2, per-y ≤3) → assign_tiers
      llm (optional): build_messages(metadata only) → provider → tolerant parse
             → canonicalize → evaluate_llm_spec (strong/weak/unverified/neutral)
             → merge with rules → re-cap → tiers; insights paired with charts
      → {charts: Recommendation[], insights, message, warnings}
  → POST /charts/render(ChartSpec) → validate_spec → render_chart → RenderResult
  → frontend buildOption(RenderResult) → ECharts
```

Principle: statistics produce evidence → LLM proposes hypotheses → statistics verify. One direction, no loop. Every LLM claim is gated by backend numbers; the LLM's own numbers are ignored.

## Backend modules (`backend/app/`)

| Module | Responsibility | Must not |
|---|---|---|
| `datasets/` | HTTP routes (`/api/*`), file loading, dataset store | contain statistics or ranking logic |
| `profiling/` | `DatasetProfile` and both evidence layers; deterministic, LLM-free | know about charts, tiers, or the LLM |
| `charts/spec.py` | `ChartSpec` model + validation matrix (shared gate for rules, LLM, manual builder) | be bypassed by any producer |
| `charts/rules.py` | candidates, evidence scoring, caps, tiers, LLM-spec verification | read raw rows (works on profile only) |
| `charts/confidence.py` | the single confidence formula, `Warning`/`Confidence` payloads, tier caps | be duplicated elsewhere |
| `charts/render.py` | aggregate a DataFrame per spec into `RenderResult`; hard point cap; sentinel-aware histogram range | change chart semantics without frontend parity tests |
| `llm/` | prompt (metadata only), provider abstraction, tolerant parsing, merge with rules | send raw data, produce code, or be required for the app to work |
| `probes/` | typed statistical tests requested by the LLM workflow, bounded and cached | accept free-form expressions (`extra="forbid"`) |
| `main.py` / `config.py` | app factory, `Settings` from `VIZPILOT_*` env, per-process caches and locks | — |

Caches and locks: `ProfileService` (per-dataset lock, atomic json, invalidated by `PROFILE_VERSION`), `RecommendationService` (per-key lock/cache), `CastedFrameCache` (casted frames for render). All process-local.

## Frontend modules (`frontend/src/`)

| Module | Responsibility |
|---|---|
| `App.tsx` | shell: sidebar + page + resizable preview panel (kept mounted); on dataset select fetches profile, recs(llm=false), recs(llm=true) in parallel; rules-only response is shown until the LLM one replaces it |
| `store.ts` | one reducer: datasets, profile, recs, preview, per-dataset saved charts (workspace), explore form |
| `lib/router.ts` | hash routes `#/<dataset>/<page>`; pages: overview, insights, explore, workspace |
| `api.ts`, `types.ts` | fetch wrappers and contract mirrors |
| `components/` | PreviewPanel (chart + warnings + save + enlarge), Recommendations (tiers, badges, empty-top text), ManualBuilder (mirrors `validate_spec` filters), DatasetOverview, Sidebar, Upload |
| `charts/echarts/` | `buildOption(RenderResult)` per type, theme (light/dark), registration, React binding; tested against 18 real-backend fixtures |
| `components/ui/` | shadcn primitives; `badge` is the single semantic colour source |

The frontend never computes statistics; it renders what the backend returns and derives axis titles from `y_label`.

## Boundaries that need an explicit decision to cross
- ChartSpec fields, RenderResult shape, render semantics, confidence formula (DECISIONS D11).
- Adding a cloud LLM provider, or making any core path depend on the LLM.
- Sending more than profile metadata + ≤5 sample rows to the LLM.
- Removing/renaming any API response field (additive only).

## Runtime
Backend `uvicorn app.main:create_app --factory --port 8100` serves `/api` and the built SPA. Dev frontend on 5173 proxies `/api`. Local LLM: any OpenAI-compatible server (Ollama with qwen2.5:14b in this environment).
