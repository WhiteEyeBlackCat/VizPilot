# VizPilot

**Local AI Data Exploration Assistant** — upload a tabular dataset, get automatic profiling, explainable chart recommendations, and interactive visualizations. Runs fully locally; raw data never leaves your machine.

> Upload data → understand data → suggest useful questions → recommend charts → visualize.

## Planned MVP

- **File formats**: CSV, Excel (.xlsx), Parquet
- **Charts**: line, bar, scatter, histogram, box plot, correlation heatmap
- **Pipeline**: data loader → profiler (Polars) → evidence layer → rule-based chart engine + local LLM reasoning → validated ChartSpec (Pydantic) → backend-rendered RenderResult → Apache ECharts
- **LLM**: optional layer, local-only, provider-agnostic (any OpenAI-compatible local server: Ollama, vLLM, llama.cpp, LM Studio). The app works fully with the LLM disabled.

## Tech Stack

| Layer | Choice |
|---|---|
| Frontend | React + TypeScript + Vite + Tailwind CSS + shadcn/ui |
| Backend | FastAPI (Python) |
| Data processing | Polars (+ Pandas, DuckDB) |
| Validation | Pydantic |
| Visualization | Apache ECharts (tree-shaken `echarts/core`, own React binding) |
| Local AI | OpenAI-compatible local endpoint |

## Getting Started

### Backend (FastAPI, port 8100)

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:create_app --factory --port 8100
```

Run tests with `.venv/bin/pytest tests -q`.

### Frontend

Two modes:

**Dev mode** (hot reload; vite proxies `/api` to `http://localhost:8100`):

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173 — backend must be running on 8100
```

**Build mode** (single port; the backend serves the built SPA at `/`):

```bash
cd frontend
npm install
npm run build      # tsc --noEmit + vite build -> frontend/dist
# restart the backend; it now serves the app at http://localhost:8100/
```

After pulling backend updates, rebuild the frontend (`npm run build`) so the
served SPA matches the current API shapes.

### Frontend tests

```bash
cd frontend
npm test                 # vitest: ECharts adapter contract + server-side SVG rendering tests
npm run e2e -- --tag dev # Playwright, headless Chromium, REAL backend on 8100, REAL files from dataset/
npm run capture-fixtures # re-record src/charts/__fixtures__ from the running backend
```

The adapter tests run against RenderResult snapshots recorded from the real
backend (`frontend/src/charts/__fixtures__/`, produced by
`npm run capture-fixtures`); re-record and commit them after any change to the
render API. The e2e run needs a built SPA served by the backend plus a
Playwright Chromium — environment notes (including a no-root workaround for
missing X11 libraries) are in [`frontend/e2e/README.md`](frontend/e2e/README.md).
Screenshots land in `frontend/e2e/screenshots/<tag>/`.

### Local LLM (optional)

The AI layer is off by default; everything works without it. To enable it,
point the backend at any OpenAI-compatible local server (Ollama, vLLM,
llama.cpp server, LM Studio) via environment variables:

```bash
VIZPILOT_LLM_ENABLED=true \
VIZPILOT_LLM_MODEL=qwen2.5:14b \
.venv/bin/uvicorn app.main:create_app --factory --port 8100
```

| Variable | Default | Purpose |
|---|---|---|
| `VIZPILOT_LLM_ENABLED` | `false` | Turn the AI suggestion layer on |
| `VIZPILOT_LLM_MODEL` | *(empty)* | Model name on the local server |
| `VIZPILOT_LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible endpoint |
| `VIZPILOT_LLM_API_KEY` | *(empty)* | Only if your local server requires one |
| `VIZPILOT_LLM_TIMEOUT_SECONDS` | `30` | Per-request timeout; on failure the app falls back to rule-based recommendations |

Only the dataset profile (schema, statistics, 5 sample rows) is ever sent to
the LLM — never the raw dataset. Sample rows can be withheld too with
`VIZPILOT_LLM_INCLUDE_SAMPLE_ROWS=false`.

## Frontend architecture

```
Dataset → Profiler → Evidence → Rules / LLM → ChartSpec → RenderResult (backend)
                                                              │
                              src/charts/echarts/option.ts ◄──┘  pure RenderResult → ECharts option
                              src/charts/echarts/theme.ts        one visual style for all six chart types
                              src/charts/echarts/EChartsView.tsx thin React binding (init / setOption / resize / dispose)
                              src/components/*                   shadcn/ui interface (upload, overview, recommendations, builder, workspace)
```

All aggregation, sampling, binning and statistics happen in the backend; the
browser only draws the RenderResult it receives. The LLM never produces UI or
chart code — only ChartSpec JSON, validated server-side.

## Status

Backend pipeline (upload → profiling → evidence → rule-based recommendations →
chart rendering → optional local-LLM re-ranking and insights) plus a React SPA
on shadcn/ui with Apache ECharts charts (tooltips, zoom, image export,
enlarged view). End-to-end tests run headless against the real backend.
