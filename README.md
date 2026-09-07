# VizPilot

**Local AI Data Exploration Assistant** — upload a tabular dataset, get automatic profiling, explainable chart recommendations, and interactive visualizations. Runs fully locally; raw data never leaves your machine.

> Upload data → understand data → suggest useful questions → recommend charts → visualize.

## Planned MVP

- **File formats**: CSV, Excel (.xlsx), Parquet
- **Charts**: line, bar, scatter, histogram, box plot, correlation heatmap
- **Pipeline**: data loader → profiler (Polars) → rule-based chart engine + local LLM reasoning → validated ChartSpec (Pydantic) → Plotly rendering
- **LLM**: optional layer, local-only, provider-agnostic (any OpenAI-compatible local server: Ollama, vLLM, llama.cpp, LM Studio). The app works fully with the LLM disabled.

## Tech Stack

| Layer | Choice |
|---|---|
| Frontend | React + TypeScript + Vite + Tailwind CSS |
| Backend | FastAPI (Python) |
| Data processing | Polars (+ Pandas, DuckDB) |
| Validation | Pydantic |
| Visualization | Plotly (react-plotly.js) |
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

`npm test` runs the chart-conversion unit tests (vitest).

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

## Status

All six stages implemented: upload → profiling → rule-based recommendations →
chart rendering → optional local-LLM re-ranking and insights → React SPA.
