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

## Status

Design phase — implementation starting with dataset upload and profiling.
