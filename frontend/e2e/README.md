# End-to-end smoke run

`npm run e2e -- --tag <name>` drives the built app in headless Chromium
against a **real backend with real files** from `dataset/` — no mocks.
Screenshots land in `e2e/screenshots/<tag>/` (git-ignored) and a structured
JSON summary is printed on stdout (`[e2e] SUMMARY …`, `ok: true|false`).

## Requirements

- Node 20 (`export PATH=/data-10/users/re6141011/opt/node20/bin:$PATH` on this host)
- Playwright's Chromium: `npx playwright install chromium` (goes to `~/.cache/ms-playwright`)
- Three X11 libraries this host lacks (`libXcomposite`, `libXdamage`,
  `libXrandr`), unpacked without root into `e2e/.xlibs/` (git-ignored). The
  script prepends that directory to `LD_LIBRARY_PATH` when it exists.
  To recreate: `apt-get download libxcomposite1 libxdamage1 libxrandr2`,
  `dpkg -x <deb> tmp`, copy `tmp/usr/lib/x86_64-linux-gnu/*.so*` into `e2e/.xlibs/`.
- A backend serving `frontend/dist` (run `npm run build` first). Default
  `http://localhost:8100`; override with `E2E_BASE_URL`, e.g. a throw-away
  backend on another port with its own data dir:

  ```bash
  cd backend && VIZPILOT_DATA_DIR=/tmp/vp-e2e .venv/bin/uvicorn app.main:create_app --factory --port 8112
  cd frontend && E2E_BASE_URL=http://localhost:8112 npm run e2e -- --tag mine
  ```

- `dataset/hour_like.csv` (generate with `backend/.venv/bin/python dataset/syn/hour_like.py`)
  and the other synthetic CSVs from `dataset/syn/`.

## What the run covers (stage 16.2 layout)

Sidebar navigation (`#/d/<dataset_id>/<page>` hash routes), one shared preview
panel (`[data-preview-panel]`), Save as the only way into the workspace:

1. empty state → **New Dataset** dialog upload → lands on Overview
2. Insights: three tiers, click a top card → preview panel (chart, confidence),
   workspace still 0; collapse / expand the panel
3. Explore: option filtering per chart type, six chart types previewed with
   tooltips; Save two → workspace 2 (Save turns into 已保存)
4. Explore form keeps its selection across page switches
5. Workspace: list, preview, Enlarge dialog (bigger second instance, disposed on
   Escape), Export (real download of a PNG data URL), Remove
6. 422 path (box without y), stable document height (no page scroll)
7. dataset switch clears the preview and keeps per-dataset workspaces;
   reload keeps the route (the workspace is in-memory and starts empty)
8. 800px viewport: top bar navigation, preview as a bottom drawer
9. outliers `display_range` note, tiny_dataset warning chips + empty-top notice,
   sales_basic derived-column disclosure, hour_like near-duplicate suppression
10. every `/api/charts/render` request body is captured (`renderPayloads`) and
    the bar / heatmap / scatter shapes are asserted (aggregation prefill, nulls)

`capture-fixtures.mjs` records real `RenderResult`s for the option-builder
unit tests (`npm run capture-fixtures`, backend on 8100).
