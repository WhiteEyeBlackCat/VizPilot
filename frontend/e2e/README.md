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
  — note that `dataset/` inputs are git-ignored: in a fresh worktree they must be present or
  symlinked from the main checkout, otherwise this run fails at the upload step and six
  backend pytest tests (the `dataset/`-backed ones) skip.
  and the other synthetic CSVs from `dataset/syn/`.

## What the run covers (stage 16.3 layout)

Sidebar navigation (`#/d/<dataset_id>/<page>` hash routes), one shared preview
panel (`[data-preview-panel]`), Save as the only way into the workspace:

1. empty state → **New Dataset** dialog upload → lands on Overview
2. Overview: summary tiles (rows / columns / missing / profiled rows / quality
   flags) match the dataset, the type chips filter the column table, the
   derived-fields and warnings sections are present; a same-name upload shows
   the confirmation prompt and cancelling changes nothing
3. Insights: three tiers (h2 headings), click a top card → preview panel
   (chart, confidence), workspace still 0; the first opening is the 55 / 45
   split and the panel chart hides its in-canvas title (the header names it);
   collapse / expand the panel
4. Explore: option filtering per chart type, six chart types previewed with
   tooltips; Save two → workspace 2 (Save turns into 已保存); the column-role
   hints and the "最近一次生成" recap update
5. Explore form and the Insights 探索 toggle survive page switches AND
   preview-panel open / close (the panel group never re-mounts)
6. Workspace: list, preview, Enlarge dialog (bigger second instance that keeps
   its title, disposed on Escape), Export (real download of a PNG data URL),
   Remove; Escape closes the panel when no dialog is open
7. 422 path (box without y), stable document height (no page scroll)
8. 1100px viewport: the preview overlays the content without squeezing it;
   800px viewport: top bar navigation, preview as a bottom drawer
9. outliers: `display_range` note in the panel and the suspected-sentinel
   quality flag on the Overview; dataset switch clears the preview and keeps
   per-dataset workspaces; reload keeps the route (the workspace is in-memory
   and starts empty)
10. hash normalisation: an unknown dataset id or `#/garbage` → empty state,
    `#/d/<id>/bogus` → `/overview` (replaced, no history entry)
11. tiny_dataset warning chips + empty-top notice, sales_basic derived-column
    disclosure (Insights banner and the Overview derived row), hour_like
    near-duplicate suppression (and its Overview row)
12. every `/api/charts/render` request body is captured (`renderPayloads`) and
    the bar / heatmap / scatter shapes are asserted (aggregation prefill, nulls)
13. stage 18 (`checkKeyFindings`, air_quality and hour_like): Key Findings rows
    render badge + headline only until the row button is activated (click,
    Enter, Space); the detail shows evidence rows and View chart drives the
    analysis panel (insight section, evidence, Save/Export/Enlarge/Close,
    chart ≈ 45–75% of the panel height); the section precedes "More charts";
    cards no longer print `x=/y=/agg=`; a `mnth` bar is read in natural
    order with a `[data-order-note]`. With the LLM off the step records
    `insights: 0` and passes.

14. stage 19.1 (hour_like): the Explore builder reads the profile's
    `column_roles` — with type `rose` the X list is the cyclic columns only
    (`season, mnth, hr`, never `weathersit`) and the page lists the cyclic /
    ordered columns (`[data-hint=cyclic]`, `[data-hint=ordered]`); the
    backend's "must be cyclic" 422 stays for hand-written specs but is no
    longer reachable from the builder, so the stage-19 nominal-rose step is
    replaced by `explore.rose-x-cyclic-only`.

`capture-fixtures.mjs` records real `RenderResult`s for the option-builder
unit tests (`npm run capture-fixtures`, backend on 8100). Stage 19.1 added
`--only a,b` (records a subset, merges into `index.json`) and the committed
inputs `e2e/data/{spread_groups,cycles,dense_xy}.csv` from
`e2e/data/make_stage191.py`; `backend/tests/test_fixture_regression.py`
re-renders every fixture whose input lives under `e2e/data/`.
