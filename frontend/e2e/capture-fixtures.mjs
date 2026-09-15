// Records RenderResult contract snapshots from the REAL backend (no browser,
// no mocks): uploads files from dataset/, calls POST /api/charts/render with
// a fixed list of specs and writes each response verbatim to
// src/charts/__fixtures__/<name>.json. index.json records which file + spec
// produced each fixture so a snapshot can be re-recorded after a backend
// change.
//
//   npm run capture-fixtures            (backend must be running on 8100)

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "..", "..");
const outDir = join(here, "..", "src", "charts", "__fixtures__");
const base = process.env.E2E_BASE_URL ?? "http://localhost:8100";
mkdirSync(outDir, { recursive: true });

// name -> { file, spec }. Every variant the ECharts adapter must honour.
const FIXTURES = {
  line_single: {
    file: "air_quality.csv",
    spec: { title: "mean temperature per day", type: "line", x: "timestamp", y: "temperature", aggregation: "mean", time_granularity: "day" },
  },
  line_grouped: {
    file: "air_quality.csv",
    spec: { title: "mean temperature per day by station", type: "line", x: "timestamp", y: "temperature", group_by: "station", aggregation: "mean", time_granularity: "day" },
  },
  line_raw_time: {
    file: "air_quality.csv",
    spec: { title: "pm25 over time", type: "line", x: "timestamp", y: "pm25" },
  },
  line_numeric_x: {
    file: "products.csv",
    spec: { title: "mean rating by price", type: "line", x: "price", y: "rating", aggregation: "mean" },
  },
  bar_single: {
    file: "air_quality.csv",
    spec: { title: "mean pm25 by station", type: "bar", x: "station", y: "pm25", aggregation: "mean" },
  },
  bar_count: {
    file: "air_quality.csv",
    spec: { title: "rows per station", type: "bar", x: "station", aggregation: "count" },
  },
  bar_grouped_null: {
    // e2e/data/sparse_groups.csv: East x Gadget never occurs -> null cell
    file: "e2e/data/sparse_groups.csv",
    spec: { title: "mean value by region and product", type: "bar", x: "region", y: "value", group_by: "product", aggregation: "mean" },
  },
  bar_truncated: {
    file: "customers_high_cardinality.csv",
    spec: { title: "top cities by lifetime value", type: "bar", x: "city", y: "lifetime_value", aggregation: "mean", top_n: 5 },
  },
  scatter_single: {
    file: "air_quality.csv",
    spec: { title: "humidity vs temperature", type: "scatter", x: "temperature", y: "humidity" },
  },
  scatter_grouped: {
    file: "air_quality.csv",
    spec: { title: "humidity vs temperature by station", type: "scatter", x: "temperature", y: "humidity", group_by: "station" },
  },
  scatter_sampled: {
    file: "large_dataset.parquet",
    spec: null, // filled below: first two numeric columns of the profile
  },
  histogram_single: {
    file: "air_quality.csv",
    spec: { title: "pm25 distribution", type: "histogram", x: "pm25" },
  },
  histogram_grouped: {
    file: "air_quality.csv",
    spec: { title: "pm25 distribution by station", type: "histogram", x: "pm25", group_by: "station" },
  },
  histogram_display_range: {
    file: "outliers.csv",
    spec: { title: "temperature distribution", type: "histogram", x: "temperature" },
  },
  box_outliers: {
    file: "outliers.csv",
    spec: { title: "value by group", type: "box", x: "group", y: "value" },
  },
  box_single: {
    file: "air_quality.csv",
    spec: { title: "pm25", type: "box", y: "pm25" },
  },
  heatmap: {
    file: "air_quality.csv",
    spec: { title: "correlation heatmap", type: "heatmap" },
  },
  heatmap_null_cell: {
    // metric_a / metric_b never overlap -> pairwise correlation is null
    file: "e2e/data/sparse_groups.csv",
    spec: { title: "correlation heatmap", type: "heatmap" },
  },
};

async function upload(file) {
  // dataset/<file> (generated synthetic data) or a small committed file
  // under frontend/e2e/data/ for shapes the synthetic sets do not produce
  const path = file.startsWith("e2e/") ? join(here, "..", file) : join(repo, "dataset", file);
  const form = new FormData();
  form.append("file", new Blob([readFileSync(path)]), basename(path));
  const resp = await fetch(`${base}/api/datasets`, { method: "POST", body: form });
  if (!resp.ok) throw new Error(`upload ${file}: HTTP ${resp.status} ${await resp.text()}`);
  return resp.json();
}

async function profile(id) {
  for (let attempt = 0; attempt < 3; attempt++) {
    const resp = await fetch(`${base}/api/datasets/${id}/profile`);
    if (resp.ok) return resp.json();
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error(`profile ${id} failed`);
}

async function render(id, spec) {
  const resp = await fetch(`${base}/api/charts/render`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_id: id, spec }),
  });
  const body = await resp.json();
  if (!resp.ok) return { error: resp.status, body };
  return { result: body };
}

const ids = {};
const index = {};
for (const [name, fx] of Object.entries(FIXTURES)) {
  if (!ids[fx.file]) {
    const meta = await upload(fx.file);
    ids[fx.file] = meta.dataset_id;
    console.log(`[fixtures] uploaded ${fx.file} -> ${meta.dataset_id} (${meta.n_rows} rows)`);
  }
  const id = ids[fx.file];
  let spec = fx.spec;
  if (spec === null) {
    const p = await profile(id);
    const numeric = p.columns.filter((c) => c.semantic_type === "numeric").map((c) => c.name);
    if (numeric.length < 2) {
      console.log(`[fixtures] ${name}: skipped, fewer than 2 numeric columns in ${fx.file}`);
      continue;
    }
    spec = { title: `${numeric[1]} vs ${numeric[0]}`, type: "scatter", x: numeric[0], y: numeric[1] };
  }
  const out = await render(id, spec);
  if (out.error) {
    console.log(`[fixtures] ${name}: HTTP ${out.error} ${JSON.stringify(out.body)} (not written)`);
    continue;
  }
  writeFileSync(join(outDir, `${name}.json`), JSON.stringify(out.result, null, 2) + "\n");
  index[name] = { file: fx.file, spec };
  const r = out.result;
  const shape =
    "series" in r.chart_data
      ? `series=${r.chart_data.series.length}`
      : "groups" in r.chart_data
        ? `groups=${r.chart_data.groups.length}`
        : "columns" in r.chart_data
          ? `columns=${r.chart_data.columns.length}`
          : `bins=${r.chart_data.bins?.counts?.length}`;
  console.log(
    `[fixtures] ${name}: n_points=${r.n_points} sampled=${r.sampled} ${shape}` +
      (r.display_range ? " display_range" : "") +
      ("truncated" in r.chart_data ? ` truncated=${r.chart_data.truncated}` : ""),
  );
}
writeFileSync(join(outDir, "index.json"), JSON.stringify(index, null, 2) + "\n");
console.log(`[fixtures] wrote ${Object.keys(index).length} fixtures to ${outDir}`);
