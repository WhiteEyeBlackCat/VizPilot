// End-to-end smoke run against the REAL backend (port 8100) with REAL files
// from dataset/. No mocks. Produces screenshots under e2e/screenshots/<tag>/
// and prints a structured JSON summary on stdout.
//
//   npm run e2e -- --tag baseline
//
// Requires: backend running on 8100 serving frontend/dist, Playwright's
// Chromium installed (npx playwright install chromium). See e2e/README.md.

import { existsSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "..", "..");
const args = process.argv.slice(2);
const tag = args.includes("--tag") ? args[args.indexOf("--tag") + 1] : "run";
const base = process.env.E2E_BASE_URL ?? "http://localhost:8100";
const outDir = join(here, "screenshots", tag);
mkdirSync(outDir, { recursive: true });

// Chromium needs three X11 libs this host lacks; they live (unpacked, no
// root) in e2e/.xlibs — child processes inherit the env set here.
const xlibs = join(here, ".xlibs");
if (existsSync(xlibs)) {
  process.env.LD_LIBRARY_PATH = process.env.LD_LIBRARY_PATH
    ? `${xlibs}:${process.env.LD_LIBRARY_PATH}`
    : xlibs;
}

const dataset = (name) => join(repo, "dataset", name);
const CHART_SEL = ".js-plotly-plot, [data-chart-view]";

const summary = { tag, base, profileRetries: 0, pageErrors: [], consoleErrors: [], steps: [] };
const step = (name, data) => {
  summary.steps.push({ name, ...data });
  console.log(`[e2e] ${name}: ${JSON.stringify(data)}`);
};

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
page.on("pageerror", (e) => summary.pageErrors.push(e.message));
page.on("console", (m) => {
  if (m.type() === "error") summary.consoleErrors.push(m.text());
});

const shot = async (name, locator) => {
  const path = join(outDir, `${name}.png`);
  if (locator) await locator.screenshot({ path });
  else await page.screenshot({ path, fullPage: true });
  return path;
};

/** Works for both a native <select> (labelled by a wrapping <label>) and a
 *  Radix/shadcn Select trigger (role=combobox with aria-label). */
async function pick(label, value) {
  const el = page.getByLabel(label).first();
  await el.waitFor({ state: "visible" });
  const tag = await el.evaluate((e) => e.tagName.toLowerCase());
  if (tag === "select") {
    await el.selectOption(value);
    return;
  }
  await el.click();
  await page.getByRole("option", { name: value, exact: true }).click();
}

async function labelledOptions(label) {
  const el = page.getByLabel(label).first();
  const tag = await el.evaluate((e) => e.tagName.toLowerCase());
  if (tag === "select") {
    return el.evaluate((e) => Array.from(e.options).map((o) => o.value).filter(Boolean));
  }
  if (await el.isDisabled()) return [];
  await el.click();
  const opts = await page.getByRole("option").allTextContents();
  await page.keyboard.press("Escape");
  return opts.map((t) => t.trim()).filter((t) => t && t !== "—");
}

async function upload(file) {
  const before = await page.locator("[id^=rec-card-]").count();
  await page.locator("input[type=file]").setInputFiles(dataset(file));
  // The profile request can hit a known backend race right after upload
  // (concurrent writers of the same profile cache tmp file -> 500); the
  // recommendations still arrive. Re-selecting the dataset re-issues the
  // profile request. Retries are counted, never hidden.
  const overview = page.getByRole("heading", { name: /B\. 資料總覽/ });
  for (let attempt = 0; ; attempt++) {
    const ok = await overview.waitFor({ timeout: 15000 }).then(() => true, () => false);
    if (ok) break;
    if (attempt >= 2) throw new Error(`overview (profile) never appeared for ${file} after ${attempt} retries`);
    summary.profileRetries += 1;
    console.log(`[e2e] profile retry ${attempt + 1} for ${file}`);
    const switcher = page.getByLabel("選擇既有資料集").first();
    await switcher.click();
    await page.getByRole("option", { name: new RegExp(`^${file.replace(/\./g, "\\.")}`) }).first().click();
  }
  await page.waitForFunction(
    (n) => document.querySelectorAll("[id^=rec-card-]").length > 0 &&
      document.querySelectorAll("[id^=rec-card-]").length !== n,
    before,
    { timeout: 60000 },
  ).catch(() => {});
  await page.waitForTimeout(500);
}

async function chartCount() {
  return page.locator(CHART_SEL).count();
}

async function generateManual({ type, x, y, group }) {
  const before = await chartCount();
  await pick("圖表類型", type);
  if (x) await pick("X 軸", x);
  if (y) await pick("Y 軸", y);
  if (group) await pick("分組", group);
  await page.getByRole("button", { name: "生成圖表" }).click();
  await page.waitForFunction(
    ([sel, n]) => document.querySelectorAll(sel).length > n,
    [CHART_SEL, before],
    { timeout: 60000 },
  );
  await page.waitForTimeout(1200); // let the chart library finish drawing
  const chart = page.locator(CHART_SEL).last();
  return { chart, count: await chartCount() };
}

try {
  await page.goto(base, { waitUntil: "networkidle" });
  step("open", { title: await page.title() });

  // ---- air_quality: overview, recommendations, six chart types ----------
  await upload("air_quality.csv");
  const recCards = await page.locator("[id^=rec-card-]").count();
  const tiers = {};
  for (const t of ["推薦重點", "次要", "探索"]) {
    tiers[t] = (await page.getByRole("heading", { level: 3, name: new RegExp(`^${t}`) }).count()) > 0;
  }
  step("air_quality.uploaded", {
    columnsInOverview: await page.locator("table tbody tr").count(),
    recCards,
    tiers,
    generateButtons: await page.getByRole("button", { name: "Generate" }).count(),
  });
  await shot("01-overview-recs");

  // recommendation-driven generation (first card)
  {
    const before = await chartCount();
    await page.getByRole("button", { name: "Generate" }).first().click();
    await page.waitForFunction(
      ([sel, n]) => document.querySelectorAll(sel).length > n,
      [CHART_SEL, before],
      { timeout: 60000 },
    );
    await page.waitForTimeout(1200);
    step("air_quality.generate-from-recommendation", { charts: await chartCount() });
    await shot("02-rec-chart", page.locator(CHART_SEL).last());
  }

  // manual builder: option filtering per type
  await pick("圖表類型", "box");
  step("builder.box.options", {
    x: await labelledOptions("X 軸"),
    y: await labelledOptions("Y 軸"),
  });
  await pick("圖表類型", "heatmap");
  step("builder.heatmap.options", {
    x: await labelledOptions("X 軸"),
    y: await labelledOptions("Y 軸"),
  });
  await pick("圖表類型", "bar");
  step("builder.bar.options", {
    x: await labelledOptions("X 軸"),
    y: await labelledOptions("Y 軸"),
    group: await labelledOptions("分組"),
    aggDisabledWithoutY: await page.getByLabel("聚合").first().isDisabled(),
  });

  const manual = [
    { name: "line", type: "line", x: "timestamp", y: "temperature" },
    { name: "bar", type: "bar", x: "station", y: "pm25" },
    { name: "scatter", type: "scatter", x: "temperature", y: "humidity" },
    { name: "histogram", type: "histogram", x: "pm25" },
    { name: "box", type: "box", x: "station", y: "pm25" },
    { name: "heatmap", type: "heatmap" },
  ];
  for (const m of manual) {
    const { chart, count } = await generateManual(m);
    step(`manual.${m.name}`, { charts: count });
    await shot(`10-manual-${m.name}`, chart.locator("xpath=.."));
  }
  await shot("11-workspace-all");

  // 422 path: line with a categorical x that needs aggregation is fine, so
  // provoke a real validation error instead: scatter with x == y is blocked
  // by the builder, so use box without y (backend rejects: y required)
  {
    await pick("圖表類型", "box");
    await pick("X 軸", "station");
    await page.getByRole("button", { name: "生成圖表" }).click();
    const err = page.locator("text=/y|Y 軸|required|必須/i").first();
    await err.waitFor({ timeout: 15000 }).catch(() => {});
    const errorShown = await page.locator("ul li, [role=alert]").filter({ hasText: /y/i }).count();
    step("builder.422", { errorShown: errorShown > 0 });
    await shot("12-builder-422");
  }

  // ---- outliers: histogram over a sentinel-laden column -> display_range --
  await upload("outliers.csv");
  {
    const { chart, count } = await generateManual({ type: "histogram", x: "temperature" });
    const note = await page.getByText(/顯示範圍/).count();
    step("outliers.histogram.display_range", { charts: count, displayRangeNote: note > 0 });
    await shot("20-outliers-histogram", chart.locator("xpath=.."));
  }

  // ---- tiny_dataset: confidence warnings on recommendation cards --------
  await upload("tiny_dataset.csv");
  {
    const chips = await page.locator("[id^=rec-card-] li").count();
    const topEmptyNotice = await page.getByText(/沒有圖表在資料中展現足夠強的證據/).count();
    step("tiny.warnings", { recCards: await page.locator("[id^=rec-card-]").count(), warningChips: chips, topEmptyNotice: topEmptyNotice > 0 });
    await shot("30-tiny-warnings");
  }
} catch (e) {
  summary.fatal = String(e && e.stack ? e.stack : e);
  await shot("99-fatal").catch(() => {});
} finally {
  await browser.close();
}

summary.ok = !summary.fatal && summary.pageErrors.length === 0;
console.log("[e2e] SUMMARY " + JSON.stringify(summary, null, 2));
process.exit(summary.ok ? 0 : 1);
