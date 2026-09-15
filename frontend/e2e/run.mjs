// End-to-end smoke run against the REAL backend with REAL files from
// dataset/. No mocks. Produces screenshots under e2e/screenshots/<tag>/ and
// prints a structured JSON summary on stdout.
//
//   npm run e2e -- --tag baseline            (backend on 8100)
//   E2E_BASE_URL=http://localhost:8112 npm run e2e -- --tag x
//
// Requires: a backend serving frontend/dist, Playwright's Chromium installed
// (npx playwright install chromium). See e2e/README.md.
//
// Layout (stage 16.2): sidebar (datasets + Overview / Insights / Explore /
// Workspace, hash routes #/d/<id>/<page>), one shared preview panel
// ([data-preview-panel]); Save is the only way into the workspace.

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
const CHART_SEL = "[data-chart-view]";
const PANEL_SEL = "[data-preview-panel]";

const summary = {
  tag,
  base,
  profileRetries: 0,
  pageErrors: [],
  consoleErrors: [],
  renderPayloads: [],
  steps: [],
};
const step = (name, data) => {
  summary.steps.push({ name, ...data });
  console.log(`[e2e] ${name}: ${JSON.stringify(data)}`);
};
const fail = (msg) => {
  throw new Error(msg);
};

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1400, height: 1000 }, acceptDownloads: true });
page.on("pageerror", (e) => summary.pageErrors.push(e.message));
page.on("console", (m) => {
  if (m.type() === "error") summary.consoleErrors.push(m.text());
});
// every chart request the UI makes, verbatim (spec shape must not change)
page.on("request", (req) => {
  if (req.method() === "POST" && req.url().includes("/api/charts/render")) {
    try {
      const body = JSON.parse(req.postData() ?? "{}");
      summary.renderPayloads.push(body.spec);
    } catch {
      summary.renderPayloads.push({ unparsable: true });
    }
  }
});

const shot = async (name, locator) => {
  const path = join(outDir, `${name}.png`);
  if (locator) await locator.screenshot({ path });
  else await page.screenshot({ path, fullPage: false });
  return path;
};

const main = () => page.locator("[data-page-content]").first();
const panel = () => page.locator(PANEL_SEL).first();
const currentHash = () => page.evaluate(() => window.location.hash);
const datasetIdFromHash = async () => (await currentHash()).split("/")[2] ?? null;

async function gotoPage(name) {
  await page.getByRole("link", { name: new RegExp(`^${name}`) }).click();
  await page.waitForFunction(
    (n) => document.querySelector("[data-page-content]")?.dataset.activePage === n,
    name.toLowerCase(),
    { timeout: 5000 },
  );
  await page.waitForTimeout(120);
}

/** Works for both a native <select> (labelled) and a Radix/shadcn Select
 *  trigger (role=combobox with aria-label). */
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

async function waitProfile(file) {
  // The profile request can hit a backend race right after upload (fixed in
  // stage 9b, kept as a guard); re-selecting the dataset re-issues it.
  // Retries are counted, never hidden.
  for (let attempt = 0; ; attempt++) {
    const ok = await page
      .waitForSelector("[data-page-content][data-profile=loaded]", { timeout: 15000 })
      .then(() => true, () => false);
    if (ok) return;
    if (attempt >= 2) fail(`profile never loaded for ${file} after ${attempt} retries`);
    summary.profileRetries += 1;
    console.log(`[e2e] profile retry ${attempt + 1} for ${file}`);
    const id = await datasetIdFromHash();
    await page.evaluate(() => (window.location.hash = "#/"));
    await page.waitForTimeout(200);
    await page.evaluate((i) => (window.location.hash = `#/d/${i}/overview`), id);
  }
}

/** New Dataset → dialog → file input. The app lands on Overview. */
async function upload(file) {
  await page.getByRole("button", { name: "New Dataset" }).first().click();
  const dialog = page.locator("[data-new-dataset-dialog]");
  await dialog.waitFor({ timeout: 5000 });
  const before = await datasetIdFromHash();
  await dialog.locator("input[type=file]").setInputFiles(dataset(file));
  await page.waitForFunction(
    (prev) => window.location.hash.startsWith("#/d/") && window.location.hash.split("/")[2] !== prev,
    before,
    { timeout: 60000 },
  );
  await dialog.waitFor({ state: "detached", timeout: 5000 });
  await waitProfile(file);
  // recommendations load in the background (pages stay mounted, hidden)
  await page.waitForFunction(() => document.querySelectorAll("[id^=rec-card-]").length > 0, null, { timeout: 60000 }).catch(() => {});
  await page.waitForTimeout(400);
}

async function panelState() {
  const p = panel();
  if ((await p.count()) === 0) return { present: false };
  return p.evaluate((el) => ({
    present: true,
    variant: el.dataset.variant,
    source: el.dataset.source,
    seq: Number(el.dataset.seq),
    open: el.dataset.open === "true",
    title: el.dataset.title,
    canvas: Boolean(el.querySelector("canvas")),
    confidence: el.querySelector("[data-preview-confidence]")?.textContent?.trim() ?? null,
    description: el.querySelector("[data-preview-description]")?.textContent?.trim().slice(0, 80) ?? null,
    saveLabel: el.querySelector("[data-preview-save]")?.textContent?.trim() ?? null,
    saveDisabled: el.querySelector("[data-preview-save]")?.hasAttribute("disabled") ?? null,
  }));
}
const panelSeq = async () => (await panelState()).seq ?? 0;
async function waitPreview(prevSeq) {
  await page.waitForFunction(
    ([sel, prev]) => Number(document.querySelector(sel)?.dataset.seq ?? 0) > prev,
    [PANEL_SEL, prevSeq],
    { timeout: 60000 },
  );
  await page.waitForTimeout(900); // let ECharts finish drawing
}
const workspaceCount = () => page.locator("[data-workspace-count]").first().evaluate((el) => Number(el.textContent));

/** The ECharts instance behind a chart host: canvas present, its pixel size,
 *  and the series types actually set. */
async function inspectChart(locator) {
  return locator.evaluate((el) => {
    const canvas = el.querySelector("canvas");
    const chart = el.__echarts;
    return {
      renderer: el.dataset.chartView,
      canvas: Boolean(canvas),
      canvasSize: canvas ? [canvas.width, canvas.height] : null,
      seriesTypes: chart ? (chart.getOption().series ?? []).map((s) => s.type) : null,
    };
  });
}

/** Ask ECharts to show a tooltip for the first datum and report the
 *  tooltip DOM it created (real dispatchAction, real DOM). */
async function probeTooltip(locator) {
  return locator.evaluate(async (el) => {
    const chart = el.__echarts;
    if (!chart) return { supported: false };
    chart.dispatchAction({ type: "showTip", seriesIndex: 0, dataIndex: 0 });
    await new Promise((r) => setTimeout(r, 300));
    const tip = Array.from(el.querySelectorAll("div")).find(
      (d) => d.style.position === "absolute" && d.style.visibility !== "hidden" && d.textContent.trim(),
    );
    chart.dispatchAction({ type: "hideTip" });
    return { supported: true, shown: Boolean(tip), text: tip ? tip.textContent.trim().slice(0, 120) : null };
  });
}

/** Explore page: pick options, generate, wait for the preview to change. */
async function generateManual({ type, x, y, group }) {
  await gotoPage("Explore");
  const before = await panelSeq();
  await pick("圖表類型", type);
  if (x) await pick("X 軸", x);
  if (y) await pick("Y 軸", y);
  if (group) await pick("分組", group);
  await page.getByRole("button", { name: "生成圖表" }).click();
  await waitPreview(before);
  return { chart: panel().locator(CHART_SEL).first(), ...(await panelState()) };
}

async function layoutProbe() {
  return page.evaluate(() => ({
    innerHeight: window.innerHeight,
    scrollHeight: document.documentElement.scrollHeight,
    bodyScrollHeight: document.body.scrollHeight,
    scrollWidth: document.documentElement.scrollWidth,
    innerWidth: window.innerWidth,
  }));
}

try {
  await page.goto(base, { waitUntil: "networkidle" });
  await page.waitForTimeout(300);
  step("open", {
    title: await page.title(),
    emptyState: (await page.locator("[data-empty-state]").count()) > 0,
    pageNavLinks: await page.locator("[data-page-nav] a").count(),
    sidebar: await page.locator("[data-sidebar]").first().getAttribute("data-sidebar"),
    newDatasetButtons: await page.getByRole("button", { name: "New Dataset" }).count(),
  });
  await shot("00-empty-state");

  // ---- air_quality: upload → Overview → Insights → preview -------------
  await upload("air_quality.csv");
  const airId = await datasetIdFromHash();
  step("air_quality.uploaded", {
    hash: await currentHash(),
    activePage: await main().getAttribute("data-active-page"),
    datasetsInSidebar: await page.locator("[data-dataset-list] a").count(),
    currentDataset: await page.locator("[data-dataset-list] a[aria-current=true]").textContent().then((t) => t?.trim()),
    overviewRows: await page.locator("[data-page-section=overview] table tbody tr").count(),
    pageNavLinks: await page.locator("[data-page-nav] a").count(),
    panelPresent: (await panelState()).present,
  });
  await shot("01-overview-page");

  await gotoPage("Insights");
  const tiers = {};
  for (const t of ["推薦重點", "次要", "探索"]) {
    tiers[t] = (await page.getByRole("heading", { level: 3, name: new RegExp(`^${t}`) }).count()) > 0;
  }
  const recCards = await page.locator("[id^=rec-card-]").count();
  const topCards = await page.locator("[id^=rec-card-][data-tier=top]").count();
  step("air_quality.insights", { recCards, topCards, tiers, previewButtons: await page.getByRole("button", { name: "Preview" }).count() });
  await shot("02-insights-page");

  // click a top card → preview panel opens with the chart, nothing saved
  {
    const before = await panelSeq();
    await page.locator("[id^=rec-card-][data-tier=top]").first().click();
    await waitPreview(before);
    const st = await panelState();
    const info = await inspectChart(panel().locator(CHART_SEL).first());
    const paneBox = await panel().boundingBox();
    const contentBox = await main().boundingBox();
    const pressed = await page.locator("[id^=rec-card-][aria-pressed=true]").count();
    step("air_quality.preview-from-insight", {
      ...st,
      ...info,
      selectedCards: pressed,
      panelRightOfContent: Boolean(paneBox && contentBox && paneBox.x >= contentBox.x + contentBox.width - 2),
      workspaceCount: await workspaceCount(),
    });
    if (!st.present || !st.canvas || st.source !== "insight" || !st.confidence) fail(`insight preview failed: ${JSON.stringify(st)}`);
    if ((await workspaceCount()) !== 0) fail("previewing must not save");
    await shot("03-insight-preview");
  }

  // collapse / expand the side panel keeps the preview
  {
    await page.getByRole("button", { name: "收合預覽面板" }).click();
    await page.waitForTimeout(300);
    const collapsed = await panelState();
    const expandBtn = await page.getByRole("button", { name: "展開預覽面板" }).count();
    await page.getByRole("button", { name: "展開預覽面板" }).click();
    await page.waitForTimeout(600);
    const expanded = await panelState();
    step("panel.collapse-expand", { collapsedPresent: collapsed.present, expandButton: expandBtn, expandedPresent: expanded.present, canvas: expanded.canvas });
    if (collapsed.present || expandBtn !== 1 || !expanded.present) fail("collapse/expand broken");
  }

  // ---- Explore: option filtering per type, six chart types --------------
  await gotoPage("Explore");
  await pick("圖表類型", "box");
  step("builder.box.options", { x: await labelledOptions("X 軸"), y: await labelledOptions("Y 軸") });
  await pick("圖表類型", "heatmap");
  step("builder.heatmap.options", { x: await labelledOptions("X 軸"), y: await labelledOptions("Y 軸") });
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
  const saveAfter = new Set(["bar", "scatter"]);
  for (const m of manual) {
    const { chart, ...state } = await generateManual(m);
    const info = await inspectChart(chart);
    step(`manual.${m.name}`, { ...state, ...info, workspaceCount: await workspaceCount() });
    if (state.source !== "explore") fail(`explore preview source ${state.source}`);
    await shot(`10-manual-${m.name}`, panel());
    const tip = await probeTooltip(chart);
    step(`manual.${m.name}.tooltip`, tip);
    if (saveAfter.has(m.name)) {
      const before = await workspaceCount();
      await page.locator("[data-preview-save]").click();
      await page.waitForTimeout(200);
      const after = await workspaceCount();
      const st = await panelState();
      step(`manual.${m.name}.save`, { before, after, saveLabel: st.saveLabel, saveDisabled: st.saveDisabled });
      if (after !== before + 1 || !st.saveDisabled) fail(`save did not add exactly one chart (${before}→${after})`);
    }
  }
  await shot("11-explore-page");
  // state survives navigation: the explore form keeps its last selection
  {
    await gotoPage("Overview");
    await gotoPage("Explore");
    const type = await page.getByLabel("圖表類型").first().textContent();
    step("explore.state-kept-across-pages", { typeAfterRoundTrip: type?.trim() });
    if (!/heatmap/.test(type ?? "")) fail("explore form reset by navigation");
  }

  // ---- Workspace: only the two saved charts -----------------------------
  await gotoPage("Workspace");
  {
    const cards = page.locator("[data-workspace-card]");
    const n = await cards.count();
    const before = await panelSeq();
    await cards.first().click();
    await waitPreview(before);
    const st = await panelState();
    step("workspace.list-and-preview", { cards: n, ...st, current: await page.locator("[data-workspace-card][aria-current=true]").count() });
    if (n !== 2 || st.source !== "workspace") fail(`workspace expected 2 saved charts / workspace preview, got ${n} / ${st.source}`);
    await shot("12-workspace-page");

    // enlarge: a second, bigger instance in a dialog; Escape disposes it
    const inline = await inspectChart(panel().locator(CHART_SEL).first());
    await page.getByRole("button", { name: "Enlarge" }).click();
    const dialog = page.locator("[data-chart-dialog]");
    await dialog.waitFor({ timeout: 10000 });
    await page.waitForTimeout(1000);
    const big = await inspectChart(dialog.locator(CHART_SEL));
    step("workspace.enlarge", { inline: inline.canvasSize, dialog: big.canvasSize, bigger: Boolean(big.canvasSize && inline.canvasSize && big.canvasSize[0] > inline.canvasSize[0]) });
    await shot("13-enlarged-dialog");
    await page.keyboard.press("Escape");
    await dialog.waitFor({ state: "detached", timeout: 10000 });
    step("workspace.enlarge.closed", { charts: await page.locator(CHART_SEL).count() });

    // export: a real download of a PNG data URL
    const dataUrl = await panel().locator(CHART_SEL).first().evaluate((el) =>
      el.__echarts.getDataURL({ type: "png", pixelRatio: 2 }),
    );
    const [download] = await Promise.all([
      page.waitForEvent("download", { timeout: 10000 }),
      page.getByRole("button", { name: "Export" }).click(),
    ]);
    step("workspace.export", {
      dataUrlPrefix: dataUrl.slice(0, 22),
      dataUrlLength: dataUrl.length,
      filename: download.suggestedFilename(),
    });
    if (!dataUrl.startsWith("data:image/png") || dataUrl.length < 1000 || !download.suggestedFilename().endsWith(".png")) fail("export failed");

    // remove from the panel → one left, panel closes
    await page.getByRole("button", { name: "Remove" }).first().click();
    await page.waitForTimeout(300);
    step("workspace.remove", { cards: await page.locator("[data-workspace-card]").count(), panel: (await panelState()).present, count: await workspaceCount() });
    if ((await workspaceCount()) !== 1) fail("remove did not leave exactly one chart");
  }

  // ---- 422 path: box without y (backend rejects: y required) ------------
  {
    await gotoPage("Explore");
    await pick("圖表類型", "box");
    await pick("X 軸", "station");
    await page.getByRole("button", { name: "生成圖表" }).click();
    const err = page.locator("[data-manual-builder] [role=alert]").first();
    await err.waitFor({ timeout: 15000 }).catch(() => {});
    const errorShown = await page.locator("[data-manual-builder] [role=alert] li").filter({ hasText: /y/i }).count();
    step("builder.422", { errorShown: errorShown > 0 });
    await shot("14-builder-422");
  }

  // ---- layout: no page scroll, stable height ----------------------------
  {
    const samples = [];
    for (let i = 0; i < 10; i++) {
      samples.push((await layoutProbe()).scrollHeight);
      await page.waitForTimeout(100);
    }
    const probe = await layoutProbe();
    step("layout.height-stable", { ...probe, samples, stable: new Set(samples).size === 1 && probe.scrollHeight === probe.innerHeight && probe.scrollWidth === probe.innerWidth });
    if (new Set(samples).size !== 1 || probe.scrollHeight !== probe.innerHeight) fail("document scrolls or height drifts");
  }

  // ---- narrow viewport: top bar + bottom drawer -------------------------
  {
    await page.setViewportSize({ width: 800, height: 1000 });
    await page.waitForTimeout(500);
    await gotoPage("Explore");
    const before = await panelSeq();
    await pick("圖表類型", "histogram");
    await pick("X 軸", "pm25");
    await page.getByRole("button", { name: "生成圖表" }).click();
    await waitPreview(before);
    const st = await panelState();
    const paneBox = await panel().boundingBox();
    const contentBox = await main().boundingBox();
    const probe = await layoutProbe();
    step("layout.narrow", {
      sidebar: await page.locator("[data-sidebar]").first().getAttribute("data-sidebar"),
      variant: st.variant,
      open: st.open,
      stacked: Boolean(paneBox && contentBox && paneBox.y >= contentBox.y + contentBox.height - 2),
      noHorizontalScroll: probe.scrollWidth === probe.innerWidth,
    });
    await shot("15-narrow-drawer");
    await page.getByRole("button", { name: "收合預覽面板" }).click();
    await page.waitForTimeout(300);
    step("layout.narrow.collapsed", { open: (await panelState()).open, height: (await panel().boundingBox())?.height });
    await page.setViewportSize({ width: 1400, height: 1000 });
    await page.waitForTimeout(500);
  }

  // ---- outliers: histogram over a sentinel-laden column -> display_range --
  await upload("outliers.csv");
  {
    step("outliers.switched", { panelPresent: (await panelState()).present, workspaceCount: await workspaceCount() });
    if ((await panelState()).present) fail("preview must clear on dataset switch");
    const { chart, ...state } = await generateManual({ type: "histogram", x: "temperature" });
    const note = await panel().getByText(/顯示範圍/).count();
    step("outliers.histogram.display_range", { title: state.title, displayRangeNote: note > 0, canvas: (await inspectChart(chart)).canvas });
    await shot("20-outliers-histogram", panel());
  }

  // switching back restores that dataset's workspace
  {
    await page.locator("[data-dataset-list] a", { hasText: "air_quality.csv" }).first().click();
    await waitProfile("air_quality.csv");
    await page.waitForTimeout(300);
    step("workspace.per-dataset", { hash: await currentHash(), workspaceCount: await workspaceCount(), panelPresent: (await panelState()).present });
    if ((await workspaceCount()) !== 1) fail("workspace of air_quality was lost on switch");
  }

  // ---- reload keeps the route (the workspace is in-memory by design and
  // starts empty after a reload) ---------------------------------------
  {
    await gotoPage("Workspace");
    const hash = await currentHash();
    await page.reload({ waitUntil: "networkidle" });
    await waitProfile("air_quality.csv");
    step("route.reload", { hashBefore: hash, hashAfter: await currentHash(), activePage: await main().getAttribute("data-active-page"), currentDataset: await page.locator("[data-dataset-list] a[aria-current=true]").count(), workspaceAfterReload: await workspaceCount() });
    if ((await currentHash()) !== hash || (await main().getAttribute("data-active-page")) !== "workspace") fail("reload lost the route");
  }


  // ---- tiny_dataset: confidence warnings on recommendation cards --------
  await upload("tiny_dataset.csv");
  await gotoPage("Insights");
  {
    const chips = await page.locator("[id^=rec-card-] li").count();
    const topEmptyNotice = await page.getByText(/沒有圖表在資料中展現足夠強的證據/).count();
    step("tiny.warnings", {
      recCards: await page.locator("[id^=rec-card-]").count(),
      warningChips: chips,
      topEmptyNotice: topEmptyNotice > 0,
      panelCleared: !(await panelState()).present,
    });
    await shot("30-tiny-warnings");
  }

  // ---- sales_basic: derived-column demotion (stage 13) -------------------
  await upload("sales_basic.csv");
  await gotoPage("Insights");
  {
    const topEmptyNotice = (await page.getByText(/沒有圖表在資料中展現足夠強的證據/).count()) > 0;
    const topCards = await page.locator("[id^=rec-card-][data-tier=top]").count();
    const derivedDisclosed = (await page.getByText(/sales appears to be computed as/).count()) > 0;
    const definitionalCards = await page.getByText(/this relationship is definitional/).count();
    step("sales_basic.derived", { recCards: await page.locator("[id^=rec-card-]").count(), topEmptyNotice, topCards, derivedDisclosed, definitionalCards });
    await shot("31-sales-basic-derived");
    if (!topEmptyNotice || topCards !== 0 || !derivedDisclosed) fail(`sales_basic derived check failed: notice=${topEmptyNotice} topCards=${topCards} disclosed=${derivedDisclosed}`);
  }

  // ---- hour_like: near-duplicate suppression (stage 14) -------------------
  await upload("hour_like.csv");
  await gotoPage("Insights");
  {
    const cards = page.locator("[id^=rec-card-]");
    const recCards = await cards.count();
    const variableLines = await cards.locator("text=/x=|y=/").allTextContents();
    const atempCards = variableLines.filter((t) => /(^|[^a-z_])atemp([^a-z_]|$)/.test(t)).length;
    const pairCards = variableLines.filter((t) => /x=temp · y=atemp|x=atemp · y=temp/.test(t)).length;
    const topTitles = await page.locator("[id^=rec-card-][data-tier=top]").allTextContents();
    const topIsPair = topTitles.some((t) => /atemp vs temp|temp vs atemp/.test(t));
    const nearDupDisclosed = (await page.getByText(/atemp is a near-duplicate of temp/).count()) > 0;
    const derivedDisclosed = (await page.getByText(/cnt appears to be computed as casual \+ registered/).count()) > 0;
    step("hour_like.near_duplicate", { recCards, atempCards, pairCards, topIsPair, nearDupDisclosed, derivedDisclosed });
    await shot("32-hour-like-near-duplicate");
    if (recCards === 0 || atempCards !== 0 || pairCards !== 0 || topIsPair || !nearDupDisclosed) {
      fail(`hour_like near-duplicate check failed: cards=${recCards} atemp=${atempCards} pair=${pairCards} topIsPair=${topIsPair} disclosed=${nearDupDisclosed}`);
    }
  }

  // ---- render payload shapes (must match the pre-16.2 UI) ----------------
  {
    const shapes = summary.renderPayloads.map((s) => ({ type: s.type, x: s.x ?? null, y: s.y ?? null, group_by: s.group_by ?? null, aggregation: s.aggregation ?? null }));
    const barNoY = shapes.find((s) => s.type === "bar" && s.x === "station" && s.y === "pm25");
    const heat = shapes.find((s) => s.type === "heatmap");
    const scatter = shapes.find((s) => s.type === "scatter" && s.x === "temperature");
    step("render.payloads", {
      count: shapes.length,
      barStationPm25: barNoY,
      heatmap: heat,
      scatter,
      keysAlwaysPresent: summary.renderPayloads.every((s) => "title" in s && "type" in s),
    });
    if (!heat || heat.x !== null || heat.y !== null || heat.aggregation !== null) fail("heatmap payload shape changed");
    if (!scatter || scatter.aggregation !== null) fail("scatter payload shape changed");
    if (!barNoY || barNoY.aggregation !== "mean") fail("bar payload aggregation prefill changed");
  }
  step("air_quality.id", { airId });
} catch (e) {
  summary.fatal = String(e && e.stack ? e.stack : e);
  await shot("99-fatal").catch(() => {});
} finally {
  await browser.close();
}

summary.ok = !summary.fatal && summary.pageErrors.length === 0;
console.log("[e2e] SUMMARY " + JSON.stringify(summary, null, 2));
process.exit(summary.ok ? 0 : 1);
