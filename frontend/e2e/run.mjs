// End-to-end smoke run against the REAL backend (port 8100) with REAL files
// from dataset/. No mocks. Produces screenshots under e2e/screenshots/<tag>/
// and prints a structured JSON summary on stdout.
//
//   npm run e2e -- --tag baseline
//
// Requires: backend running on 8100 serving frontend/dist, Playwright's
// Chromium installed (npx playwright install chromium). See e2e/README.md.
//
// Layout (stage 15): tabs 總覽 / 推薦圖表 / 手動建圖; the two chart pages are
// split with a shared RStudio-style plots pane on the right
// ([data-plots-pane], data-count / data-current attributes).

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
const PANE_SEL = "[data-plots-pane]";

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

async function gotoTab(name) {
  const tab = page.getByRole("tab", { name });
  await tab.click();
  await page.waitForFunction(
    (n) => document.querySelector(`[role=tab][aria-selected=true]`)?.textContent?.trim() === n,
    name,
    { timeout: 5000 },
  );
  await page.waitForTimeout(150);
}

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

/** Upload through the header button's file input; the app then switches to
 *  the 推薦圖表 page. Profile readiness is exposed as data-profile=loaded. */
async function upload(file) {
  const before = await page.locator("[id^=rec-card-]").count();
  await page.locator("input[type=file]").first().setInputFiles(dataset(file));
  // The profile request can hit a backend race right after upload (fixed in
  // stage 9b, kept as a guard); re-selecting the dataset re-issues it.
  // Retries are counted, never hidden.
  for (let attempt = 0; ; attempt++) {
    const ok = await page
      .waitForSelector("[data-profile=loaded]", { timeout: 15000 })
      .then(() => true, () => false);
    if (ok) break;
    if (attempt >= 2) throw new Error(`profile never loaded for ${file} after ${attempt} retries`);
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

const pane = () => page.locator(PANE_SEL).first();
async function paneState() {
  const p = pane();
  if ((await p.count()) === 0) return { count: 0, current: 0, counter: null };
  return p.evaluate((el) => ({
    count: Number(el.dataset.count),
    current: Number(el.dataset.current),
    counter: el.querySelector("[data-plots-counter]")?.textContent?.trim() ?? null,
  }));
}
async function chartCount() {
  return (await paneState()).count;
}

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
    return { supported: true, shown: Boolean(tip), text: tip ? tip.textContent.trim().slice(0, 120) : null };
  });
}

async function waitForCount(n) {
  await page.waitForFunction(
    ([sel, want]) => Number(document.querySelector(sel)?.dataset.count) === want,
    [PANE_SEL, n],
    { timeout: 60000 },
  );
  await page.waitForTimeout(1200); // let the chart library finish drawing
}

/** Manual builder page: pick options, generate, and return the plot that is
 *  now current in the pane (the newest chart becomes current). */
async function generateManual({ type, x, y, group }) {
  await gotoTab("手動建圖");
  const before = await chartCount();
  await pick("圖表類型", type);
  if (x) await pick("X 軸", x);
  if (y) await pick("Y 軸", y);
  if (group) await pick("分組", group);
  await page.getByRole("button", { name: "生成圖表" }).click();
  await waitForCount(before + 1);
  const chart = pane().locator(CHART_SEL).first();
  return { chart, ...(await paneState()) };
}

try {
  await page.goto(base, { waitUntil: "networkidle" });
  const tabsBefore = await page.getByRole("tab").evaluateAll((els) =>
    els.map((e) => ({ name: e.textContent.trim(), disabled: e.hasAttribute("disabled") || e.getAttribute("aria-disabled") === "true" })),
  );
  step("open", { title: await page.title(), tabsBefore });

  // ---- air_quality: overview, recommendations, six chart types ----------
  await upload("air_quality.csv");
  const activeTab = await page.locator("[role=tab][aria-selected=true]").textContent();
  const recCards = await page.locator("[id^=rec-card-]").count();
  const tiers = {};
  for (const t of ["推薦重點", "次要", "探索"]) {
    tiers[t] = (await page.getByRole("heading", { level: 3, name: new RegExp(`^${t}`) }).count()) > 0;
  }
  const split = await page.locator("[data-split]").first().getAttribute("data-split");
  step("air_quality.uploaded", {
    activeTab: activeTab?.trim(),
    recCards,
    tiers,
    split,
    paneVisible: (await pane().count()) > 0,
    generateButtons: await page.getByRole("button", { name: "Generate" }).count(),
  });
  await shot("01-recommend-page");

  await gotoTab("總覽");
  step("air_quality.overview", { columnsInOverview: await page.locator("table tbody tr").count() });
  await shot("00-overview-page");
  await gotoTab("推薦圖表");

  // recommendation-driven generation (first card) -> shows in the right pane
  {
    const before = await chartCount();
    await page.getByRole("button", { name: "Generate" }).first().click();
    await waitForCount(before + 1);
    const info = await inspectChart(pane().locator(CHART_SEL).first());
    const paneBox = await pane().boundingBox();
    const contentBox = await page.locator("[data-page-content]").first().boundingBox();
    step("air_quality.generate-from-recommendation", {
      ...(await paneState()),
      ...info,
      paneRightOfContent: Boolean(paneBox && contentBox && paneBox.x >= contentBox.x + contentBox.width - 1),
    });
    await shot("02-rec-chart-in-pane");
  }

  // manual builder: option filtering per type
  await gotoTab("手動建圖");
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
  // the pane is shared: the chart generated on the recommendations page is here too
  step("builder.pane-shared", await paneState());

  const manual = [
    { name: "line", type: "line", x: "timestamp", y: "temperature" },
    { name: "bar", type: "bar", x: "station", y: "pm25" },
    { name: "scatter", type: "scatter", x: "temperature", y: "humidity" },
    { name: "histogram", type: "histogram", x: "pm25" },
    { name: "box", type: "box", x: "station", y: "pm25" },
    { name: "heatmap", type: "heatmap" },
  ];
  for (const m of manual) {
    const { chart, ...state } = await generateManual(m);
    const info = await inspectChart(chart);
    step(`manual.${m.name}`, { ...state, ...info });
    await shot(`10-manual-${m.name}`, pane());
    const tip = await probeTooltip(chart);
    step(`manual.${m.name}.tooltip`, tip);
    if (tip.shown) await shot(`10-manual-${m.name}-tooltip`, pane());
  }
  await shot("11-manual-page-all");

  // plots pane navigation: ◀ ▶, thumbnail strip, remove
  {
    const total = await chartCount();
    await page.getByRole("button", { name: "上一張" }).click();
    const afterPrev = await paneState();
    await page.getByRole("button", { name: "下一張" }).click();
    const afterNext = await paneState();
    await page.locator("[data-plots-strip] button").first().click();
    await page.waitForTimeout(400);
    const afterThumb = await paneState();
    const firstTitle = await pane().locator("[data-plots-strip] button").first().textContent();
    await shot("12-pane-navigation", pane());
    await page.getByRole("button", { name: "移除" }).click();
    await waitForCount(total - 1);
    const afterRemove = await paneState();
    step("pane.navigation", {
      total,
      afterPrev,
      afterNext,
      afterThumb,
      firstThumb: firstTitle?.trim().slice(0, 60),
      afterRemove,
      ok:
        afterPrev.current === total - 1 &&
        afterNext.current === total &&
        afterThumb.current === 1 &&
        afterRemove.count === total - 1 &&
        afterRemove.current === 1,
    });
  }

  // enlarged view: the "放大" button opens a dialog that draws the same
  // RenderResult in a second, bigger ECharts instance; Escape closes it and
  // the instance is disposed with the dialog
  {
    const inline = await inspectChart(pane().locator(CHART_SEL).first());
    await page.getByRole("button", { name: "放大" }).click();
    const dialog = page.locator("[data-chart-dialog]");
    await dialog.waitFor({ timeout: 10000 });
    await page.waitForTimeout(1200);
    const info = await inspectChart(dialog.locator(CHART_SEL));
    step("pane.enlarge", {
      dialogOpen: true,
      inlineCanvas: inline.canvasSize,
      dialogCanvas: info.canvasSize,
      bigger: Boolean(info.canvasSize && inline.canvasSize && info.canvasSize[0] > inline.canvasSize[0]),
      seriesTypes: info.seriesTypes,
    });
    await shot("13-enlarged-dialog");
    await page.keyboard.press("Escape");
    await dialog.waitFor({ state: "detached", timeout: 10000 });
    step("pane.enlarge.closed", { ...(await paneState()), inlineCharts: await page.locator(CHART_SEL).count() });
  }

  // 422 path: box without y (backend rejects: y required)
  {
    await gotoTab("手動建圖");
    await pick("圖表類型", "box");
    await pick("X 軸", "station");
    await page.getByRole("button", { name: "生成圖表" }).click();
    const err = page.locator("text=/y|Y 軸|required|必須/i").first();
    await err.waitFor({ timeout: 15000 }).catch(() => {});
    const errorShown = await page.locator("ul li, [role=alert]").filter({ hasText: /y/i }).count();
    step("builder.422", { errorShown: errorShown > 0 });
    await shot("14-builder-422");
  }

  // narrow viewport: the split stacks vertically (plots pane below)
  {
    await page.setViewportSize({ width: 800, height: 1100 });
    await page.waitForTimeout(600);
    const split = await page.locator("[data-split]").first().getAttribute("data-split");
    const paneBox = await pane().boundingBox();
    const contentBox = await page.locator("[data-page-content]").first().boundingBox();
    step("layout.narrow", {
      split,
      stacked: Boolean(paneBox && contentBox && paneBox.y >= contentBox.y + contentBox.height - 1),
      chartsStillThere: await chartCount(),
    });
    await shot("15-narrow-stacked");
    await page.setViewportSize({ width: 1400, height: 1000 });
    await page.waitForTimeout(600);
  }

  // ---- outliers: histogram over a sentinel-laden column -> display_range --
  await upload("outliers.csv");
  {
    const { chart, ...state } = await generateManual({ type: "histogram", x: "temperature" });
    const note = await page.getByText(/顯示範圍/).count();
    step("outliers.histogram.display_range", { ...state, displayRangeNote: note > 0, canvas: (await inspectChart(chart)).canvas });
    await shot("20-outliers-histogram", pane());
  }

  // ---- tiny_dataset: confidence warnings on recommendation cards --------
  await upload("tiny_dataset.csv");
  {
    const chips = await page.locator("[id^=rec-card-] li").count();
    const topEmptyNotice = await page.getByText(/沒有圖表在資料中展現足夠強的證據/).count();
    step("tiny.warnings", {
      recCards: await page.locator("[id^=rec-card-]").count(),
      warningChips: chips,
      topEmptyNotice: topEmptyNotice > 0,
      paneCleared: await chartCount(),
    });
    await shot("30-tiny-warnings");
  }

  // ---- sales_basic: derived-column demotion (stage 13) -------------------
  await upload("sales_basic.csv");
  {
    const topEmptyNotice = (await page.getByText(/沒有圖表在資料中展現足夠強的證據/).count()) > 0;
    const topCards = await page
      .locator("xpath=//h3[normalize-space()='推薦重點']/..//*[starts-with(@id,'rec-card-')]")
      .count();
    const derivedDisclosed = (await page.getByText(/sales appears to be computed as/).count()) > 0;
    const definitionalCards = await page.getByText(/this relationship is definitional/).count();
    step("sales_basic.derived", {
      recCards: await page.locator("[id^=rec-card-]").count(),
      topEmptyNotice,
      topCards,
      derivedDisclosed,
      definitionalCards,
    });
    await shot("31-sales-basic-derived");
    if (!topEmptyNotice || topCards !== 0 || !derivedDisclosed) {
      throw new Error(`sales_basic derived check failed: notice=${topEmptyNotice} topCards=${topCards} disclosed=${derivedDisclosed}`);
    }
  }

  // ---- hour_like: near-duplicate suppression (stage 14) -------------------
  // atemp is a near-duplicate of temp: no card may use atemp, the pair
  // chart must not exist, and the response discloses the suppression.
  // Generated by dataset/syn/hour_like.py (backend/.venv python is enough).
  await upload("hour_like.csv");
  {
    const cards = page.locator("[id^=rec-card-]");
    const recCards = await cards.count();
    const variableLines = await cards.locator("text=/x=|y=/").allTextContents();
    const atempCards = variableLines.filter((t) => /(^|[^a-z_])atemp([^a-z_]|$)/.test(t)).length;
    const pairCards = variableLines.filter((t) => /x=temp · y=atemp|x=atemp · y=temp/.test(t)).length;
    const topTitles = await page
      .locator("xpath=//h3[normalize-space()='推薦重點']/..//*[starts-with(@id,'rec-card-')]")
      .allTextContents();
    const topIsPair = topTitles.some((t) => /atemp vs temp|temp vs atemp/.test(t));
    const nearDupDisclosed = (await page.getByText(/atemp is a near-duplicate of temp/).count()) > 0;
    const derivedDisclosed = (await page.getByText(/cnt appears to be computed as casual \+ registered/).count()) > 0;
    step("hour_like.near_duplicate", { recCards, atempCards, pairCards, topIsPair, nearDupDisclosed, derivedDisclosed });
    await shot("32-hour-like-near-duplicate");
    if (recCards === 0 || atempCards !== 0 || pairCards !== 0 || topIsPair || !nearDupDisclosed) {
      throw new Error(
        `hour_like near-duplicate check failed: cards=${recCards} atemp=${atempCards} pair=${pairCards} topIsPair=${topIsPair} disclosed=${nearDupDisclosed}`,
      );
    }
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
