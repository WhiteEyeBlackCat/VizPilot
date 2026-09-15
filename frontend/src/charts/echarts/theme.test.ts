import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { COLORS, DIVERGING, PALETTE } from "./theme";

// The CSS design tokens are the single source of colour for the UI; the
// ECharts theme cannot read CSS variables in SSR, so it mirrors the hex
// values. This test pins the two together: change one, and this fails.
const css = readFileSync(resolve(dirname(fileURLToPath(import.meta.url)), "../../index.css"), "utf8");

/** A hex token (`--danger: #f87171`) or an HSL triplet (`--card: 222 18% 11%`)
 *  resolved to lowercase hex; `--surface: hsl(var(--card))` follows the alias. */
function token(name: string): string {
  const hex = css.match(new RegExp(`--${name}:\\s*(#[0-9a-fA-F]{6})`));
  if (hex) return hex[1].toLowerCase();
  const alias = css.match(new RegExp(`--${name}:\\s*hsl\\(var\\(--([a-z-]+)\\)\\)`));
  if (alias) return token(alias[1]);
  const hsl = css.match(new RegExp(`--${name}:\\s*(\\d+)\\s+(\\d+)%\\s+(\\d+)%`));
  if (!hsl) throw new Error(`token --${name} not found in index.css`);
  return hslToHex(Number(hsl[1]), Number(hsl[2]) / 100, Number(hsl[3]) / 100);
}

function hslToHex(h: number, s: number, l: number): string {
  const k = (n: number) => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  return "#" + [f(0), f(8), f(4)].map((v) => Math.round(v * 255).toString(16).padStart(2, "0")).join("");
}

// WCAG 2.x relative luminance / contrast ratio
function luminance(hex: string): number {
  const c = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  const lin = c.map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
  return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
}
export const contrast = (a: string, b: string): number => {
  const [l1, l2] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (l1 + 0.05) / (l2 + 0.05);
};

describe("theme tokens mirror src/index.css", () => {
  it("chart palette equals --chart-1..8", () => {
    expect(PALETTE.map((c) => c.toLowerCase())).toEqual(PALETTE.map((_, i) => token(`chart-${i + 1}`)));
  });

  it("surface, elevated, grid and danger colours equal the CSS tokens", () => {
    expect(COLORS.surface).toBe(token("surface"));
    expect(COLORS.elevated).toBe(token("elevated"));
    expect(COLORS.splitLine).toBe(token("border-subtle"));
    expect(COLORS.outlier).toBe(token("danger"));
    expect(COLORS.cellBorder).toBe(token("surface"));
  });

  it("chart text colours equal --foreground / --muted-foreground", () => {
    expect(COLORS.text).toBe(token("foreground"));
    expect(COLORS.muted).toBe(token("muted-foreground"));
    expect(COLORS.surface).toBe(token("card"));
    expect(COLORS.elevated).toBe(token("popover"));
  });

  it("every palette colour and both diverging ends read on the surface (≥ 3:1)", () => {
    for (const c of PALETTE) expect(contrast(c, COLORS.surface)).toBeGreaterThanOrEqual(3);
    expect(contrast(DIVERGING[0], COLORS.surface)).toBeGreaterThanOrEqual(3);
    expect(contrast(DIVERGING[2], COLORS.surface)).toBeGreaterThanOrEqual(3);
  });

  it("chart text keeps AA contrast on the surface and on the diverging ends", () => {
    expect(contrast(COLORS.text, COLORS.surface)).toBeGreaterThanOrEqual(4.5);
    expect(contrast(COLORS.muted, COLORS.surface)).toBeGreaterThanOrEqual(4.5);
    expect(contrast(COLORS.text, COLORS.elevated)).toBeGreaterThanOrEqual(4.5);
    // heatmap cell labels sit on the ends of the scale
    expect(contrast(COLORS.text, DIVERGING[0])).toBeGreaterThanOrEqual(3);
    expect(contrast(COLORS.text, DIVERGING[2])).toBeGreaterThanOrEqual(3);
  });
});
