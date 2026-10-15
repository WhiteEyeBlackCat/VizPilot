// ECharts renders the string returned by a custom tooltip `formatter` as
// HTML, unescaped. Column names, category values, group / series names and
// node labels all come from the user's data, so every dynamic text spliced
// into tooltip HTML goes through `esc`. Do not escape `item.marker` (ECharts
// generates it) and do not use this for canvas text (axis / series labels).

const ENTITIES: Record<string, string> = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

export function escapeHtml(value: unknown): string {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ENTITIES[c]);
}

export const esc = escapeHtml;
