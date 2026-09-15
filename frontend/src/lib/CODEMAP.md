---
mode: learning
generated_at: 2026-09-15
---

> Small framework-free helpers: hash router, DOM hooks, `cn`.

## Task Guide

| Task | Domain | Target | Also Check |
|---|---|---|---|
| Understand `#/<dataset>/<page>` routing and canonical hashes | Router | `router.ts` | `../store.ts` (`PAGES`), `../App.tsx` |
| Media query / element size hooks | Hooks | `hooks.ts` | — |

## Files

| File | Domain | Deps | Function |
|---|---|---|---|
| `router.ts` | Router | `← ../store.ts \| → ../App.tsx, ../components/Sidebar.tsx` | `parseHash`, `buildHash`, `isCanonicalHash`, `useHashRoute`. |
| `hooks.ts` | Hooks | `→ ../App.tsx, ../components/PreviewPanel.tsx` | `useMediaQuery`, `useElementSize`. |
| `utils.ts` | Utils | `→ many` | `cn` (clsx + tailwind-merge). |
