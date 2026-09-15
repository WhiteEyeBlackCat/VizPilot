---
mode: learning
generated_at: 2026-09-15
---

> Vite + React 18 + TypeScript + Tailwind 3.4 + shadcn/ui frontend; built SPA is served by the backend at `/`.

## Task Guide

| Task | Domain | Target | Also Check |
|---|---|---|---|
| Any UI/chart source question | Frontend | `src/CODEMAP.md` | — |
| Run e2e / record fixtures | E2E | `e2e/CODEMAP.md` | — |
| Build/test commands | Frontend | `package.json` scripts | `../README.md` |
| Path alias `@/`, proxy `/api` → 8100 | Frontend | `vite.config.ts`, `tsconfig*.json` | — |

## Subdirectories

| Dir | Domain | Depends On | Purpose |
|---|---|---|---|
| `src/` | Frontend | react, echarts, radix, tailwind | App source. |
| `e2e/` | E2E | playwright, running backend | Smoke tests + fixture capture. |

## Files

| File | Domain | Function |
|---|---|---|
| `package.json` | Frontend | Scripts: `dev`, `build` (tsc + vite), `typecheck`, `test` (vitest), `e2e`, `capture-fixtures`. |
| `vite.config.ts` | Frontend | Alias + dev proxy. |
| `tailwind.config.*`, `postcss.config.*`, `components.json` | Frontend | Tailwind/shadcn config. |
| `index.html`, `tsconfig*.json` | Frontend | Entry + TS config. |
