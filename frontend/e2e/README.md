# Frontend E2E (real backend, real files, headless Chromium)

`npm run e2e -- --tag <name>` drives the built SPA at `http://localhost:8100/`
through the real API: uploads files from `dataset/`, reads the overview and
recommendations, generates all six chart types through the manual builder,
checks the `display_range` note (outliers.csv) and confidence warning chips
(tiny_dataset.csv). Screenshots land in `e2e/screenshots/<tag>/` (git-ignored)
and a JSON summary is printed at the end (`[e2e] SUMMARY …`); exit code is
non-zero on a fatal step or any page error.

Nothing is mocked. The run needs:

1. **Backend on 8100** serving `frontend/dist` (`npm run build` first, then
   `cd backend && .venv/bin/uvicorn app.main:create_app --factory --port 8100`).
   Override with `E2E_BASE_URL`.
2. **Node 20** — on this host: `export PATH=/data-10/users/re6141011/opt/node20/bin:$PATH`.
3. **Playwright Chromium** — `npx playwright install chromium` (downloads to
   `~/.cache/ms-playwright`, no root needed).
4. **Three X11 libraries** the host lacks (`libXcomposite.so.1`,
   `libXdamage.so.1`, `libXrandr.so.2`). Without root, download the Ubuntu
   debs and unpack them into `e2e/.xlibs/` (git-ignored); `run.mjs` prepends
   that directory to `LD_LIBRARY_PATH` before launching Chromium:

   ```bash
   mkdir -p /tmp/xlibs && cd /tmp/xlibs
   apt-get download libxcomposite1 libxdamage1 libxrandr2
   for d in *.deb; do dpkg -x "$d" root; done
   cp root/usr/lib/x86_64-linux-gnu/*.so* <repo>/frontend/e2e/.xlibs/
   ```

The selectors are written to work with both the native `<select>` form
controls and Radix/shadcn `Select` triggers (matched by accessible label), so
the same script validates the UI before and after the shadcn migration.
Compare `screenshots/baseline/` (pre-migration) against a later tag.
