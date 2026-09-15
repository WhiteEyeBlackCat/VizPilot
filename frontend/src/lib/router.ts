import { useCallback, useEffect, useState } from "react";

import { PAGES, type Page } from "../store";

/** `#/d/<dataset_id>/<page>`; anything else is the empty state. A reload
 *  lands on the same dataset and page. */
export interface Route {
  datasetId: string | null;
  page: Page;
}

const PAGE_SET = new Set<string>(PAGES.map((p) => p.page));

export function parseHash(hash: string): Route {
  const m = /^#\/d\/([^/]+)(?:\/([a-z]+))?/.exec(hash);
  if (!m) return { datasetId: null, page: "overview" };
  const page = m[2] && PAGE_SET.has(m[2]) ? (m[2] as Page) : "overview";
  return { datasetId: decodeURIComponent(m[1]), page };
}

export function buildHash(route: Route): string {
  return route.datasetId ? `#/d/${encodeURIComponent(route.datasetId)}/${route.page}` : "#/";
}

export function useHashRoute(): [Route, (route: Route) => void] {
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash));
  useEffect(() => {
    const onChange = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  const navigate = useCallback((next: Route) => {
    const hash = buildHash(next);
    if (window.location.hash === hash) return;
    window.location.hash = hash; // hashchange updates the state
  }, []);
  return [route, navigate];
}
