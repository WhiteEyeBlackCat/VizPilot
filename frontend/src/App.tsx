import { PanelRightOpen, Plus } from "lucide-react";
import { useCallback, useEffect, useReducer, useRef, useState } from "react";

import { api } from "./api";
import { NewDatasetDialog } from "./components/NewDatasetDialog";
import { PreviewPanel } from "./components/PreviewPanel";
import { Sidebar } from "./components/Sidebar";
import { ExplorePage } from "./pages/ExplorePage";
import { InsightsPage } from "./pages/InsightsPage";
import { OverviewPage } from "./pages/OverviewPage";
import { WorkspacePage } from "./pages/WorkspacePage";
import {
  initialState,
  isSaved,
  reducer,
  workspaceOf,
  type ExploreField,
  type FieldUpdate,
  type Page,
  type Preview,
  type SavedChart,
} from "./store";
import type { ChartSpec, DatasetMeta, Recommendation } from "./types";
import { Button } from "@/components/ui/button";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup, type GroupHandle, type Layout } from "@/components/ui/resizable";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useMediaQuery } from "@/lib/hooks";
import { isCanonicalHash, useHashRoute } from "@/lib/router";

const PAGE_ORDER: Page[] = ["overview", "insights", "explore", "workspace"];
const DEFAULT_LAYOUT: Layout = { content: 55, preview: 45 };

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [route, navigate] = useHashRoute();
  const [newOpen, setNewOpen] = useState(false);
  // three layouts: ≥1280 sidebar + content + resizable preview column;
  // 1024–1279 sidebar + content with the preview overlaying the right edge
  // (the content keeps its width); <1024 top bar + bottom drawer
  const wide = useMediaQuery("(min-width: 1024px)");
  const split = useMediaQuery("(min-width: 1280px)");
  const saveSeq = useRef(0);
  // remembered across collapse / expand (the preview panel re-mounts inside
  // the ALWAYS-mounted group) and page switches; v4 has no autoSaveId
  const layoutRef = useRef<Layout>(DEFAULT_LAYOUT);
  const groupRef = useRef<GroupHandle>(null);
  // while the preview panel is (re)joining the group, the library reports
  // its own provisional layout (an even split); those events must not
  // overwrite the remembered width that is about to be restored
  const restoringRef = useRef(false);

  const { datasets, meta, profile, recs, aiPending, preview, panelOpen } = state;
  const datasetId = meta?.dataset_id ?? null;
  const workspace = workspaceOf(state, datasetId);

  useEffect(() => {
    api
      .listDatasets()
      .then((list) => dispatch({ type: "DATASETS_LOADED", datasets: list }))
      .catch(() => dispatch({ type: "DATASETS_LOADED", datasets: [] }));
  }, []);

  /** Load a dataset: profile plus the two-phase recommendations. Every
   *  response is tagged with its dataset id; the reducer drops stale ones. */
  const selectDataset = useCallback((m: DatasetMeta) => {
    const id = m.dataset_id;
    dispatch({ type: "SELECT_DATASET", meta: m });
    api
      .profile(id)
      .then((p) => dispatch({ type: "PROFILE_LOADED", datasetId: id, profile: p }))
      .catch(() => {});
    // two-phase load: rules instantly, then the full (possibly LLM) result
    api
      .recommendations(id, false)
      .then((r) => dispatch({ type: "RECS_LOADED", datasetId: id, recs: r, final: false }))
      .catch(() => {});
    api
      .recommendations(id, true)
      .then((r) => dispatch({ type: "RECS_LOADED", datasetId: id, recs: r, final: true }))
      .catch(() => dispatch({ type: "AI_DONE", datasetId: id }));
  }, []);

  // the hash is the source of truth for the selected dataset; malformed
  // hashes are normalised (replace, no history entry) and an unknown id
  // falls back to the empty state
  useEffect(() => {
    if (!datasets) return;
    const raw = window.location.hash;
    if (!route.datasetId) {
      if (!isCanonicalHash(raw)) navigate({ datasetId: null, page: "overview" }, { replace: true });
      if (datasetId) dispatch({ type: "CLEAR_SELECTION" });
      return;
    }
    const m = datasets.find((d) => d.dataset_id === route.datasetId);
    if (!m) {
      navigate({ datasetId: null, page: "overview" }, { replace: true }); // unknown id in the hash
      return;
    }
    if (!isCanonicalHash(raw)) navigate(route, { replace: true }); // e.g. #/d/<id>/bogus → /overview
    if (route.datasetId !== datasetId) selectDataset(m);
  }, [datasets, route, datasetId, selectDataset, navigate]);

  const onUploaded = useCallback(
    (m: DatasetMeta) => {
      dispatch({ type: "DATASET_ADDED", meta: m });
      navigate({ datasetId: m.dataset_id, page: "overview" });
    },
    [navigate],
  );

  /** Render a spec and show it in the preview panel. Never touches the
   *  workspace. Errors propagate to the caller's error list. */
  const previewSpec = useCallback(
    async (spec: ChartSpec, extra: Omit<Preview, "result" | "seq">) => {
      const id = datasetId;
      if (!id) return;
      const result = await api.render(id, spec);
      dispatch({ type: "SET_PREVIEW", datasetId: id, preview: { result, ...extra } });
    },
    [datasetId],
  );

  const previewRecommendation = useCallback(
    (rec: Recommendation) => {
      const insightText = recs?.insights.find((i) => i.chart_priority === rec.spec.priority)?.text;
      return previewSpec(rec.spec, { source: "insight", rec, insightText });
    },
    [previewSpec, recs],
  );

  const previewManual = useCallback((spec: ChartSpec) => previewSpec(spec, { source: "explore" }), [previewSpec]);

  const previewSaved = useCallback(
    (chart: SavedChart) => {
      if (!datasetId) return;
      dispatch({
        type: "SET_PREVIEW",
        datasetId,
        preview: {
          result: chart.result,
          source: "workspace",
          rec: chart.rec,
          insightText: chart.insightText,
          savedKey: chart.key,
        },
      });
    },
    [datasetId],
  );

  const savePreview = useCallback(() => {
    saveSeq.current += 1;
    dispatch({ type: "SAVE_PREVIEW", key: `saved-${Date.now()}-${saveSeq.current}`, savedAt: new Date().toISOString() });
  }, []);

  const removeSaved = useCallback(
    (key: string) => {
      if (datasetId) dispatch({ type: "REMOVE_SAVED", datasetId, key });
    },
    [datasetId],
  );

  const onExploreField = useCallback(
    (field: ExploreField, value: FieldUpdate<string>) => dispatch({ type: "EXPLORE_FIELD", field, value }),
    [],
  );

  const page: Page = route.datasetId && datasetId ? route.page : "overview";
  const panelVisible = preview !== null && panelOpen;
  const splitOpen = split && panelVisible;
  const overlayOpen = wide && !split && panelVisible;
  const panelMode = !wide ? "drawer" : split ? "split" : "overlay";

  // the preview panel joins the persistent group: restore the remembered
  // width (the group itself never re-mounts, so the pages keep their state).
  // The library's provisional layout events are ignored until then.
  const wasSplitOpen = useRef(false);
  if (splitOpen && !wasSplitOpen.current) restoringRef.current = true;
  wasSplitOpen.current = splitOpen;
  useEffect(() => {
    if (!splitOpen) return;
    let second = 0;
    const first = requestAnimationFrame(() => {
      groupRef.current?.setLayout(layoutRef.current);
      second = requestAnimationFrame(() => {
        restoringRef.current = false;
      });
    });
    return () => {
      cancelAnimationFrame(first);
      cancelAnimationFrame(second);
      restoringRef.current = false;
    };
  }, [splitOpen]);

  // Esc closes the preview unless a dialog (enlarge / new dataset) is open —
  // those own the key
  useEffect(() => {
    if (!preview) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      // Radix dismisses an open dialog on Escape and marks the event handled
      if (e.defaultPrevented) return;
      if (document.querySelector("[role=dialog][data-state=open]")) return;
      dispatch({ type: "CLOSE_PREVIEW" });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [preview]);
  const selectedPriority =
    preview && preview.source !== "explore" && preview.rec ? (preview.rec.spec.priority ?? null) : null;
  const currentSavedKey =
    preview?.source === "workspace" ? (preview.savedKey ?? null) : null;

  const panel = preview && (
    <PreviewPanel
      preview={preview}
      saved={isSaved(state, preview.result.spec)}
      variant={wide ? "side" : "drawer"}
      mode={panelMode}
      open={panelOpen}
      onToggle={() => dispatch({ type: "SET_PANEL_OPEN", open: !panelOpen })}
      onClose={() => dispatch({ type: "CLOSE_PREVIEW" })}
      onSave={savePreview}
      onRemove={() => preview.savedKey && removeSaved(preview.savedKey)}
    />
  );

  // pages stay mounted (hidden) so their local state survives navigation
  const pages = meta && (
    <>
      {PAGE_ORDER.map((p) => (
        <section key={p} hidden={page !== p} aria-hidden={page !== p} className="px-6 py-5" data-page-section={p}>
          {p === "overview" && <OverviewPage meta={meta} profile={profile} warnings={recs ? (recs.warnings ?? []) : null} />}
          {p === "insights" && (
            <InsightsPage
              recs={recs}
              aiPending={aiPending}
              selectedPriority={selectedPriority}
              onPreview={previewRecommendation}
              exploratoryOpen={state.insightsExploratoryOpen}
              onExploratoryOpenChange={(open) => dispatch({ type: "SET_EXPLORATORY_OPEN", open })}
            />
          )}
          {p === "explore" && (
            <ExplorePage
              profile={profile}
              form={state.explore}
              onField={onExploreField}
              onGenerate={previewManual}
              lastSpec={state.lastExploreSpec}
            />
          )}
          {p === "workspace" && (
            <WorkspacePage
              charts={workspace}
              currentKey={currentSavedKey}
              onPreview={previewSaved}
              onRemove={removeSaved}
            />
          )}
        </section>
      ))}
    </>
  );

  const emptyState = (
    <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center" data-empty-state>
      <h1 className="text-xl font-semibold tracking-tight">VizPilot</h1>
      <p className="max-w-md text-sm text-muted-foreground">
        上傳一份 CSV、XLSX 或 Parquet。系統會建立 profile、計算證據並推薦圖表；再用 Explore
        自行建圖，把值得留下的圖 Save 到 Workspace。
      </p>
      <Button size="sm" onClick={() => setNewOpen(true)}>
        <Plus className="mr-1.5 h-3.5 w-3.5" />
        New Dataset
      </Button>
      {datasets === null && <p className="text-xs text-muted-foreground">載入資料集清單中…</p>}
    </div>
  );

  const main = (
    <main
      className="relative h-full min-h-0 overflow-y-auto"
      data-page-content
      data-profile={profile ? "loaded" : "none"}
      data-dataset={datasetId ?? ""}
      data-active-page={meta ? page : "none"}
      data-panel-mode={panelMode}
    >
      {meta ? pages : emptyState}
      {/* collapsed side panel: a slim edge control brings it back */}
      {wide && preview && !panelOpen && (
        <Button
          variant="outline"
          size="sm"
          className="absolute right-3 top-3 h-7 px-2 text-xs"
          aria-label="展開預覽面板"
          onClick={() => dispatch({ type: "SET_PANEL_OPEN", open: true })}
        >
          <PanelRightOpen className="mr-1 h-3.5 w-3.5" />
          預覽
        </Button>
      )}
    </main>
  );

  return (
    <TooltipProvider delayDuration={200}>
      <div className="flex h-screen w-full overflow-hidden bg-background text-foreground" data-app-shell>
        {wide && (
          <Sidebar
            variant="side"
            datasets={datasets ?? []}
            currentId={datasetId}
            page={page}
            workspaceCount={workspace.length}
            onNewDataset={() => setNewOpen(true)}
          />
        )}
        <div className="flex min-h-0 min-w-0 flex-1 flex-col">
          {!wide && (
            <Sidebar
              variant="top"
              datasets={datasets ?? []}
              currentId={datasetId}
              page={page}
              workspaceCount={workspace.length}
              onNewDataset={() => setNewOpen(true)}
            />
          )}
          {split ? (
            <ResizablePanelGroup
              groupRef={groupRef}
              orientation="horizontal"
              className="min-h-0 flex-1"
              onLayoutChanged={(layout) => {
                if (restoringRef.current) return;
                if (layout.preview !== undefined) layoutRef.current = layout;
              }}
            >
              <ResizablePanel id="content" minSize={360} className="min-h-0">
                {main}
              </ResizablePanel>
              {splitOpen && (
                <>
                  <ResizableHandle withHandle aria-label="調整預覽面板寬度" />
                  <ResizablePanel id="preview" minSize={420} className="min-h-0">
                    {panel}
                  </ResizablePanel>
                </>
              )}
            </ResizablePanelGroup>
          ) : wide ? (
            // overlay: the content keeps its full width; the panel floats over
            // its right edge and can be collapsed to the edge control
            <div className="relative min-h-0 flex-1" data-panel-overlay-host>
              {main}
              {overlayOpen && (
                <div className="absolute inset-y-0 right-0 z-20 w-[480px] max-w-[62%] shadow-2xl" data-panel-overlay>
                  {panel}
                </div>
              )}
            </div>
          ) : (
            <div className="flex min-h-0 flex-1 flex-col">
              <div className="min-h-0 flex-1">{main}</div>
              {panel}
            </div>
          )}
        </div>
      </div>
      <NewDatasetDialog
        open={newOpen}
        onOpenChange={setNewOpen}
        onUploaded={onUploaded}
        existingNames={(datasets ?? []).map((d) => d.filename)}
      />
    </TooltipProvider>
  );
}
