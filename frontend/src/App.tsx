import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "./api";
import { DatasetOverview } from "./components/DatasetOverview";
import { ErrorList } from "./components/ErrorList";
import { ManualBuilder } from "./components/ManualBuilder";
import { PlotsPane, type WorkspaceChart } from "./components/PlotsPane";
import { Recommendations } from "./components/Recommendations";
import { SplitPage } from "./components/SplitPage";
import { DatasetSwitcher, UploadButton, UploadPanel, useUploader } from "./components/UploadPanel";
import type { ChartSpec, DatasetMeta, DatasetProfile, RecommendationsResponse } from "./types";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TooltipProvider } from "@/components/ui/tooltip";

type Page = "overview" | "recommend" | "manual";

export default function App() {
  const [datasets, setDatasets] = useState<DatasetMeta[]>([]);
  const [meta, setMeta] = useState<DatasetMeta | null>(null);
  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [recs, setRecs] = useState<RecommendationsResponse | null>(null);
  const [aiPending, setAiPending] = useState(false);
  const [charts, setCharts] = useState<WorkspaceChart[]>([]);
  const [currentKey, setCurrentKey] = useState<string | null>(null);
  const [page, setPage] = useState<Page>("overview");
  const currentId = useRef<string | null>(null);
  const fullRecsLoaded = useRef<Record<string, boolean>>({});
  const chartSeq = useRef(0);

  useEffect(() => {
    api.listDatasets().then(setDatasets).catch(() => {});
  }, []);

  const selectDataset = useCallback((m: DatasetMeta) => {
    const id = m.dataset_id;
    currentId.current = id;
    setMeta(m);
    setProfile(null);
    setRecs(null);
    setCharts([]);
    setCurrentKey(null);
    setAiPending(true);
    setPage("recommend");

    api
      .profile(id)
      .then((p) => currentId.current === id && setProfile(p))
      .catch(() => {});
    // two-phase load: rules instantly, then the full (possibly LLM) result
    api
      .recommendations(id, false)
      .then((r) => {
        if (currentId.current === id && !fullRecsLoaded.current[id]) setRecs(r);
      })
      .catch(() => {});
    api
      .recommendations(id, true)
      .then((r) => {
        if (currentId.current !== id) return;
        fullRecsLoaded.current[id] = true;
        setRecs(r);
        setAiPending(false);
      })
      .catch(() => {
        if (currentId.current === id) setAiPending(false);
      });
  }, []);

  const onUploaded = useCallback(
    (m: DatasetMeta) => {
      setDatasets((prev) => [m, ...prev.filter((d) => d.dataset_id !== m.dataset_id)]);
      selectDataset(m);
    },
    [selectDataset],
  );
  const uploader = useUploader(onUploaded);

  const generateChart = useCallback(async (spec: ChartSpec) => {
    const id = currentId.current;
    if (!id) return;
    const result = await api.render(id, spec);
    if (currentId.current !== id) return; // dataset switched while rendering
    chartSeq.current += 1;
    const key = `chart-${chartSeq.current}`;
    setCharts((prev) => [...prev, { key, result }]);
    setCurrentKey(key); // the newest chart becomes the current plot
  }, []);

  const removeChart = useCallback((key: string) => {
    setCharts((prev) => {
      const idx = prev.findIndex((c) => c.key === key);
      const next = prev.filter((c) => c.key !== key);
      // keep the position: show the neighbour that took the removed slot
      setCurrentKey((cur) =>
        cur !== key ? cur : next.length === 0 ? null : next[Math.min(idx, next.length - 1)].key,
      );
      return next;
    });
  }, []);

  const clearCharts = useCallback(() => {
    setCharts([]);
    setCurrentKey(null);
  }, []);

  const plots = (
    <PlotsPane
      charts={charts}
      currentKey={currentKey}
      onSelect={setCurrentKey}
      onRemove={removeChart}
      onClear={clearCharts}
    />
  );

  return (
    <TooltipProvider delayDuration={200}>
      <div
        className="mx-auto max-w-[1600px] space-y-4 p-4 md:p-6"
        data-profile={profile ? "loaded" : "none"}
      >
        <header className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">VizPilot</h1>
            <p className="text-sm text-muted-foreground">
              本地 AI 資料探索助手 — 上傳資料、理解資料、獲得可解釋的圖表推薦
            </p>
          </div>
          <div className="flex items-center gap-2">
            <DatasetSwitcher datasets={datasets} current={meta} onSelect={selectDataset} />
            <UploadButton busy={uploader.busy} onFile={uploader.upload} />
          </div>
        </header>

        <Tabs value={page} onValueChange={(v) => setPage(v as Page)}>
          <TabsList>
            <TabsTrigger value="overview">總覽</TabsTrigger>
            <TabsTrigger value="recommend" disabled={!meta}>
              推薦圖表
            </TabsTrigger>
            <TabsTrigger value="manual" disabled={!profile}>
              手動建圖
            </TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            <UploadPanel
              current={meta}
              busy={uploader.busy}
              errors={uploader.errors}
              onFile={uploader.upload}
            />
            {profile && <DatasetOverview profile={profile} />}
          </TabsContent>

          <TabsContent value="recommend">
            {meta && (
              <SplitPage plots={plots}>
                <Recommendations recs={recs} aiPending={aiPending} onGenerate={generateChart} />
              </SplitPage>
            )}
          </TabsContent>

          <TabsContent value="manual">
            {profile && (
              <SplitPage plots={plots}>
                <ManualBuilder profile={profile} onGenerate={generateChart} />
              </SplitPage>
            )}
          </TabsContent>
        </Tabs>

        {/* upload errors raised from the header button while not on the overview page */}
        {page !== "overview" && <ErrorList errors={uploader.errors} />}
      </div>
    </TooltipProvider>
  );
}
