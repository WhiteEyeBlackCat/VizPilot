import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "./api";
import { ChartsWorkspace, type WorkspaceChart } from "./components/Workspace";
import { DatasetOverview } from "./components/DatasetOverview";
import { ManualBuilder } from "./components/ManualBuilder";
import { Recommendations } from "./components/Recommendations";
import { UploadPanel } from "./components/UploadPanel";
import type { ChartSpec, DatasetMeta, DatasetProfile, RecommendationsResponse } from "./types";

export default function App() {
  const [datasets, setDatasets] = useState<DatasetMeta[]>([]);
  const [meta, setMeta] = useState<DatasetMeta | null>(null);
  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [recs, setRecs] = useState<RecommendationsResponse | null>(null);
  const [aiPending, setAiPending] = useState(false);
  const [charts, setCharts] = useState<WorkspaceChart[]>([]);
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
    setAiPending(true);

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

  const generateChart = useCallback(
    async (spec: ChartSpec) => {
      const id = currentId.current;
      if (!id) return;
      const result = await api.render(id, spec);
      if (currentId.current !== id) return; // dataset switched while rendering
      chartSeq.current += 1;
      setCharts((prev) => [...prev, { key: `chart-${chartSeq.current}`, result }]);
    },
    [],
  );

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      <header>
        <h1 className="text-2xl font-bold text-slate-900">VizPilot</h1>
        <p className="text-sm text-slate-500">
          本地 AI 資料探索助手 — 上傳資料、理解資料、獲得可解釋的圖表推薦
        </p>
      </header>

      <UploadPanel
        datasets={datasets}
        current={meta}
        onUploaded={onUploaded}
        onSelect={selectDataset}
      />

      {profile && <DatasetOverview profile={profile} />}

      {meta && (
        <Recommendations recs={recs} aiPending={aiPending} onGenerate={generateChart} />
      )}

      {profile && <ManualBuilder profile={profile} onGenerate={generateChart} />}

      {charts.length > 0 && (
        <ChartsWorkspace
          charts={charts}
          onRemove={(key) => setCharts((prev) => prev.filter((c) => c.key !== key))}
        />
      )}
    </div>
  );
}
