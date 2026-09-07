import { useRef, useState, type DragEvent } from "react";

import { api, ApiError } from "../api";
import type { DatasetMeta } from "../types";

interface Props {
  datasets: DatasetMeta[];
  current: DatasetMeta | null;
  onUploaded: (meta: DatasetMeta) => void;
  onSelect: (meta: DatasetMeta) => void;
}

export function UploadPanel({ datasets, current, onUploaded, onSelect }: Props) {
  const [busy, setBusy] = useState(false);
  const [errors, setErrors] = useState<string[]>([]);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const upload = async (file: File) => {
    setBusy(true);
    setErrors([]);
    try {
      onUploaded(await api.upload(file));
    } catch (e) {
      setErrors(e instanceof ApiError ? e.errors : [String(e)]);
    } finally {
      setBusy(false);
    }
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) void upload(file);
  };

  return (
    <section className="rounded-lg bg-white p-4 shadow">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="font-semibold">A. 上傳資料集</h2>
        {datasets.length > 0 && (
          <select
            className="rounded border border-slate-300 px-2 py-1 text-sm"
            value={current?.dataset_id ?? ""}
            onChange={(e) => {
              const m = datasets.find((d) => d.dataset_id === e.target.value);
              if (m) onSelect(m);
            }}
          >
            <option value="" disabled>
              選擇既有資料集…
            </option>
            {datasets.map((d) => (
              <option key={d.dataset_id} value={d.dataset_id}>
                {d.filename} ({d.n_rows}×{d.n_cols})
              </option>
            ))}
          </select>
        )}
      </div>

      <div
        className={`flex cursor-pointer flex-col items-center justify-center rounded border-2 border-dashed p-6 text-sm ${
          dragging ? "border-blue-400 bg-blue-50" : "border-slate-300"
        }`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        {busy ? (
          <span className="text-blue-600">上傳中…</span>
        ) : (
          <>
            <span>拖放檔案到這裡，或點擊選擇</span>
            <span className="mt-1 text-xs text-slate-400">支援 CSV / XLSX / Parquet，上限 200MB</span>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept=".csv,.xlsx,.parquet"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void upload(file);
            e.target.value = "";
          }}
        />
      </div>

      {errors.length > 0 && (
        <ul className="mt-2 list-inside list-disc text-sm text-red-600">
          {errors.map((err, i) => (
            <li key={i}>{err}</li>
          ))}
        </ul>
      )}

      {current && (
        <div className="mt-3 text-sm">
          <span className="font-medium">{current.filename}</span>
          <span className="ml-2 text-slate-500">
            {current.n_rows} 列 × {current.n_cols} 欄
          </span>
          <div className="mt-1 flex flex-wrap gap-1">
            {current.columns.map((c) => (
              <span key={c.name} className="rounded bg-slate-100 px-1.5 py-0.5 text-xs text-slate-600">
                {c.name} <span className="text-slate-400">({c.dtype})</span>
              </span>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
