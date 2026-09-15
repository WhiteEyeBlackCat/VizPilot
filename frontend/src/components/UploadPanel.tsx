import { useCallback, useRef, useState, type DragEvent } from "react";

import { api, ApiError } from "../api";
import type { DatasetMeta } from "../types";
import { cn } from "@/lib/utils";

/** Upload state shared by every place that can start an upload. */
export function useUploader(onUploaded: (meta: DatasetMeta) => void) {
  const [busy, setBusy] = useState(false);
  const [errors, setErrors] = useState<string[]>([]);
  const upload = useCallback(
    async (file: File) => {
      setBusy(true);
      setErrors([]);
      try {
        onUploaded(await api.upload(file));
      } catch (e) {
        setErrors(e instanceof ApiError ? e.errors : [String(e)]);
      } finally {
        setBusy(false);
      }
    },
    [onUploaded],
  );
  const clearErrors = useCallback(() => setErrors([]), []);
  return { busy, errors, upload, clearErrors };
}

const ACCEPT = ".csv,.xlsx,.parquet";

interface DropZoneProps {
  busy: boolean;
  onFile: (file: File) => void;
  className?: string;
}

/** Drag-and-drop / click-to-pick file area (CSV, XLSX, Parquet). */
export function DropZone({ busy, onFile, className }: DropZoneProps) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) onFile(file);
  };

  return (
    <div
      className={cn(
        "flex cursor-pointer flex-col items-center justify-center rounded-md border border-dashed p-8 text-sm transition-colors",
        dragging ? "border-primary bg-accent" : "border-border hover:bg-accent/50",
        className,
      )}
      data-drop-zone
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      {busy ? (
        <span className="text-ring">上傳中…</span>
      ) : (
        <>
          <span>拖放檔案到這裡，或點擊選擇</span>
          <span className="mt-1 text-xs text-muted-foreground">
            支援 CSV / XLSX / Parquet，上限 200MB
          </span>
        </>
      )}
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        className="hidden"
        aria-label="上傳資料集檔案"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFile(file);
          e.target.value = "";
        }}
      />
    </div>
  );
}
