import { useRef, useState, type DragEvent } from "react";

import { api, ApiError } from "../api";
import type { DatasetMeta } from "../types";
import { ErrorList } from "./ErrorList";
import { SectionCard } from "./SectionCard";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

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

  const switcher = datasets.length > 0 && (
    <Select
      value={current?.dataset_id ?? ""}
      onValueChange={(id) => {
        const m = datasets.find((d) => d.dataset_id === id);
        if (m) onSelect(m);
      }}
    >
      <SelectTrigger aria-label="選擇既有資料集" className="h-8 w-64 text-sm">
        <SelectValue placeholder="選擇既有資料集…" />
      </SelectTrigger>
      <SelectContent>
        {datasets.map((d) => (
          <SelectItem key={d.dataset_id} value={d.dataset_id}>
            {d.filename} ({d.n_rows}×{d.n_cols})
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );

  return (
    <SectionCard title="A. 上傳資料集" aside={switcher}>
      <div
        className={cn(
          "flex cursor-pointer flex-col items-center justify-center rounded-md border-2 border-dashed p-6 text-sm transition-colors",
          dragging ? "border-primary bg-accent" : "border-border hover:bg-accent/50",
        )}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        {busy ? (
          <span className="text-primary">上傳中…</span>
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
          accept=".csv,.xlsx,.parquet"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void upload(file);
            e.target.value = "";
          }}
        />
      </div>

      <ErrorList errors={errors} />

      {current && (
        <div className="mt-3 text-sm">
          <span className="font-medium">{current.filename}</span>
          <span className="ml-2 text-muted-foreground">
            {current.n_rows} 列 × {current.n_cols} 欄
          </span>
          <div className="mt-1.5 flex flex-wrap gap-1">
            {current.columns.map((c) => (
              <Badge key={c.name} variant="muted" className="font-normal">
                {c.name}
                <span className="ml-1 text-muted-foreground">({c.dtype})</span>
              </Badge>
            ))}
          </div>
        </div>
      )}
    </SectionCard>
  );
}
