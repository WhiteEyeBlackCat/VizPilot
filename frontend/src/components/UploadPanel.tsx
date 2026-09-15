import { Upload } from "lucide-react";
import { useCallback, useRef, useState, type DragEvent } from "react";

import { api, ApiError } from "../api";
import type { DatasetMeta } from "../types";
import { ErrorList } from "./ErrorList";
import { SectionCard } from "./SectionCard";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

/** Upload state shared by the header button and the overview drop zone. */
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
  return { busy, errors, upload };
}

const ACCEPT = ".csv,.xlsx,.parquet";

interface SwitcherProps {
  datasets: DatasetMeta[];
  current: DatasetMeta | null;
  onSelect: (meta: DatasetMeta) => void;
}

export function DatasetSwitcher({ datasets, current, onSelect }: SwitcherProps) {
  if (datasets.length === 0) return null;
  return (
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
}

interface ButtonProps {
  busy: boolean;
  onFile: (file: File) => void;
}

/** Header upload button: a hidden file input behind a shadcn Button. */
export function UploadButton({ busy, onFile }: ButtonProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  return (
    <>
      <Button size="sm" className="h-8" disabled={busy} onClick={() => inputRef.current?.click()}>
        <Upload className="mr-1.5 h-3.5 w-3.5" />
        {busy ? "上傳中…" : "上傳"}
      </Button>
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
    </>
  );
}

interface PanelProps {
  current: DatasetMeta | null;
  busy: boolean;
  errors: string[];
  onFile: (file: File) => void;
}

/** Overview-page drop zone plus the current dataset's column chips. */
export function UploadPanel({ current, busy, errors, onFile }: PanelProps) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) onFile(file);
  };

  return (
    <SectionCard title="上傳資料集">
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
          accept={ACCEPT}
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onFile(file);
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
