import { useState } from "react";

import type { DatasetMeta } from "../types";
import { ErrorList } from "./ErrorList";
import { DropZone, useUploader } from "./UploadPanel";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onUploaded: (meta: DatasetMeta) => void;
  /** filenames already uploaded; a same-name file asks before uploading again */
  existingNames: string[];
}

/** "+ New Dataset": a dialog with the drop zone. Closes itself on success;
 *  upload errors stay visible inside. A file whose name is already in the
 *  list is held back until the user confirms (each upload is a new dataset). */
export function NewDatasetDialog({ open, onOpenChange, onUploaded, existingNames }: Props) {
  const [pending, setPending] = useState<File | null>(null);
  const uploader = useUploader((meta) => {
    onOpenChange(false);
    onUploaded(meta);
  });

  const onFile = (file: File) => {
    if (existingNames.includes(file.name)) setPending(file);
    else void uploader.upload(file);
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (!next) {
          uploader.clearErrors();
          setPending(null);
        }
        onOpenChange(next);
      }}
    >
      <DialogContent className="max-w-md" data-new-dataset-dialog>
        <DialogTitle>New Dataset</DialogTitle>
        <DialogDescription>
          上傳一份表格資料；系統會建立 profile 並產生推薦。原始資料不會離開這台機器。
        </DialogDescription>
        {pending ? (
          <div className="space-y-3 rounded-md border border-warning/50 p-3 text-sm" role="alertdialog" aria-live="polite" data-duplicate-prompt>
            <p>
              已有同名資料集 <span className="font-medium">{pending.name}</span>
              。再上傳一次會建立第二份獨立的資料集（不會覆蓋）。
            </p>
            <div className="flex gap-2">
              <Button
                size="sm"
                className="h-7 px-2.5 text-xs"
                onClick={() => {
                  const file = pending;
                  setPending(null);
                  void uploader.upload(file);
                }}
              >
                仍要上傳
              </Button>
              <Button variant="outline" size="sm" className="h-7 px-2.5 text-xs" onClick={() => setPending(null)}>
                取消
              </Button>
            </div>
          </div>
        ) : (
          <DropZone busy={uploader.busy} onFile={onFile} />
        )}
        <ErrorList errors={uploader.errors} className="mt-0" />
      </DialogContent>
    </Dialog>
  );
}
