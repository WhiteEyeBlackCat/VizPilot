import type { DatasetMeta } from "../types";
import { ErrorList } from "./ErrorList";
import { DropZone, useUploader } from "./UploadPanel";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onUploaded: (meta: DatasetMeta) => void;
}

/** "+ New Dataset": a dialog with the drop zone. Closes itself on success;
 *  upload errors stay visible inside. */
export function NewDatasetDialog({ open, onOpenChange, onUploaded }: Props) {
  const uploader = useUploader((meta) => {
    onOpenChange(false);
    onUploaded(meta);
  });
  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (!next) uploader.clearErrors();
        onOpenChange(next);
      }}
    >
      <DialogContent className="max-w-md" data-new-dataset-dialog>
        <DialogTitle>New Dataset</DialogTitle>
        <DialogDescription>
          上傳一份表格資料；系統會建立 profile 並產生推薦。原始資料不會離開這台機器。
        </DialogDescription>
        <DropZone busy={uploader.busy} onFile={uploader.upload} />
        <ErrorList errors={uploader.errors} className="mt-0" />
      </DialogContent>
    </Dialog>
  );
}
