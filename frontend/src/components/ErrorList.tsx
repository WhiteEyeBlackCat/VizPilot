import { AlertCircle } from "lucide-react";

import { Alert, AlertDescription } from "@/components/ui/alert";

/** API / validation errors (422 error lists, upload failures) rendered the
 *  same way everywhere. Keeps the <ul><li> structure so each error is one
 *  list item. */
export function ErrorList({ errors, className }: { errors: string[]; className?: string }) {
  if (errors.length === 0) return null;
  return (
    <Alert variant="destructive" className={className ?? "mt-3"}>
      <AlertCircle className="h-4 w-4" />
      <AlertDescription>
        <ul className="list-inside list-disc">
          {errors.map((err, i) => (
            <li key={i}>{err}</li>
          ))}
        </ul>
      </AlertDescription>
    </Alert>
  );
}
