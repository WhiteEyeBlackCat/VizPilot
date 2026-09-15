import type { ReactNode } from "react";

import { Card, CardContent, CardHeader } from "@/components/ui/card";

interface Props {
  title: ReactNode;
  /** Right-aligned header content (dataset switcher, status badge …). */
  aside?: ReactNode;
  children: ReactNode;
}

/** The one layout used by every top-level area (A–D + workspace) so the
 *  page reads as a single system: same header row, padding and radius. */
export function SectionCard({ title, aside, children }: Props) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between gap-3 space-y-0 pb-3">
        <h2 className="text-base font-semibold leading-none tracking-tight">{title}</h2>
        {aside}
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
}
