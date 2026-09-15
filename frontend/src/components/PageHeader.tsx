import type { ReactNode } from "react";

interface Props {
  title: string;
  subtitle?: ReactNode;
  aside?: ReactNode;
}

/** Page title row: typography and spacing carry the hierarchy, no card. */
export function PageHeader({ title, subtitle, aside }: Props) {
  return (
    <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
        {subtitle && <p className="mt-1 text-sm text-muted-foreground">{subtitle}</p>}
      </div>
      {aside}
    </div>
  );
}
