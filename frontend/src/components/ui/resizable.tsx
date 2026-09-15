import { GripVertical } from "lucide-react";
import * as ResizablePrimitive from "react-resizable-panels";

import { cn } from "@/lib/utils";

// react-resizable-panels v4 API (Group / Panel / Separator, `orientation`).
// The shadcn template targets the v2 names; this wrapper keeps the shadcn
// component names so call sites read the same.

const ResizablePanelGroup = ({
  className,
  orientation = "horizontal",
  ...props
}: React.ComponentProps<typeof ResizablePrimitive.Group>) => (
  <ResizablePrimitive.Group
    orientation={orientation}
    className={cn("flex h-full w-full", orientation === "vertical" && "flex-col", className)}
    {...props}
  />
);

const ResizablePanel = ResizablePrimitive.Panel;

const ResizableHandle = ({
  withHandle,
  orientation = "horizontal",
  className,
  ...props
}: React.ComponentProps<typeof ResizablePrimitive.Separator> & {
  withHandle?: boolean;
  orientation?: "horizontal" | "vertical";
}) => (
  <ResizablePrimitive.Separator
    className={cn(
      "relative flex items-center justify-center bg-border focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-offset-1",
      orientation === "horizontal"
        ? "w-px after:absolute after:inset-y-0 after:left-1/2 after:w-2 after:-translate-x-1/2"
        : "h-px w-full after:absolute after:inset-x-0 after:top-1/2 after:h-2 after:-translate-y-1/2",
      className,
    )}
    {...props}
  >
    {withHandle && (
      <div
        className={cn(
          "z-10 flex h-4 w-3 items-center justify-center rounded-sm border bg-border",
          orientation === "vertical" && "rotate-90",
        )}
      >
        <GripVertical className="h-2.5 w-2.5" />
      </div>
    )}
  </ResizablePrimitive.Separator>
);

export type Layout = ResizablePrimitive.Layout;

export { ResizablePanelGroup, ResizablePanel, ResizableHandle };
