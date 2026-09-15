import type { ReactNode } from "react";

import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import { useMediaQuery } from "@/lib/hooks";

interface Props {
  /** Page content (left, scrollable). */
  children: ReactNode;
  /** The plots pane (right). */
  plots: ReactNode;
}

/** Page body split like RStudio: content on the left, plots on the right,
 *  draggable divider; below 1024px the two stack vertically.
 *
 *  The outer div owns the definite height: react-resizable-panels sets its
 *  own inline height on the group, and a content-driven height would let
 *  the plots pane grow with the chart it sizes to (a feedback loop). */
export function SplitPage({ children, plots }: Props) {
  const wide = useMediaQuery("(min-width: 1024px)");
  const orientation = wide ? "horizontal" : "vertical";
  return (
    <div className="h-[calc(100vh-9.5rem)] min-h-[520px] overflow-hidden" data-split={orientation}>
      <ResizablePanelGroup
        key={orientation} // remount when the direction flips so panel sizes reset
        orientation={orientation}
        style={{ height: "100%" }}
      >
        <ResizablePanel defaultSize="50" minSize="25" className="h-full min-h-0 overflow-hidden">
          <div className="h-full overflow-y-auto pr-2" data-page-content>
            {children}
          </div>
        </ResizablePanel>
        <ResizableHandle withHandle orientation={orientation} className={wide ? "mx-1" : "my-1"} />
        <ResizablePanel defaultSize="50" minSize="25" className="h-full min-h-0 overflow-hidden">
          <div className="h-full pl-2">{plots}</div>
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  );
}
