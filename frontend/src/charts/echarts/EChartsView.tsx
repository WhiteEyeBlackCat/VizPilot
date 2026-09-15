import { useEffect, useRef } from "react";

import { echarts, type ECharts, type VizOption } from "./echarts";

interface Props {
  option: VizOption;
  /** px number or any CSS height value */
  height?: number | string;
}

/** The chart instance is reachable from the host element (e2e / debugging). */
export interface EChartsHost extends HTMLDivElement {
  __echarts?: ECharts;
}

/** Hide every visible tooltip of the ECharts instances on the page (e.g.
 *  before an overlay opens, so a hovered tooltip cannot float above it). */
export function hideAllTips(): void {
  document.querySelectorAll<HTMLDivElement>("[data-chart-view]").forEach((host) => {
    echarts.getInstanceByDom(host)?.dispatchAction({ type: "hideTip" });
  });
}

/** Thin React binding over echarts/core: one instance per mount, option
 *  replaced wholesale on change (notMerge), resized with its container. */
export function EChartsView({ option, height = 360 }: Props) {
  const hostRef = useRef<EChartsHost>(null);
  const chartRef = useRef<ECharts | null>(null);

  useEffect(() => {
    const host = hostRef.current!;
    const chart = echarts.init(host, undefined, { renderer: "canvas" });
    chartRef.current = chart;
    host.__echarts = chart;
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(host);
    return () => {
      observer.disconnect();
      chart.dispose();
      chartRef.current = null;
      delete host.__echarts;
    };
  }, []);

  useEffect(() => {
    chartRef.current?.setOption(option, { notMerge: true });
  }, [option]);

  return <div ref={hostRef} data-chart-view="echarts" style={{ width: "100%", height }} />;
}
