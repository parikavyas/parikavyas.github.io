"use client";

/**
 * Visualization registry.
 *
 * To add a demo:
 *   1. Create src/components/visualizations/<name>-demo.tsx with a default export.
 *   2. Add an entry below (the key is what projects reference).
 *   3. Set `visualization: "<key>"` on the project in src/data/projects.ts,
 *      or use <Visualization id="<key>" /> inside an MDX file.
 * Nothing in the project-page architecture needs to change.
 */
import dynamic from "next/dynamic";
import type { ComponentType } from "react";

function Loading() {
  return (
    <div className="my-8 grid h-72 place-items-center rounded-[var(--radius-card)] border border-border bg-surface font-mono text-xs text-muted">
      Loading interactive demo…
    </div>
  );
}

export interface VisualizationEntry {
  title: string;
  Component: ComponentType;
}

export const visualizations: Record<string, VisualizationEntry> = {
  "linear-regression": {
    title: "Interactive linear regression",
    Component: dynamic(() => import("./linear-regression-demo"), { ssr: false, loading: Loading }),
  },
  "k-means": {
    title: "Interactive k-means clustering",
    Component: dynamic(() => import("./k-means-demo"), { ssr: false, loading: Loading }),
  },
};
