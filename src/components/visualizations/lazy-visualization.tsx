"use client";

import { useEffect, useRef, useState } from "react";
import { visualizations } from "./registry";

/**
 * Mounts a registered visualization only when it scrolls near the viewport,
 * so its JavaScript is never downloaded by readers who don't reach it.
 */
export function LazyVisualization({ id }: { id: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  const entry = visualizations[id];

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (!("IntersectionObserver" in window)) {
      setVisible(true);
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setVisible(true);
          io.disconnect();
        }
      },
      { rootMargin: "400px 0px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  if (!entry) return null; // unknown ids render nothing rather than a fake control
  const { Component } = entry;
  return (
    <div ref={ref} className="min-h-72">
      {visible ? (
        <Component />
      ) : (
        <div className="my-8 grid h-72 place-items-center rounded-[var(--radius-card)] border border-border bg-surface font-mono text-xs text-muted">
          {entry.title} loads as you scroll
        </div>
      )}
    </div>
  );
}
