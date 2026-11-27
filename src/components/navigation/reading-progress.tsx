"use client";

import { useEffect, useRef } from "react";

/**
 * Thin yellow bar under the sticky header showing how far through the
 * article (the element with `targetId`) the reader is — not the whole page.
 */
export function ReadingProgress({ targetId }: { targetId: string }) {
  const bar = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const target = document.getElementById(targetId);
    if (!target || !bar.current) return;
    let frame: number | null = null;
    const update = () => {
      frame = null;
      const rect = target.getBoundingClientRect();
      const header = 64;
      const total = rect.height - (window.innerHeight - header);
      const progressed = header - rect.top;
      const p = total <= 0 ? (rect.top < header ? 1 : 0) : Math.min(1, Math.max(0, progressed / total));
      if (bar.current) {
        bar.current.style.transform = `scaleX(${p})`;
        bar.current.parentElement?.setAttribute("aria-valuenow", String(Math.round(p * 100)));
      }
    };
    const schedule = () => {
      if (frame === null) frame = requestAnimationFrame(update);
    };
    update();
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    return () => {
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
      if (frame !== null) cancelAnimationFrame(frame);
    };
  }, [targetId]);

  return (
    <div
      className="pointer-events-none fixed inset-x-0 top-[var(--header-height)] z-40 h-[2px] bg-transparent"
      role="progressbar"
      aria-label="Reading progress"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={0}
    >
      <div ref={bar} className="h-full origin-left scale-x-0 bg-accent will-change-transform" />
    </div>
  );
}
