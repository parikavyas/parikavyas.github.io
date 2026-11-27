"use client";

import { LazyMotion, domAnimation, m, useReducedMotion } from "motion/react";

/** A restrained one-time fade/rise on first render. Disabled under reduced motion. */
export function Reveal({ children, delay = 0, className }: { children: React.ReactNode; delay?: number; className?: string }) {
  const reduce = useReducedMotion();
  if (reduce) return <div className={className}>{children}</div>;
  return (
    <LazyMotion features={domAnimation} strict>
      <m.div
        className={className}
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, delay, ease: [0.2, 0.7, 0.2, 1] }}
      >
        {children}
      </m.div>
    </LazyMotion>
  );
}
