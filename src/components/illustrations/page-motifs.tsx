/**
 * One abstract motif per primary page, composed from the primitives.
 * Hello: origin portal · Work: interconnected pathways · Projects: node universe
 * Wonder & Purpose: constellations · Learn: expanding orbits · Beyond: organic forms
 */
import { cn } from "@/lib/utils";
import {
  Constellation,
  DimensionalFrame,
  FloatingStructure,
  NodeNetwork,
  OrbitalPaths,
  OrganicForms,
  PerspectiveGrid,
  Portal,
} from "./primitives";

export type MotifName = "hello" | "work" | "projects" | "wonder" | "learn" | "beyond";

export function HelloBackdrop({ className }: { className?: string }) {
  return (
    <div className={cn("pointer-events-none absolute inset-0 overflow-hidden", className)} aria-hidden="true">
      <PerspectiveGrid className="absolute inset-x-0 bottom-0 h-[55%] w-full opacity-60" />
      <Constellation className="absolute left-[2%] top-[6%] w-[38%] max-w-md opacity-50" seed={42} count={7} />
      <Constellation className="absolute right-[4%] top-[58%] hidden w-[22%] opacity-40 md:block" seed={8} count={6} accent={false} />
    </div>
  );
}

function WorkPathways({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 400 300" className={className} aria-hidden="true" fill="none">
      {[0, 1, 2, 3].map((i) => (
        <path
          key={i}
          d={`M ${10 + i * 6} ${270 - i * 48} C ${130 + i * 10} ${262 - i * 48}, ${170 - i * 12} ${110 - i * 14}, ${390} ${70 + i * 22}`}
          className={i === 1 ? "stroke-accent" : "stroke-blue"}
          strokeOpacity={i === 1 ? 0.85 : 0.5}
          strokeWidth={1}
        />
      ))}
      {[
        [60, 250],
        [150, 160],
        [272, 92],
        [205, 120],
        [340, 60],
      ].map(([x, y], i) => (
        <g key={i}>
          <rect x={x - 6} y={y - 6} width={12} height={12} transform={`rotate(45 ${x} ${y})`} className={i === 2 ? "fill-accent" : "fill-bg stroke-blue-soft"} strokeWidth={1} />
        </g>
      ))}
    </svg>
  );
}

export function PageMotif({ name, className }: { name: MotifName; className?: string }) {
  const base = cn("w-full h-auto", className);
  switch (name) {
    case "hello":
      return <Portal className={base} />;
    case "work":
      return <WorkPathways className={base} />;
    case "projects":
      return (
        <div className={cn("relative", className)}>
          <NodeNetwork className="h-auto w-full" seed={17} nodes={16} highlight={3} />
          <DimensionalFrame className="absolute right-0 top-0 h-auto w-1/3 opacity-70" depth={5} />
        </div>
      );
    case "wonder":
      return (
        <div className={cn("relative", className)}>
          <Constellation className="h-auto w-full" seed={27} count={10} field={70} />
        </div>
      );
    case "learn":
      return <OrbitalPaths className={base} rings={5} seed={4} />;
    case "beyond":
      return <OrganicForms className={base} />;
  }
}

export { FloatingStructure };
