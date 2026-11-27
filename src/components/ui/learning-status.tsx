import type { LearningStatus } from "@/lib/schemas";
import { cn } from "@/lib/utils";

const ring: Record<LearningStatus, number> = { Curious: 1, Exploring: 2, Learning: 3, Implementing: 4, Revisiting: 5 };

/** Status pill with a tiny orbit gauge (how far along the orbit the topic is). */
export function LearningStatusBadge({ status, className }: { status: LearningStatus; className?: string }) {
  const n = ring[status];
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-full border border-blue/50 bg-bg/70 px-2.5 py-0.5 font-mono text-[0.7rem] text-blue-soft", className)}>
      <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true" fill="none">
        <circle cx="7" cy="7" r="5.5" className="stroke-border" strokeWidth="1.5" />
        <circle cx="7" cy="7" r="5.5" className="stroke-accent" strokeWidth="1.5" strokeDasharray={`${(n / 5) * 34.6} 34.6`} transform="rotate(-90 7 7)" />
      </svg>
      <span>
        <span className="sr-only">Learning status: </span>
        {status}
      </span>
    </span>
  );
}
