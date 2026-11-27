import type { ProjectStatus } from "@/lib/schemas";
import { cn } from "@/lib/utils";

const styles: Record<ProjectStatus, string> = {
  Planned: "border-border text-muted",
  "In Progress": "border-blue/60 text-blue-soft",
  Published: "border-accent/50 text-accent",
};

const dot: Record<ProjectStatus, string> = {
  Planned: "border border-muted bg-transparent",
  "In Progress": "bg-blue-soft",
  Published: "bg-accent",
};

export function StatusBadge({ status, className }: { status: ProjectStatus; className?: string }) {
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 font-mono text-[0.7rem]", styles[status], className)}>
      <span className={cn("h-1.5 w-1.5 rounded-full", dot[status])} aria-hidden="true" />
      <span>
        <span className="sr-only">Status: </span>
        {status}
      </span>
    </span>
  );
}
