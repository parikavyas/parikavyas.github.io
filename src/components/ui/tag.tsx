import { cn } from "@/lib/utils";

export function Tag({ children, className, tone = "default" }: { children: React.ReactNode; className?: string; tone?: "default" | "blue" | "accent" | "warm" }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-md border px-2 py-0.5 font-mono text-[0.72rem] leading-5",
        tone === "default" && "border-border bg-bg/60 text-muted",
        tone === "blue" && "border-blue/50 bg-blue/10 text-blue-soft",
        tone === "accent" && "border-accent/40 bg-accent/10 text-accent",
        tone === "warm" && "border-warm/40 bg-warm/10 text-warm",
        className,
      )}
    >
      {children}
    </span>
  );
}

export function TagList({ tags, className }: { tags: string[]; className?: string }) {
  if (!tags.length) return null;
  return (
    <ul className={cn("flex flex-wrap gap-1.5", className)} aria-label="Tags">
      {tags.map((t) => (
        <li key={t}>
          <Tag>{t}</Tag>
        </li>
      ))}
    </ul>
  );
}
