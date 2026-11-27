import { cn } from "@/lib/utils";

/**
 * A numbered page section. The `id` doubles as the scrollspy anchor.
 */
export function Section({
  id,
  number,
  title,
  description,
  children,
  className,
  aside,
}: {
  id: string;
  number?: string;
  title: string;
  description?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  aside?: React.ReactNode;
}) {
  return (
    <section id={id} aria-labelledby={`${id}-title`} className={cn(" py-12 first:pt-4", className)}>
      <div className={cn("mb-8", aside && "grid items-end gap-6 sm:grid-cols-[1fr_12rem]")}>
        <div>
          {number && <p className="font-mono text-sm text-accent">{number}</p>}
          <h2 id={`${id}-title`} className="mt-1 font-display text-2xl font-semibold tracking-tight sm:text-3xl">
            {title}
          </h2>
          {description && <div className="mt-3 max-w-2xl text-muted">{description}</div>}
        </div>
        {aside && <div className="hidden sm:block">{aside}</div>}
      </div>
      {children}
    </section>
  );
}
