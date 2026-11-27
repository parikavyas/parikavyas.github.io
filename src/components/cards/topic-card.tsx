import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { Constellation, OrbitalPaths } from "@/components/illustrations/primitives";
import { hashString } from "@/lib/utils";
import { formatDate } from "@/lib/dates";

/**
 * Card for a Wonder & Purpose topic or a Learn topic. Each gets its own
 * seeded constellation (Wonder) or orbit (Learn) so no two look alike.
 */
export function TopicCard({
  href,
  title,
  description,
  meta,
  updated,
  variant,
  badge,
}: {
  href: string;
  title: string;
  description: string;
  meta?: string;
  updated?: string;
  variant: "constellation" | "orbit";
  badge?: React.ReactNode;
}) {
  const seed = hashString(href);
  return (
    <article className="group relative flex h-full flex-col overflow-hidden rounded-[var(--radius-card)] border border-border bg-surface transition-colors hover:border-accent/45 focus-within:border-accent/45">
      <div className="relative h-36 overflow-hidden border-b border-border bg-gradient-to-b from-bg-secondary/80 to-bg/40">
        {variant === "constellation" ? (
          <Constellation className="absolute inset-0 h-full w-full opacity-80 transition-opacity group-hover:opacity-100" seed={seed % 997} count={6 + (seed % 5)} field={30} />
        ) : (
          <OrbitalPaths className="absolute left-1/2 top-1/2 h-56 w-56 -translate-x-1/2 -translate-y-1/2 opacity-80" rings={3 + (seed % 3)} seed={seed % 991} tilt={-10 - (seed % 25)} />
        )}
        {badge && <div className="absolute left-4 top-4">{badge}</div>}
      </div>
      <div className="flex flex-1 flex-col p-5">
        <h3 className="font-display text-lg font-semibold leading-snug">
          <Link href={href} className="after:absolute after:inset-0 after:content-[''] focus-visible:outline-none">
            {title}
          </Link>
        </h3>
        <p className="mt-2 flex-1 text-[0.93rem] leading-relaxed text-muted">{description}</p>
        <div className="mt-4 flex items-center justify-between gap-3 border-t border-border/70 pt-3 font-mono text-[0.7rem] text-muted">
          <span>{meta}</span>
          <span className="inline-flex items-center gap-1 text-accent">
            {updated ? <time dateTime={updated}>{formatDate(updated)}</time> : "Open"} <ArrowRight size={13} aria-hidden="true" />
          </span>
        </div>
      </div>
    </article>
  );
}
