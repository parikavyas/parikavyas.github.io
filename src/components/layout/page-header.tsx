import { Container } from "./container";
import { PageMotif, type MotifName } from "@/components/illustrations/page-motifs";
import { cn } from "@/lib/utils";

/** The opening band of every inner page: eyebrow, title, subtitle and the page motif. */
export function PageHeader({
  eyebrow,
  title,
  subtitle,
  motif,
  children,
  className,
}: {
  eyebrow: string;
  title: string;
  subtitle?: string;
  motif: MotifName;
  children?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("relative overflow-hidden border-b border-border/70", className)}>
      <div className="bg-grid mask-fade-b absolute inset-0 opacity-50" aria-hidden="true" />
      <Container className="relative grid items-center gap-6 py-12 md:gap-8 md:py-20 md:grid-cols-[1.4fr_1fr] md:py-20">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.22em] text-accent">{eyebrow}</p>
          <h1 className="mt-4 font-display text-4xl font-semibold leading-[1.05] tracking-tight sm:text-5xl lg:text-6xl">
            {title}
          </h1>
          {subtitle && <p className="mt-5 max-w-2xl text-lg leading-relaxed text-muted">{subtitle}</p>}
          {children}
        </div>
        <div className="mx-auto w-full max-w-[13rem] opacity-90 sm:max-w-[18rem] md:max-w-none">
          <PageMotif name={motif} />
        </div>
      </Container>
    </div>
  );
}
