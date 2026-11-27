import { FlaskConical } from "lucide-react";

/** Shared chrome for every interactive demo, with an explicit simulation label. */
export function VisualizationFrame({
  title,
  description,
  children,
  footnote,
}: {
  title: string;
  description: string;
  children: React.ReactNode;
  footnote?: string;
}) {
  return (
    <figure className="not-prose my-8 overflow-hidden rounded-[var(--radius-card)] border border-border bg-surface">
      <figcaption className="flex flex-col gap-2 border-b border-border px-5 py-4 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <p className="font-display text-lg font-semibold">{title}</p>
          <p className="mt-1 text-sm text-muted">{description}</p>
        </div>
        <span className="inline-flex shrink-0 items-center gap-1.5 self-start rounded-full border border-blue/50 bg-blue/10 px-2.5 py-1 font-mono text-[0.68rem] text-blue-soft">
          <FlaskConical size={12} aria-hidden="true" />
          Educational simulation · synthetic data
        </span>
      </figcaption>
      <div className="p-4 sm:p-5">{children}</div>
      <p className="border-t border-border px-5 py-3 font-mono text-[0.7rem] text-muted">
        {footnote ?? "Runs entirely in your browser. Not a production model."}
      </p>
    </figure>
  );
}

export function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border bg-bg/60 px-3 py-2">
      <dt className="font-mono text-[0.66rem] uppercase tracking-[0.14em] text-muted">{label}</dt>
      <dd className="mt-0.5 font-mono text-sm text-text">{value}</dd>
    </div>
  );
}
