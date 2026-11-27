import { Constellation } from "@/components/illustrations/primitives";

export function EmptyState({ title, children, seed = 5 }: { title: string; children?: React.ReactNode; seed?: number }) {
  return (
    <div className="relative overflow-hidden rounded-[var(--radius-card)] border border-dashed border-border bg-surface/50 px-6 py-10 text-center">
      <Constellation className="mx-auto h-24 w-48 opacity-70" seed={seed} count={6} field={10} />
      <p className="mt-4 font-display text-lg font-semibold">{title}</p>
      {children && <div className="mx-auto mt-2 max-w-md text-sm text-muted">{children}</div>}
    </div>
  );
}
