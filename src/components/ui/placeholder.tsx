import { cn, isDev } from "@/lib/utils";

/**
 * Marks content that has not been written yet.
 *
 * In development it shows a dashed, warm-coloured note telling you where to
 * fill the content in. On the live site it shows the neutral `publicText`
 * (or nothing), so placeholder wording is never published as fact.
 */
export function Placeholder({
  hint,
  publicText,
  className,
  inline = false,
}: {
  /** Developer note, e.g. "Add role in src/data/experience.ts" */
  hint: string;
  /** Neutral text shown in production. Omit to render nothing. */
  publicText?: string;
  className?: string;
  inline?: boolean;
}) {
  const Tag = inline ? "span" : "p";
  if (isDev) {
    return (
      <Tag
        className={cn(
          "rounded-md border border-dashed border-warm/50 bg-warm/5 px-2 py-1 font-mono text-[0.72rem] text-warm",
          inline ? "inline-block" : "block",
          className,
        )}
      >
        Placeholder · {hint}
      </Tag>
    );
  }
  if (!publicText) return null;
  return <Tag className={cn("text-sm italic text-muted", className)}>{publicText}</Tag>;
}

/** Banner at the top of draft pages (visible only where drafts are shown). */
export function DraftBanner() {
  return (
    <p className="mb-6 rounded-lg border border-dashed border-warm/60 bg-warm/5 px-4 py-3 font-mono text-xs text-warm">
      Draft — visible only in local preview. Set <code>draft: false</code> (or <code>status: &quot;published&quot;</code>) to publish.
    </p>
  );
}
