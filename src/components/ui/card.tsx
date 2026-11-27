import { cn } from "@/lib/utils";

/**
 * Base rectangular card: elevated navy, thin border, ~14px radius,
 * restrained hover with a subtle yellow accent. Use `interactive` for
 * cards that respond to hover/focus.
 */
export function Card({
  as: Tag = "div",
  className,
  interactive = false,
  children,
  ...rest
}: {
  as?: "div" | "article" | "li" | "section";
  className?: string;
  interactive?: boolean;
  children: React.ReactNode;
} & React.HTMLAttributes<HTMLElement>) {
  return (
    <Tag
      className={cn(
        "relative rounded-[var(--radius-card)] border border-border bg-surface p-6",
        interactive &&
          "transition-[border-color,background-color,transform] duration-200 hover:border-accent/45 hover:bg-surface-elevated/60 focus-within:border-accent/45",
        className,
      )}
      {...rest}
    >
      {children}
    </Tag>
  );
}
