import { Scrollspy, type ScrollspyItem } from "@/components/navigation/scrollspy";
import { cn } from "@/lib/utils";
import { Container } from "./container";

/**
 * Documentation-style two-column layout: content on the left, sticky
 * right-hand scrollspy on large screens (an "On this page" menu on smaller ones).
 */
export function DocLayout({
  toc,
  children,
  className,
  tocTitle,
}: {
  toc: ScrollspyItem[];
  children: React.ReactNode;
  className?: string;
  tocTitle?: string;
}) {
  return (
    <Container className={cn("lg:grid lg:grid-cols-[minmax(0,1fr)_13.5rem] lg:gap-14 xl:gap-20", className)}>
      <div className="contents lg:order-2 lg:block">
        <Scrollspy items={toc} title={tocTitle} />
      </div>
      <div className="min-w-0 lg:order-1">{children}</div>
    </Container>
  );
}
