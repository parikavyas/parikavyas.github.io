import { Clock, RefreshCw } from "lucide-react";
import { Container } from "@/components/layout/container";
import { Breadcrumbs, type Crumb } from "@/components/navigation/breadcrumbs";
import { Tag } from "@/components/ui/tag";
import { formatDate } from "@/lib/dates";

/** Header for long-form pages: articles, learning notes, creative writing. */
export function ArticleHeader({
  crumbs,
  kind,
  title,
  description,
  date,
  updated,
  readingMinutes,
  tags = [],
}: {
  crumbs: Crumb[];
  kind?: string;
  title: string;
  description?: string;
  date?: string;
  updated?: string;
  readingMinutes?: number;
  tags?: string[];
}) {
  return (
    <div className="relative overflow-hidden border-b border-border/70">
      <div className="bg-grid mask-fade-b absolute inset-0 opacity-35" aria-hidden="true" />
      <Container className="relative py-10 md:py-14">
        <Breadcrumbs items={crumbs} />
        {kind && <p className="mt-6 font-mono text-xs uppercase tracking-[0.2em] text-accent">{kind}</p>}
        <h1 className="mt-3 max-w-4xl font-display text-3xl font-semibold leading-tight tracking-tight sm:text-5xl">{title}</h1>
        {description && <p className="mt-4 max-w-2xl text-lg text-muted">{description}</p>}
        <p className="mt-6 flex flex-wrap gap-x-5 gap-y-1 font-mono text-xs text-muted">
          {date && (
            <span>
              Published <time dateTime={date}>{formatDate(date)}</time>
            </span>
          )}
          {updated && updated !== date && (
            <span className="inline-flex items-center gap-1.5">
              <RefreshCw size={12} aria-hidden="true" /> Updated <time dateTime={updated}>{formatDate(updated)}</time>
            </span>
          )}
          {readingMinutes && (
            <span className="inline-flex items-center gap-1.5">
              <Clock size={12} aria-hidden="true" /> {readingMinutes} min read
            </span>
          )}
        </p>
        {tags.length > 0 && (
          <ul className="mt-4 flex flex-wrap gap-1.5">
            {tags.map((t) => (
              <li key={t}>
                <Tag>{t}</Tag>
              </li>
            ))}
          </ul>
        )}
      </Container>
    </div>
  );
}
