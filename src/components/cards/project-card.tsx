import Link from "next/link";
import { ArrowRight, ExternalLink } from "lucide-react";
import { Github } from "@/components/ui/brand-icons";
import type { Project } from "@/lib/schemas";
import { StatusBadge } from "@/components/ui/status-badge";
import { Tag, TagList } from "@/components/ui/tag";
import { formatDate } from "@/lib/dates";

/**
 * Project card. GitHub and demo actions only render when the project is not
 * "Planned" AND the link exists — no dead buttons.
 */
export function ProjectCard({
  project: p,
  illustration,
  categoryLabels,
}: {
  project: Project;
  illustration?: React.ReactNode;
  categoryLabels: string[];
}) {
  const live = p.status !== "Planned";
  const repoHref = live && p.repo ? `https://github.com/${p.repo.owner}/${p.repo.name}` : null;
  const demoHref = live ? p.demoUrl : null;
  return (
    <article className="group relative flex h-full flex-col overflow-hidden rounded-[var(--radius-card)] border border-border bg-surface transition-colors hover:border-accent/45 focus-within:border-accent/45">
      {illustration && <div className="border-b border-border bg-bg/50 px-6 pt-4 opacity-80 transition-opacity group-hover:opacity-100">{illustration}</div>}
      <div className="flex flex-1 flex-col p-5">
        <div className="flex flex-wrap items-center gap-2">
          <StatusBadge status={p.status} />
          {p.implementationTypes.map((t) => (
            <Tag key={t} tone="blue">
              {t}
            </Tag>
          ))}
        </div>
        <h3 className="mt-3 font-display text-lg font-semibold leading-snug">
          <Link href={`/projects/${p.slug}`} className="after:absolute after:inset-0 after:content-[''] focus-visible:outline-none">
            {p.title}
          </Link>
        </h3>
        <p className="mt-1 font-mono text-[0.7rem] text-muted">{categoryLabels.join(" · ")}</p>
        <p className="mt-2.5 flex-1 text-[0.93rem] leading-relaxed text-muted">{p.summary}</p>
        <TagList tags={p.tags} className="mt-4" />
        <div className="relative z-10 mt-5 flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-border/70 pt-4 text-sm">
          <span className="inline-flex items-center gap-1 font-medium text-accent">
            Explore project <ArrowRight size={14} aria-hidden="true" className="transition-transform group-hover:translate-x-0.5" />
          </span>
          {repoHref && (
            <a href={repoHref} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 text-muted hover:text-accent">
              <Github size={14} aria-hidden="true" /> GitHub<span className="sr-only"> repository for {p.title}</span>
            </a>
          )}
          {demoHref && (
            <a href={demoHref} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 text-muted hover:text-accent">
              <ExternalLink size={14} aria-hidden="true" /> Live demo<span className="sr-only"> of {p.title}</span>
            </a>
          )}
          <span className="ml-auto font-mono text-[0.7rem] text-muted">
            Updated <time dateTime={p.updated}>{formatDate(p.updated)}</time>
          </span>
        </div>
      </div>
    </article>
  );
}
