import type { Metadata } from "next";
import { canonical } from "@/lib/seo";
import Link from "next/link";
import { notFound } from "next/navigation";
import { CheckCircle2, Circle, ExternalLink } from "lucide-react";
import { DocLayout } from "@/components/layout/doc-layout";
import { Section } from "@/components/layout/section";
import { Container } from "@/components/layout/container";
import { Breadcrumbs } from "@/components/navigation/breadcrumbs";
import { OrbitalPaths } from "@/components/illustrations/primitives";
import { LearningStatusBadge } from "@/components/ui/learning-status";
import { DraftBanner } from "@/components/ui/placeholder";
import { EmptyState } from "@/components/content/empty-state";
import { projects } from "@/data/projects";
import { getLearningNotes } from "@/lib/content";
import { visibleLearningTopics, visibleWonderTopics } from "@/lib/visibility";
import { formatDate } from "@/lib/dates";
import { hashString } from "@/lib/utils";

export const dynamicParams = false;

export function generateStaticParams() {
  return visibleLearningTopics().map((t) => ({ topic: t.slug }));
}

type Params = { params: Promise<{ topic: string }> };

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const { topic } = await params;
  const t = visibleLearningTopics().find((x) => x.slug === topic);
  if (!t) return {};
  return { ...canonical(`/learn/${t.slug}`), title: t.title, description: t.description, robots: t.draft ? { index: false, follow: false } : undefined };
}

export default async function LearningTopicPage({ params }: Params) {
  const { topic } = await params;
  const t = visibleLearningTopics().find((x) => x.slug === topic);
  if (!t) notFound();
  const notes = getLearningNotes(t.slug);
  const relProjects = projects.filter((p) => t.related.projects.includes(p.slug) || t.related.projects.includes(p.id) || p.related.learn.includes(t.slug));
  const relWonder = visibleWonderTopics().filter((w) => t.related.wonder.includes(w.slug) || w.related.learn.includes(t.slug));

  const toc = [
    { id: "notes", label: "Notes", show: true },
    { id: "resources", label: "Resources", show: t.resources.length > 0 },
    { id: "reading", label: "Reading list", show: t.readingList.length > 0 },
    { id: "roadmap", label: "Roadmap", show: t.roadmap.length > 0 },
    { id: "questions", label: "Questions", show: t.questions.length > 0 },
    { id: "learned", label: "Recently learned", show: t.learned.length > 0 },
    { id: "related", label: "Related", show: relProjects.length + relWonder.length > 0 },
  ].filter((s) => s.show);

  return (
    <>
      <div className="relative overflow-hidden border-b border-border/70">
        <OrbitalPaths className="absolute -right-24 -top-20 h-[28rem] w-[28rem] opacity-40" rings={5} seed={hashString(t.slug) % 97} />
        <Container className="relative py-12 md:py-16">
          <Breadcrumbs items={[{ label: "Learn", href: "/learn" }, { label: t.title }]} />
          <div className="mt-6">
            <LearningStatusBadge status={t.status} />
          </div>
          <h1 className="mt-4 max-w-3xl font-display text-4xl font-semibold tracking-tight sm:text-5xl">{t.title}</h1>
          <p className="mt-4 max-w-2xl text-lg text-muted">{t.description}</p>
          <p className="mt-5 font-mono text-xs text-muted">
            Updated <time dateTime={t.updated}>{formatDate(t.updated)}</time>
          </p>
        </Container>
      </div>
      <DocLayout toc={toc} className="pt-8">
        {t.draft && <DraftBanner />}
        <Section id="notes" title="Notes">
          {notes.length ? (
            <ul className="space-y-3">
              {notes.map((n) => (
                <li key={n.slug}>
                  <Link href={`/learn/${t.slug}/${n.slug}`} className="block rounded-[var(--radius-card)] border border-border bg-surface px-5 py-4 transition-colors hover:border-accent/45">
                    <p className="font-mono text-[0.7rem] text-muted">
                      {formatDate(n.frontmatter.updated ?? n.frontmatter.date)} · {n.readingMinutes} min read
                      {n.frontmatter.draft && <span className="text-warm"> · draft</span>}
                    </p>
                    <p className="mt-1 font-display text-lg font-semibold">{n.frontmatter.title}</p>
                    {n.frontmatter.description && <p className="mt-1 text-sm text-muted">{n.frontmatter.description}</p>}
                  </Link>
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState title="No notes yet" seed={3} />
          )}
        </Section>
        {t.resources.length > 0 && (
          <Section id="resources" title="Resources">
            <ul className="space-y-2">
              {t.resources.map((r) => (
                <li key={r.title} className="rounded-lg border border-border bg-surface px-4 py-3">
                  <p className="font-mono text-[0.7rem] text-muted">{r.kind}</p>
                  {r.url ? (
                    <a href={r.url} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 font-semibold hover:text-accent">
                      {r.title} <ExternalLink size={13} aria-hidden="true" />
                    </a>
                  ) : (
                    <p className="font-semibold">{r.title}</p>
                  )}
                  {r.note && <p className="text-sm text-muted">{r.note}</p>}
                </li>
              ))}
            </ul>
          </Section>
        )}
        {t.readingList.length > 0 && (
          <Section id="reading" title="Reading list">
            <ul className="space-y-2">
              {t.readingList.map((r) => (
                <li key={r.title} className="flex flex-wrap items-baseline justify-between gap-2 rounded-lg border border-border bg-surface px-4 py-3">
                  <span>
                    {r.url ? (
                      <a href={r.url} target="_blank" rel="noopener noreferrer" className="font-semibold hover:text-accent">
                        {r.title}
                      </a>
                    ) : (
                      <span className="font-semibold">{r.title}</span>
                    )}
                    {r.author && <span className="text-sm text-muted"> — {r.author}</span>}
                  </span>
                  <span className="font-mono text-[0.7rem] text-blue-soft">{r.state}</span>
                </li>
              ))}
            </ul>
          </Section>
        )}
        {t.roadmap.length > 0 && (
          <Section id="roadmap" title="Roadmap">
            <ol className="space-y-2">
              {t.roadmap.map((s) => (
                <li key={s.step} className="flex items-start gap-2 text-muted">
                  {s.done ? <CheckCircle2 size={17} className="mt-1 shrink-0 text-accent" aria-label="Done" /> : <Circle size={17} className="mt-1 shrink-0" aria-label="Not yet" />}
                  {s.step}
                </li>
              ))}
            </ol>
          </Section>
        )}
        {t.questions.length > 0 && (
          <Section id="questions" title="Questions">
            <ul className="space-y-3">
              {t.questions.map((q) => (
                <li key={q} className="border-l-2 border-accent pl-4 font-display text-lg">
                  {q}
                </li>
              ))}
            </ul>
          </Section>
        )}
        {t.learned.length > 0 && (
          <Section id="learned" title="Recently learned">
            <ul className="space-y-2">
              {t.learned.map((l) => (
                <li key={`${l.date}${l.text}`} className="grid gap-1 sm:grid-cols-[7rem_1fr]">
                  <time dateTime={l.date} className="font-mono text-xs text-muted">
                    {formatDate(l.date)}
                  </time>
                  <span>{l.text}</span>
                </li>
              ))}
            </ul>
          </Section>
        )}
        {relProjects.length + relWonder.length > 0 && (
          <Section id="related" title="Related">
            <ul className="space-y-2">
              {relProjects.map((p) => (
                <li key={p.slug}>
                  <Link href={`/projects/${p.slug}`} className="text-accent hover:underline">
                    Project · {p.title}
                  </Link>
                </li>
              ))}
              {relWonder.map((w) => (
                <li key={w.slug}>
                  <Link href={`/wonder-and-purpose/${w.slug}`} className="text-accent hover:underline">
                    Wonder &amp; Purpose · {w.title}
                  </Link>
                </li>
              ))}
            </ul>
          </Section>
        )}
      </DocLayout>
    </>
  );
}
