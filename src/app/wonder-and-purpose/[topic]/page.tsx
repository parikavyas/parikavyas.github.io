import type { Metadata } from "next";
import { canonical } from "@/lib/seo";
import Link from "next/link";
import { notFound } from "next/navigation";
import { ExternalLink } from "lucide-react";
import { DocLayout } from "@/components/layout/doc-layout";
import { Section } from "@/components/layout/section";
import { Container } from "@/components/layout/container";
import { Breadcrumbs } from "@/components/navigation/breadcrumbs";
import { Constellation } from "@/components/illustrations/primitives";
import { DocumentPreview } from "@/components/content/document-preview";
import { VideoEmbed } from "@/components/content/video-embed";
import { EmptyState } from "@/components/content/empty-state";
import { DraftBanner } from "@/components/ui/placeholder";
import { projects } from "@/data/projects";
import { getWonderArticles } from "@/lib/content";
import { visibleLearningTopics, visibleWonderTopics } from "@/lib/visibility";
import { formatDate } from "@/lib/dates";
import { hashString } from "@/lib/utils";
import type { WonderResource } from "@/lib/schemas";

export const dynamicParams = false;

export function generateStaticParams() {
  return visibleWonderTopics().map((t) => ({ topic: t.slug }));
}

type Params = { params: Promise<{ topic: string }> };

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const { topic } = await params;
  const t = visibleWonderTopics().find((x) => x.slug === topic);
  if (!t) return {};
  return {
    ...canonical(`/wonder-and-purpose/${t.slug}`),
    title: t.title,
    description: t.description,
    robots: t.status === "draft" ? { index: false, follow: false } : undefined,
  };
}

function youtubeId(url: string) {
  const m = url.match(/(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([\w-]{6,})/);
  return m?.[1] ?? null;
}

const groups: { id: string; title: string; types: WonderResource["type"][] }[] = [
  { id: "papers", title: "Research papers", types: ["Research paper"] },
  { id: "documents", title: "Decks & documents", types: ["Presentation deck", "PDF"] },
  { id: "videos", title: "Videos", types: ["Video"] },
  { id: "resources", title: "External resources", types: ["External resource"] },
];

function ResourceItem({ r }: { r: WonderResource }) {
  if ((r.type === "PDF" || r.type === "Presentation deck") && r.preview) return <DocumentPreview href={r.href} title={r.title} />;
  const yt = r.type === "Video" ? youtubeId(r.href) : null;
  if (yt && r.preview) {
    return (
      <div className="space-y-2">
        <VideoEmbed provider="youtube" id={yt} title={r.title} />
        <p className="text-sm text-muted">{r.description}</p>
      </div>
    );
  }
  const external = /^https?:\/\//.test(r.href);
  return (
    <a
      href={r.href}
      {...(external ? { target: "_blank", rel: "noopener noreferrer" } : {})}
      className="flex items-start justify-between gap-4 rounded-[var(--radius-card)] border border-border bg-surface px-5 py-4 transition-colors hover:border-accent/45"
    >
      <span>
        <span className="font-mono text-[0.7rem] text-muted">
          {r.type}
          {r.date && ` · ${formatDate(r.date)}`}
        </span>
        <span className="mt-1 block font-display font-semibold">{r.title}</span>
        {r.description && <span className="mt-1 block text-sm text-muted">{r.description}</span>}
      </span>
      <ExternalLink size={15} className="mt-1 shrink-0 text-accent" aria-hidden="true" />
    </a>
  );
}

export default async function WonderTopicPage({ params }: Params) {
  const { topic } = await params;
  const t = visibleWonderTopics().find((x) => x.slug === topic);
  if (!t) notFound();

  const articles = getWonderArticles(t.slug);
  const presentGroups = groups.map((g) => ({ ...g, items: t.resources.filter((r) => g.types.includes(r.type)) })).filter((g) => g.items.length);
  const relProjects = projects.filter((p) => t.related.projects.includes(p.slug) || t.related.projects.includes(p.id) || p.related.wonder.includes(t.slug));
  const relLearn = visibleLearningTopics().filter((l) => t.related.learn.includes(l.slug) || l.related.wonder.includes(t.slug));
  const empty = !articles.length && !presentGroups.length && !t.openQuestions.length;

  const toc = [
    ...(articles.length ? [{ id: "writing", label: "Articles & perspectives" }] : []),
    ...presentGroups.map((g) => ({ id: g.id, label: g.title })),
    ...(t.openQuestions.length ? [{ id: "questions", label: "Open questions" }] : []),
    ...(relProjects.length || relLearn.length ? [{ id: "related", label: "Related" }] : []),
  ];

  return (
    <>
      <div className="relative overflow-hidden border-b border-border/70">
        <Constellation className="absolute -right-10 top-0 h-full w-[60%] opacity-50" seed={hashString(t.slug) % 997} count={9} field={50} />
        <Container className="relative py-12 md:py-16">
          <Breadcrumbs items={[{ label: "Wonder & Purpose", href: "/wonder-and-purpose" }, { label: t.title }]} />
          <h1 className="mt-6 max-w-3xl font-display text-4xl font-semibold tracking-tight sm:text-5xl">{t.title}</h1>
          <p className="mt-4 max-w-2xl text-lg text-muted">{t.description}</p>
          <p className="mt-5 font-mono text-xs text-muted">
            Updated <time dateTime={t.updated}>{formatDate(t.updated)}</time>
          </p>
        </Container>
      </div>
      <DocLayout toc={toc} className="pt-8">
        {t.status === "draft" && <DraftBanner />}
        {empty && (
          <EmptyState title="This collection is still being gathered" seed={hashString(t.slug) % 50}>
            Writing, papers and resources on this theme will appear here.
          </EmptyState>
        )}
        {articles.length > 0 && (
          <Section id="writing" title="Articles & perspectives">
            <ul className="space-y-3">
              {articles.map((a) => (
                <li key={a.slug}>
                  <Link href={`/wonder-and-purpose/${t.slug}/${a.slug}`} className="block rounded-[var(--radius-card)] border border-border bg-surface px-5 py-4 transition-colors hover:border-accent/45">
                    <p className="font-mono text-[0.7rem] text-muted">
                      {a.frontmatter.kind} · {formatDate(a.frontmatter.date)} · {a.readingMinutes} min read
                      {a.frontmatter.draft && <span className="text-warm"> · draft</span>}
                    </p>
                    <p className="mt-1 font-display text-lg font-semibold">{a.frontmatter.title}</p>
                    {a.frontmatter.description && <p className="mt-1 text-sm text-muted">{a.frontmatter.description}</p>}
                  </Link>
                </li>
              ))}
            </ul>
          </Section>
        )}
        {presentGroups.map((g) => (
          <Section key={g.id} id={g.id} title={g.title}>
            <div className="space-y-3">
              {g.items.map((r) => (
                <ResourceItem key={r.href} r={r} />
              ))}
            </div>
          </Section>
        ))}
        {t.openQuestions.length > 0 && (
          <Section id="questions" title="Open questions">
            <ul className="space-y-3">
              {t.openQuestions.map((q) => (
                <li key={q} className="border-l-2 border-accent pl-4 font-display text-lg">
                  {q}
                </li>
              ))}
            </ul>
          </Section>
        )}
        {(relProjects.length > 0 || relLearn.length > 0) && (
          <Section id="related" title="Related">
            <ul className="space-y-2">
              {relProjects.map((p) => (
                <li key={p.slug}>
                  <Link href={`/projects/${p.slug}`} className="text-accent hover:underline">
                    Project · {p.title}
                  </Link>
                </li>
              ))}
              {relLearn.map((l) => (
                <li key={l.slug}>
                  <Link href={`/learn/${l.slug}`} className="text-accent hover:underline">
                    Learning · {l.title}
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
