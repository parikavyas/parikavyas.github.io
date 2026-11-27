import type { Metadata } from "next";
import { canonical } from "@/lib/seo";
import Link from "next/link";
import { notFound } from "next/navigation";
import { Clock, ExternalLink, RefreshCw } from "lucide-react";
import { Github } from "@/components/ui/brand-icons";
import { DocLayout } from "@/components/layout/doc-layout";
import { Container } from "@/components/layout/container";
import { Breadcrumbs } from "@/components/navigation/breadcrumbs";
import { ReadingProgress } from "@/components/navigation/reading-progress";
import { StatusBadge } from "@/components/ui/status-badge";
import { Tag, TagList } from "@/components/ui/tag";
import { LinkButton } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ProjectGlyph } from "@/components/illustrations/project-glyph";
import { CategoryIllustration } from "@/components/illustrations/category-illustrations";
import { LazyVisualization } from "@/components/visualizations/lazy-visualization";
import { getProject, projects } from "@/data/projects";
import { getCategory } from "@/data/project-categories";
import { learningTopics } from "@/data/learning-topics";
import { wonderTopics } from "@/data/wonder-topics";
import { getProjectDoc } from "@/lib/content";
import { getReadme, repoUrl } from "@/lib/github";
import { renderDocument, type RenderedDocument } from "@/lib/render-document";
import { projectTemplates } from "@/lib/project-templates";
import { formatDate } from "@/lib/dates";
import { showDrafts } from "@/lib/utils";
import type { Project } from "@/lib/schemas";

export const revalidate = 86400; // keep in sync with README_REVALIDATE_SECONDS
export const dynamicParams = false;

export function generateStaticParams() {
  return projects.map((p) => ({ slug: p.slug }));
}

type Params = { params: Promise<{ slug: string }> };

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const { slug } = await params;
  const p = getProject(slug);
  if (!p) return {};
  return {
    title: p.title,
    description: p.summary,
    openGraph: { title: p.title, description: p.summary, type: "article" },
    ...canonical(`/projects/${p.slug}`),
    // Planned projects have no real content yet: keep them out of search.
    robots: p.status === "Planned" ? { index: false, follow: true } : undefined,
  };
}

interface LoadedDoc {
  doc: RenderedDocument | null;
  source: "readme" | "local" | "none";
  updated: string;
  readmeUrl?: string;
  readmeFailed: boolean;
}

async function loadDoc(p: Project): Promise<LoadedDoc> {
  let readmeFailed = false;
  if (p.repo && p.status !== "Planned") {
    const readme = await getReadme(p.repo);
    if (readme) {
      const doc = await renderDocument(readme.markdown, {
        format: "md",
        trusted: false,
        repo: p.repo,
        dropLeadingH1: true,
      });
      return {
        doc,
        source: "readme",
        updated: readme.lastCommit?.slice(0, 10) ?? p.updated,
        readmeUrl: readme.url,
        readmeFailed,
      };
    }
    readmeFailed = true;
  }
  if (p.localDoc) {
    const local = getProjectDoc(p.localDoc);
    if (local) {
      const doc = await renderDocument(local.body, { format: local.format, trusted: true });
      return { doc, source: "local", updated: local.frontmatter.updated ?? p.updated, readmeFailed };
    }
  }
  return { doc: null, source: "none", updated: p.updated, readmeFailed };
}

function relatedProjects(p: Project) {
  const explicit = p.related.projects
    .map((ref) => projects.find((q) => q.slug === ref || q.id === ref))
    .filter((q): q is Project => !!q && q.id !== p.id);
  const sameCategory = projects.filter((q) => q.id !== p.id && q.categories.some((c) => p.categories.includes(c)) && !explicit.includes(q));
  return [...explicit, ...sameCategory].slice(0, 3);
}

export default async function ProjectPage({ params }: Params) {
  const { slug } = await params;
  const p = getProject(slug);
  if (!p) notFound();

  const loaded = await loadDoc(p);
  const categories = p.categories.map(getCategory).filter((c) => !!c);
  const primary = categories[0];
  const live = p.status !== "Planned";
  const template = projectTemplates[p.template];

  const toc = [
    ...(loaded.doc?.headings ?? []).map((h) => ({ id: h.id, label: h.text, level: h.level })),
    ...(!loaded.doc ? [{ id: "what-this-page-will-cover", label: "What this page will cover" }] : []),
    ...(p.visualization ? [{ id: "interactive-visualization", label: "Interactive visualization" }] : []),
    { id: "related", label: "Related" },
  ];

  const learn = learningTopics.filter((t) => (showDrafts || !t.draft) && (p.related.learn.includes(t.slug) || t.related.projects.some((r) => r === p.slug || r === p.id)));
  const wonder = wonderTopics.filter((t) => (showDrafts || t.status === "published") && (p.related.wonder.includes(t.slug) || t.related.projects.some((r) => r === p.slug || r === p.id)));
  const related = relatedProjects(p);

  return (
    <>
      <ReadingProgress targetId="article" />
      <div className="relative overflow-hidden border-b border-border/70">
        <div className="bg-grid mask-fade-b absolute inset-0 opacity-40" aria-hidden="true" />
        <Container className="relative py-10 md:py-14">
          <Breadcrumbs
            items={[
              { label: "Projects", href: "/projects" },
              ...(primary ? [{ label: primary.title, href: `/projects#${primary.id}` }] : []),
              { label: p.title },
            ]}
          />
          <div className="mt-6 grid items-end gap-8 md:grid-cols-[minmax(0,1fr)_16rem]">
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <StatusBadge status={p.status} />
                {p.implementationTypes.map((t) => (
                  <Tag key={t} tone="blue">
                    {t}
                  </Tag>
                ))}
              </div>
              <h1 className="mt-4 font-display text-3xl font-semibold leading-tight tracking-tight sm:text-5xl">{p.title}</h1>
              <p className="mt-4 max-w-2xl text-lg text-muted">{p.summary}</p>
              <TagList tags={p.tags} className="mt-5" />
              <div className="mt-6 flex flex-wrap items-center gap-3">
                {live && p.repo && (
                  <LinkButton href={repoUrl(p.repo)} variant="primary">
                    <Github size={16} aria-hidden="true" /> View on GitHub
                  </LinkButton>
                )}
                {live && p.demoUrl && (
                  <LinkButton href={p.demoUrl}>
                    <ExternalLink size={16} aria-hidden="true" /> Live demo
                  </LinkButton>
                )}
              </div>
              <p className="mt-6 flex flex-wrap gap-x-5 gap-y-1 font-mono text-xs text-muted">
                <span className="inline-flex items-center gap-1.5">
                  <RefreshCw size={12} aria-hidden="true" /> Updated <time dateTime={loaded.updated}>{formatDate(loaded.updated)}</time>
                </span>
                {loaded.doc && (
                  <span className="inline-flex items-center gap-1.5">
                    <Clock size={12} aria-hidden="true" /> {loaded.doc.readingMinutes} min read
                  </span>
                )}
                {loaded.source === "readme" && <span>Synced from the repository README</span>}
              </p>
            </div>
            <div className="hidden opacity-85 md:block">{primary && <CategoryIllustration kind={primary.illustration} />}</div>
          </div>
        </Container>
      </div>

      <DocLayout toc={toc} className="pt-10">
        <article id="article" className="min-w-0 pb-4">
          {loaded.readmeFailed && (
            <p className="mb-6 rounded-lg border border-border bg-surface px-4 py-3 text-sm text-muted">
              The repository README couldn’t be loaded right now{loaded.doc ? ", so a local copy is shown" : ""}. You can still read it{" "}
              {p.repo && (
                <a className="text-accent underline" href={repoUrl(p.repo)} target="_blank" rel="noopener noreferrer">
                  on GitHub
                </a>
              )}
              .
            </p>
          )}

          {loaded.doc ? (
            loaded.doc.node
          ) : (
            <section id="what-this-page-will-cover" >
              <Card className="border-dashed">
                <ProjectGlyph slug={p.slug} />
                <h2 className="mt-4 font-display text-xl font-semibold">
                  {p.status === "Planned" ? "This project is planned" : "Write-up in progress"}
                </h2>
                <p className="mt-2 text-muted">
                  {p.status === "Planned"
                    ? "Nothing has been implemented or published yet. When it is, this page will follow the structure below."
                    : "The implementation is under way. The full write-up will follow the structure below."}
                </p>
                <p className="mt-5 font-mono text-[0.7rem] uppercase tracking-[0.18em] text-muted">{template.name}</p>
                <ol className="mt-3 grid gap-x-6 gap-y-1.5 sm:grid-cols-2">
                  {template.sections.map((s, i) => (
                    <li key={s} className="flex gap-3 text-sm text-muted">
                      <span className="font-mono text-accent">{String(i + 1).padStart(2, "0")}</span> {s}
                    </li>
                  ))}
                </ol>
              </Card>
            </section>
          )}

          {p.visualization && (
            <section id="interactive-visualization" aria-labelledby="viz-title" className="mt-12">
              <h2 id="viz-title" className="font-display text-2xl font-semibold tracking-tight">
                Interactive visualization
              </h2>
              <LazyVisualization id={p.visualization} />
            </section>
          )}
        </article>

        <section id="related" aria-labelledby="related-title" className="mt-16 border-t border-border pt-10">
          <h2 id="related-title" className="font-display text-2xl font-semibold tracking-tight">
            Related
          </h2>
          {related.length > 0 && (
            <>
              <h3 className="mt-6 font-mono text-[0.7rem] uppercase tracking-[0.18em] text-muted">Projects</h3>
              <ul className="mt-3 grid gap-3 sm:grid-cols-3">
                {related.map((r) => (
                  <li key={r.id}>
                    <Link href={`/projects/${r.slug}`} className="block h-full rounded-[var(--radius-card)] border border-border bg-surface p-4 transition-colors hover:border-accent/45">
                      <StatusBadge status={r.status} />
                      <p className="mt-2 font-display font-semibold leading-snug">{r.title}</p>
                    </Link>
                  </li>
                ))}
              </ul>
            </>
          )}
          {learn.length > 0 && (
            <>
              <h3 className="mt-8 font-mono text-[0.7rem] uppercase tracking-[0.18em] text-muted">Learning notes</h3>
              <ul className="mt-3 space-y-2">
                {learn.map((t) => (
                  <li key={t.slug}>
                    <Link href={`/learn/${t.slug}`} className="text-accent hover:underline">
                      {t.title}
                    </Link>
                  </li>
                ))}
              </ul>
            </>
          )}
          {wonder.length > 0 && (
            <>
              <h3 className="mt-8 font-mono text-[0.7rem] uppercase tracking-[0.18em] text-muted">Wonder &amp; Purpose</h3>
              <ul className="mt-3 space-y-2">
                {wonder.map((t) => (
                  <li key={t.slug}>
                    <Link href={`/wonder-and-purpose/${t.slug}`} className="text-accent hover:underline">
                      {t.title}
                    </Link>
                  </li>
                ))}
              </ul>
            </>
          )}
        </section>
      </DocLayout>
    </>
  );
}
