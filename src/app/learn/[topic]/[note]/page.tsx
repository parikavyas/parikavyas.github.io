import type { Metadata } from "next";
import { canonical } from "@/lib/seo";
import { notFound } from "next/navigation";
import { ArticleHeader } from "@/components/content/article-header";
import { DocLayout } from "@/components/layout/doc-layout";
import { ReadingProgress } from "@/components/navigation/reading-progress";
import { DraftBanner } from "@/components/ui/placeholder";
import { getLearningNote, getLearningNotes } from "@/lib/content";
import { visibleLearningTopics } from "@/lib/visibility";
import { renderDocument } from "@/lib/render-document";

export const dynamicParams = false;

export function generateStaticParams() {
  return visibleLearningTopics().flatMap((t) => getLearningNotes(t.slug).map((n) => ({ topic: t.slug, note: n.slug })));
}

type Params = { params: Promise<{ topic: string; note: string }> };

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const { topic, note } = await params;
  const n = getLearningNote(topic, note);
  const t = visibleLearningTopics().find((x) => x.slug === topic);
  if (!n || !t) return {};
  return {
    ...canonical(`/learn/${topic}/${note}`),
    title: n.frontmatter.title,
    description: n.frontmatter.description,
    robots: n.frontmatter.draft || t.draft ? { index: false, follow: false } : undefined,
  };
}

export default async function LearningNotePage({ params }: Params) {
  const { topic, note } = await params;
  const t = visibleLearningTopics().find((x) => x.slug === topic);
  const n = getLearningNote(topic, note);
  if (!t || !n) notFound();
  const doc = await renderDocument(n.body, { format: n.format, trusted: true });
  return (
    <>
      <ReadingProgress targetId="article" />
      <ArticleHeader
        crumbs={[{ label: "Learn", href: "/learn" }, { label: t.title, href: `/learn/${t.slug}` }, { label: n.frontmatter.title }]}
        kind="Learning note"
        title={n.frontmatter.title}
        description={n.frontmatter.description}
        date={n.frontmatter.date}
        updated={n.frontmatter.updated}
        readingMinutes={doc.readingMinutes}
        tags={n.frontmatter.tags}
      />
      <DocLayout toc={doc.headings.map((h) => ({ id: h.id, label: h.text, level: h.level }))} className="pt-10">
        {(n.frontmatter.draft || t.draft) && <DraftBanner />}
        <article id="article">{doc.node}</article>
      </DocLayout>
    </>
  );
}
