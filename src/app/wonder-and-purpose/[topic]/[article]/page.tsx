import type { Metadata } from "next";
import { canonical } from "@/lib/seo";
import { notFound } from "next/navigation";
import { ArticleHeader } from "@/components/content/article-header";
import { DocLayout } from "@/components/layout/doc-layout";
import { ReadingProgress } from "@/components/navigation/reading-progress";
import { DraftBanner } from "@/components/ui/placeholder";
import { getWonderArticle, getWonderArticles } from "@/lib/content";
import { visibleWonderTopics } from "@/lib/visibility";
import { renderDocument } from "@/lib/render-document";

export const dynamicParams = false;

export function generateStaticParams() {
  return visibleWonderTopics().flatMap((t) => getWonderArticles(t.slug).map((a) => ({ topic: t.slug, article: a.slug })));
}

type Params = { params: Promise<{ topic: string; article: string }> };

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const { topic, article } = await params;
  const a = getWonderArticle(topic, article);
  if (!a) return {};
  return {
    ...canonical(`/wonder-and-purpose/${topic}/${article}`),
    title: a.frontmatter.title,
    description: a.frontmatter.description,
    openGraph: { type: "article", title: a.frontmatter.title, description: a.frontmatter.description },
    robots: a.frontmatter.draft ? { index: false, follow: false } : undefined,
  };
}

export default async function WonderArticlePage({ params }: Params) {
  const { topic, article } = await params;
  const t = visibleWonderTopics().find((x) => x.slug === topic);
  const a = getWonderArticle(topic, article);
  if (!t || !a) notFound();
  const doc = await renderDocument(a.body, { format: a.format, trusted: true });
  return (
    <>
      <ReadingProgress targetId="article" />
      <ArticleHeader
        crumbs={[{ label: "Wonder & Purpose", href: "/wonder-and-purpose" }, { label: t.title, href: `/wonder-and-purpose/${t.slug}` }, { label: a.frontmatter.title }]}
        kind={a.frontmatter.kind}
        title={a.frontmatter.title}
        description={a.frontmatter.description}
        date={a.frontmatter.date}
        updated={a.frontmatter.updated}
        readingMinutes={doc.readingMinutes}
        tags={a.frontmatter.tags}
      />
      <DocLayout toc={doc.headings.map((h) => ({ id: h.id, label: h.text, level: h.level }))} className="pt-10">
        {a.frontmatter.draft && <DraftBanner />}
        <article id="article">{doc.node}</article>
      </DocLayout>
    </>
  );
}
