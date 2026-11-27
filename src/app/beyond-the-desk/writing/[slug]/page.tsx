import type { Metadata } from "next";
import { canonical } from "@/lib/seo";
import { notFound } from "next/navigation";
import { ArticleHeader } from "@/components/content/article-header";
import { Container } from "@/components/layout/container";
import { ReadingProgress } from "@/components/navigation/reading-progress";
import { DraftBanner } from "@/components/ui/placeholder";
import { getWriting, getWritings } from "@/lib/content";
import { renderDocument } from "@/lib/render-document";

export const dynamicParams = false;

export function generateStaticParams() {
  return getWritings().map((w) => ({ slug: w.slug }));
}

type Params = { params: Promise<{ slug: string }> };

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const { slug } = await params;
  const w = getWriting(slug);
  if (!w) return {};
  return {
    ...canonical(`/beyond-the-desk/writing/${slug}`),
    title: w.frontmatter.title,
    description: w.frontmatter.excerpt,
    robots: w.frontmatter.draft ? { index: false, follow: false } : undefined,
  };
}

/** Typography-first reading page for poetry, stories and reflections. */
export default async function WritingPage({ params }: Params) {
  const { slug } = await params;
  const w = getWriting(slug);
  if (!w) notFound();
  const doc = await renderDocument(w.body, { format: w.format, trusted: true, className: "prose-writing mx-auto" });
  return (
    <>
      <ReadingProgress targetId="article" />
      <ArticleHeader
        crumbs={[{ label: "Beyond the Desk", href: "/beyond-the-desk" }, { label: "Between the Lines", href: "/beyond-the-desk#between-the-lines" }, { label: w.frontmatter.title }]}
        kind={w.frontmatter.form}
        title={w.frontmatter.title}
        date={w.frontmatter.date}
        updated={w.frontmatter.updated}
        readingMinutes={doc.readingMinutes}
      />
      <Container className="pt-12">
        {w.frontmatter.draft && (
          <div className="mx-auto max-w-[62ch]">
            <DraftBanner />
          </div>
        )}
        <article id="article" className="mx-auto max-w-[62ch]">
          {doc.node}
        </article>
      </Container>
    </>
  );
}
