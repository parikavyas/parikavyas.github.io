import type { Metadata } from "next";
import { canonical } from "@/lib/seo";
import Link from "next/link";
import { PageHeader } from "@/components/layout/page-header";
import { DocLayout } from "@/components/layout/doc-layout";
import { Section } from "@/components/layout/section";
import { TopicCard } from "@/components/cards/topic-card";
import { EmptyState } from "@/components/content/empty-state";
import { Tag } from "@/components/ui/tag";
import { getWonderArticles } from "@/lib/content";
import { visibleWonderTopics } from "@/lib/visibility";
import { formatDate } from "@/lib/dates";

export const metadata: Metadata = {
  ...canonical("/wonder-and-purpose"),
  title: "Wonder & Purpose",
  description: "Things I care deeply about, questions I keep coming back to, and ideas I can’t stop exploring.",
};

export default function WonderPage() {
  const topics = visibleWonderTopics();
  const articles = topics
    .flatMap((t) => getWonderArticles(t.slug).map((a) => ({ ...a, topic: t })))
    .sort((a, b) => b.frontmatter.date.localeCompare(a.frontmatter.date))
    .slice(0, 6);
  const questions = topics.flatMap((t) => t.openQuestions.map((q) => ({ q, topic: t })));

  const toc = [
    { id: "collections", label: "Collections" },
    ...(articles.length ? [{ id: "recent-writing", label: "Recent writing" }] : []),
    ...(questions.length ? [{ id: "open-questions", label: "Open questions" }] : []),
  ];

  return (
    <>
      <PageHeader
        eyebrow="04 · Wonder & Purpose"
        title="Wonder & Purpose"
        subtitle="Things I care deeply about, questions I keep coming back to, and ideas I can’t stop exploring."
        motif="wonder"
      />
      <DocLayout toc={toc} className="pt-6">
        <Section id="collections" number="01" title="Collections">
          {topics.length ? (
            <div className="grid gap-5 sm:grid-cols-2">
              {topics.map((t) => {
                const count = getWonderArticles(t.slug).length + t.resources.length;
                return (
                  <TopicCard
                    key={t.slug}
                    href={`/wonder-and-purpose/${t.slug}`}
                    title={t.title}
                    description={t.description}
                    meta={count ? `${count} item${count === 1 ? "" : "s"}` : "Being gathered"}
                    updated={t.updated}
                    variant="constellation"
                    badge={t.status === "draft" ? <Tag tone="warm">Draft</Tag> : undefined}
                  />
                );
              })}
            </div>
          ) : (
            <EmptyState title="Constellations are still forming">
              Essays, perspectives and resources on the questions that matter most to me will appear here soon.
            </EmptyState>
          )}
        </Section>

        {articles.length > 0 && (
          <Section id="recent-writing" number="02" title="Recent writing">
            <ul className="divide-y divide-border rounded-[var(--radius-card)] border border-border bg-surface">
              {articles.map((a) => (
                <li key={`${a.topic.slug}/${a.slug}`}>
                  <Link href={`/wonder-and-purpose/${a.topic.slug}/${a.slug}`} className="block px-5 py-4 transition-colors hover:bg-surface-elevated/50">
                    <p className="font-mono text-[0.7rem] text-muted">
                      {a.frontmatter.kind} · {a.topic.title} · {formatDate(a.frontmatter.date)}
                    </p>
                    <p className="mt-1 font-display font-semibold">{a.frontmatter.title}</p>
                    {a.frontmatter.description && <p className="mt-1 text-sm text-muted">{a.frontmatter.description}</p>}
                  </Link>
                </li>
              ))}
            </ul>
          </Section>
        )}

        {questions.length > 0 && (
          <Section id="open-questions" number={articles.length ? "03" : "02"} title="Open questions">
            <ul className="space-y-3">
              {questions.map(({ q, topic }) => (
                <li key={q} className="rounded-[var(--radius-card)] border border-border bg-surface px-5 py-4">
                  <p className="font-display text-[1.05rem]">{q}</p>
                  <Link href={`/wonder-and-purpose/${topic.slug}`} className="mt-1 inline-block font-mono text-[0.7rem] text-accent hover:underline">
                    {topic.title}
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
