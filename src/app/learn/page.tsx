import type { Metadata } from "next";
import { canonical } from "@/lib/seo";
import Link from "next/link";
import { CheckCircle2, Circle } from "lucide-react";
import { PageHeader } from "@/components/layout/page-header";
import { DocLayout } from "@/components/layout/doc-layout";
import { Section } from "@/components/layout/section";
import { TopicCard } from "@/components/cards/topic-card";
import { EmptyState } from "@/components/content/empty-state";
import { LearningStatusBadge } from "@/components/ui/learning-status";
import { Card } from "@/components/ui/card";
import { getLearningNotes } from "@/lib/content";
import { visibleLearningTopics } from "@/lib/visibility";
import { learningStatuses } from "@/lib/schemas";
import { formatDate } from "@/lib/dates";

export const metadata: Metadata = {
  ...canonical("/learn"),
  title: "Learn",
  description: "A living record of what I’m currently learning, exploring and trying to understand.",
};

export default function LearnPage() {
  const topics = visibleLearningTopics().sort((a, b) => b.updated.localeCompare(a.updated));
  const current = topics.filter((t) => t.status === "Exploring" || t.status === "Learning" || t.status === "Implementing");
  const notes = topics
    .flatMap((t) => getLearningNotes(t.slug).map((n) => ({ ...n, topic: t })))
    .sort((a, b) => (b.frontmatter.updated ?? b.frontmatter.date).localeCompare(a.frontmatter.updated ?? a.frontmatter.date));
  const roadmaps = topics.filter((t) => t.roadmap.length);
  const reading = topics.flatMap((t) => t.readingList.map((r) => ({ ...r, topic: t })));
  const questions = topics.flatMap((t) => t.questions.map((q) => ({ q, topic: t })));
  const learned = topics.flatMap((t) => t.learned.map((l) => ({ ...l, topic: t }))).sort((a, b) => b.date.localeCompare(a.date)).slice(0, 8);

  const sections = [
    { id: "currently-learning", label: "Currently learning", show: true },
    { id: "learning-notes", label: "Learning notes", show: notes.length > 0 },
    { id: "roadmaps", label: "Learning roadmaps", show: roadmaps.length > 0 },
    { id: "reading-list", label: "Reading list", show: reading.length > 0 },
    { id: "open-questions", label: "Open questions", show: questions.length > 0 },
    { id: "recently-learned", label: "Recently learned", show: learned.length > 0 },
    { id: "library", label: "All topics", show: topics.length > 0 },
  ].filter((s) => s.show);
  const num = (id: string) => String(sections.findIndex((s) => s.id === id) + 1).padStart(2, "0");

  return (
    <>
      <PageHeader
        eyebrow="05 · Learn"
        title="Expanding orbits"
        subtitle="A living record of what I’m learning, exploring and trying to understand — notes, roadmaps, reading and the questions that come up along the way."
        motif="learn"
      >
        <ul className="mt-8 flex flex-wrap gap-2" aria-label="Learning statuses">
          {learningStatuses.map((s) => (
            <li key={s}>
              <LearningStatusBadge status={s} />
            </li>
          ))}
        </ul>
      </PageHeader>
      <DocLayout toc={sections.map((s) => ({ id: s.id, label: s.label }))} className="pt-6">
        <Section id="currently-learning" number={num("currently-learning")} title="Currently learning">
          {current.length ? (
            <div className="grid gap-5 sm:grid-cols-2">
              {current.map((t) => (
                <TopicCard
                  key={t.slug}
                  href={`/learn/${t.slug}`}
                  title={t.title}
                  description={t.description}
                  meta={t.status}
                  updated={t.updated}
                  variant="orbit"
                  badge={<LearningStatusBadge status={t.status} />}
                />
              ))}
            </div>
          ) : (
            <EmptyState title="The first orbits are being drawn" seed={14}>
              Topics I’m exploring will show up here with notes, resources and open questions.
            </EmptyState>
          )}
        </Section>

        {notes.length > 0 && (
          <Section id="learning-notes" number={num("learning-notes")} title="Learning notes">
            <ul className="divide-y divide-border rounded-[var(--radius-card)] border border-border bg-surface">
              {notes.map((n) => (
                <li key={`${n.topic.slug}/${n.slug}`}>
                  <Link href={`/learn/${n.topic.slug}/${n.slug}`} className="block px-5 py-4 transition-colors hover:bg-surface-elevated/50">
                    <p className="font-mono text-[0.7rem] text-muted">
                      {n.topic.title} · {formatDate(n.frontmatter.updated ?? n.frontmatter.date)} · {n.readingMinutes} min
                      {n.frontmatter.draft && <span className="text-warm"> · draft</span>}
                    </p>
                    <p className="mt-1 font-display font-semibold">{n.frontmatter.title}</p>
                  </Link>
                </li>
              ))}
            </ul>
          </Section>
        )}

        {roadmaps.length > 0 && (
          <Section id="roadmaps" number={num("roadmaps")} title="Learning roadmaps">
            <div className="grid gap-5 sm:grid-cols-2">
              {roadmaps.map((t) => (
                <Card key={t.slug}>
                  <Link href={`/learn/${t.slug}`} className="font-display font-semibold hover:text-accent">
                    {t.title}
                  </Link>
                  <ol className="mt-3 space-y-2">
                    {t.roadmap.map((s) => (
                      <li key={s.step} className="flex items-start gap-2 text-sm text-muted">
                        {s.done ? <CheckCircle2 size={16} className="mt-0.5 shrink-0 text-accent" aria-label="Done" /> : <Circle size={16} className="mt-0.5 shrink-0" aria-label="Not yet" />}
                        {s.step}
                      </li>
                    ))}
                  </ol>
                </Card>
              ))}
            </div>
          </Section>
        )}

        {reading.length > 0 && (
          <Section id="reading-list" number={num("reading-list")} title="Reading list">
            <ul className="grid gap-3 sm:grid-cols-2">
              {reading.map((r) => (
                <li key={`${r.topic.slug}-${r.title}`}>
                  <Card className="p-4">
                    <p className="font-mono text-[0.7rem] text-muted">
                      {r.state} · {r.topic.title}
                    </p>
                    <p className="mt-1 font-display font-semibold">
                      {r.url ? (
                        <a href={r.url} target="_blank" rel="noopener noreferrer" className="hover:text-accent">
                          {r.title}
                        </a>
                      ) : (
                        r.title
                      )}
                    </p>
                    {r.author && <p className="text-sm text-muted">{r.author}</p>}
                  </Card>
                </li>
              ))}
            </ul>
          </Section>
        )}

        {questions.length > 0 && (
          <Section id="open-questions" number={num("open-questions")} title="Open questions">
            <ul className="space-y-3">
              {questions.map(({ q, topic }) => (
                <li key={q} className="border-l-2 border-accent pl-4">
                  <p className="font-display text-[1.05rem]">{q}</p>
                  <Link href={`/learn/${topic.slug}`} className="font-mono text-[0.7rem] text-accent hover:underline">
                    {topic.title}
                  </Link>
                </li>
              ))}
            </ul>
          </Section>
        )}

        {learned.length > 0 && (
          <Section id="recently-learned" number={num("recently-learned")} title="Recently learned">
            <ul className="space-y-3">
              {learned.map((l) => (
                <li key={`${l.date}-${l.text}`} className="grid gap-1 sm:grid-cols-[7rem_1fr]">
                  <time dateTime={l.date} className="font-mono text-xs text-muted">
                    {formatDate(l.date)}
                  </time>
                  <p className="text-[0.95rem]">
                    {l.text}{" "}
                    <Link href={`/learn/${l.topic.slug}`} className="font-mono text-[0.7rem] text-accent hover:underline">
                      {l.topic.title}
                    </Link>
                  </p>
                </li>
              ))}
            </ul>
          </Section>
        )}

        {topics.length > 0 && (
          <Section id="library" number={num("library")} title="All topics">
            <div className="grid gap-5 sm:grid-cols-2">
              {topics.map((t) => (
                <TopicCard
                  key={t.slug}
                  href={`/learn/${t.slug}`}
                  title={t.title}
                  description={t.description}
                  meta={`${getLearningNotes(t.slug).length} notes`}
                  updated={t.updated}
                  variant="orbit"
                  badge={<LearningStatusBadge status={t.status} />}
                />
              ))}
            </div>
          </Section>
        )}
      </DocLayout>
    </>
  );
}
