import type { Metadata } from "next";
import { canonical } from "@/lib/seo";
import Image from "next/image";
import Link from "next/link";
import { PageHeader } from "@/components/layout/page-header";
import { DocLayout } from "@/components/layout/doc-layout";
import { Section } from "@/components/layout/section";
import { KitchenGallery } from "@/components/gallery/kitchen-gallery";
import { VideoEmbed } from "@/components/content/video-embed";
import { EmptyState } from "@/components/content/empty-state";
import { OrganicForms } from "@/components/illustrations/primitives";
import { Tag } from "@/components/ui/tag";
import { danceItems, kitchenItems } from "@/data/gallery";
import { getWritings } from "@/lib/content";
import { formatDate } from "@/lib/dates";
import { showDrafts } from "@/lib/utils";

export const metadata: Metadata = {
  ...canonical("/beyond-the-desk"),
  title: "Beyond the Desk",
  description: "Cooking, dancing and creative writing — the parts of life that happen away from the keyboard.",
};

const toc = [
  { id: "from-my-kitchen", label: "From My Kitchen" },
  { id: "on-the-dance-floor", label: "On the Dance Floor" },
  { id: "between-the-lines", label: "Between the Lines" },
];

export default function BeyondTheDeskPage() {
  const kitchen = kitchenItems.filter((k) => showDrafts || !k.draft);
  const dance = danceItems.filter((d) => showDrafts || !d.draft);
  const writing = getWritings();
  return (
    <>
      <PageHeader
        eyebrow="06 · Beyond the Desk"
        title="Beyond the Desk"
        subtitle="Food, movement and words — the parts of life that happen away from the keyboard."
        motif="beyond"
      />
      <DocLayout toc={toc} className="pt-6">
        <Section id="from-my-kitchen" number="01" title="From My Kitchen" aside={<OrganicForms className="h-auto w-full opacity-80" seed={31} />}>
          {kitchen.length ? (
            <KitchenGallery items={kitchen} />
          ) : (
            <EmptyState title="The kitchen is warming up" seed={31}>
              Photos of dishes, recipes and the stories behind them are on their way.
            </EmptyState>
          )}
        </Section>

        <Section id="on-the-dance-floor" number="02" title="On the Dance Floor" aside={<OrganicForms className="h-auto w-full opacity-80" seed={47} />}>
          {dance.length ? (
            <ul className="grid gap-5 md:grid-cols-2">
              {dance.map((d) => (
                <li key={d.id} className="overflow-hidden rounded-[var(--radius-card)] border border-border bg-surface">
                  {d.video ? (
                    <VideoEmbed provider={d.video.provider} id={d.video.id} title={d.title} thumbnail={d.video.thumbnail} />
                  ) : d.image ? (
                    <Image src={d.image.src} alt={d.image.alt} width={d.image.width ?? 1200} height={d.image.height ?? 800} sizes="(min-width: 768px) 45vw, 100vw" className="h-auto w-full" />
                  ) : null}
                  <div className="p-5">
                    <div className="flex flex-wrap gap-1.5">
                      {d.style && <Tag tone="warm">{d.style}</Tag>}
                      {d.occasion && <Tag>{d.occasion}</Tag>}
                      {d.draft && <Tag tone="warm">Draft</Tag>}
                    </div>
                    <h3 className="mt-3 font-display text-lg font-semibold">{d.title}</h3>
                    {d.date && <p className="font-mono text-[0.7rem] text-muted">{formatDate(d.date)}</p>}
                    {d.description && <p className="mt-2 text-sm text-muted">{d.description}</p>}
                    {d.choreography && (
                      <p className="mt-2 text-sm text-muted">
                        <span className="text-text">Choreography:</span> {d.choreography}
                      </p>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState title="Music’s about to start" seed={47}>
              Performances, photographs and videos will be shared here.
            </EmptyState>
          )}
        </Section>

        <Section id="between-the-lines" number="03" title="Between the Lines" aside={<OrganicForms className="h-auto w-full opacity-80" seed={59} />}>
          {writing.length ? (
            <ul className="divide-y divide-border border-y border-border">
              {writing.map((w) => (
                <li key={w.slug}>
                  <Link href={`/beyond-the-desk/writing/${w.slug}`} className="group grid gap-1 py-5 sm:grid-cols-[8rem_1fr]">
                    <span className="font-mono text-[0.7rem] uppercase tracking-[0.15em] text-warm">{w.frontmatter.form}</span>
                    <span>
                      <span className="font-display text-xl font-semibold transition-colors group-hover:text-accent">{w.frontmatter.title}</span>
                      {w.frontmatter.draft && <span className="ml-2 font-mono text-[0.7rem] text-warm">draft</span>}
                      {w.frontmatter.excerpt && <span className="mt-1 block italic text-muted">{w.frontmatter.excerpt}</span>}
                      <span className="mt-1 block font-mono text-[0.7rem] text-muted">{formatDate(w.frontmatter.date)}</span>
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState title="The first page is still blank" seed={59}>
              Poems, reflections and short stories will live here.
            </EmptyState>
          )}
        </Section>
      </DocLayout>
    </>
  );
}
