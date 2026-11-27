import type { Metadata } from "next";
import { canonical } from "@/lib/seo";
import { PageHeader } from "@/components/layout/page-header";
import { DocLayout } from "@/components/layout/doc-layout";
import { Section } from "@/components/layout/section";
import { ExperienceCard } from "@/components/cards/experience-card";
import { AwardCard } from "@/components/cards/award-card";
import { EducationCard } from "@/components/cards/education-card";
import { experiences, focusAreas } from "@/data/experience";
import { awards } from "@/data/awards";
import { education } from "@/data/education";
import { projects } from "@/data/projects";
import { formatPeriod } from "@/lib/dates";
import { isDev } from "@/lib/utils";

export const metadata: Metadata = {
  ...canonical("/work"),
  title: "Work",
  description: "Professional experience, awards and education — technical depth, client work, product building and leadership.",
};

const toc = [
  { id: "experience", label: "Experience" },
  { id: "awards", label: "Awards & Recognitions" },
  { id: "education", label: "Education" },
];

export default function WorkPage() {
  return (
    <>
      <PageHeader
        eyebrow="02 · Work"
        title="Pathways, problems and people"
        subtitle="The places I’ve worked, what I built there, and how I approached the problems in front of me. Flip a card for the story behind it."
        motif="work"
      />
      <DocLayout toc={toc} className="pt-6">
        <Section
          id="experience"
          number="01"
          title="Experience"
          description={
            focusAreas.length > 0 && (
              <ul className="mt-1 flex flex-wrap gap-1.5" aria-label="Areas of work">
                {focusAreas.map((a) => (
                  <li key={a} className="rounded-md border border-border bg-surface px-2 py-0.5 font-mono text-[0.72rem] text-muted">
                    {a}
                  </li>
                ))}
              </ul>
            )
          }
        >
          <div className="grid gap-5 md:grid-cols-2">
            {experiences.map((e) => {
              const period = [formatPeriod(e.start), formatPeriod(e.end)].filter(Boolean).join(" – ") || null;
              const related = e.relatedProjects
                .map((slug) => projects.find((p) => p.slug === slug || p.id === slug))
                .filter((p): p is (typeof projects)[number] => !!p)
                .map((p) => ({ slug: p.slug, title: p.title }));
              return <ExperienceCard key={e.id} experience={e} period={period} related={related} devHints={isDev} />;
            })}
          </div>
        </Section>

        <Section id="awards" number="02" title="Awards & Recognitions">
          <ul className="grid gap-3 sm:grid-cols-2">
            {awards.map((a) => (
              <AwardCard key={a.id} award={a} />
            ))}
          </ul>
        </Section>

        <Section id="education" number="03" title="Education">
          <div className="grid gap-5 md:grid-cols-2">
            {education.map((e) => (
              <EducationCard key={e.id} education={e} />
            ))}
          </div>
        </Section>
      </DocLayout>
    </>
  );
}
