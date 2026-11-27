import type { Metadata } from "next";
import { canonical } from "@/lib/seo";
import Image from "next/image";
import { PageHeader } from "@/components/layout/page-header";
import { Container } from "@/components/layout/container";
import { ProjectsExplorer } from "@/components/projects/projects-explorer";
import { CategoryIllustration } from "@/components/illustrations/category-illustrations";
import { ProjectGlyph } from "@/components/illustrations/project-glyph";
import { projectCategories } from "@/data/project-categories";
import { projects } from "@/data/projects";
import { implementationTypes, projectStatuses } from "@/lib/schemas";

export const metadata: Metadata = {
  ...canonical("/projects"),
  title: "Projects",
  description:
    "A technical implementation library: machine learning from scratch, graph neural networks, federated and privacy-preserving ML, RAG and agentic AI.",
};

export default function ProjectsPage() {
  const sectionArt = Object.fromEntries(projectCategories.map((c) => [c.id, <CategoryIllustration key={c.id} kind={c.illustration} />]));
  const cardArt = Object.fromEntries(
    projects.map((p) => [
      p.id,
      p.thumbnail ? (
        <Image
          key={p.id}
          src={p.thumbnail.src}
          alt={p.thumbnail.alt}
          width={p.thumbnail.width ?? 640}
          height={p.thumbnail.height ?? 360}
          className="mb-4 h-32 w-full rounded-md object-cover"
        />
      ) : (
        <ProjectGlyph key={p.id} slug={p.slug} />
      ),
    ]),
  );
  const counts = {
    published: projects.filter((p) => p.status === "Published").length,
    progress: projects.filter((p) => p.status === "In Progress").length,
    planned: projects.filter((p) => p.status === "Planned").length,
  };
  return (
    <>
      <PageHeader
        eyebrow="03 · Projects"
        title="A universe of models and systems"
        subtitle="An organised library of implementations — from the mathematics of foundational algorithms to applied AI systems. Each project has its own page; most have their own repository."
        motif="projects"
      >
        <dl className="mt-8 flex flex-wrap gap-6 font-mono text-xs text-muted">
          <div>
            <dt className="inline">Published </dt>
            <dd className="inline text-accent">{counts.published}</dd>
          </div>
          <div>
            <dt className="inline">In progress </dt>
            <dd className="inline text-blue-soft">{counts.progress}</dd>
          </div>
          <div>
            <dt className="inline">Planned </dt>
            <dd className="inline text-text">{counts.planned}</dd>
          </div>
        </dl>
      </PageHeader>
      <Container className="pt-2">
        <ProjectsExplorer
          categories={projectCategories}
          projects={projects}
          sectionArt={sectionArt}
          cardArt={cardArt}
          implementationTypes={implementationTypes}
          statuses={projectStatuses}
        />
      </Container>
    </>
  );
}
