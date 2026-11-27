"use client";

import { useMemo, useState } from "react";
import type { ImplementationType, Project, ProjectCategory, ProjectStatus } from "@/lib/schemas";
import { Scrollspy } from "@/components/navigation/scrollspy";
import { ProjectCard } from "@/components/cards/project-card";
import { cn } from "@/lib/utils";

function Chip({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={onClick}
      className={cn(
        "rounded-md border px-2.5 py-1 font-mono text-[0.72rem] transition-colors",
        active ? "border-accent/60 bg-accent/10 text-accent" : "border-border text-muted hover:border-blue-soft/60 hover:text-text",
      )}
    >
      {children}
    </button>
  );
}

/**
 * The /projects library: 11 category sections, light filtering
 * (category, implementation type, status) and the right-hand scrollspy.
 * Illustrations are rendered on the server and passed in.
 */
export function ProjectsExplorer({
  categories,
  projects,
  sectionArt,
  cardArt,
  implementationTypes,
  statuses,
  intro,
}: {
  categories: ProjectCategory[];
  projects: Project[];
  sectionArt: Record<string, React.ReactNode>;
  cardArt: Record<string, React.ReactNode>;
  implementationTypes: readonly ImplementationType[];
  statuses: readonly ProjectStatus[];
  intro?: React.ReactNode;
}) {
  const [category, setCategory] = useState<string>("all");
  const [type, setType] = useState<ImplementationType | null>(null);
  const [status, setStatus] = useState<ProjectStatus | null>(null);

  const labels = useMemo(() => Object.fromEntries(categories.map((c) => [c.id, c.title])), [categories]);
  const visibleCategories = categories.filter((c) => category === "all" || c.id === category);
  const matches = (p: Project) => (!type || p.implementationTypes.includes(type)) && (!status || p.status === status);
  const filtering = category !== "all" || type !== null || status !== null;
  const total = projects.filter(matches).filter((p) => category === "all" || p.categories.includes(category)).length;

  const toc = useMemo(() => visibleCategories.map((c) => ({ id: c.id, label: `${c.number} ${c.title}` })), [visibleCategories]);

  return (
    <div className="lg:grid lg:grid-cols-[minmax(0,1fr)_13.5rem] lg:gap-14 xl:gap-20">
      <div className="contents lg:order-2 lg:block">
        <Scrollspy items={toc} title="Categories" />
      </div>
      <div className="min-w-0 lg:order-1">
        {intro}
        <div role="search" aria-label="Filter projects" className="mt-8 space-y-3 rounded-[var(--radius-card)] border border-border bg-surface/60 p-4">
          <div className="flex flex-wrap items-center gap-3">
            <label htmlFor="category-filter" className="font-mono text-[0.7rem] uppercase tracking-[0.16em] text-muted">
              Category
            </label>
            <select
              id="category-filter"
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="min-w-0 max-w-full rounded-md border border-border bg-bg px-2.5 py-1.5 text-sm text-text"
            >
              <option value="all">All categories</option>
              {categories.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.number} · {c.title}
                </option>
              ))}
            </select>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="mr-1 font-mono text-[0.7rem] uppercase tracking-[0.16em] text-muted">Type</span>
            {implementationTypes.map((t) => (
              <Chip key={t} active={type === t} onClick={() => setType(type === t ? null : t)}>
                {t}
              </Chip>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="mr-1 font-mono text-[0.7rem] uppercase tracking-[0.16em] text-muted">Status</span>
            {statuses.map((s) => (
              <Chip key={s} active={status === s} onClick={() => setStatus(status === s ? null : s)}>
                {s}
              </Chip>
            ))}
            {filtering && (
              <button
                type="button"
                onClick={() => {
                  setCategory("all");
                  setType(null);
                  setStatus(null);
                }}
                className="ml-auto text-xs text-accent underline-offset-2 hover:underline"
              >
                Clear filters
              </button>
            )}
          </div>
          <p className="font-mono text-[0.7rem] text-muted" aria-live="polite">
            {total} project{total === 1 ? "" : "s"} shown
          </p>
        </div>

        {visibleCategories.map((c) => {
          const list = projects.filter((p) => p.categories.includes(c.id) && matches(p));
          const hasAny = projects.some((p) => p.categories.includes(c.id));
          return (
            <section key={c.id} id={c.id} aria-labelledby={`${c.id}-title`} className=" border-b border-border/60 py-12 last:border-0">
              <div className="grid items-center gap-6 sm:grid-cols-[minmax(0,1fr)_13rem]">
                <div>
                  <p className="font-mono text-sm text-accent">{c.number}</p>
                  <h2 id={`${c.id}-title`} className="mt-1 font-display text-2xl font-semibold tracking-tight sm:text-3xl">
                    {c.title}
                  </h2>
                  <p className="mt-2 max-w-xl text-muted">{c.description}</p>
                </div>
                <div className="hidden opacity-80 sm:block">{sectionArt[c.id]}</div>
              </div>
              {list.length > 0 ? (
                <div className="mt-8 grid gap-5 md:grid-cols-2">
                  {list.map((p) => (
                    <ProjectCard
                      key={p.id}
                      project={p}
                      illustration={cardArt[p.id]}
                      categoryLabels={p.categories.map((id) => labels[id]).filter(Boolean)}
                    />
                  ))}
                </div>
              ) : (
                <p className="mt-8 rounded-[var(--radius-card)] border border-dashed border-border px-5 py-6 text-sm text-muted">
                  {hasAny ? "No projects in this category match the current filters." : c.emptyMessage ?? "Projects for this category are on the way."}
                </p>
              )}
            </section>
          );
        })}
      </div>
    </div>
  );
}
