import type { MetadataRoute } from "next";
import { siteConfig } from "@/config/site";
import { primaryNav } from "@/config/navigation";
import { projects } from "@/data/projects";
import { wonderTopics } from "@/data/wonder-topics";
import { learningTopics } from "@/data/learning-topics";
import { getLearningNotes, getWonderArticles, getWritings } from "@/lib/content";

/**
 * Only real, published content is listed. Planned projects, draft topics,
 * draft articles and sample content are deliberately left out.
 */
export default function sitemap(): MetadataRoute.Sitemap {
  const base = siteConfig.url ?? "http://localhost:3000";
  const entry = (path: string, lastModified?: string): MetadataRoute.Sitemap[number] => ({
    url: `${base}${path}`,
    ...(lastModified ? { lastModified } : {}),
  });
  const published = (d: { draft: boolean }) => !d.draft;
  const topics = wonderTopics.filter((t) => t.status === "published");
  const learn = learningTopics.filter((t) => !t.draft);
  return [
    ...primaryNav.map((n) => entry(n.href === "/" ? "" : n.href, siteConfig.lastUpdated)),
    ...projects.filter((p) => p.status !== "Planned").map((p) => entry(`/projects/${p.slug}`, p.updated)),
    ...topics.map((t) => entry(`/wonder-and-purpose/${t.slug}`, t.updated)),
    ...topics.flatMap((t) =>
      getWonderArticles(t.slug)
        .filter((a) => published(a.frontmatter))
        .map((a) => entry(`/wonder-and-purpose/${t.slug}/${a.slug}`, a.frontmatter.updated ?? a.frontmatter.date)),
    ),
    ...learn.map((t) => entry(`/learn/${t.slug}`, t.updated)),
    ...learn.flatMap((t) =>
      getLearningNotes(t.slug)
        .filter((n) => published(n.frontmatter))
        .map((n) => entry(`/learn/${t.slug}/${n.slug}`, n.frontmatter.updated ?? n.frontmatter.date)),
    ),
    ...getWritings()
      .filter((w) => published(w.frontmatter))
      .map((w) => entry(`/beyond-the-desk/writing/${w.slug}`, w.frontmatter.updated ?? w.frontmatter.date)),
  ];
}
