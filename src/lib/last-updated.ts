import "server-only";
import { siteConfig } from "@/config/site";
import { projects } from "@/data/projects";
import { wonderTopics } from "@/data/wonder-topics";
import { learningTopics } from "@/data/learning-topics";
import { getLearningNotes, getWonderArticles, getWritings } from "@/lib/content";
import { newestDate } from "@/lib/dates";
import { showDrafts } from "@/lib/utils";

/**
 * The site-wide "last updated" date: the newest explicit content date across
 * the manual siteConfig.lastUpdated value and every visible, dated entry.
 * It never uses the build time, so a redeploy alone does not change it.
 */
export function getSiteLastUpdated() {
  const topics = wonderTopics.filter((t) => showDrafts || t.status === "published");
  const learn = learningTopics.filter((t) => showDrafts || !t.draft);
  return newestDate([
    siteConfig.lastUpdated,
    ...projects.map((p) => p.updated),
    ...topics.map((t) => t.updated),
    ...topics.flatMap((t) => getWonderArticles(t.slug).map((a) => a.frontmatter.updated ?? a.frontmatter.date)),
    ...learn.map((t) => t.updated),
    ...learn.flatMap((t) => getLearningNotes(t.slug).map((n) => n.frontmatter.updated ?? n.frontmatter.date)),
    ...getWritings().map((w) => w.frontmatter.updated ?? w.frontmatter.date),
  ]);
}
