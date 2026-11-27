/**
 * Loads local Markdown/MDX content from the /content folder (server-only).
 * Drafts (`draft: true` in front matter) are only returned in development or
 * when SHOW_DRAFTS=true.
 */
import "server-only";
import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import { z } from "zod";
import readingTime from "reading-time";
import {
  articleFrontmatterSchema,
  projectDocFrontmatterSchema,
  writingFrontmatterSchema,
  type ArticleFrontmatter,
  type WritingFrontmatter,
} from "@/lib/schemas";
import { showDrafts } from "@/lib/utils";

const CONTENT_ROOT = path.join(process.cwd(), "content");

export interface LocalDoc<T> {
  slug: string;
  format: "md" | "mdx";
  frontmatter: T;
  body: string;
  readingMinutes: number;
}

function listFiles(dir: string) {
  const full = path.join(CONTENT_ROOT, dir);
  if (!fs.existsSync(full)) return [];
  return fs
    .readdirSync(full)
    .filter((f) => /\.(md|mdx)$/.test(f) && !f.startsWith("_"))
    .sort();
}

/** YAML turns 2026-01-31 into a Date; convert back to "YYYY-MM-DD". */
function normaliseDates(data: Record<string, unknown>) {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(data)) {
    out[k] = v instanceof Date ? v.toISOString().slice(0, 10) : v;
  }
  return out;
}

function readDoc<T>(
  rel: string,
  parse: (data: unknown, file: string) => T,
): LocalDoc<T> | null {
  const full = path.join(CONTENT_ROOT, rel);
  if (!fs.existsSync(full)) return null;
  const raw = fs.readFileSync(full, "utf8");
  const { data, content } = matter(raw);
  return {
    slug: path.basename(rel).replace(/\.(md|mdx)$/, ""),
    format: rel.endsWith(".mdx") ? "mdx" : "md",
    frontmatter: parse(normaliseDates(data), `content/${rel}`),
    body: content,
    readingMinutes: Math.max(1, Math.round(readingTime(content).minutes)),
  };
}

function parser<S extends z.ZodTypeAny>(schema: S) {
  return (data: unknown, file: string): z.infer<S> => {
    const result = schema.safeParse(data);
    if (!result.success) {
      const issues = result.error.issues.map((i) => `  • ${i.path.join(".")}: ${i.message}`).join("\n");
      throw new Error(`Invalid front matter in ${file}:\n${issues}`);
    }
    return result.data;
  };
}

const parseArticle: (d: unknown, f: string) => ArticleFrontmatter = parser(articleFrontmatterSchema);
const parseWriting: (d: unknown, f: string) => WritingFrontmatter = parser(writingFrontmatterSchema);

function visible<T extends { draft: boolean }>(doc: LocalDoc<T> | null): doc is LocalDoc<T> {
  return !!doc && (showDrafts || !doc.frontmatter.draft);
}

function byDateDesc<T extends { date: string }>(a: LocalDoc<T>, b: LocalDoc<T>) {
  return b.frontmatter.date.localeCompare(a.frontmatter.date);
}

/* Projects: local fallback documentation */
export function getProjectDoc(file: string) {
  return readDoc(`projects/${file}`, parser(projectDocFrontmatterSchema));
}

/* Wonder & Purpose articles: content/wonder-and-purpose/<topic>/<slug>.md(x) */
export function getWonderArticles(topic: string) {
  return listFiles(`wonder-and-purpose/${topic}`)
    .map((f) => readDoc(`wonder-and-purpose/${topic}/${f}`, parseArticle))
    .filter(visible)
    .sort(byDateDesc);
}
export function getWonderArticle(topic: string, slug: string) {
  return getWonderArticles(topic).find((a) => a.slug === slug) ?? null;
}

/* Learning notes: content/learn/<topic>/<slug>.md(x) */
export function getLearningNotes(topic: string) {
  return listFiles(`learn/${topic}`)
    .map((f) => readDoc(`learn/${topic}/${f}`, parseArticle))
    .filter(visible)
    .sort(byDateDesc);
}
export function getLearningNote(topic: string, slug: string) {
  return getLearningNotes(topic).find((n) => n.slug === slug) ?? null;
}

/* Creative writing: content/writing/<slug>.md(x) */
export function getWritings() {
  return listFiles("writing")
    .map((f) => readDoc(`writing/${f}`, parseWriting))
    .filter(visible)
    .sort(byDateDesc);
}
export function getWriting(slug: string) {
  return getWritings().find((w) => w.slug === slug) ?? null;
}
