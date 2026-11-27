/**
 * Schemas for every structured content type.
 *
 * Data files in src/data/ are validated against these at build time, so a typo
 * (a missing title, a misspelled status) fails the build with a readable
 * message instead of silently breaking a page.
 */
import { z } from "zod";

const isoDate = z
  .string()
  .regex(/^\d{4}-\d{2}-\d{2}$/, "Use the format YYYY-MM-DD");

/** "2021", "2021-06" or "present" */
const period = z
  .string()
  .regex(/^(\d{4}(-\d{2})?|present)$/i, 'Use "YYYY", "YYYY-MM" or "present"');

const imageSchema = z.object({
  src: z.string(),
  alt: z.string().min(1, "Every image needs alt text"),
  width: z.number().optional(),
  height: z.number().optional(),
});
export type ImageAsset = z.infer<typeof imageSchema>;

/* ---------------- Work ---------------- */

export const experienceSchema = z.object({
  id: z.string(),
  organization: z.string(),
  /** Path under /public, e.g. "/images/organizations/mckinsey.svg" */
  logo: z.string().nullable().default(null),
  role: z.string().nullable().default(null),
  location: z.string().nullable().default(null),
  start: period.nullable().default(null),
  end: period.nullable().default(null),
  summary: z.string().nullable().default(null),
  themes: z.array(z.string()).default([]),
  details: z
    .object({
      problems: z.array(z.string()).default([]),
      built: z.array(z.string()).default([]),
      owned: z.array(z.string()).default([]),
      decisions: z.array(z.string()).default([]),
      leadership: z.array(z.string()).default([]),
      stakeholders: z.array(z.string()).default([]),
      impact: z.array(z.string()).default([]),
    })
    .prefault({}),
  /** Project slugs from src/data/projects.ts */
  relatedProjects: z.array(z.string()).default([]),
  /** Lower numbers appear first. */
  order: z.number(),
});
export type Experience = z.infer<typeof experienceSchema>;

export const awardSchema = z.object({
  id: z.string(),
  name: z.string(),
  year: z.string().nullable().default(null),
  issuer: z.string().nullable().default(null),
  description: z.string().nullable().default(null),
  order: z.number(),
});
export type Award = z.infer<typeof awardSchema>;

export const educationSchema = z.object({
  id: z.string(),
  institution: z.string(),
  school: z.string().nullable().default(null),
  degree: z.string(),
  logo: z.string().nullable().default(null),
  start: period.nullable().default(null),
  end: period.nullable().default(null),
  activities: z.array(z.string()).default([]),
  interests: z.array(z.string()).default([]),
  accomplishments: z.array(z.string()).default([]),
  order: z.number(),
});
export type Education = z.infer<typeof educationSchema>;

/* ---------------- Projects ---------------- */

export const implementationTypes = [
  "From Scratch",
  "Applied AI",
  "Research",
  "Benchmark",
  "End-to-End",
  "Interactive Demo",
] as const;
export type ImplementationType = (typeof implementationTypes)[number];

export const projectStatuses = ["Planned", "In Progress", "Published"] as const;
export type ProjectStatus = (typeof projectStatuses)[number];

export const illustrationKinds = [
  "regression",
  "clusters",
  "ensemble",
  "neural",
  "vision",
  "graph",
  "federated",
  "privacy",
  "retrieval",
  "generative",
  "agentic",
] as const;
export type IllustrationKind = (typeof illustrationKinds)[number];

export const projectCategorySchema = z.object({
  id: z.string(),
  number: z.string(),
  title: z.string(),
  description: z.string(),
  illustration: z.enum(illustrationKinds),
  /** Message shown when the category has no projects yet. */
  emptyMessage: z.string().optional(),
});
export type ProjectCategory = z.infer<typeof projectCategorySchema>;

export const projectSchema = z.object({
  id: z.string(),
  slug: z.string().regex(/^[a-z0-9-]+$/, "Slugs use lowercase letters, numbers and dashes"),
  title: z.string(),
  summary: z.string(),
  /** Category ids. A project can live in several categories but has one page. */
  categories: z.array(z.string()).min(1),
  /** "foundational" = Template A, "applied" = Template B */
  template: z.enum(["foundational", "applied"]),
  implementationTypes: z.array(z.enum(implementationTypes)).default([]),
  status: z.enum(projectStatuses),
  tags: z.array(z.string()).default([]),
  repo: z
    .object({
      owner: z.string(),
      name: z.string(),
      branch: z.string().default("main"),
      readmePath: z.string().default("README.md"),
    })
    .nullable()
    .default(null),
  demoUrl: z.string().url().nullable().default(null),
  /** Key from src/components/visualizations/registry.ts */
  visualization: z.string().nullable().default(null),
  /** Optional local Markdown/MDX fallback: content/projects/<file> */
  localDoc: z.string().nullable().default(null),
  updated: isoDate,
  thumbnail: imageSchema.nullable().default(null),
  related: z
    .object({
      projects: z.array(z.string()).default([]),
      learn: z.array(z.string()).default([]),
      wonder: z.array(z.string()).default([]),
    })
    .prefault({}),
});
export type Project = z.infer<typeof projectSchema>;
export type ProjectInput = z.input<typeof projectSchema>;

/* ---------------- Wonder & Purpose ---------------- */

export const wonderResourceTypes = [
  "Research paper",
  "Presentation deck",
  "PDF",
  "External resource",
  "Video",
] as const;

export const wonderResourceSchema = z.object({
  type: z.enum(wonderResourceTypes),
  title: z.string(),
  description: z.string().optional(),
  /** For PDFs and decks: a path under /public/documents. For others: a URL. */
  href: z.string(),
  date: isoDate.optional(),
  /** Show an inline preview (PDF/deck) or player (video). */
  preview: z.boolean().default(false),
});
export type WonderResource = z.infer<typeof wonderResourceSchema>;

export const wonderTopicSchema = z.object({
  slug: z.string().regex(/^[a-z0-9-]+$/),
  title: z.string(),
  /** One-line description of what the collection covers. */
  description: z.string(),
  /** Draft topics appear only in local development and previews. */
  status: z.enum(["draft", "published"]),
  order: z.number(),
  updated: isoDate,
  openQuestions: z.array(z.string()).default([]),
  resources: z.array(wonderResourceSchema).default([]),
  related: z
    .object({
      projects: z.array(z.string()).default([]),
      learn: z.array(z.string()).default([]),
    })
    .prefault({}),
});
export type WonderTopic = z.infer<typeof wonderTopicSchema>;

/* ---------------- Learn ---------------- */

export const learningStatuses = [
  "Curious",
  "Exploring",
  "Learning",
  "Implementing",
  "Revisiting",
] as const;
export type LearningStatus = (typeof learningStatuses)[number];

export const learningTopicSchema = z.object({
  slug: z.string().regex(/^[a-z0-9-]+$/),
  title: z.string(),
  description: z.string(),
  status: z.enum(learningStatuses),
  draft: z.boolean().default(false),
  updated: isoDate,
  resources: z
    .array(
      z.object({
        title: z.string(),
        url: z.string().url().optional(),
        kind: z.string().optional(),
        note: z.string().optional(),
      }),
    )
    .default([]),
  readingList: z
    .array(
      z.object({
        title: z.string(),
        author: z.string().optional(),
        url: z.string().url().optional(),
        state: z.enum(["To read", "Reading", "Read"]),
      }),
    )
    .default([]),
  roadmap: z
    .array(z.object({ step: z.string(), done: z.boolean().default(false) }))
    .default([]),
  questions: z.array(z.string()).default([]),
  /** Short "today I learned" style entries, newest first. */
  learned: z.array(z.object({ date: isoDate, text: z.string() })).default([]),
  related: z
    .object({
      projects: z.array(z.string()).default([]),
      wonder: z.array(z.string()).default([]),
    })
    .prefault({}),
});
export type LearningTopic = z.infer<typeof learningTopicSchema>;

/* ---------------- Beyond the Desk ---------------- */

export const kitchenItemSchema = z.object({
  id: z.string(),
  dish: z.string(),
  image: imageSchema,
  description: z.string().optional(),
  recipe: z.string().optional(),
  story: z.string().optional(),
  date: isoDate.optional(),
  draft: z.boolean().default(false),
});
export type KitchenItem = z.infer<typeof kitchenItemSchema>;

export const danceItemSchema = z.object({
  id: z.string(),
  title: z.string(),
  style: z.string().optional(),
  occasion: z.string().optional(),
  date: isoDate.optional(),
  image: imageSchema.optional(),
  video: z
    .object({
      provider: z.enum(["youtube", "vimeo", "file"]),
      /** YouTube/Vimeo id, or a path under /public for "file". */
      id: z.string(),
      thumbnail: z.string().optional(),
    })
    .optional(),
  choreography: z.string().optional(),
  description: z.string().optional(),
  draft: z.boolean().default(false),
});
export type DanceItem = z.infer<typeof danceItemSchema>;

/* ---------------- Markdown front matter ---------------- */

export const articleFrontmatterSchema = z.object({
  title: z.string(),
  description: z.string().optional(),
  /** Date first published */
  date: isoDate,
  /** Date of the last meaningful content change */
  updated: isoDate.optional(),
  kind: z
    .enum(["Article", "Perspective", "Essay", "Reflection", "Note"])
    .default("Article"),
  draft: z.boolean().default(false),
  tags: z.array(z.string()).default([]),
});
export type ArticleFrontmatter = z.infer<typeof articleFrontmatterSchema>;

export const writingFrontmatterSchema = z.object({
  title: z.string(),
  form: z.enum(["Poetry", "Reflection", "Short story", "Essay", "Other"]),
  date: isoDate,
  updated: isoDate.optional(),
  excerpt: z.string().optional(),
  draft: z.boolean().default(false),
});
export type WritingFrontmatter = z.infer<typeof writingFrontmatterSchema>;

export const projectDocFrontmatterSchema = z.object({
  updated: isoDate.optional(),
});

/** Parse a list with a friendly error that names the file. */
export function parseList<T extends z.ZodTypeAny>(
  schema: T,
  items: unknown[],
  file: string,
): z.infer<T>[] {
  return items.map((item, i) => {
    const result = schema.safeParse(item);
    if (!result.success) {
      const issues = result.error.issues
        .map((iss) => `  • ${iss.path.join(".") || "(root)"}: ${iss.message}`)
        .join("\n");
      throw new Error(`Invalid entry #${i + 1} in ${file}:\n${issues}`);
    }
    return result.data;
  });
}
