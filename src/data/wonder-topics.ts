/**
 * Wonder & Purpose topic collections.
 *
 * These eight are starting suggestions. They are marked "draft", which means
 * they show up while you preview locally but are hidden on the live site.
 * Set `status: "published"` on a topic once it has something in it.
 *
 * Articles for a topic go in content/wonder-and-purpose/<slug>/*.md(x).
 * Papers, decks, PDFs, videos and links go in `resources` below.
 * Reorder by changing `order`. Rename by changing `title` (keep `slug` stable
 * so links keep working). See docs/CONTENT_GUIDE.md → "Wonder & Purpose".
 */
import { parseList, wonderTopicSchema } from "@/lib/schemas";

const D = "2026-11-27";

const entries = [
  { slug: "ai-for-social-good", title: "AI for Social Good", description: "Where AI can widen access, opportunity and dignity — and where it can’t.", status: "draft", order: 1, updated: D },
  { slug: "privacy-ethics-and-technology", title: "Privacy, Ethics & Technology", description: "Data, consent, and the systems we build around people.", status: "draft", order: 2, updated: D },
  { slug: "ai-and-sustainability", title: "AI & Sustainability", description: "Technology, energy and the planet.", status: "draft", order: 3, updated: D },
  { slug: "human-behaviour-and-technology", title: "Human Behaviour & Technology", description: "How people and the tools they use shape each other.", status: "draft", order: 4, updated: D },
  { slug: "healthcare-and-technology", title: "Healthcare & Technology", description: "Care, access and evidence in a data-rich world.", status: "draft", order: 5, updated: D },
  { slug: "education-and-opportunity", title: "Education & Opportunity", description: "Learning, mobility and who gets a chance to build.", status: "draft", order: 6, updated: D },
  { slug: "the-ai-economy", title: "The AI Economy", description: "Work, markets and value as AI spreads.", status: "draft", order: 7, updated: D },
  { slug: "entrepreneurship-and-innovation", title: "Entrepreneurship & Innovation", description: "Turning ideas into things people use.", status: "draft", order: 8, updated: D },
  /*
   * Example with resources:
   * {
   *   slug: "my-topic", title: "My Topic", description: "One line.", status: "published", order: 9, updated: "2026-12-01",
   *   openQuestions: ["A question I keep returning to?"],
   *   resources: [
   *     { type: "PDF", title: "My paper", href: "/documents/my-paper.pdf", preview: true },
   *     { type: "Presentation deck", title: "Talk slides", href: "/documents/talk.pdf", preview: true },
   *     { type: "Video", title: "A talk I loved", href: "https://www.youtube.com/watch?v=VIDEO_ID" },
   *     { type: "External resource", title: "An essay", href: "https://example.com", description: "Why it matters" },
   *   ],
   *   related: { projects: ["differential-privacy"], learn: ["some-learning-topic"] },
   * },
   */
];

export const wonderTopics = parseList(wonderTopicSchema, entries, "src/data/wonder-topics.ts").sort(
  (a, b) => a.order - b.order,
);
