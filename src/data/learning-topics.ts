/**
 * Learn — the learning library.
 *
 * Each topic has a status: Curious, Exploring, Learning, Implementing, Revisiting.
 * Notes for a topic go in content/learn/<slug>/*.md(x).
 * `draft: true` topics only appear in local previews.
 *
 * The single entry below is a template that demonstrates every field. It is a
 * draft, so it never appears on the live site. Copy it for real topics.
 */
import { learningTopicSchema, parseList } from "@/lib/schemas";

const entries = [
  {
    slug: "example-topic",
    title: "Example topic (template)",
    description: "A sample showing every field a learning topic supports. Copy it, then delete it.",
    status: "Exploring",
    draft: true,
    updated: "2026-11-27",
    resources: [{ title: "A resource title", kind: "Course", note: "Why it is useful" }],
    readingList: [{ title: "A book or paper", author: "Author name", state: "To read" }],
    roadmap: [
      { step: "First milestone", done: false },
      { step: "Second milestone", done: false },
    ],
    questions: ["A question you are trying to answer?"],
    learned: [{ date: "2026-11-27", text: "A short thing you learned recently." }],
    related: { projects: ["linear-regression"] },
  },
];

export const learningTopics = parseList(learningTopicSchema, entries, "src/data/learning-topics.ts");
