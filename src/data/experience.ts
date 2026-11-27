/**
 * Work → Experience.
 *
 * Every field except `id`, `organization` and `order` is optional. Anything
 * left empty shows as "to be added" (only visible as a marked placeholder in
 * development) — nothing is invented. Fill entries in your own words.
 *
 * Order: lower `order` appears first. Change the numbers to reorder.
 * Dates: "2021", "2021-06" or "present".
 * See docs/CONTENT_GUIDE.md → "Add or edit a Work experience".
 */
import { experienceSchema, parseList } from "@/lib/schemas";

const entries = [
  { id: "mckinsey", organization: "McKinsey & Company", order: 1 },
  { id: "bloom-ai", organization: "Bloom AI", order: 2 },
  { id: "zs-associates", organization: "ZS Associates", order: 3 },
  { id: "pari-udaan", organization: "Pari-Udaan", order: 4 },
  { id: "the-apprentice-project", organization: "The Apprentice Project", order: 5 },
  { id: "sukoon", organization: "Sukoon", order: 6 },
  { id: "illinois-tech", organization: "Illinois Institute of Technology, Chicago", order: 7 },
  { id: "microtek", organization: "Microtek", order: 8 },
  { id: "forbes-marshall", organization: "Forbes Marshall", order: 9 },
  /*
   * Template — copy, fill in, and delete what you don't need:
   * {
   *   id: "example-org",
   *   organization: "Example Org",
   *   logo: "/images/organizations/example.svg",
   *   role: "Your role",
   *   location: "City, Country",
   *   start: "2022-06",
   *   end: "present",
   *   summary: "One or two sentences on the essence of the work.",
   *   themes: ["Recommendation systems", "LLMs"],
   *   details: {
   *     problems: ["The problem in plain words"],
   *     built: ["What you built or contributed"],
   *     owned: ["What you were accountable for"],
   *     decisions: ["A technical or product decision and why"],
   *     leadership: ["How you led or collaborated"],
   *     stakeholders: ["Who you worked with and how"],
   *     impact: ["Only impact you can stand behind publicly"],
   *   },
   *   relatedProjects: ["intelligent-knowledge-assistant"],
   *   order: 10,
   * },
   */
];

export const experiences = parseList(experienceSchema, entries, "src/data/experience.ts").sort(
  (a, b) => a.order - b.order,
);

/**
 * Areas the professional narrative spans (shown above the experience cards).
 * Edit freely; leave empty to hide.
 */
export const focusAreas: string[] = [
  "Marketing AI",
  "Personalization",
  "Sales intelligence",
  "Recommendation systems",
  "Healthcare",
  "Pharmaceutical pricing",
  "Asset management",
  "Graph neural networks",
  "Enterprise AI",
  "Large language models",
  "Privacy-preserving ML",
  "AI product development",
  "Technical delivery & deployment",
  "Client-facing problem solving",
  "Team leadership",
];
