/**
 * Work → Awards & Recognitions.
 * Add `year`, `issuer` and `description` once verified. Empty fields are hidden.
 */
import { awardSchema, parseList } from "@/lib/schemas";

const entries = [
  { id: "pets-prize", name: "Privacy-Enhancing Technologies Prize — Third Place", order: 1 },
  { id: "agile-innovation", name: "Agile Innovation of the Year", order: 2 },
  { id: "early-adopter", name: "Early Adopter Young Champion", order: 3 },
  { id: "client-innovation", name: "Client Innovation Impact Award", order: 4 },
  { id: "ug-research-grant", name: "Undergraduate Research Grant", order: 5 },
  { id: "best-poster", name: "Best Research Poster", order: 6 },
  { id: "summer-fellowship", name: "Immersive Summer Research Fellowship", order: 7 },
  { id: "maharashtra-merit", name: "Maharashtra State Merit Scholarship", order: 8 },
  { id: "best-girl-academic", name: "Best Girl Academic Achiever", order: 9 },
  { id: "most-promising", name: "Most Promising Student", order: 10 },
  // { id: "new-award", name: "Award name", year: "2025", issuer: "Organization", description: "Optional one-liner", order: 11 },
];

export const awards = parseList(awardSchema, entries, "src/data/awards.ts").sort(
  (a, b) => a.order - b.order,
);
