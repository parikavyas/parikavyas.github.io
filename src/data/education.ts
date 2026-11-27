/** Work → Education. Empty lists and dates are hidden. */
import { educationSchema, parseList } from "@/lib/schemas";

const entries = [
  {
    id: "duke-fuqua",
    institution: "Duke University",
    school: "The Fuqua School of Business",
    degree: "MBA",
    order: 1,
  },
  {
    id: "bit-mesra",
    institution: "Birla Institute of Technology, Mesra",
    degree: "Bachelor of Engineering in Electronics and Communication Engineering",
    order: 2,
  },
];

export const education = parseList(educationSchema, entries, "src/data/education.ts").sort(
  (a, b) => a.order - b.order,
);
