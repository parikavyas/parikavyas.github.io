/**
 * The technical categories on /projects, in display order.
 * `id` is also the section anchor, e.g. /projects#graph-neural-networks.
 * `illustration` picks the section's abstract artwork.
 */
import { parseList, projectCategorySchema } from "@/lib/schemas";

const entries = [
  {
    id: "supervised-learning",
    number: "01",
    title: "Supervised Learning",
    description: "Learning a mapping from labelled examples — the foundation most applied models build on.",
    illustration: "regression",
  },
  {
    id: "unsupervised-learning",
    number: "02",
    title: "Unsupervised Learning",
    description: "Finding structure — clusters, densities and directions of variance — without labels.",
    illustration: "clusters",
  },
  {
    id: "ensemble-learning",
    number: "03",
    title: "Ensemble Learning & Gradient Boosting",
    description: "Combining many weak learners into one strong one, by averaging or by sequential correction.",
    illustration: "ensemble",
  },
  {
    id: "deep-learning",
    number: "04",
    title: "Deep Learning",
    description: "Neural networks, from forward and backward passes to sequence models with memory.",
    illustration: "neural",
  },
  {
    id: "computer-vision",
    number: "05",
    title: "Computer Vision",
    description: "Systems that see: detection, classification and assistive perception.",
    illustration: "vision",
  },
  {
    id: "graph-neural-networks",
    number: "06",
    title: "Graph Neural Networks",
    description: "Learning on relationships through message passing — and using it for recommendation.",
    illustration: "graph",
  },
  {
    id: "federated-learning",
    number: "07",
    title: "Federated Learning",
    description: "Training across many clients without centralising their data.",
    illustration: "federated",
  },
  {
    id: "privacy-preserving-ml",
    number: "08",
    title: "Privacy-Preserving Machine Learning",
    description: "Techniques that let models learn from data while protecting the people behind it.",
    illustration: "privacy",
  },
  {
    id: "llms-and-rag",
    number: "09",
    title: "LLMs & Retrieval-Augmented Generation",
    description: "Grounding language models in your own documents, with citations and evaluation.",
    illustration: "retrieval",
  },
  {
    id: "generative-ai",
    number: "10",
    title: "Generative AI",
    description: "Models that create. The first project for this category is still being chosen.",
    illustration: "generative",
    emptyMessage: "A project for this category will be selected soon. The template is ready for it.",
  },
  {
    id: "agentic-ai",
    number: "11",
    title: "Agentic AI",
    description: "Systems that plan, call tools and act — with transparent, inspectable execution.",
    illustration: "agentic",
  },
];

export const projectCategories = parseList(
  projectCategorySchema,
  entries,
  "src/data/project-categories.ts",
);

export function getCategory(id: string) {
  return projectCategories.find((c) => c.id === id);
}
