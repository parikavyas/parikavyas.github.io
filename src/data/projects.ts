/**
 * The project manifest — one entry per project, one page per entry.
 *
 * • A project may list several `categories` (e.g. XGBoost) but has exactly one
 *   page at /projects/<slug> and one repository.
 * • `repo` points at the project's own GitHub repository. Its README becomes
 *   the project page at build time. Leave `repo: null` until the repo exists.
 * • `status`: "Planned" (nothing built yet), "In Progress", "Published".
 *   Planned projects never show GitHub or demo buttons.
 * • `updated`: the date this entry or its write-up last meaningfully changed.
 *   When a README is synced, the README's last commit date is used instead
 *   (if GitHub returns one).
 *
 * Statuses below default to "Planned" wherever no public repository was found.
 * Change them as work progresses. See docs/PROJECT_GUIDE.md.
 */
import { parseList, projectSchema, type ProjectInput } from "@/lib/schemas";

const OWNER = "parikavyas";
const D = "2026-11-27";

const entries: ProjectInput[] = [
  /* 01 Supervised Learning */
  {
    id: "linear-regression",
    slug: "linear-regression",
    title: "Linear Regression",
    summary: "Fitting a line by least squares — the closed-form solution, gradient descent, and an interactive playground.",
    categories: ["supervised-learning"],
    template: "foundational",
    implementationTypes: ["From Scratch", "Interactive Demo"],
    status: "In Progress",
    tags: ["Least squares", "Gradient descent", "NumPy"],
    visualization: "linear-regression",
    localDoc: "linear-regression.md",
    updated: D,
    related: { projects: ["logistic-regression", "gradient-boosting"] },
  },
  {
    id: "logistic-regression",
    slug: "logistic-regression",
    title: "Logistic Regression",
    summary: "Binary classification with the sigmoid, cross-entropy loss and decision boundaries.",
    categories: ["supervised-learning"],
    template: "foundational",
    implementationTypes: ["From Scratch"],
    status: "Planned",
    tags: ["Classification", "Cross-entropy"],
    updated: D,
  },
  {
    id: "decision-trees",
    slug: "decision-trees",
    title: "Decision Trees",
    summary: "Recursive splitting by impurity, depth control and interpretability.",
    categories: ["supervised-learning"],
    template: "foundational",
    implementationTypes: ["From Scratch"],
    status: "Planned",
    tags: ["Gini", "Entropy", "Pruning"],
    updated: D,
    related: { projects: ["random-forest", "xgboost"] },
  },
  {
    id: "random-forest",
    slug: "random-forest",
    title: "Random Forest",
    summary: "Bootstrap-aggregated decision trees with random feature subsets.",
    categories: ["supervised-learning"],
    template: "foundational",
    implementationTypes: ["From Scratch"],
    status: "Planned",
    tags: ["Bagging", "Feature importance"],
    updated: D,
    related: { projects: ["bagging", "decision-trees"] },
  },
  {
    id: "xgboost",
    slug: "xgboost",
    title: "XGBoost",
    summary: "Regularised gradient-boosted trees: second-order objectives, split finding and tuning.",
    categories: ["supervised-learning", "ensemble-learning"],
    template: "foundational",
    implementationTypes: ["From Scratch"],
    status: "Planned",
    tags: ["Gradient boosting", "Regularisation", "Trees"],
    updated: D,
    related: { projects: ["gradient-boosting", "ensemble-benchmark"] },
  },
  {
    id: "knn",
    slug: "k-nearest-neighbors",
    title: "K-Nearest Neighbors",
    summary: "Instance-based classification and the effect of K and distance metrics.",
    categories: ["supervised-learning"],
    template: "foundational",
    implementationTypes: ["From Scratch"],
    status: "Planned",
    tags: ["Distance metrics", "Classification"],
    updated: D,
  },

  /* 02 Unsupervised Learning */
  {
    id: "k-means",
    slug: "k-means-clustering",
    title: "K-Means Clustering",
    summary: "Lloyd's algorithm step by step — assignment, update, convergence — with an interactive demo.",
    categories: ["unsupervised-learning"],
    template: "foundational",
    implementationTypes: ["From Scratch", "Interactive Demo"],
    status: "In Progress",
    tags: ["Clustering", "Lloyd's algorithm", "Inertia"],
    visualization: "k-means",
    localDoc: "k-means-clustering.md",
    updated: D,
    related: { projects: ["dbscan", "hierarchical-clustering"] },
  },
  {
    id: "dbscan",
    slug: "dbscan",
    title: "DBSCAN",
    summary: "Density-based clustering: core points, epsilon neighbourhoods and noise.",
    categories: ["unsupervised-learning"],
    template: "foundational",
    implementationTypes: ["From Scratch"],
    status: "Planned",
    tags: ["Density", "Clustering"],
    updated: D,
  },
  {
    id: "hierarchical-clustering",
    slug: "hierarchical-clustering",
    title: "Hierarchical Clustering",
    summary: "Hierarchical clustering applied to e-commerce customer data.",
    categories: ["unsupervised-learning"],
    template: "foundational",
    implementationTypes: ["Applied AI"],
    status: "Published",
    tags: ["Dendrograms", "Customer segmentation"],
    repo: { owner: OWNER, name: "hierarchical_clustering", branch: "main", readmePath: "README.md" },
    updated: "2021-11-26",
  },
  {
    id: "pca",
    slug: "principal-component-analysis",
    title: "Principal Component Analysis",
    summary: "Dimensionality reduction via eigen-decomposition and explained variance.",
    categories: ["unsupervised-learning"],
    template: "foundational",
    implementationTypes: ["From Scratch"],
    status: "Planned",
    tags: ["Linear algebra", "Dimensionality reduction"],
    updated: D,
  },

  /* 03 Ensemble Learning & Gradient Boosting (XGBoost lives above) */
  {
    id: "bagging",
    slug: "bagging",
    title: "Bagging",
    summary: "Bootstrap aggregation and why averaging reduces variance.",
    categories: ["ensemble-learning"],
    template: "foundational",
    implementationTypes: ["From Scratch"],
    status: "Planned",
    tags: ["Bootstrap", "Variance reduction"],
    updated: D,
  },
  {
    id: "adaboost",
    slug: "adaboost",
    title: "AdaBoost",
    summary: "Adaptive boosting: re-weighting hard examples across sequential learners.",
    categories: ["ensemble-learning"],
    template: "foundational",
    implementationTypes: ["From Scratch"],
    status: "Planned",
    tags: ["Boosting", "Weak learners"],
    updated: D,
  },
  {
    id: "gradient-boosting",
    slug: "gradient-boosting",
    title: "Gradient Boosting",
    summary: "Sequential residual correction as gradient descent in function space.",
    categories: ["ensemble-learning"],
    template: "foundational",
    implementationTypes: ["From Scratch"],
    status: "Planned",
    tags: ["Residuals", "Boosting"],
    updated: D,
  },
  {
    id: "ensemble-benchmark",
    slug: "ensemble-learning-benchmark",
    title: "Ensemble Learning Benchmark",
    summary: "A like-for-like comparison of ensemble methods across datasets.",
    categories: ["ensemble-learning"],
    template: "foundational",
    implementationTypes: ["Benchmark"],
    status: "Planned",
    tags: ["Benchmarking", "Evaluation"],
    updated: D,
  },

  /* 04 Deep Learning */
  {
    id: "nn-from-scratch",
    slug: "neural-networks-from-scratch",
    title: "Neural Networks from Scratch",
    summary: "Forward propagation, backpropagation, activation functions, optimisation and training — without a framework.",
    categories: ["deep-learning"],
    template: "foundational",
    implementationTypes: ["From Scratch"],
    status: "Planned",
    tags: ["Backpropagation", "Optimisers", "NumPy"],
    updated: D,
  },
  {
    id: "lstm",
    slug: "long-short-term-memory",
    title: "Long Short-Term Memory Networks",
    summary: "Gated recurrent memory for sequences: input, forget and output gates.",
    categories: ["deep-learning"],
    template: "foundational",
    implementationTypes: ["From Scratch"],
    status: "Planned",
    tags: ["RNN", "Sequence modelling"],
    updated: D,
  },

  /* 05 Computer Vision */
  {
    id: "advanced-surveillance",
    slug: "advanced-surveillance-system",
    title: "Advanced Surveillance System",
    summary: "A computer-vision surveillance system. The full write-up is being prepared.",
    categories: ["computer-vision"],
    template: "applied",
    implementationTypes: ["Applied AI"],
    status: "Published",
    tags: ["Computer vision"],
    repo: { owner: OWNER, name: "Advanced-Surveillance-System", branch: "main", readmePath: "README.md" },
    updated: "2021-12-29",
  },
  {
    id: "broken-screen",
    slug: "broken-phone-screen-detection",
    title: "Broken Phone Screen Detection",
    summary: "Detecting broken phone screens using image classification.",
    categories: ["computer-vision"],
    template: "applied",
    implementationTypes: ["Applied AI"],
    status: "Published",
    tags: ["Image classification"],
    repo: { owner: OWNER, name: "Broken-Screen-Detection", branch: "master", readmePath: "README.md" },
    updated: "2020-07-07",
  },
  {
    id: "smart-glasses",
    slug: "assistive-smart-glasses",
    title: "AI-Powered Assistive Smart Glasses for Visually Impaired Individuals",
    summary: "Assistive perception to help visually impaired people navigate their surroundings.",
    categories: ["computer-vision"],
    template: "applied",
    implementationTypes: ["Applied AI", "End-to-End"],
    status: "Planned",
    tags: ["Assistive technology", "Edge AI"],
    updated: D,
    related: { wonder: ["ai-for-social-good"] },
  },

  /* 06 Graph Neural Networks */
  {
    id: "gnn-fundamentals",
    slug: "graph-neural-network-fundamentals",
    title: "Graph Neural Network Fundamentals",
    summary: "Graphs as data, message passing and neighbourhood aggregation.",
    categories: ["graph-neural-networks"],
    template: "foundational",
    implementationTypes: ["From Scratch"],
    status: "Planned",
    tags: ["Message passing", "Graphs"],
    updated: D,
  },
  {
    id: "gcn",
    slug: "graph-convolutional-networks",
    title: "Graph Convolutional Networks",
    summary: "Spectral intuition and the normalised-adjacency propagation rule.",
    categories: ["graph-neural-networks"],
    template: "foundational",
    implementationTypes: ["From Scratch"],
    status: "Planned",
    tags: ["GCN", "Node classification"],
    updated: D,
  },
  {
    id: "graphsage",
    slug: "graphsage",
    title: "GraphSAGE",
    summary: "Inductive representation learning by sampling and aggregating neighbourhoods.",
    categories: ["graph-neural-networks"],
    template: "foundational",
    implementationTypes: ["From Scratch"],
    status: "Planned",
    tags: ["Inductive learning", "Sampling"],
    updated: D,
  },
  {
    id: "sage-comparison",
    slug: "graphsage-hinsage-pinsage",
    title: "Comparative Study of GraphSAGE, HinSAGE and PinSAGE",
    summary: "Homogeneous, heterogeneous and web-scale variants compared for recommendation.",
    categories: ["graph-neural-networks"],
    template: "foundational",
    implementationTypes: ["Research", "Benchmark"],
    status: "Planned",
    tags: ["Recommendation systems", "Heterogeneous graphs"],
    updated: D,
  },

  /* 07 Federated Learning */
  {
    id: "federated-learning",
    slug: "federated-learning",
    title: "Federated Learning",
    summary: "Client training, Federated Averaging and secure aggregation — starting from a proof-of-concept with two clients, a secure aggregator and a main server.",
    categories: ["federated-learning"],
    template: "applied",
    implementationTypes: ["End-to-End"],
    status: "In Progress",
    tags: ["FedAvg", "Secure aggregation", "Distributed training"],
    repo: { owner: OWNER, name: "Federated-Learning-Experiment", branch: "main", readmePath: "README.md" },
    updated: "2022-07-01",
    related: { projects: ["secure-aggregation", "differential-privacy"] },
  },

  /* 08 Privacy-Preserving ML */
  {
    id: "differential-privacy",
    slug: "differential-privacy",
    title: "Differential Privacy",
    summary: "Calibrated noise, privacy budgets and the privacy–utility trade-off.",
    categories: ["privacy-preserving-ml"],
    template: "foundational",
    implementationTypes: ["From Scratch"],
    status: "Planned",
    tags: ["Epsilon", "Laplace mechanism"],
    updated: D,
    related: { wonder: ["privacy-ethics-and-technology"] },
  },
  {
    id: "homomorphic-encryption",
    slug: "homomorphic-encryption",
    title: "Homomorphic Encryption",
    summary: "Computing on encrypted values — an educational walk-through of additive schemes.",
    categories: ["privacy-preserving-ml"],
    template: "foundational",
    implementationTypes: ["From Scratch"],
    status: "Planned",
    tags: ["Cryptography", "Paillier"],
    updated: D,
  },
  {
    id: "smpc",
    slug: "secure-multiparty-computation",
    title: "Secure Multiparty Computation",
    summary: "Joint computation over private inputs with secret sharing.",
    categories: ["privacy-preserving-ml"],
    template: "foundational",
    implementationTypes: ["From Scratch"],
    status: "Planned",
    tags: ["Secret sharing", "Cryptography"],
    updated: D,
  },
  {
    id: "secure-aggregation",
    slug: "secure-aggregation",
    title: "Secure Aggregation",
    summary: "Summing client updates so the server learns only the aggregate.",
    categories: ["privacy-preserving-ml"],
    template: "foundational",
    implementationTypes: ["From Scratch"],
    status: "Planned",
    tags: ["Masking", "Federated learning"],
    updated: D,
    related: { projects: ["federated-learning"] },
  },

  /* 09 LLMs & RAG */
  {
    id: "knowledge-assistant",
    slug: "intelligent-knowledge-assistant",
    title: "Intelligent Knowledge Assistant",
    summary: "Retrieval-augmented generation over documents: ingestion, chunking, embeddings, vector search, grounded answers with citations, and evaluation.",
    categories: ["llms-and-rag"],
    template: "applied",
    implementationTypes: ["Applied AI", "End-to-End"],
    status: "Planned",
    tags: ["RAG", "Embeddings", "Vector search", "Evaluation"],
    updated: D,
  },

  /* 10 Generative AI — intentionally empty until a project is chosen */

  /* 11 Agentic AI */
  {
    id: "whats-flying-above-me",
    slug: "whats-flying-above-me",
    title: "What’s Flying Above Me?",
    summary: "Point your phone at the sky and see which plane is passing above you — live ADS-B data, geospatial reasoning and a radar sky view.",
    categories: ["agentic-ai"],
    template: "applied",
    implementationTypes: ["End-to-End", "Interactive Demo"],
    status: "Published",
    tags: ["Geospatial", "ADS-B", "Tool integration"],
    repo: { owner: OWNER, name: "what-plane", branch: "main", readmePath: "README.md" },
    demoUrl: "https://parikavyas.github.io/what-plane/",
    updated: "2026-10-16",
  },
  {
    id: "agentic-trading",
    slug: "agentic-trading-system",
    title: "Agentic Trading System",
    summary: "Simulated trading only: research, strategy evaluation, backtesting, risk controls and transparent agent execution.",
    categories: ["agentic-ai"],
    template: "applied",
    implementationTypes: ["Applied AI", "End-to-End"],
    status: "Planned",
    tags: ["Agents", "Backtesting", "Risk controls", "Simulation"],
    updated: D,
  },
];

export const projects = parseList(projectSchema, entries, "src/data/projects.ts");

export function getProject(slug: string) {
  return projects.find((p) => p.slug === slug);
}

export function getProjectsByCategory(categoryId: string) {
  return projects.filter((p) => p.categories.includes(categoryId));
}

/* Integrity checks so broken references fail the build early. */
{
  const slugs = new Set<string>();
  for (const p of projects) {
    if (slugs.has(p.slug)) throw new Error(`Duplicate project slug "${p.slug}" in src/data/projects.ts`);
    slugs.add(p.slug);
  }
}
