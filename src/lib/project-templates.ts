/** The two project page templates. Sections are optional — use the ones that fit. */
export const projectTemplates = {
  foundational: {
    name: "Foundational technical implementation",
    sections: [
      "Overview",
      "Why It Matters",
      "Mathematical Intuition",
      "How It Works",
      "Implementation from Scratch",
      "Library-Based Implementation",
      "Evaluation and Results",
      "Interactive Visualization",
      "Real-World Applications",
      "Strengths and Limitations",
      "References",
    ],
  },
  applied: {
    name: "Applied AI system",
    sections: [
      "Problem Statement",
      "Users and Requirements",
      "Constraints",
      "Proposed Solution",
      "System Architecture",
      "Technical Implementation",
      "Product and Engineering Decisions",
      "Evaluation",
      "Deployment",
      "Results and Impact",
      "Limitations and Learnings",
      "Future Improvements",
    ],
  },
} as const;
