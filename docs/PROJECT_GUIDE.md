# Project guide — structuring each technical repository

Each technical project lives in its **own public GitHub repository**. The
portfolio reads that repository's `README.md` at build time and renders it as
the project page, so the README is the single source of truth.

## 1. Repository layout

```
my-project/
├── README.md            ← becomes the project page
├── src/ or notebooks/   ← the implementation
├── tests/
├── data/README.md       ← how to obtain data (never commit private data)
├── docs/images/         ← figures referenced from the README
├── requirements.txt / pyproject.toml / package.json
└── LICENSE
```

## 2. README structure

Start from `content/templates/project-readme.md`. Pick the template that fits:

**Template A — foundational implementation** (algorithms, from-scratch work):
Overview · Why It Matters · Mathematical Intuition · How It Works ·
Implementation from Scratch · Library-Based Implementation · Evaluation and
Results · Interactive Visualization · Real-World Applications · Strengths and
Limitations · References

**Template B — applied AI system** (products, pipelines, agents):
Problem Statement · Users and Requirements · Constraints · Proposed Solution ·
System Architecture · Technical Implementation · Product and Engineering
Decisions · Evaluation · Deployment · Results and Impact · Limitations and
Learnings · Future Improvements

All sections are optional. Use `##` for sections and `###` for sub-sections —
they become the right-hand "On this page" menu. The first `# Title` line is
dropped on the site (the page already shows the title).

## 3. What renders

| Feature | Syntax |
| --- | --- |
| Maths | `$inline$`, `$$ display $$` |
| Code | fenced blocks with a language (` ```python `) — syntax highlighted with a copy button |
| Diagrams | ` ```mermaid ` blocks |
| Tables, task lists, footnotes | GitHub-flavoured Markdown |
| Images | relative paths like `docs/images/arch.png` (rewritten to GitHub raw URLs) |
| Links | relative links to repo files are rewritten to github.com |

For safety, README HTML is sanitised: scripts, iframes, inline event handlers
and styles are removed. MDX components are **not** available in READMEs.

## 4. Connect it to the portfolio

In `src/data/projects.ts`:

```ts
repo: { owner: "parikavyas", name: "my-project", branch: "main", readmePath: "README.md" },
status: "In Progress",   // or "Published"
```

Planned projects ignore `repo` until their status changes, so you can set it early.

## 5. Interactive visualization (optional)

1. Build a client component in `src/components/visualizations/<name>-demo.tsx`
   (default export; wrap it in `VisualizationFrame` so it's labelled as an
   educational simulation).
2. Register it in `registry.tsx` and `known.ts`.
3. Set `visualization: "<name>"` on the project.

Demos load lazily when scrolled into view and must use synthetic or public data.

## 6. Good practice

- No confidential client data, proprietary code or unverifiable metrics.
- Pin dependencies and include a one-command way to run it.
- Put the results you actually measured in "Evaluation", with the setup to reproduce them.
