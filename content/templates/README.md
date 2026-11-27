# Content templates

Copy a template into the right folder, rename it, fill it in.
Files in this folder are never shown on the site.

| Template | Copy to |
| --- | --- |
| `article.mdx` | `content/wonder-and-purpose/<topic-slug>/<article-slug>.mdx` |
| `learning-note.md` | `content/learn/<topic-slug>/<note-slug>.md` |
| `writing.md` | `content/writing/<piece-slug>.md` |
| `project-doc.md` (Template A) | `content/projects/<file>.md`, then set `localDoc` in `src/data/projects.ts` |
| `project-doc-applied.md` (Template B) | same as above |
| `project-readme.md` | the `README.md` of a project's own GitHub repository |

Structured entries (experience, awards, education, projects, topics, galleries)
live in `src/data/*.ts`. Each of those files has a commented template at the bottom.
