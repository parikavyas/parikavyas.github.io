# Content guide

This guide explains how to change everything on the site **without touching
the design or components**. You only ever edit three kinds of files:

| What | Where | Format |
| --- | --- | --- |
| Personal details, links, footer text | `src/config/site.ts` | TypeScript object |
| Structured lists (work, awards, projects, topics, galleries) | `src/data/*.ts` | TypeScript objects |
| Long-form writing (articles, notes, write-ups, poems) | `content/**` | Markdown (`.md`) or MDX (`.mdx`) |
| Images, PDFs, videos | `public/**` | Files |

> **Golden rule:** if you can't stand behind a sentence publicly, leave the
> field empty. Empty fields are hidden on the live site. While you preview
> locally, they show up as dashed orange **"Placeholder"** notes telling you
> which file to edit — those notes never appear on the live site.

If you make a typo in a data file (for example a misspelled status), the
preview and the build stop with a message that names the file and the field.
Fix it and save — the page reloads.

---

## Contents

1. [Preview changes locally](#preview-changes-locally)
2. [Homepage introduction](#change-the-homepage-introduction)
3. [Profile photograph](#replace-the-profile-photograph)
4. [Social links and email](#update-social-links)
5. [Work experience](#add-or-edit-a-work-experience)
6. [Awards](#add-an-award)
7. [Education](#update-education)
8. [Projects](#add-a-project)
9. [Connect a GitHub repository](#connect-a-github-repository)
10. [New project category](#add-a-new-project-category)
11. [Wonder & Purpose topic](#add-a-wonder--purpose-topic)
12. [Publish an article](#publish-an-article)
13. [PDFs and slide decks](#add-a-pdf-or-slide-deck)
14. [Learning topic](#create-a-learning-topic)
15. [Learning notes](#add-learning-notes)
16. [Cooking photographs](#upload-cooking-photographs)
17. [Dance videos](#add-dance-videos)
18. [Creative writing](#publish-creative-writing)
19. [Dates](#update-content-dates)
20. [Commit and deploy](#commit-and-deploy-changes)

---

## Preview changes locally

You need [Node.js](https://nodejs.org) 20 or newer.

```bash
npm install        # first time only
npm run dev        # starts the site at http://localhost:3000
```

Leave it running. Every time you save a file, the browser refreshes.
Drafts and placeholder hints are **visible in this preview only**.

To see exactly what visitors will see (no drafts, no hints):

```bash
npm run build && npm start
```

## Change the homepage introduction

Open `src/config/site.ts` and edit `hero`:

```ts
hero: {
  greetingBefore: "Hello, I’m ",
  greetingHighlight: "Parika",   // shown in yellow
  greetingAfter: "!",
  paragraphs: [
    "First paragraph…",
    "Second paragraph…",
  ],
},
```

Each string in `paragraphs` is one paragraph. Bump `lastUpdated` at the
bottom of the file to today's date when you change it.

## Replace the profile photograph

1. Save your photo as `public/images/profile/parika.jpg` (portrait orientation, at least 1000px tall; JPG or WebP).
2. In `src/config/site.ts` set:

```ts
portrait: {
  src: "/images/profile/parika.jpg",
  alt: "Parika smiling, standing in front of …",   // describe the photo
},
```

The image is optimised automatically (resized, converted to modern formats).

## Update social links

In `src/config/site.ts`:

```ts
social: {
  github: "https://github.com/parikavyas",
  linkedin: "https://www.linkedin.com/in/your-handle/",
  email: "you@example.com",       // no "mailto:"
},
```

Set any of them to `null` to hide it everywhere (homepage and footer). The
footer wording lives in `footer` in the same file.

## Add or edit a Work experience

Open `src/data/experience.ts`. Each organisation is one entry. Only `id`,
`organization` and `order` are required:

```ts
{
  id: "zs-associates",
  organization: "ZS Associates",
  logo: "/images/organizations/zs.svg",   // optional, put the file in public/images/organizations/
  role: "Your title",
  location: "City, Country",
  start: "2019-07",                        // "YYYY", "YYYY-MM" or "present"
  end: "2023-06",
  summary: "One or two sentences — the front of the card.",
  themes: ["Marketing AI", "Personalization"],
  details: {                               // the back of the card; each is a list of bullet points
    problems: ["…"],
    built: ["…"],
    owned: ["…"],
    decisions: ["…"],
    leadership: ["…"],
    stakeholders: ["…"],
    impact: ["…"],                         // only results you can share publicly
  },
  relatedProjects: ["federated-learning"], // project slugs from src/data/projects.ts
  order: 3,
},
```

- **Reorder** cards by changing `order` (1 is first).
- **Long text is fine** — cards grow to fit; nothing is cut off.
- The list of tags above the cards ("areas of work") is `focusAreas` at the bottom of the same file.
- Never include confidential client names or internal metrics.

## Add an award

`src/data/awards.ts`:

```ts
{ id: "my-award", name: "Award name", year: "2024", issuer: "Organisation", description: "Optional one-liner", order: 11 },
```

## Update education

`src/data/education.ts` — fill `start`, `end`, `activities`, `interests`,
`accomplishments` (each a list of strings). Empty lists are hidden.

## Add a project

Open `src/data/projects.ts` and add an entry (copy an existing one):

```ts
{
  id: "my-project",
  slug: "my-project",                       // URL: /projects/my-project
  title: "My Project",
  summary: "One or two sentences for the card.",
  categories: ["llms-and-rag"],             // one or more category ids
  template: "applied",                      // "foundational" (Template A) or "applied" (Template B)
  implementationTypes: ["Applied AI", "End-to-End"],
  status: "In Progress",                    // "Planned" | "In Progress" | "Published"
  tags: ["RAG", "Evaluation"],
  repo: { owner: "parikavyas", name: "my-project", branch: "main", readmePath: "README.md" },
  demoUrl: "https://my-demo.vercel.app",    // optional
  visualization: null,                      // or an id from the visualization registry
  localDoc: null,                           // or "my-project.md" in content/projects/
  updated: "2026-01-31",
},
```

Rules the site enforces for you:

- **Planned** projects never show GitHub or demo buttons, show "This project
  is planned", and are hidden from search engines.
- A project in several categories (like XGBoost) still has **one** page.
- Implementation types: From Scratch, Applied AI, Research, Benchmark, End-to-End, Interactive Demo.

## Connect a GitHub repository

1. Make the project repository **public** and give it a good `README.md`
   (see `docs/PROJECT_GUIDE.md`).
2. Set `repo` on the project entry (owner, name, default branch, README path).
3. Set `status` to `"In Progress"` or `"Published"`.

The README becomes the project page at build time: headings become the
right-hand menu, maths/code/tables/Mermaid render, and relative images and
links are rewritten to point at GitHub. If GitHub can't be reached, the page
falls back to `localDoc` (if set) and shows a "view it on GitHub" note.

Updating a README does **not** redeploy the website by itself. Pages refresh
from GitHub at most once a day, or immediately on the next deploy. See
`docs/DEPLOYMENT.md → Rebuild when a project repo changes`.

## Add a new project category

Add an entry to `src/data/project-categories.ts`:

```ts
{ id: "reinforcement-learning", number: "12", title: "Reinforcement Learning",
  description: "One line.", illustration: "neural" },
```

`illustration` must be one of: regression, clusters, ensemble, neural,
vision, graph, federated, privacy, retrieval, generative, agentic. The
section, filter option and right-hand menu entry appear automatically.

## Add a Wonder & Purpose topic

`src/data/wonder-topics.ts`:

```ts
{
  slug: "ai-for-social-good",          // URL: /wonder-and-purpose/ai-for-social-good
  title: "AI for Social Good",
  description: "One line about the collection.",
  status: "published",                 // "draft" = only visible in local preview
  order: 1,
  updated: "2026-01-31",
  openQuestions: ["A question I keep coming back to?"],
  resources: [ /* papers, decks, PDFs, videos, links — see below */ ],
  related: { projects: ["differential-privacy"], learn: [] },
},
```

The eight starting topics are drafts. Publish one by switching its `status`.
Rename freely (keep the `slug` if the topic is already live, so links keep working).

## Publish an article

1. Copy `content/templates/article.mdx` to
   `content/wonder-and-purpose/<topic-slug>/<article-slug>.mdx`.
2. Fill in the front matter (title, description, date, kind) and write.
3. Preview it, then set `draft: false`.

Use `.md` for plain Markdown or `.mdx` if you want components:

| Component | Example |
| --- | --- |
| Callout | `<Callout title="Note">Text</Callout>` |
| Figure | `<Figure src="/images/x.png" alt="…" caption="…" />` |
| PDF / deck | `<Document href="/documents/paper.pdf" title="Paper" />` |
| Video | `<Video provider="youtube" id="VIDEO_ID" title="Talk" />` |
| Interactive demo | `<Visualization id="k-means" />` |

Maths: `$inline$` and `$$display$$`. Diagrams: a code block with the language `mermaid`.

## Add a PDF or slide deck

1. Export the deck as PDF and put it in `public/documents/` (e.g. `public/documents/talk.pdf`).
2. Either add it to a topic's `resources`:

```ts
{ type: "Presentation deck", title: "My talk", href: "/documents/talk.pdf", preview: true, date: "2026-01-31" },
```

…or embed it in an article with `<Document href="/documents/talk.pdf" title="My talk" />`.
Visitors get a preview on request plus a download link.

Other resource types: `"Research paper"`, `"PDF"`, `"External resource"`, `"Video"`
(YouTube links with `preview: true` become an embedded player).

## Create a learning topic

`src/data/learning-topics.ts` — copy the example entry (then delete it):

```ts
{
  slug: "diffusion-models",
  title: "Diffusion models",
  description: "What I'm trying to understand.",
  status: "Exploring",        // Curious | Exploring | Learning | Implementing | Revisiting
  draft: false,
  updated: "2026-01-31",
  resources: [{ title: "Course", url: "https://…", kind: "Course", note: "Why" }],
  readingList: [{ title: "Paper", author: "Author", state: "Reading" }],   // To read | Reading | Read
  roadmap: [{ step: "Read the DDPM paper", done: false }],
  questions: ["Why does the noise schedule matter?"],
  learned: [{ date: "2026-01-31", text: "Something small I learned." }],
  related: { projects: [], wonder: [] },
},
```

Topics with status Exploring, Learning or Implementing appear under
"Currently learning". Only record progress you've actually made.

## Add learning notes

Copy `content/templates/learning-note.md` to `content/learn/<topic-slug>/<note-slug>.md`.
Notes appear on the topic page and in "Learning notes" on `/learn`.

## Upload cooking photographs

1. Put photos in `public/images/cooking/` (JPG/WebP, ideally under 2 MB each).
2. Add an entry to `kitchenEntries` in `src/data/gallery.ts`:

```ts
{
  id: "dish-id",
  dish: "Dish name",
  image: { src: "/images/cooking/dish.jpg", alt: "Describe the photo", width: 1200, height: 1500 },
  description: "Optional",
  recipe: "Optional — shown in the enlarged view",
  story: "Optional memory",
  date: "2026-01-31",
},
```

`width`/`height` are the photo's pixel size (keeps the layout from jumping).
Delete the two `sample-dish` placeholder entries once you have real photos.

## Add dance videos

In `danceEntries` (`src/data/gallery.ts`):

```ts
{
  id: "annual-showcase-2025",
  title: "Performance name",
  style: "Dance style",
  occasion: "Occasion",
  date: "2025-04-12",
  video: { provider: "youtube", id: "VIDEO_ID" },        // the part after watch?v=
  // or image: { src: "/images/dancing/photo.jpg", alt: "…", width: 1200, height: 800 },
  description: "Optional",
  choreography: "Optional",
},
```

Videos load only when a visitor presses play (YouTube's privacy-enhanced mode).
For your own video files, use `provider: "file"` and `id: "/images/dancing/clip.mp4"`
(keep files small — large videos are better on YouTube/Vimeo).

## Publish creative writing

Copy `content/templates/writing.md` to `content/writing/<slug>.md`. `form` is
one of Poetry, Reflection, Short story, Essay, Other. Line breaks are kept
exactly as typed. Set `draft: false` to publish. Delete `content/writing/sample-piece.md`.

## Update content dates

Dates are **never** taken from "today" or from deploy time. They come from:

| Where | Field |
| --- | --- |
| Project card & page | `updated` in `src/data/projects.ts` — or, for synced READMEs, the date of the README's last commit on GitHub |
| Articles, notes, writing | `date` (first published) and `updated` (last meaningful change) in front matter |
| Topics | `updated` in the data file |
| Footer "Last updated" | the newest of all the dates above and `lastUpdated` in `src/config/site.ts` |

Bump `updated` only for meaningful content changes, not typo fixes.

## Commit and deploy changes

```bash
git checkout -b update-content          # optional but tidy
git add .
git commit -m "Add a new dish to the kitchen gallery"
git push
```

If the site is connected to Vercel (see `docs/DEPLOYMENT.md`), pushing to the
production branch publishes automatically in about a minute; pushing another
branch creates a private preview link first.
