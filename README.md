# Parika Vyas — personal portfolio

A personal universe: what I do, build, learn and wonder about. Six pages —
**Hello, Work, Projects, Wonder & Purpose, Learn, Beyond the Desk** — built
with Next.js (App Router), TypeScript, Tailwind CSS and MDX, deployed on Vercel.

## Run it locally

```bash
npm install
npm run dev          # http://localhost:3000 (drafts + placeholder hints visible)
npm run check        # lint + type check + production build
```

Requires Node.js 20+.

## Where things live

```
src/
  app/                 routes (one folder per page; [slug] folders are nested pages)
  components/
    layout/            header, footer, page header, sections, doc layout
    navigation/        scrollspy, reading progress, breadcrumbs
    cards/             experience (flip), award, education, project, topic cards
    illustrations/     original procedural SVG art
    content/           Markdown enhancer, portrait, PDF/video embeds, MDX components
    visualizations/    interactive demos + registry
    projects/          project library explorer (filters)
    gallery/           kitchen gallery + lightbox
    ui/                buttons, tags, badges, placeholders
  config/              site.ts (name, intro, links, footer) and navigation.ts
  data/                structured content: experience, awards, education, projects, topics, galleries
  lib/                 schemas, Markdown pipeline, GitHub README sync, content loading, dates, SEO
content/
  projects/            local fallback write-ups for projects
  wonder-and-purpose/  articles by topic
  learn/               learning notes by topic
  writing/             creative writing
  templates/           copy-and-fill templates for every content type
public/                images, documents, icons
docs/                  CONTENT_GUIDE, DESIGN_SYSTEM, PROJECT_GUIDE, DEPLOYMENT
```

## Guides

- [Content guide](docs/CONTENT_GUIDE.md) — edit anything without touching components
- [Project guide](docs/PROJECT_GUIDE.md) — how to structure each project repository
- [Design system](docs/DESIGN_SYSTEM.md) — colours, type, cards, illustrations
- [Deployment](docs/DEPLOYMENT.md) — GitHub + Vercel, rebuilds, environment variables
