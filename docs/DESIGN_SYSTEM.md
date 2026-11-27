# Design system

**Concept:** a personal universe — each page a different dimension of the same
person. Bold personal identity (hero), clean documentation structure (inner
pages), warm editorial reading (long-form), and a quiet cosmic / parallel-world
geometry layer.

## Colour tokens

Defined once in `src/app/globals.css` (`:root`) and exposed to Tailwind via `@theme`.

| Token | Hex | Tailwind | Use |
| --- | --- | --- | --- |
| `--bg` | `#080F20` | `bg-bg` | Page background |
| `--bg-secondary` | `#001C40` | `bg-bg-secondary` | Duke-blue depth, gradients, footer |
| `--surface` | `#111E33` | `bg-surface` | Cards |
| `--surface-elevated` | `#192C48` | `bg-surface-elevated` | Hover / flipped card back |
| `--accent` | `#E8FF47` | `text-accent`, `stroke-accent` | Lemon yellow — sparingly |
| `--accent-blue` | `#4169E1` | `text-blue`, `stroke-blue` | Secondary accent, illustration lines |
| `--accent-blue-soft` | `#7B9CF5` | `text-blue-soft` | Blue text on navy (meets contrast) |
| `--text` | `#F5F5F0` | `text-text` | Primary text |
| `--text-muted` | `#A5B2C8` | `text-muted` | Secondary text |
| `--border` | `#263650` | `border-border` | Hairlines |
| `--warm` | `#F2B880` | `text-warm` | Beyond the Desk warmth, dev placeholders |

**Yellow is reserved for:** active navigation and its underline, section
numbers, important links, primary buttons, the active scrollspy marker,
reading progress, small decorative points, and text selection. Never use it
for paragraphs of body text.

Contrast (WCAG AA): text `#F5F5F0` on `#080F20` ≈ 17:1; muted `#A5B2C8` on
`#111E33` ≈ 8:1; yellow `#E8FF47` on `#080F20` ≈ 17:1; blue-soft on cards ≈ 6.3:1.
Pure `#4169E1` is used for lines and fills, not small text.

## Typography

| Role | Font | Where |
| --- | --- | --- |
| Display | Space Grotesk (`font-display`) | Hero, page and section headings, card titles |
| Body | Plus Jakarta Sans (`font-sans`, default) | Paragraphs, navigation, UI |
| Mono | Geist Mono (`font-mono`) | Code, dates, metadata, section numbers, labels |

Fonts are self-hosted at build time by `next/font` (no runtime requests to Google).
The hero uses `clamp(3rem, 9vw, 6.75rem)`; long-form prose is capped at
72 characters per line (`.prose`), creative writing at 62 (`.prose-writing`).

## Spacing & layout

- Content width: `max-w-7xl` with 16/24/32px side gutters (`Container`).
- Doc pages: content + 13.5rem right scrollspy (`DocLayout`), single column below 1024px.
- Section rhythm: `py-12` per `Section`, numbered with a mono yellow index.
- Two-column card grids on desktop (`md:grid-cols-2`), one column on mobile.

## Cards

`src/components/ui/card.tsx` is the base: navy surface, 1px border,
`--radius-card` (14px), 24px padding, minimal shadow (none), hover lifts the
border toward yellow (`hover:border-accent/45`). Specialised cards:

| Component | File |
| --- | --- |
| Experience flip card | `components/cards/experience-card.tsx` |
| Award card | `components/cards/award-card.tsx` |
| Education card | `components/cards/education-card.tsx` |
| Project card | `components/cards/project-card.tsx` |
| Topic card (Wonder & Learn) | `components/cards/topic-card.tsx` |

## Illustration language

Original, procedural SVG — thin blue and yellow lines on navy, seeded so the
server and browser draw the same thing. All decorative (`aria-hidden`).

| Primitive (`components/illustrations/primitives.tsx`) | Idea |
| --- | --- |
| `PerspectiveGrid` | A floor receding to a horizon |
| `OrbitalPaths` | Tilted orbits with small bodies |
| `Constellation` | Stars joined by a path |
| `FloatingStructure` | Isometric floating rooms and stairs (impossible architecture) |
| `NodeNetwork` | Graph of nodes and edges |
| `DimensionalFrame` | Nested receding frames (doorways) |
| `Portal` | Concentric rotating geometry around an origin |
| `OrganicForms` | Warm, soft blobs and waves |

Page motifs (`page-motifs.tsx`): Hello → portal/grid · Work → pathways ·
Projects → node universe · Wonder → constellations · Learn → orbits ·
Beyond → organic forms. Category diagrams (`category-illustrations.tsx`)
give each project category its own technical sketch; `ProjectGlyph` makes a
unique strip per project from its slug.

## Motion

- Motion (`motion/react`) only for the one-time hero reveal.
- CSS for flip cards and very slow illustration drift/twinkle.
- Everything respects `prefers-reduced-motion` (flip becomes a crossfade,
  illustrations stop, demos jump straight to the result).
- No constantly moving backgrounds, no neon glows.

## Component usage

- New page section → `<Section id number title>` inside `<DocLayout toc={…}>`; add the same `id` to `toc`.
- Placeholder text → `<Placeholder hint publicText>` (dev hint, neutral public text).
- External link styled as a button → `<LinkButton href>`; it opens in a new tab.
- Status → `<StatusBadge>` (projects) or `<LearningStatusBadge>` (learning).
