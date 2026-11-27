# Deployment — GitHub + Vercel (free)

The site is a static-first Next.js app. It needs no database, paid API or
server of its own and fits Vercel's free Hobby plan.

## 1. One-time setup

1. Push this repository to GitHub (it lives in `parikavyas/parikavyas.github.io`).
2. Go to <https://vercel.com/new>, sign in with GitHub, and **Import** the repository.
3. Framework preset: **Next.js** (auto-detected). Build command `next build`,
   output: default. Root directory: the repository root.
4. Environment variables (Project → Settings → Environment Variables):

| Name | Value | Why |
| --- | --- | --- |
| `NEXT_PUBLIC_SITE_URL` | `https://<your-project>.vercel.app` | Canonical URLs, sitemap, social previews |
| `GITHUB_TOKEN` *(optional)* | fine-grained token, **public repositories read-only** | Avoids GitHub API rate limits when reading README commit dates. Server-only; never exposed to browsers |
| `README_REVALIDATE_SECONDS` *(optional)* | `86400` | How often README pages refresh (default: daily) |
| `SHOW_DRAFTS` *(Preview environment only, optional)* | `true` | Show drafts on preview links, never in Production |

5. Deploy. Your site is live at `https://<your-project>.vercel.app`.
   You can rename the subdomain under Project → Settings → Domains.

## 2. Everyday publishing

- Push to the production branch → production deploy (about a minute).
- Push any other branch or open a pull request → a private **preview** URL.
  Previews are automatically `noindex` (see `src/app/robots.ts`).

## 3. Rebuild when a project repo changes

Updating a project's README does **not** automatically redeploy the portfolio.
Project pages refresh from GitHub on their own at most once a day
(`README_REVALIDATE_SECONDS`). To publish a README change immediately:

**Option A — manual:** Vercel dashboard → Deployments → ⋯ → *Redeploy*.

**Option B — deploy hook (automatic):**

1. Vercel → Project → Settings → Git → **Deploy Hooks** → create one (e.g. "readme-sync") and copy the URL.
2. In each project repository: Settings → Secrets and variables → Actions → add `PORTFOLIO_DEPLOY_HOOK` with that URL.
3. Add `.github/workflows/notify-portfolio.yml` to the project repository:

```yaml
name: Refresh portfolio
on:
  push:
    branches: [main]
    paths: ["README.md", "docs/**"]
jobs:
  ping:
    runs-on: ubuntu-latest
    steps:
      - run: curl -fsS -X POST "${{ secrets.PORTFOLIO_DEPLOY_HOOK }}"
```

Keep the hook URL secret — anyone with it can trigger builds.

## 4. Checks before deploying

```bash
npm run lint
npm run typecheck
npm run build
```

## 5. About GitHub Pages

This repository previously hosted a Jekyll site on GitHub Pages. The new site
is meant for Vercel. If GitHub Pages is still enabled for this repository
(Settings → Pages), it will no longer show a working site; you can turn it off
there. Other repositories' Pages sites (for example the What's Flying Above
Me demo at `/what-plane/`) are separate and keep working.

## 6. Limits to keep in mind

- No paid services are used. Images are optimised by Vercel's built-in image
  optimisation (the Hobby plan includes a monthly allowance; keep photos
  reasonably sized).
- Large videos belong on YouTube/Vimeo, not in `public/`.
