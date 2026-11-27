/**
 * Build-time GitHub README sync (server-only).
 *
 * • README text comes from raw.githubusercontent.com (no token needed).
 * • The README's last commit date comes from the GitHub API. Set GITHUB_TOKEN
 *   (a read-only, public-repo token) in Vercel to avoid anonymous rate limits.
 *   The token is only read on the server and never sent to the browser.
 * • Results are cached and refreshed at most once per README_REVALIDATE_SECONDS
 *   (default: daily). A redeploy always refetches. See docs/DEPLOYMENT.md.
 * • Any failure returns null so the page can fall back gracefully.
 */
import "server-only";
import type { Project } from "@/lib/schemas";

export const README_REVALIDATE = Number(process.env.README_REVALIDATE_SECONDS ?? 86400);

export interface ReadmeResult {
  markdown: string;
  /** ISO date of the last commit that touched the README, when available. */
  lastCommit: string | null;
  url: string;
}

type Repo = NonNullable<Project["repo"]>;

const TIMEOUT_MS = 10_000;

function headers(): HeadersInit {
  const h: Record<string, string> = {
    "User-Agent": "portfolio-site-build",
    Accept: "application/vnd.github+json",
  };
  if (process.env.GITHUB_TOKEN) h.Authorization = `Bearer ${process.env.GITHUB_TOKEN}`;
  return h;
}

export function repoUrl(repo: Repo) {
  return `https://github.com/${repo.owner}/${repo.name}`;
}

async function fetchReadmeText(repo: Repo) {
  const url = `https://raw.githubusercontent.com/${repo.owner}/${repo.name}/${repo.branch}/${repo.readmePath}`;
  const res = await fetch(url, {
    next: { revalidate: README_REVALIDATE, tags: [`readme:${repo.owner}/${repo.name}`] },
    signal: AbortSignal.timeout(TIMEOUT_MS),
  });
  if (!res.ok) return null;
  return res.text();
}

async function fetchLastCommit(repo: Repo) {
  try {
    const url = `https://api.github.com/repos/${repo.owner}/${repo.name}/commits?path=${encodeURIComponent(
      repo.readmePath,
    )}&sha=${encodeURIComponent(repo.branch)}&per_page=1`;
    const res = await fetch(url, {
      headers: headers(),
      next: { revalidate: README_REVALIDATE },
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
    if (!res.ok) return null;
    const data = (await res.json()) as { commit?: { committer?: { date?: string } } }[];
    return data[0]?.commit?.committer?.date ?? null;
  } catch {
    return null;
  }
}

export async function getReadme(repo: Repo): Promise<ReadmeResult | null> {
  try {
    const markdown = await fetchReadmeText(repo);
    if (markdown === null || markdown.trim() === "") return null;
    const lastCommit = await fetchLastCommit(repo);
    return { markdown, lastCommit, url: `${repoUrl(repo)}/blob/${repo.branch}/${repo.readmePath}` };
  } catch (err) {
    console.warn(`[github] Could not load README for ${repo.owner}/${repo.name}:`, (err as Error).message);
    return null;
  }
}
